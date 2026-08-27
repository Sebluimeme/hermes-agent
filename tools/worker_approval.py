"""File-based approval bridge between Kanban worker processes and the gateway.

Background (see Kanban task t_f98bc92d): the existing dangerous-approval flow
(``tools/approval.py``'s ``_gateway_queues`` / ``_gateway_notify_cbs``, driven
by Telegram's inline Approve/Refuse buttons) lives entirely in the gateway
process's memory and is registered per agent turn
(``register_gateway_notify`` / ``unregister_gateway_notify`` in
``gateway/run.py``). A Kanban worker is a *separate OS process* spawned by
the dispatcher — it can neither register a notify callback nor read
``_gateway_queues``, so a protected-instruction-write approval request from
a worker was previously delivered to nobody and simply expired (observed:
19 requests, 19 timeouts, zero human answers).

This module is the missing cross-process transport: a small durable JSON
file per request under ``~/.hermes/state/worker-approvals/``.

* The worker (protected instruction writes *and* the generic command/tool
  approval gate) calls :func:`request_decision`, which writes (or reattaches
  to) a request file and blocks polling it.
* The gateway (``gateway/kanban_watchers.py``'s
  ``_kanban_worker_approval_bridge``) polls :func:`list_pending_requests`,
  claims one with :func:`claim_for_dispatch`, turns it into a REAL Telegram/
  Discord/etc. button prompt via the adapter's existing
  ``send_exec_approval`` + ``resolve_gateway_approval`` machinery, and calls
  :func:`write_decision` with the human's answer.

Contract, matching the task's explicit constraints:

* **Durable dedup** — the request id is derived deterministically from
  ``task_id`` + the sorted target path set, so a worker retry (crash,
  respawn, re-attempt) reattaches to the *same* in-flight request instead of
  firing a second solicitation. Process-memory dedup is not enough; this is
  why the id is a stable hash, not a random uuid.
* **Silence is not consent** — a worker-side timeout, a missing/unreadable
  file, or no reachable delivery channel all resolve to a refusal. Nothing
  in this module can produce an implicit approval; only a human clicking a
  real button (relayed through ``write_decision``) can.
* **Transport failures never decide** — if the gateway can't find a
  subscriber for the task, or the send itself fails, it leaves the request
  file untouched rather than writing a decision; the worker's own timeout
  closes it out as a refusal. The gateway only ever writes a real human
  choice or its own bridge-side "timeout".
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# A human has to notice a phone buzz, not a terminal prompt — 60s (the
# generic ``approvals.timeout``) is far too short. Default to 15 minutes;
# configurable via env or ``kanban.worker_approval_timeout_seconds``.
DEFAULT_TIMEOUT_SECONDS = 900
POLL_INTERVAL_SECONDS = 2.0
# Safety ceiling: a pending request older than this is treated as orphaned
# (e.g. the gateway was down for a day) rather than reattached forever.
_STALE_PENDING_SECONDS = 24 * 3600


def _state_dir() -> Path:
    # Approval transport is a control-plane mailbox shared by the gateway and
    # every worker profile.  A worker runs with
    # HERMES_HOME=<root>/profiles/<spark|coder|...>; using that path here made
    # it write into a mailbox the default-profile gateway never polled.  Resolve
    # the installation root explicitly, the same way the shared Kanban DB does.
    from hermes_constants import get_default_hermes_root

    base = get_default_hermes_root()
    d = base / "state" / "worker-approvals"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _request_id(task_id: str, targets: list[str]) -> str:
    """Stable id: same task + same approval target set == same demand.

    Deliberately NOT a uuid — a fresh random id per call is exactly what
    would let a worker retry produce a second Telegram prompt for a demand
    that already has one in flight.
    """
    key = task_id.strip() + "|" + "|".join(sorted({p.strip() for p in targets}))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]


def _request_path(request_id: str) -> Path:
    return _state_dir() / f"{request_id}.json"


def _dispatch_marker(request_id: str) -> Path:
    return _state_dir() / f"{request_id}.dispatched"


def _atomic_write(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _configured_timeout() -> int:
    raw = os.environ.get("HERMES_WORKER_APPROVAL_TIMEOUT_SECONDS")
    if raw:
        try:
            return max(int(raw), 30)
        except ValueError:
            pass
    try:
        from hermes_cli.config import load_config

        cfg = (load_config() or {}).get("kanban") or {}
        val = cfg.get("worker_approval_timeout_seconds")
        if val:
            return max(int(val), 30)
    except Exception:
        pass
    return DEFAULT_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Worker-side API
# ---------------------------------------------------------------------------

def request_decision(
    *,
    task_id: str,
    title: str,
    description: str,
    paths: Optional[list[str]] = None,
    request_key: Optional[str] = None,
    command: Optional[str] = None,
    pattern_keys: Optional[list[str]] = None,
    allow_session: bool = False,
    allow_permanent: bool = False,
    timeout_seconds: Optional[int] = None,
) -> dict:
    """Ask a human for a worker action and block for the decision.

    Returns ``{"resolved": bool, "choice": str|None, "reason": str|None}``.
    ``resolved`` is True only when a real decision (human or explicit
    bridge-side closure) was recorded; ``choice`` is one of
    ``"once"|"session"|"always"|"deny"`` when resolved. Any failure to reach
    a decision — including "resolved" but the file expired — must be treated
    by the caller as a refusal; this function never returns an implicit
    approval.
    """
    clean_paths = sorted({str(path).strip() for path in (paths or []) if str(path).strip()})
    clean_key = str(request_key or "").strip()
    if not task_id or (not clean_paths and not clean_key):
        return {
            "resolved": False,
            "choice": None,
            "reason": (
                "missing task_id/paths"
                if clean_paths or paths is not None
                else "missing task_id/approval target"
            ),
        }

    timeout = timeout_seconds if timeout_seconds is not None else _configured_timeout()
    targets = clean_paths or [clean_key]
    request_id = _request_id(task_id, targets)
    path = _request_path(request_id)
    now = time.time()

    existing = _read(path)
    reattach = (
        existing is not None
        and existing.get("decision") is None
        and (now - float(existing.get("created_at", 0) or 0)) < _STALE_PENDING_SECONDS
    )

    if reattach:
        expires_at = float(existing.get("expires_at") or (now + timeout))
    else:
        # Either no prior request, a resolved one (a fresh ask needs fresh
        # consent), or an orphaned one past the safety ceiling. Clear any
        # leftover dispatch marker so this new demand can be delivered —
        # otherwise the deterministic id would look "already claimed"
        # forever for what is, from the operator's point of view, a new ask.
        try:
            _dispatch_marker(request_id).unlink()
        except FileNotFoundError:
            pass
        expires_at = now + max(timeout, 1)
        payload = {
            "request_id": request_id,
            "task_id": task_id,
            "board": os.environ.get("HERMES_KANBAN_BOARD") or "default",
            "title": (title or "")[:200],
            "description": description,
            "paths": clean_paths,
            "request_key": clean_key or None,
            "command": str(command or "").strip() or None,
            "pattern_keys": sorted({
                str(value).strip()
                for value in (pattern_keys or [])
                if str(value).strip()
            }),
            "allow_session": bool(allow_session),
            "allow_permanent": bool(allow_permanent),
            "created_at": now,
            "expires_at": expires_at,
            "worker_pid": os.getpid(),
            "decision": None,
            "reason": None,
            "decided_by": None,
            "decided_at": None,
        }
        try:
            _atomic_write(path, payload)
        except OSError as exc:
            logger.warning("worker_approval: could not write request %s: %s", request_id, exc)
            return {"resolved": False, "choice": None, "reason": f"local write failed: {exc}"}

    while True:
        current = _read(path)
        if current is None:
            return {"resolved": False, "choice": None, "reason": "request file disappeared"}
        decision = current.get("decision")
        if decision is not None:
            return {
                "resolved": decision != "timeout",
                "choice": decision if decision != "timeout" else None,
                "reason": current.get("reason"),
            }
        if time.time() >= expires_at:
            _close_as_timeout(path)
            return {
                "resolved": False,
                "choice": None,
                "reason": "no response within the approval window; silence is not consent",
            }
        time.sleep(POLL_INTERVAL_SECONDS)


def _close_as_timeout(path: Path) -> None:
    """Best-effort local closure so a late gateway write can't flip this to
    an approval after the caller has already treated it as refused."""
    current = _read(path)
    if current is None or current.get("decision") is not None:
        return
    current["decision"] = "timeout"
    current["reason"] = "worker-side timeout: no human response in time"
    current["decided_at"] = time.time()
    try:
        _atomic_write(path, current)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Gateway-side API
# ---------------------------------------------------------------------------

def list_pending_requests() -> list[dict]:
    """Return requests awaiting dispatch: undecided, unclaimed, unexpired."""
    out = []
    now = time.time()
    for p in sorted(_state_dir().glob("*.json")):
        data = _read(p)
        if not data:
            continue
        if data.get("decision") is not None:
            continue
        if now >= float(data.get("expires_at", 0) or 0):
            continue  # let the worker's own timeout close this one out
        request_id = data.get("request_id")
        if not request_id or _dispatch_marker(request_id).exists():
            continue
        out.append(data)
    return out


def claim_for_dispatch(request_id: str) -> bool:
    """Atomically claim a request for delivery. False if already claimed.

    Cross-process/cross-tick dedup for the "one solicitation per demand"
    contract: only the first caller to create the marker file wins, even if
    two gateway ticks (or, in a multi-gateway deployment, two processes)
    observe the same pending request in the same instant.
    """
    marker = _dispatch_marker(request_id)
    try:
        fd = os.open(str(marker), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except OSError as exc:
        logger.warning("worker_approval: claim failed for %s: %s", request_id, exc)
        return False


def write_decision(
    request_id: str,
    choice: str,
    reason: Optional[str] = None,
    decided_by: Optional[str] = None,
) -> bool:
    """Record the human's (or bridge's own timeout) decision. Idempotent."""
    path = _request_path(request_id)
    current = _read(path)
    if current is None or current.get("decision") is not None:
        return False
    current["decision"] = choice
    current["reason"] = reason
    current["decided_by"] = decided_by
    current["decided_at"] = time.time()
    try:
        _atomic_write(path, current)
        return True
    except OSError:
        return False
