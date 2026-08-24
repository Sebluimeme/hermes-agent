"""Read-only aggregated diagnostics across Kanban, sessions, and context handoffs.

Motivation (2026-08-22 audit, "diagnostic agrégé lecture seule natif"): the
non-interactive heredoc/inline-script guard correctly blocks arbitrary Python
in unattended sessions, but that also blocked several legitimate read-only
audits (counting tasks by status, active sessions, context-handoff events).
Rather than loosen that guard, this module gives a narrow, strictly-typed
native path for exactly those counters, so a worker never needs inline
Python or a heredoc to answer "how many / since when / which state".

Contract:

* Inputs are named, typed parameters only (``since``/``until``/``board``) —
  there is no raw SQL / free-query parameter anywhere in this module. A
  caller that wants a different aggregate must add a new named, reviewed
  field here rather than pass free-form text through
  (:func:`run_aggregate_diagnostics` raises :class:`DiagnosticsError` if
  handed anything resembling one, see ``_reject_free_form_query``).
* Output is non-sensitive aggregates only: counters and time windows. No
  task body/comment text, no session transcript, no secret material is ever
  read or returned.
* Every function here only reads (SQL ``SELECT``, JSON file reads). Nothing
  in this module executes ``INSERT``/``UPDATE``/``DELETE`` or writes to any
  file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home


class DiagnosticsError(ValueError):
    """Raised for an invalid or unsupported diagnostic request.

    This includes malformed ``since``/``until`` values and any attempt to
    pass a free-form query through a keyword this module does not define
    (see ``_reject_free_form_query``) — both fail closed rather than
    guessing at intent.
    """


# Keywords that would signal an attempt to smuggle a free-form query through
# ``**kwargs``-style callers (e.g. a wrapper CLI naively forwarding flags).
# Rejected unconditionally: this module has no query language to accept one.
_FREE_FORM_QUERY_KEYS = frozenset({"sql", "query", "where", "raw", "filter_expr"})


def _reject_free_form_query(extra: dict[str, Any]) -> None:
    offending = sorted(_FREE_FORM_QUERY_KEYS & extra.keys())
    if offending:
        raise DiagnosticsError(
            "aggregate_diagnostics has no free-form query surface; "
            f"unsupported argument(s): {', '.join(offending)}"
        )
    if extra:
        # Any other unknown keyword is also rejected (fail closed) rather
        # than silently ignored, so a typo never looks like it "worked".
        raise DiagnosticsError(
            f"unsupported argument(s): {', '.join(sorted(extra.keys()))}"
        )


def _parse_window(
    since: Optional[Any], until: Optional[Any]
) -> tuple[Optional[int], Optional[int]]:
    """Normalize ``since``/``until`` to epoch seconds, or raise."""
    from hermes_cli.kanban_db import _to_epoch

    since_ts = None
    until_ts = None
    if since is not None:
        since_ts = _to_epoch(since)
        if since_ts is None:
            raise DiagnosticsError(f"invalid --since value: {since!r}")
    if until is not None:
        until_ts = _to_epoch(until)
        if until_ts is None:
            raise DiagnosticsError(f"invalid --until value: {until!r}")
    if since_ts is not None and until_ts is not None and since_ts > until_ts:
        raise DiagnosticsError("--since must not be after --until")
    return since_ts, until_ts


def kanban_counts_by_status(
    conn,
    *,
    since_ts: Optional[int] = None,
    until_ts: Optional[int] = None,
) -> dict[str, int]:
    """Task counts by status within ``[since_ts, until_ts]`` on ``created_at``.

    Read-only: a single ``SELECT ... GROUP BY status``. Archived tasks are
    excluded (matches ``board_stats``'s existing convention).
    """
    query = "SELECT status, COUNT(*) AS n FROM tasks WHERE status != 'archived'"
    params: list[Any] = []
    if since_ts is not None:
        query += " AND created_at >= ?"
        params.append(since_ts)
    if until_ts is not None:
        query += " AND created_at <= ?"
        params.append(until_ts)
    query += " GROUP BY status"
    counts: dict[str, int] = {}
    for row in conn.execute(query, params):
        counts[row["status"]] = int(row["n"])
    return counts


def session_counts(
    *, since_ts: Optional[int] = None, until_ts: Optional[int] = None
) -> dict[str, Any]:
    """Active session lease counts, total and by surface.

    Reads the same on-disk registry as ``hermes_cli.active_sessions``
    (``runtime/active_sessions.json``) and reuses its dead-lease pruning so
    a crashed process's stale entry is never counted as "active". No file
    is written back here — pruning only filters the in-memory list.
    """
    from hermes_cli.active_sessions import _prune_dead, _read_entries, _state_path

    entries = _prune_dead(_read_entries(_state_path()))

    def _in_window(entry: dict[str, Any]) -> bool:
        if since_ts is None and until_ts is None:
            return True
        started = entry.get("started_at")
        try:
            started_f = float(started)
        except (TypeError, ValueError):
            return False
        if since_ts is not None and started_f < since_ts:
            return False
        if until_ts is not None and started_f > until_ts:
            return False
        return True

    entries = [e for e in entries if _in_window(e)]
    by_surface: dict[str, int] = {}
    for entry in entries:
        surface = str(entry.get("surface") or "unknown")
        by_surface[surface] = by_surface.get(surface, 0) + 1
    return {"total": len(entries), "by_surface": by_surface}


def _context_handoff_log_path() -> Path:
    return Path(get_hermes_home()) / "state" / "context-handoff" / "events.jsonl"


def context_handoff_counts(
    *, since_ts: Optional[int] = None, until_ts: Optional[int] = None
) -> dict[str, Any]:
    """Context compaction/handoff event counts, total and by event type.

    Reads ``state/context-handoff/events.jsonl`` (written by the
    context-handoff-guard plugin): one compact JSON object per line with
    ``at`` (epoch seconds), ``session_id`` (opaque id, no content),
    ``event``, and non-sensitive numeric fields. Malformed lines are
    skipped rather than raising, since this is a diagnostic reading a log
    another process appends to. Never writes to the log.
    """
    total = 0
    by_event: dict[str, int] = {}
    path = _context_handoff_log_path()
    if not path.exists():
        return {"total": 0, "by_event": {}}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            at = row.get("at")
            try:
                at_f = float(at)
            except (TypeError, ValueError):
                continue
            if since_ts is not None and at_f < since_ts:
                continue
            if until_ts is not None and at_f > until_ts:
                continue
            event = str(row.get("event") or "unknown")
            by_event[event] = by_event.get(event, 0) + 1
            total += 1
    return {"total": total, "by_event": by_event}


def run_aggregate_diagnostics(
    *,
    since: Optional[Any] = None,
    until: Optional[Any] = None,
    board: Optional[str] = None,
    include_sessions: bool = True,
    include_context: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    """Compute the strictly-typed, read-only aggregated diagnostic snapshot.

    ``since``/``until`` accept an epoch-seconds int/float or an ISO-8601
    string (delegated to ``kanban_db._to_epoch``, the same parser the rest
    of the kanban CLI already trusts for time filters). ``board`` selects a
    kanban board by slug; ``None`` uses the current/default board.

    Any keyword outside this explicit signature is rejected — see
    ``_reject_free_form_query`` — so this stays a fixed-shape aggregate, not
    a query interface.
    """
    _reject_free_form_query(extra)
    since_ts, until_ts = _parse_window(since, until)

    from hermes_cli import kanban_db as kb

    with kb.connect_closing(board=board) as conn:
        kanban = kanban_counts_by_status(conn, since_ts=since_ts, until_ts=until_ts)

    result: dict[str, Any] = {
        "generated_at": int(time.time()),
        "window": {"since": since_ts, "until": until_ts},
        "kanban": kanban,
    }
    if include_sessions:
        result["sessions"] = session_counts(since_ts=since_ts, until_ts=until_ts)
    if include_context:
        result["context_handoffs"] = context_handoff_counts(
            since_ts=since_ts, until_ts=until_ts
        )
    return result
