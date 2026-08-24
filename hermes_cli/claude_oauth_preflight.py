"""Deterministic, credential-free Claude 2 OAuth canary for Kanban dispatch.

The probe intentionally uses the same isolated Claude CLI profile as the Claude 2
lane. It never reads, prints, or changes credentials: the CLI resolves OAuth from
``CLAUDE_CONFIG_DIR=/home/seb/.claude2`` itself.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

CLAUDE2_CONFIG_DIR = Path("/home/seb/.claude2")
CANARY_PROMPT = "Reply exactly PONG"
CANARY_TIMEOUT_SECONDS = 30
OAUTH_FAILURE_MARKERS = (
    "oauth session expired and could not be refreshed",
    "oauth token expired",
    "not logged in",
    "please run /login",
    "authentication failed and could not be refreshed",
)


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    reconnect_required: bool
    reason: str


def credentials_fingerprint() -> Optional[float]:
    """Content-free signal for "OAuth reconnected" (t_b0bc4445 LOT 2).

    Returns the mtime of the isolated Claude 2 credentials file, or ``None``
    when it doesn't exist. Never opens/reads the file's contents -- a
    changed mtime is enough to tell the dispatcher's transition fingerprint
    (``kanban_db.claude2_oauth_dispatch_guard``) that a human re-logged in
    (or logged in for the first time), which is one of the four events that
    justifies spending a fresh paid canary. Falls back to ``.claude.json``'s
    mtime when the dedicated credentials file is absent (older profiles).
    """
    for name in (".credentials.json", ".claude.json"):
        candidate = CLAUDE2_CONFIG_DIR / name
        try:
            return candidate.stat().st_mtime
        except OSError:
            continue
    return None


def canary_command() -> list[str]:
    """Exact no-secret command used both for dispatch and manual recovery."""
    return [
        "claude", "-p", "--output-format", "json", "--no-chrome",
        "--model", "claude-sonnet-4-6", CANARY_PROMPT,
    ]


def probe_claude2_oauth(
    *,
    run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    environ: Optional[Mapping[str, str]] = None,
) -> ProbeResult:
    """Probe the real Claude 2 OAuth store without exposing credentials.

    A confirmed OAuth failure returns ``reconnect_required=True``. Any other
    failed canary remains fail-closed for dispatch, but is described separately
    so an operator is not told to re-login for an unrelated local failure.
    """
    if not (CLAUDE2_CONFIG_DIR / ".claude.json").is_file():
        return ProbeResult(False, True, "profil OAuth Claude 2 introuvable")

    env = dict(os.environ if environ is None else environ)
    env["CLAUDE_CONFIG_DIR"] = str(CLAUDE2_CONFIG_DIR)
    try:
        completed = run_fn(
            canary_command(),
            cwd=str(Path.home()),
            env=env,
            text=True,
            capture_output=True,
            timeout=CANARY_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ProbeResult(False, False, "canari OAuth Claude 2 expiré après 30 s")
    except OSError as exc:
        return ProbeResult(False, False, f"canari OAuth Claude 2 impossible: {type(exc).__name__}")

    if completed.returncode == 0:
        return ProbeResult(True, False, "canari PONG réussi")
    output = "\n".join((completed.stdout or "", completed.stderr or "")).casefold()
    if any(marker in output for marker in OAUTH_FAILURE_MARKERS):
        return ProbeResult(False, True, "OAuth Claude 2 expiré ou déconnecté")
    return ProbeResult(False, False, f"canari OAuth Claude 2 échoué (code {completed.returncode})")


def dispatch_block_reason(result: ProbeResult) -> str:
    """Human-safe reason persisted on a blocked card; never includes CLI output."""
    if result.reconnect_required:
        return "reconnexion Claude 2 requise — OAuth expiré ou déconnecté; aucun worker n’a été lancé."
    return f"vérification OAuth Claude 2 impossible — {result.reason}; aucun worker n’a été lancé."
