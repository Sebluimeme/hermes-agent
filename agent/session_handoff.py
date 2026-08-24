"""Deterministic session-relay mechanism (LOT 3).

Rule this module encodes (Sébastien, 2026-08-21): Hermes must never ask the
user to open a new Topic/thread just because a conversation grew too large or
a durable transition occurred. Instead, when a context threshold is crossed
or a durable transition happens, Hermes must:

  1. write a short checkpoint into the relevant project's TASKS.md
     (see ``write_relay_checkpoint``),
  2. preserve any Kanban workers already running for the session (this module
     never touches worker/dispatch state — that is an invariant, not a
     runtime check),
  3. perform an in-place technical resume ("summary") or, only when a summary
     cannot repair the situation, a full reset — in BOTH cases inside the
     exact same Topic/session routing identity. No reset may create a new
     *visible* conversation or drop the task the user was in the middle of.

This module is pure/deterministic and side-effect-free except for
``write_relay_checkpoint`` (a single idempotent file append) and
``perform_session_relay`` (which only calls the caller-supplied callbacks —
it never imports gateway/session/kanban modules itself, so it cannot
accidentally interrupt a worker or rotate a topic binding on its own).

Callers (gateway session boundary, automatic compaction hook) are
responsible for supplying ``compress_fn``/``reset_fn`` and the real
``tasks_md_path``; this module only supplies the decision table and the
checkpoint/identity-preservation plumbing so the decision itself is unit
testable without spinning up the gateway.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Optional

RelayAction = Literal["none", "summary", "reset"]

# Transition reasons that a mid-conversation compaction/"résumé" can NEVER
# repair, because they invalidate assumptions the running conversation state
# depends on (a different model's tokenizer/tool schema, a different
# provider's auth/session semantics, or state Hermes already knows is
# corrupted). These always force a full reset, regardless of how small the
# current context is.
DURABLE_TRANSITIONS: frozenset[str] = frozenset(
    {
        "model_change",
        "provider_change",
        "corrupted_context",
    }
)

# Transition reasons that are "durable" (mark a real boundary worth a
# checkpoint) but that an in-place technical summary can fully absorb — no
# need to blow away the conversation/topic just because the user explicitly
# asked to start fresh content-wise.
SUMMARY_ELIGIBLE_TRANSITIONS: frozenset[str] = frozenset(
    {
        "explicit_new_topic_request",
        "idle_boundary",
    }
)


def decide_relay_action(
    *,
    context_tokens: int,
    threshold_tokens: int,
    transition_reason: Optional[str] = None,
) -> RelayAction:
    """Pure decision table: summary vs reset vs no-op.

    Rules (in priority order):
      1. A durable transition in ``DURABLE_TRANSITIONS`` always forces
         "reset" — no amount of headroom makes a stale model/provider/
         corrupted context safe to keep summarizing.
      2. Otherwise, if the context threshold is crossed
         (``context_tokens >= threshold_tokens``), a "summary" (in-place
         compaction) is the action — it is always tried before a reset,
         per the "si un résumé suffit, ne pas reset" rule.
      3. Otherwise, if there is a recognized-but-summary-eligible transition
         (e.g. an explicit request or an idle boundary) even under
         threshold, it still gets a checkpointed "summary" — cheap and
         keeps the handoff trail honest — but never a reset.
      4. Otherwise "none": nothing durable happened, do nothing.

    ``threshold_tokens <= 0`` disables the token-threshold branch (treated
    as "no threshold configured") so callers cannot accidentally force
    permanent summarization by mis-configuring a zero/negative threshold.
    """
    if transition_reason in DURABLE_TRANSITIONS:
        return "reset"

    if threshold_tokens > 0 and context_tokens >= threshold_tokens:
        return "summary"

    if transition_reason in SUMMARY_ELIGIBLE_TRANSITIONS:
        return "summary"

    return "none"


@dataclass(frozen=True)
class SessionIdentity:
    """The routing identity that MUST survive a relay unchanged.

    ``topic_key`` is the platform-level binding (e.g. Telegram
    chat_id/thread_id, Discord channel id) — the thing the user actually
    sees as "the conversation". ``session_id`` is Hermes's internal session
    id; a "summary" relay keeps it identical, a "reset" relay is allowed to
    rotate it (a fresh internal session id is how /reset already works) but
    ``topic_key`` must never move, otherwise the reset becomes visible to
    the user as a new conversation.
    """

    topic_key: str
    session_id: str


@dataclass(frozen=True)
class RelayResult:
    action: RelayAction
    checkpoint_written: bool
    identity_before: SessionIdentity
    identity_after: SessionIdentity


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def format_relay_checkpoint(
    *,
    task_label: str,
    note: str,
    action: RelayAction,
) -> str:
    """Render the short checkpoint line appended to TASKS.md.

    Kept to a single line by design — this is a handoff breadcrumb, not a
    report. Anything longer belongs in the task's own history/comments.
    """
    verb = {"summary": "résumé technique", "reset": "reset", "none": "aucun"}[action]
    return f"- [{_utc_now_iso()}] checkpoint ({verb}) {task_label}: {note}".rstrip()


def write_relay_checkpoint(
    tasks_md_path: Path,
    *,
    task_label: str,
    note: str,
    action: RelayAction,
) -> bool:
    """Append a checkpoint line to ``tasks_md_path`` under a stable heading.

    Idempotent in the narrow sense that matters here: if the exact same
    line (same timestamp granularity would differ across calls in
    practice, but tests may pass a frozen clock) is already the last
    checkpoint line recorded, it is not duplicated. Creates the file and
    the ``## Checkpoints`` section if either is missing. Returns True if a
    line was written, False if skipped as a duplicate.
    """
    heading = "## Checkpoints"
    line = format_relay_checkpoint(task_label=task_label, note=note, action=action)

    tasks_md_path = Path(tasks_md_path)
    existing = tasks_md_path.read_text(encoding="utf-8") if tasks_md_path.exists() else ""

    # Duplicate guard: same task_label + note + action already the most
    # recent checkpoint line (ignoring the timestamp prefix), skip.
    if existing:
        for existing_line in reversed(existing.splitlines()):
            stripped = existing_line.strip()
            if not stripped.startswith("- ["):
                if stripped.startswith("#"):
                    break
                continue
            # Compare everything after the "] " that follows the timestamp.
            marker = "] "
            idx = stripped.find(marker)
            body = stripped[idx + len(marker):] if idx != -1 else stripped
            new_marker = "] "
            new_idx = line.find(new_marker)
            new_body = line[new_idx + len(new_marker):] if new_idx != -1 else line
            if body == new_body:
                return False
            break

    if heading not in existing:
        separator = "\n\n" if existing and not existing.endswith("\n\n") else ""
        if existing and not existing.endswith("\n"):
            separator = "\n" + separator
        new_content = existing + separator + heading + "\n" + line + "\n"
    else:
        # Append under the existing heading, right after its last entry.
        lines = existing.splitlines()
        heading_idx = next(i for i, l in enumerate(lines) if l.strip() == heading)
        insert_at = heading_idx + 1
        while insert_at < len(lines) and lines[insert_at].strip().startswith("- ["):
            insert_at += 1
        lines.insert(insert_at, line)
        new_content = "\n".join(lines) + "\n"

    tasks_md_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_md_path.write_text(new_content, encoding="utf-8")
    return True


def perform_session_relay(
    *,
    identity: SessionIdentity,
    action: RelayAction,
    tasks_md_path: Path,
    task_label: str,
    note: str,
    compress_fn: Optional[Callable[[], None]] = None,
    reset_fn: Optional[Callable[[], "SessionIdentity"]] = None,
) -> RelayResult:
    """Execute the decided relay action while preserving the Topic identity.

    Contract:
      - "none": no checkpoint, no callback, identity unchanged.
      - "summary": checkpoint written first, then ``compress_fn()`` runs.
        ``identity`` (topic_key AND session_id) is required to be unchanged
        afterwards — a summary is an in-place operation.
      - "reset": checkpoint written first, then ``reset_fn()`` runs and MAY
        return a new identity, but ``topic_key`` must be identical to the
        one passed in (a reset that moves the topic is exactly the
        "ask the user to open a new Topic" failure mode this module exists
        to prevent) — this is asserted, not just documented.

    This function never imports gateway/session/kanban modules and never
    touches worker/dispatch state itself, by construction — "preserve the
    workers" is satisfied by NOT reaching for them at all; only the
    caller-supplied callbacks may do gateway-side work.
    """
    if action == "none":
        return RelayResult(
            action="none",
            checkpoint_written=False,
            identity_before=identity,
            identity_after=identity,
        )

    checkpoint_written = write_relay_checkpoint(
        tasks_md_path, task_label=task_label, note=note, action=action
    )

    if action == "summary":
        if compress_fn is not None:
            compress_fn()
        return RelayResult(
            action="summary",
            checkpoint_written=checkpoint_written,
            identity_before=identity,
            identity_after=identity,
        )

    # action == "reset"
    new_identity = identity
    if reset_fn is not None:
        returned = reset_fn()
        if returned is not None:
            new_identity = returned

    if new_identity.topic_key != identity.topic_key:
        raise ValueError(
            "session relay reset must not move the Topic/session routing "
            f"identity: before={identity.topic_key!r} after={new_identity.topic_key!r}"
        )

    return RelayResult(
        action="reset",
        checkpoint_written=checkpoint_written,
        identity_before=identity,
        identity_after=new_identity,
    )
