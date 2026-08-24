"""Task authority — LOT 4 (authority & closure proof).

Hermes tracks work through two independent, deliberately un-merged stores:

- **Kanban SQLite** (``hermes_cli.kanban_db`` / ``kanban.db``) is the sole
  authority for the lifecycle of an *active Kanban card*: todo / running /
  review / changes_requested / blocked / done. Nothing outside
  ``kanban_db.py`` may assign or infer a card's status.
- **TASKS.md** (per-project, read/written by
  ``scripts/hermes_next_task.py``) is the sole authority for the *linked
  project queue*: TODO / IN_PROGRESS / DONE / BLOCKED / NEEDS_DECISION. It
  is a human-facing resumption ledger, not a mirror of Kanban.

The two stores have no shared foreign key today and neither may silently
overwrite the other. The one place they legitimately meet is when a human
or worker manually notes, in the same breath, "this Kanban card finished
X" and "TASKS.md row X says Y" — at that point the two claims can be
compared. This module is that comparison: a pure function that names a
contradiction instead of letting either side win by default (e.g. by
"last write wins" or by one script silently trusting whichever value it
read most recently).

No I/O, no DB access: callers own reading both sides and pass the two
status strings in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Kanban SQLite lifecycle (hermes_cli.kanban_db) — source of truth for
# active Kanban cards. Mirrors kanban_db.VALID_STATUSES plus
# "changes_requested", a logical status kanban_db derives from the latest
# task_events row rather than storing directly in the status column
# (see kanban_db.py's terminal-status resolution around VALID_STATUSES).
KANBAN_STATUSES = (
    "triage", "todo", "scheduled", "ready", "running", "blocked",
    "review", "changes_requested", "done", "archived",
)
# "archived" is terminal (no further work happens on it) even though it
# does not mean "completed successfully" the way "done" does; for
# authority-contradiction purposes both close out the card.
KANBAN_CLOSED = ("done", "archived")
KANBAN_OPEN = tuple(s for s in KANBAN_STATUSES if s not in KANBAN_CLOSED)

# TASKS.md lifecycle (scripts/hermes_next_task.py) — source of truth for
# the linked project queue.
TASKS_MD_STATUSES = ("TODO", "IN_PROGRESS", "DONE", "BLOCKED", "NEEDS_DECISION")
TASKS_MD_CLOSED = ("DONE",)
TASKS_MD_OPEN = tuple(s for s in TASKS_MD_STATUSES if s not in TASKS_MD_CLOSED)


@dataclass(frozen=True)
class AuthorityContradiction:
    """A named disagreement between the two authoritative stores."""

    kanban_status: str
    tasks_md_status: str
    message: str


def check_authority_contradiction(
    *,
    kanban_status: Optional[str],
    tasks_md_status: Optional[str],
) -> Optional[AuthorityContradiction]:
    """Compare a Kanban card's status against a TASKS.md row's status for
    what is asserted to be the same piece of work.

    Returns ``None`` when either side is unknown (``None``) — with only one
    authority present there is nothing to contradict — or when both sides
    agree on open/closed. Returns an :class:`AuthorityContradiction`
    naming which store says what when they disagree, so the caller can
    surface it instead of silently picking one.

    This function never decides *which* store is right: Kanban stays
    authoritative for the card, TASKS.md stays authoritative for the
    project row. It only refuses to let a mismatch pass unnoticed.
    """
    if kanban_status is None or tasks_md_status is None:
        return None

    if kanban_status not in KANBAN_STATUSES:
        raise ValueError(f"unknown kanban status: {kanban_status!r}")
    if tasks_md_status not in TASKS_MD_STATUSES:
        raise ValueError(f"unknown TASKS.md status: {tasks_md_status!r}")

    kanban_closed = kanban_status in KANBAN_CLOSED
    tasks_md_closed = tasks_md_status in TASKS_MD_CLOSED

    if kanban_closed and not tasks_md_closed:
        return AuthorityContradiction(
            kanban_status=kanban_status,
            tasks_md_status=tasks_md_status,
            message=(
                f"Kanban card is 'done' but TASKS.md still shows "
                f"'{tasks_md_status}' — TASKS.md was not updated, or the "
                f"Kanban closure was premature. Reconcile explicitly; do "
                f"not assume either side."
            ),
        )

    if tasks_md_closed and not kanban_closed:
        return AuthorityContradiction(
            kanban_status=kanban_status,
            tasks_md_status=tasks_md_status,
            message=(
                f"TASKS.md row is 'DONE' but the linked Kanban card is "
                f"still '{kanban_status}' — TASKS.md was marked done ahead "
                f"of the card, or the card regressed after DONE was "
                f"written. Reconcile explicitly; do not assume either "
                f"side."
            ),
        )

    return None
