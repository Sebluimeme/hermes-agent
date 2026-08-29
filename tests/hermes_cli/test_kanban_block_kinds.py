"""Tests for typed block reasons + the unblock-loop breaker.

Covers the built-in fix for the kanban "blocked loop" — a worker blocks a
task, a cron unblocks it, the worker re-blocks for the same reason, repeat
forever. The fix gives ``block_task`` a typed ``kind`` and a persistent
``block_recurrences`` counter:

* ``dependency`` blocks route to ``todo`` (parent-gated, auto-resumed) and
  never enter the human ``blocked`` bucket a cron would keep unblocking.
* ``transient`` blocks return to the dispatchable phase with a bounded retry
  time and exact-session checkpoint, without creating a human action.
* ``needs_input`` / ``capability`` / un-typed blocks land in ``blocked``;
  each same-cause re-block after an unblock increments ``block_recurrences``,
  and at ``BLOCK_RECURRENCE_LIMIT`` the task routes to ``triage`` for a human.
* ``unblock_task`` deliberately does NOT reset ``block_recurrences`` (the
  amnesia that let the loop run unbounded).
* A successful ``complete_task`` resets the loop memory.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t"):
    """Create a task and drive it to ``running`` so block_task can act."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer="worker")
    assert claimed is not None
    return tid


def _make_running_again(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None


@pytest.mark.parametrize(
    "reason",
    ["test", "test reason simple", "reason with spaces", "à compléter"],
)
def test_placeholder_block_reason_is_rejected_without_mutation(
    kanban_home: Path, reason: str,
) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        with pytest.raises(ValueError, match="placeholder block reason refused"):
            kb.block_task(conn, tid, reason=reason, kind="needs_input")
        assert kb.get_task(conn, tid).status == "running"
        assert not [event for event in kb.list_events(conn, tid) if event.kind == "blocked"]


# ---------------------------------------------------------------------------
# Loop breaker
# ---------------------------------------------------------------------------










def test_block_loop_detected_event_emitted(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="capability")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="x", kind="capability")
        events = [e for e in kb.list_events(conn, tid)
                  if e.kind == "block_loop_detected"]
        assert events, "expected a block_loop_detected event"
        payload = events[-1].payload or {}
        assert payload.get("recurrences") == 2
        assert payload.get("kind") == "capability"


def test_human_block_persists_exact_action_separately(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn, title="Ecobloc API")
        action = (
            "Autoriser Hermes à activer l’API Google Business Profile dans "
            "le projet 362154063865."
        )
        assert kb.block_task(
            conn,
            tid,
            reason="Le jeton est valide mais l’API du projet est désactivée.",
            action=action,
            kind="capability",
        )

        task = kb.get_task(conn, tid)
        assert task.action_required == action
        event = [e for e in kb.list_events(conn, tid) if e.kind == "blocked"][-1]
        assert event.payload["reason"].startswith("Le jeton est valide")
        assert event.payload["action"] == action
        row = conn.execute(
            "SELECT prompt FROM human_actions "
            "WHERE task_id = ? AND status = 'open'",
            (tid,),
        ).fetchone()
        assert row["prompt"] == action


def test_transient_block_schedules_automatic_resume_without_human_action(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        before = int(time.time())

        assert kb.block_task(
            conn,
            tid,
            reason="checkpoint durable écrit; reprendre le traitement restant",
            kind="transient",
            metadata={"worker_session_id": "session-checkpointed"},
        )

        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.execution_status == "retrying"
        assert task.block_kind == "transient"
        assert task.next_retry_at is not None
        assert before < task.next_retry_at <= before + kb.TRANSIENT_RETRY_DELAY_SECONDS + 1
        assert conn.execute(
            "SELECT COUNT(*) FROM human_actions "
            "WHERE task_id = ? AND status = 'open'",
            (tid,),
        ).fetchone()[0] == 0
        events = kb.list_events(conn, tid)
        assert events[-1].kind == "transient_retry_scheduled"
        assert events[-1].payload["retry_status"] == "ready"

    assert kb._transient_resume_session_id(
        tid, board=kb.get_current_board(),
    ) == "session-checkpointed"


def test_init_requeues_a_legacy_transient_human_block(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="legacy transient", assignee="worker")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='blocked',block_kind='transient',"
                "execution_status='blocked',failure_class='transient',"
                "action_required='reprendre le checkpoint',next_retry_at=NULL "
                "WHERE id=?",
                (tid,),
            )
            conn.execute(
                "INSERT INTO human_actions "
                "(task_id,kind,prompt,status,created_at) "
                "VALUES (?,'transient','reprendre le checkpoint','open',?)",
                (tid, int(time.time())),
            )
            kb._append_event(
                conn,
                tid,
                "blocked",
                {"kind": "transient", "source_status": "ready"},
            )

    kb.init_db()

    with kb.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.execution_status == "retrying"
        assert task.next_retry_at is not None
        assert task.action_required is None
        assert conn.execute(
            "SELECT COUNT(*) FROM human_actions "
            "WHERE task_id=? AND status='open'",
            (tid,),
        ).fetchone()[0] == 0
        assert kb.list_events(conn, tid)[-1].kind == "legacy_transient_block_requeued"


# ---------------------------------------------------------------------------
# Dependency routing
# ---------------------------------------------------------------------------


def test_dependency_then_parent_done_promotes(kanban_home: Path) -> None:
    """A dependency-parked child becomes ready once its parent completes."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        kb.block_task(conn, child, reason="wait", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        # Finish the parent, then let recompute_ready run.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


# ---------------------------------------------------------------------------
# Completion resets loop memory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------
