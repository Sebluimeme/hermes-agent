"""Regressions for late worker failures racing a successful completion."""

from __future__ import annotations

from pathlib import Path

from hermes_cli import kanban_db as kb


def _connection(tmp_path: Path):
    return kb.connect(tmp_path / "kanban.db")


def test_late_timeout_cannot_downgrade_completed_task(tmp_path: Path) -> None:
    conn = _connection(tmp_path)
    try:
        task_id = kb.create_task(conn, title="Audit", assignee="coder")
        claimed = kb.claim_task(conn, task_id, claimer="test")
        assert claimed is not None and claimed.current_run_id is not None
        run_id = claimed.current_run_id
        assert kb.complete_task(
            conn,
            task_id,
            summary="Audit terminé avec preuve.",
            metadata={"evidence": {"kind": "test", "detail": "OK"}},
            expected_run_id=run_id,
        )

        tripped = kb._record_task_failure(
            conn,
            task_id,
            "Iteration budget exhausted (30/30)",
            outcome="timed_out",
            release_claim=True,
            end_run=True,
            expected_run_id=run_id,
        )

        assert tripped is False
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done"
        assert task.execution_status == "done"
        assert task.failure_class is None
        row = conn.execute(
            "SELECT next_retry_at FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        assert row["next_retry_at"] is None
        assert [event.kind for event in kb.list_events(conn, task_id)].count("timed_out") == 0
        runs = kb.list_runs(conn, task_id)
        assert len(runs) == 1 and runs[0].outcome == "completed"
    finally:
        conn.close()


def test_stale_failure_cannot_close_newer_retry(tmp_path: Path) -> None:
    conn = _connection(tmp_path)
    try:
        task_id = kb.create_task(conn, title="Retry", assignee="coder")
        first = kb.claim_task(conn, task_id, claimer="first")
        assert first is not None and first.current_run_id is not None
        assert kb._record_task_failure(
            conn,
            task_id,
            "first timeout",
            outcome="timed_out",
            release_claim=True,
            end_run=True,
            expected_run_id=first.current_run_id,
        ) is False
        second = kb.claim_task(conn, task_id, claimer="second")
        assert second is not None and second.current_run_id is not None

        assert kb._record_task_failure(
            conn,
            task_id,
            "late timeout from first run",
            outcome="timed_out",
            release_claim=True,
            end_run=True,
            expected_run_id=first.current_run_id,
        ) is False

        current = kb.get_task(conn, task_id)
        assert current is not None
        assert current.status == "running"
        assert current.current_run_id == second.current_run_id
        assert kb.latest_run(conn, task_id).ended_at is None
    finally:
        conn.close()


def test_init_repairs_done_phase_drift_and_records_reconciliation(tmp_path: Path) -> None:
    db_path = tmp_path / "kanban.db"
    conn = kb.connect(db_path)
    task_id = kb.create_task(conn, title="Ancien audit", assignee="coder")
    claimed = kb.claim_task(conn, task_id, claimer="test")
    assert claimed is not None
    assert kb.complete_task(conn, task_id, summary="Terminé")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET execution_status='retrying',failure_class='timed_out',"
            "next_retry_at=123,action_required='faux délai' WHERE id=?",
            (task_id,),
        )
    conn.close()

    kb.init_db(db_path)
    conn = kb.connect(db_path)
    try:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done"
        assert task.execution_status == "done"
        assert task.failure_class is None
        row = conn.execute(
            "SELECT next_retry_at,action_required FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        assert row["next_retry_at"] is None
        assert row["action_required"] is None
        reconciled = [
            event for event in kb.list_events(conn, task_id)
            if event.kind == "terminal_state_reconciled"
        ]
        assert len(reconciled) == 1
        assert reconciled[0].payload["previous_failure_class"] == "timed_out"
    finally:
        conn.close()
