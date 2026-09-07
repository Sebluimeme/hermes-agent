"""Regression tests for structured replacement-card reconciliation."""

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def conn(tmp_path: Path):
    db = kb.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def test_successful_replacement_archives_blocked_original_atomically(conn):
    original = kb.create_task(conn, title="original", assignee="worker")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='triage', execution_status='blocked', "
            "failure_class='needs_input', action_required='autorisation' WHERE id=?",
            (original,),
        )
    replacement = kb.create_task(conn, title="replacement", assignee="worker")

    assert kb.declare_task_replacements(conn, replacement, [original]) == (original,)
    assert kb.complete_task(conn, replacement, summary="action exécutée")

    row = conn.execute(
        "SELECT status, execution_status, verification_status, delivery_status, "
        "action_required, result FROM tasks WHERE id=?",
        (original,),
    ).fetchone()
    assert row["status"] == "archived"
    assert row["execution_status"] == "done"
    assert row["verification_status"] == "verified"
    assert row["delivery_status"] == "delivered"
    assert row["action_required"] is None
    assert replacement in row["result"]
    events = kb.list_events(conn, original)
    assert any(
        event.kind == "resolved_by_replacement"
        and event.payload["resolver_task_id"] == replacement
        for event in events
    )


def test_replacement_declaration_rejects_executable_target(conn):
    original = kb.create_task(conn, title="still executable", assignee="worker")
    replacement = kb.create_task(conn, title="replacement", assignee="worker")

    with pytest.raises(ValueError, match="only blocked or triage"):
        kb.declare_task_replacements(conn, replacement, [original])


def test_replacement_does_not_archive_target_unblocked_before_completion(conn):
    original = kb.create_task(conn, title="original", assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (original,))
    replacement = kb.create_task(conn, title="replacement", assignee="worker")
    kb.declare_task_replacements(conn, replacement, [original])
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (original,))

    assert kb.complete_task(conn, replacement, summary="action exécutée")

    assert kb.get_task(conn, original).status == "ready"
