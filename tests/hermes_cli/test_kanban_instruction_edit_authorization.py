"""Regression coverage for durable, task-scoped instruction-edit consent."""

import json


def _write(path, content="authorized content", task_id="default"):
    from tools.file_tools import write_file_tool

    return json.loads(write_file_tool(str(path), content, task_id=task_id))


def test_headless_worker_rejects_protected_write_without_authorization(
    monkeypatch, tmp_path
):
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / ".hermes" / "kanban.db"))
    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn, title="edit instructions", assignee="worker", workspace_kind="dir",
            workspace_path=str(tmp_path), created_by="dashboard",
        )
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setattr(
        "tools.worker_approval.request_decision",
        lambda **_: {"resolved": False, "choice": "timeout"},
    )

    result = _write(tmp_path / "AGENTS.md", task_id=task_id)

    assert "BLOCKED" in result["error"]
    assert not (tmp_path / "AGENTS.md").exists()


def test_durable_task_authorization_allows_only_its_target(
    monkeypatch, tmp_path
):
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / ".hermes" / "kanban.db"))
    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn, title="edit instructions", assignee="worker", workspace_kind="dir",
            workspace_path=str(tmp_path), created_by="dashboard",
        )
        assert kb.authorize_instruction_edit(
            conn, task_id, str(tmp_path / "AGENTS.md"), granted_by="dashboard-user",
            reason="Sébastien explicitly authorized this instruction edit.",
        )
        assert any(
            "INSTRUCTION EDIT AUTHORIZED" in comment.body
            for comment in kb.list_comments(conn, task_id)
        )
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setattr(
        "tools.worker_approval.request_decision",
        lambda **_: {"resolved": False, "choice": "timeout"},
    )

    allowed = _write(tmp_path / "AGENTS.md", task_id=task_id)
    denied = _write(tmp_path / "CLAUDE.md", task_id=task_id)

    assert not allowed.get("error"), allowed
    assert (tmp_path / "AGENTS.md").read_text(encoding="utf-8") == "authorized content"
    assert "BLOCKED" in denied["error"]
    assert not (tmp_path / "CLAUDE.md").exists()


def test_authorized_triage_task_recovers_through_controlled_transition(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / ".hermes" / "kanban.db"))
    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn, title="edit instructions", assignee="worker", workspace_kind="dir",
            workspace_path=str(tmp_path), created_by="dashboard", triage=True,
        )
        assert kb.recover_triage_after_instruction_authorization(conn, task_id) is False
        assert kb.authorize_instruction_edit(
            conn, task_id, str(tmp_path / "AGENTS.md"), granted_by="dashboard-user",
            reason="Sébastien explicitly authorized this instruction edit.",
        )
        assert kb.recover_triage_after_instruction_authorization(conn, task_id) is True
        assert kb.get_task(conn, task_id).status == "ready"
        assert any(
            event.kind == "instruction_edit_authorized_recovery"
            for event in kb.list_events(conn, task_id)
        )
    finally:
        conn.close()
