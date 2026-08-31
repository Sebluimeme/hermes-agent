from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.auth import AuthError, CODEX_RATE_LIMITED_CODE
from hermes_cli.kanban_auth_incident import (
    mark_kanban_auth_healthy,
    report_kanban_auth_required,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(monkeypatch) -> tuple[str, int]:
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="auth probe", assignee="claude1")
        task = kb.claim_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        run_id = task.current_run_id
    monkeypatch.setenv("HERMES_PROFILE", "claude1")
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return task_id, run_id


def test_auth_incident_notifies_once_while_fallback_keeps_task_running(
    kanban_home, monkeypatch,
):
    task_id, _ = _running_task(monkeypatch)
    error = AuthError(
        "OAuth token expired",
        provider="anthropic",
        code="invalid_token",
        relogin_required=True,
    )

    assert report_kanban_auth_required(
        error, provider="anthropic", fallback="openai-codex/gpt-5.5",
    )
    assert report_kanban_auth_required(
        error, provider="anthropic", fallback="openai-codex/gpt-5.5",
    )

    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        event_count = conn.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id=? AND kind='provider_auth_required'",
            (task_id,),
        ).fetchone()[0]
        incident = conn.execute(
            "SELECT status, fallback_active, fallback "
            "FROM provider_auth_incidents "
            "WHERE profile='claude1' AND provider='anthropic'",
        ).fetchone()

    assert task is not None and task.status == "running"
    assert event_count == 1
    assert incident["status"] == "open"
    assert incident["fallback_active"] == 1
    assert incident["fallback"] == "openai-codex/gpt-5.5"


def test_auth_incident_reopens_after_primary_auth_recovers(kanban_home, monkeypatch):
    task_id, _ = _running_task(monkeypatch)
    error = AuthError(
        "token revoked", provider="anthropic", code="invalid_token",
        relogin_required=True,
    )
    assert report_kanban_auth_required(
        error, provider="anthropic", fallback="openai-codex/gpt-5.5",
    )
    assert mark_kanban_auth_healthy(provider="anthropic")
    assert report_kanban_auth_required(
        error, provider="anthropic", fallback="openai-codex/gpt-5.5",
    )

    with kb.connect() as conn:
        event_count = conn.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id=? AND kind='provider_auth_required'",
            (task_id,),
        ).fetchone()[0]
    assert event_count == 2


def test_unrecovered_auth_blocks_with_action_but_quota_never_does(
    kanban_home, monkeypatch,
):
    task_id, _ = _running_task(monkeypatch)
    quota = AuthError(
        "usage limit reached",
        provider="openai-codex",
        code=CODEX_RATE_LIMITED_CODE,
    )
    assert not report_kanban_auth_required(quota, provider="openai-codex")

    auth = AuthError(
        "credentials missing",
        provider="anthropic",
        code="auth_missing",
        relogin_required=True,
    )
    assert report_kanban_auth_required(auth, provider="anthropic")

    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        provider_events = conn.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id=? AND kind='provider_auth_required'",
            (task_id,),
        ).fetchone()[0]
        blocked = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id=? AND kind='blocked' ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()

    assert task is not None and task.status == "blocked"
    assert task.block_kind == "capability"
    assert provider_events == 0, "blocked is the one user notification"
    assert "provider-auth" in blocked["payload"]


def test_respawn_guard_does_not_apply_previous_profile_failure(
    kanban_home, all_assignees_spawnable, configured_handoff_routes, monkeypatch,
):
    now = 5_000_000
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Spark fallback",
            assignee="spark",
            routing_tier="simple",
        )
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None and claimed.current_run_id is not None
        conn.execute(
            "UPDATE task_runs SET outcome='rate_limited', status='rate_limited', "
            "ended_at=? WHERE id=?",
            (now, claimed.current_run_id),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, "
            "last_failure_error='Spark quota wall' WHERE id=?",
            (task_id,),
        )
        conn.commit()
        assert kb.fallback_simple_route(
            conn, task_id, "Spark quota wall", provider_proven=True,
        )
        task = kb.get_task(conn, task_id)
        assert task is not None and task.assignee == "claude2"
        monkeypatch.setattr(kb.time, "time", lambda: now + 10)
        assert kb.check_respawn_guard(conn, task_id) is None
