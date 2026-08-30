"""Tests for the Kanban pre-request-review handoff projection seam."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import plugins as plugins_mod
from hermes_cli.kanban import _cmd_request_review
from hermes_cli.kanban_review_handoff_validators import (
    KanbanReviewHandoffResult,
    KanbanReviewHandoffValidationError,
    _normalize_validator_result,
)
from hermes_cli.plugins import get_plugin_manager
from tools.kanban_tools import _handle_request_review


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def review_handoff_validators(kanban_home, monkeypatch):
    mgr = get_plugin_manager()
    saved_hooks = {name: list(callbacks) for name, callbacks in mgr._hooks.items()}
    saved_discovered = getattr(mgr, "_discovered", True)
    callbacks: list = []
    mgr._hooks["validate_kanban_review_handoff"] = callbacks
    mgr._discovered = True
    has_hook_calls: list[str] = []

    def _has_hook(name: str) -> bool:
        has_hook_calls.append(name)
        return bool(mgr._hooks.get(name))

    monkeypatch.setattr(plugins_mod, "has_hook", _has_hook)
    try:
        yield callbacks, has_hook_calls
    finally:
        mgr._hooks = saved_hooks
        mgr._discovered = saved_discovered


@pytest.fixture
def fake_visual_handoff(monkeypatch):
    calls = []

    def _prepare_review_handoff(*, task_id, title, body, metadata, reviewer):
        calls.append(
            {
                "task_id": task_id,
                "title": title,
                "body": body,
                "metadata": dict(metadata or {}),
                "reviewer": reviewer,
            }
        )
        projected = dict(metadata or {})
        projected.setdefault("visual_review", {"normalized": True})
        return projected, reviewer or "coder"

    monkeypatch.setattr(
        "hermes_cli.visual_review.prepare_review_handoff",
        _prepare_review_handoff,
    )
    return calls


def _task(conn, *, title: str = "T", body: str = "B", created_by: str = "creator") -> str:
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee="worker",
        created_by=created_by,
        initial_status="running",
    )


def _status(conn, tid: str) -> str:
    task = kb.get_task(conn, tid)
    assert task is not None
    return task.status


def _last_run_metadata(conn, tid: str) -> dict:
    run = conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()
    assert run is not None
    return json.loads(run["metadata"])


def test_absent_plugin_keeps_visual_handoff_and_uses_lazy_has_hook(
    review_handoff_validators, fake_visual_handoff, kanban_home
):
    callbacks, has_hook_calls = review_handoff_validators
    conn = kb.connect()
    try:
        tid = _task(conn)
        assert callbacks == []
        assert kb.request_review(conn, tid, summary="ready", metadata={"x": 1}) is True
        assert _status(conn, tid) == "review"
        assert fake_visual_handoff[0]["metadata"] == {"x": 1}
        assert _last_run_metadata(conn, tid)["visual_review"] == {"normalized": True}
        assert has_hook_calls == ["validate_kanban_review_handoff"]
    finally:
        conn.close()


def test_direct_request_review_projects_metadata_reviewer_and_context(
    review_handoff_validators, fake_visual_handoff, kanban_home
):
    callbacks, _ = review_handoff_validators
    seen = []

    def validator(context):
        seen.append(context)
        projected = dict(context.metadata)
        projected["validator"] = {
            "task_id": context.task_id,
            "title": context.title,
            "body": context.body,
            "created_by": context.created_by,
            "source": context.source,
            "surface": context.surface,
            "reviewer": context.reviewer,
        }
        return KanbanReviewHandoffResult.accept(projected, reviewer="Claude2")

    callbacks.append(validator)
    conn = kb.connect()
    try:
        tid = _task(conn, title="Visible title", body="Visible body", created_by="alice")
        assert kb.request_review(conn, tid, summary="ready", metadata={"base": True}) is True
        assert len(fake_visual_handoff) == 1
        assert _last_run_metadata(conn, tid)["validator"] == {
            "task_id": tid,
            "title": "Visible title",
            "body": "Visible body",
            "created_by": "alice",
            "source": "core",
            "surface": "core",
            "reviewer": "coder",
        }
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.assignee == "claude2"
        assert len(seen) == 1
    finally:
        conn.close()


def test_metadata_is_redacted_before_plugin_boundary(
    review_handoff_validators, fake_visual_handoff, kanban_home, monkeypatch
):
    callbacks, _ = review_handoff_validators
    seen = []
    callbacks.append(lambda context: seen.append(dict(context.metadata)) or {"accepted": True})

    def fake_redact(value):
        if isinstance(value, dict):
            return {k: fake_redact(v) for k, v in value.items()}
        if isinstance(value, str):
            return value.replace("sensitive fixture value", "[redacted]")
        return value

    monkeypatch.setattr(kb, "redact_review_value", fake_redact)
    conn = kb.connect()
    try:
        tid = _task(conn)
        assert kb.request_review(
            conn,
            tid,
            summary="ready",
            metadata={"token": "sensitive fixture value"},
        ) is True
        assert seen
        assert "sensitive fixture value" not in json.dumps(seen[0])
        assert seen[0]["token"] == "[redacted]"
    finally:
        conn.close()


def test_veto_error_malformed_and_exception_fail_closed(
    review_handoff_validators, fake_visual_handoff, kanban_home
):
    callbacks, _ = review_handoff_validators
    conn = kb.connect()
    try:
        for callback in [
            lambda context: {"accepted": False, "reason": "blocked"},
            lambda context: True,
            lambda context: (_ for _ in ()).throw(RuntimeError("boom")),
        ]:
            callbacks[:] = [callback]
            tid = _task(conn)
            prior = _status(conn, tid)
            ok, reason = kb.request_review(conn, tid, summary="ready", with_reason=True)
            assert ok is False
            assert reason
            assert _status(conn, tid) == prior
    finally:
        conn.close()


def test_reentrant_request_review_poisons_outer_validation_fail_closed(
    review_handoff_validators, fake_visual_handoff, kanban_home
):
    callbacks, _ = review_handoff_validators
    conn = kb.connect()
    try:
        tid = _task(conn)
        prior = _status(conn, tid)

        def validator(context):
            ok, reason = kb.request_review(
                conn,
                context.task_id,
                summary="recursive handoff",
                with_reason=True,
            )
            assert ok is False
            assert "recursive" in reason
            return {"accepted": True}

        callbacks.append(validator)
        ok, reason = kb.request_review(conn, tid, summary="ready", with_reason=True)
        assert ok is False
        assert "recursive" in reason
        assert _status(conn, tid) == prior
    finally:
        conn.close()


def test_tool_path_invokes_validator_exactly_once_with_tool_source(
    review_handoff_validators, fake_visual_handoff, kanban_home, monkeypatch
):
    callbacks, _ = review_handoff_validators
    seen = []
    callbacks.append(lambda context: seen.append(context.source) or {"accepted": True})
    conn = kb.connect()
    try:
        tid = _task(conn)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)

    payload = json.loads(_handle_request_review({"summary": "ready"}))
    assert payload["ok"] is True
    assert seen == ["tool"]
    assert len(fake_visual_handoff) == 1


def test_cli_path_invokes_validator_exactly_once_with_cli_source(
    review_handoff_validators, fake_visual_handoff, kanban_home
):
    callbacks, _ = review_handoff_validators
    seen = []
    callbacks.append(lambda context: seen.append(context.source) or {"accepted": True})
    conn = kb.connect()
    try:
        tid = _task(conn)
    finally:
        conn.close()

    args = argparse.Namespace(
        task_id=tid,
        summary="ready",
        reviewer=None,
        metadata=None,
        force=False,
    )
    assert _cmd_request_review(args) == 0
    assert seen == ["cli"]
    assert len(fake_visual_handoff) == 1


def test_live_claim_diagnostic_remains_when_plugin_absent(
    review_handoff_validators, fake_visual_handoff, kanban_home
):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="claimed", assignee="worker")
        assert kb.claim_task(conn, tid, claimer="worker:test") is not None
        ok, reason = kb.request_review(conn, tid, summary="ready", with_reason=True)
        assert ok is False
        assert "live claim" in reason
        assert len(fake_visual_handoff) == 1
    finally:
        conn.close()


def test_active_reviewer_diagnostic_remains_when_plugin_absent(
    review_handoff_validators, fake_visual_handoff, kanban_home
):
    conn = kb.connect()
    try:
        tid = _task(conn)
        assert kb.request_review(conn, tid, summary="ready") is True
        reviewer_claim = kb.claim_review_task(conn, tid, claimer="reviewer:test")
        assert reviewer_claim is not None
        ok, reason = kb.request_review(conn, tid, summary="again", with_reason=True)
        assert ok is False
        assert "active reviewer" in reason
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("claim_lock", "competing-worker"),
        ("title", "mutated title"),
        ("body", "mutated body"),
    ],
)
def test_race_sensitive_fields_are_revalidated_after_external_projection(
    review_handoff_validators, fake_visual_handoff, kanban_home, field, value
):
    callbacks, _ = review_handoff_validators
    conn = kb.connect()
    try:
        tid = _task(conn, title="stable title", body="stable body")
        prior = _status(conn, tid)

        def validator(context):
            conn.execute(f"UPDATE tasks SET {field} = ? WHERE id = ?", (value, context.task_id))
            return {"accepted": True}

        callbacks.append(validator)
        ok, reason = kb.request_review(conn, tid, summary="ready", with_reason=True)
        assert ok is False
        assert reason is not None
        assert "changed while projecting review handoff" in str(reason)
        assert _status(conn, tid) == prior
    finally:
        conn.close()


def test_normalizer_rejects_ambiguous_mapping_and_non_string_reviewer():
    with pytest.raises(KanbanReviewHandoffValidationError):
        _normalize_validator_result(
            {"accepted": True, "ok": True},
            current_metadata={},
            current_reviewer=None,
        )
    with pytest.raises(KanbanReviewHandoffValidationError):
        _normalize_validator_result(
            {"accepted": True, "reviewer": 123},
            current_metadata={},
            current_reviewer=None,
        )
