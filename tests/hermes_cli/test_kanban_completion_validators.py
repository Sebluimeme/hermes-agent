"""Tests for the Kanban pre-completion validator seam."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import plugins as plugins_mod
from hermes_cli.kanban import _cmd_complete
from hermes_cli.kanban_completion_validators import (
    KanbanCompletionValidationError,
    KanbanCompletionValidationResult,
    _normalize_validator_result,
)
from hermes_cli.plugins import get_plugin_manager
from tools.kanban_tools import _handle_complete, _handle_completion_ready


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
def completion_validators(kanban_home, monkeypatch):
    mgr = get_plugin_manager()
    saved_hooks = {name: list(callbacks) for name, callbacks in mgr._hooks.items()}
    saved_discovered = getattr(mgr, "_discovered", True)
    callbacks: list = []
    mgr._hooks["validate_kanban_completion"] = callbacks
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


def _task(conn, *, title: str = "T", body: str = "B", created_by: str = "creator") -> str:
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee="worker",
        created_by=created_by,
        initial_status="running",
    )


def _evidence() -> dict:
    return {"evidence": {"kind": "test", "detail": "unit test fixture"}}


def _status(conn, tid: str) -> str:
    task = kb.get_task(conn, tid)
    assert task is not None
    return task.status


def test_absent_plugin_accepts_without_legacy_closure_or_visual_policy(
    kanban_home, completion_validators
):
    callbacks, has_hook_calls = completion_validators
    conn = kb.connect()
    try:
        tid = _task(conn)
        assert callbacks == []
        assert kb.complete_task(conn, tid, summary="ok", metadata={}) is True
        assert _status(conn, tid) == "done"
        assert has_hook_calls == ["validate_kanban_completion"]
    finally:
        conn.close()


def test_direct_complete_projects_metadata_and_context(completion_validators, kanban_home):
    callbacks, _ = completion_validators
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
            "dry_run": context.dry_run,
        }
        return KanbanCompletionValidationResult.accept(projected)

    callbacks.append(validator)
    conn = kb.connect()
    try:
        tid = _task(conn, title="Visible title", body="Visible body", created_by="alice")
        assert kb.complete_task(conn, tid, summary="ok", metadata=_evidence()) is True
        run = conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        metadata = json.loads(run["metadata"])
        assert metadata["validator"] == {
            "task_id": tid,
            "title": "Visible title",
            "body": "Visible body",
            "created_by": "alice",
            "source": "core",
            "surface": "core",
            "dry_run": False,
        }
        assert len(seen) == 1
    finally:
        conn.close()


def test_veto_and_plugin_errors_block_before_mutation(completion_validators, kanban_home):
    callbacks, _ = completion_validators
    conn = kb.connect()
    try:
        tid = _task(conn)
        before = _status(conn, tid)
        callbacks.append(lambda **_: {"accepted": False, "reason": "not enough proof", "code": "nope"})
        with pytest.raises(kb.CompletionValidationError) as excinfo:
            kb.complete_task(conn, tid, summary="ok", metadata=_evidence())
        assert excinfo.value.code == "nope"
        assert "not enough proof" in str(excinfo.value)
        assert _status(conn, tid) == before

        before = _status(conn, tid)
        callbacks[:] = [lambda **_: (_ for _ in ()).throw(RuntimeError("boom"))]
        with pytest.raises(kb.CompletionValidationError) as excinfo:
            kb.complete_task(conn, tid, summary="ok", metadata=_evidence())
        assert excinfo.value.code == "validator_exception"
        assert _status(conn, tid) == before
    finally:
        conn.close()


@pytest.mark.parametrize(
    "bad",
    [
        None,
        True,
        False,
        {},
        {"metadata": {"x": 1}},
        {"ok": "yes"},
        {"action": "allow", "ok": False},
        {"action": "block", "allow": True},
        KanbanCompletionValidationResult(accepted=cast(Any, "yes")),
        KanbanCompletionValidationResult(accepted=True, metadata=cast(Any, [])),
    ],
)
def test_ambiguous_results_fail_closed(bad):
    with pytest.raises(KanbanCompletionValidationError) as excinfo:
        _normalize_validator_result(bad, current_metadata={})
    assert excinfo.value.code == "validator_bad_result"


@pytest.mark.parametrize(
    "raw, accepted",
    [
        ({"accepted": True}, True),
        ({"ok": False, "reason": "blocked"}, False),
        ({"allow": True, "metadata": {"x": 1}}, True),
        ({"action": "allow"}, True),
        ({"action": "block", "reason": "blocked"}, False),
    ],
)
def test_explicit_mapping_contract(raw, accepted):
    result = _normalize_validator_result(raw, current_metadata={})
    assert result.accepted is accepted


def test_completion_ready_is_dry_run_and_does_not_mutate(completion_validators, kanban_home, monkeypatch):
    callbacks, _ = completion_validators
    seen = []

    def validator(context):
        seen.append((context.source, context.surface, context.dry_run))
        projected = dict(context.metadata)
        projected["ready_projection"] = True
        return {"accepted": True, "metadata": projected}

    callbacks.append(validator)
    conn = kb.connect()
    try:
        tid = _task(conn)
        before = _status(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)

    payload = json.loads(
        _handle_completion_ready({"metadata": _evidence(), "summary": "ok"})
    )
    assert payload["ok"] is True
    assert payload["ready"] is True
    assert seen == [("tool", "tool", True)]

    conn = kb.connect()
    try:
        assert _status(conn, tid) == before
    finally:
        conn.close()


def test_tool_path_invokes_validator_exactly_once_in_complete_task(
    completion_validators, kanban_home, monkeypatch
):
    callbacks, _ = completion_validators
    seen = []

    def validator(context):
        seen.append((context.source, context.dry_run))
        return {"accepted": True}

    callbacks.append(validator)
    conn = kb.connect()
    try:
        tid = _task(conn)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)

    payload = json.loads(_handle_complete({"summary": "ok", "metadata": _evidence()}))
    assert payload["ok"] is True
    assert seen == [("tool", False)]


def test_cli_path_invokes_validator_exactly_once_with_cli_source(
    completion_validators, kanban_home
):
    callbacks, _ = completion_validators
    seen = []
    callbacks.append(lambda context: seen.append((context.source, context.dry_run)) or {"accepted": True})
    conn = kb.connect()
    try:
        tid = _task(conn)
    finally:
        conn.close()

    args = argparse.Namespace(
        task_id=tid,
        task_ids=[tid],
        result=None,
        summary="ok",
        metadata=json.dumps(_evidence()),
    )
    assert _cmd_complete(args) == 0
    assert seen == [("cli", False)]


def test_created_cards_generator_is_seen_by_validator_and_completion_event(
    completion_validators, kanban_home
):
    callbacks, _ = completion_validators
    seen = []

    def validator(context):
        seen.append(context.created_cards)
        return {"accepted": True}

    callbacks.append(validator)
    conn = kb.connect()
    try:
        tid = _task(conn, created_by="worker")
        child = kb.create_task(conn, title="child", created_by="worker")

        def cards():
            yield child

        assert (
            kb.complete_task(
                conn,
                tid,
                summary="ok",
                metadata=_evidence(),
                created_cards=cards(),
            )
            is True
        )
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'completed' ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        payload = json.loads(event["payload"])
        assert seen == [(child,)]
        assert payload["verified_cards"] == [child]
    finally:
        conn.close()


def test_reentrant_validation_fails_closed(completion_validators, kanban_home):
    callbacks, _ = completion_validators
    conn = kb.connect()
    try:
        tid = _task(conn)
        before = _status(conn, tid)

        def validator(context):
            kb.completion_validation_projection(
                conn,
                context.task_id,
                dict(context.metadata),
                summary=context.summary,
                source=context.source,
                dry_run=True,
            )
            return {"accepted": True}

        callbacks.append(validator)
        with pytest.raises(kb.CompletionValidationError) as excinfo:
            kb.complete_task(conn, tid, summary="ok", metadata=_evidence())
        assert excinfo.value.code == "validator_reentrant"
        assert _status(conn, tid) == before
    finally:
        conn.close()


def test_reentrant_validation_failure_does_not_contaminate_next_validation(
    completion_validators, kanban_home
):
    callbacks, _ = completion_validators
    conn = kb.connect()
    try:
        tid = _task(conn)
        before = _status(conn, tid)

        def reentrant_validator(context):
            kb.completion_validation_projection(
                conn,
                context.task_id,
                dict(context.metadata),
                summary=context.summary,
                source=context.source,
                dry_run=True,
            )
            return {"accepted": True}

        callbacks.append(reentrant_validator)
        with pytest.raises(kb.CompletionValidationError) as excinfo:
            kb.complete_task(conn, tid, summary="ok", metadata=_evidence())
        assert excinfo.value.code == "validator_reentrant"
        assert _status(conn, tid) == before

        callbacks[:] = [lambda context: {"accepted": True}]
        assert kb.complete_task(conn, tid, summary="ok", metadata=_evidence()) is True
        assert _status(conn, tid) == "done"
    finally:
        conn.close()
