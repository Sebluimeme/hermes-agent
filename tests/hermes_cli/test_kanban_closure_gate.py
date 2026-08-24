"""End-to-end wiring tests for the LOT 4 closure evidence gate.

test_closure_evidence.py covers the pure decision table in isolation; this
file proves the gate is actually reached from both real entry points a
worker can use to close a card — the `kanban_complete` tool
(tools/kanban_tools.py::_handle_complete) and the CLI
(`hermes kanban complete`, hermes_cli/kanban.py::_cmd_complete) — against a
real (temp) sqlite kanban DB, so a future refactor that quietly bypasses the
gate at one of the two call sites gets caught here rather than in
production.
"""
from __future__ import annotations

import argparse
import json

import pytest


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """Isolated HERMES_HOME with one claimed, plain (non-goal_mode) task."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="closure-gate-test", assignee="test-worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    return tid


# ---------------------------------------------------------------------------
# Tool path: tools/kanban_tools.py::_handle_complete
# ---------------------------------------------------------------------------


def test_tool_complete_without_evidence_is_rejected(worker_env):
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_complete({"summary": "trust me, it's done"}))
    assert "error" in out
    assert "structured closure evidence" in out["error"]

    conn = kb.connect()
    try:
        assert kb.get_task(conn, worker_env).status == "running"
    finally:
        conn.close()


def test_tool_complete_with_artifact_evidence_succeeds(worker_env):
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_complete({
        "summary": "shipped the report",
        "metadata": {"artifacts": ["/tmp/report.pdf"]},
    }))
    assert out.get("ok") is True

    conn = kb.connect()
    try:
        assert kb.get_task(conn, worker_env).status == "done"
    finally:
        conn.close()


def test_tool_complete_with_test_evidence_succeeds(worker_env):
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_complete({
        "summary": "shipped the feature",
        "metadata": {"evidence": {"kind": "test", "detail": "pytest -q: 9 passed"}},
    }))
    assert out.get("ok") is True

    conn = kb.connect()
    try:
        assert kb.get_task(conn, worker_env).status == "done"
    finally:
        conn.close()


def test_tool_complete_after_review_lane_needs_no_extra_evidence(worker_env):
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    conn = kb.connect()
    try:
        assert kb.request_review(
            conn, worker_env, summary="ready for a look", force=True,
        )
    finally:
        conn.close()

    out = json.loads(kt._handle_complete({"summary": "approved by reviewer"}))
    assert out.get("ok") is True

    conn = kb.connect()
    try:
        assert kb.get_task(conn, worker_env).status == "done"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI path: hermes_cli/kanban.py::_cmd_complete
# ---------------------------------------------------------------------------


def test_cli_complete_without_evidence_is_rejected(worker_env, capsys):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban import _cmd_complete

    args = argparse.Namespace(
        task_ids=[worker_env], summary=None, result="trust me, it's done", metadata=None,
    )
    rc = _cmd_complete(args)
    assert rc != 0
    err = capsys.readouterr().err
    assert "structured closure evidence" in err

    conn = kb.connect()
    try:
        assert kb.get_task(conn, worker_env).status == "running"
    finally:
        conn.close()


def test_cli_complete_with_evidence_succeeds(worker_env):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban import _cmd_complete

    args = argparse.Namespace(
        task_ids=[worker_env],
        summary=None,
        result="done",
        metadata=json.dumps({"evidence": {"kind": "canary", "detail": "probe: OK"}}),
    )
    rc = _cmd_complete(args)
    assert rc == 0

    conn = kb.connect()
    try:
        assert kb.get_task(conn, worker_env).status == "done"
    finally:
        conn.close()
