"""Kanban worker → gateway approval bridge, from the ``file_tools`` call site.

Before t_f98bc92d, ``_present_protected_instruction_approval`` running inside
a Kanban worker process (no live gateway turn, no interactive CLI) fell
straight to a BLOCKED refusal with a durable-grant hint — the request never
reached a human. It now first tries ``tools.worker_approval`` when
``HERMES_KANBAN_TASK`` is set. These tests pin that branch's contract without
touching the real file transport (covered separately in
``test_worker_approval.py``):

* a Kanban worker with no interactive channel calls into the bridge and
  honors whatever it returns (approve / deny / timeout);
* anything NOT a Kanban worker (no ``HERMES_KANBAN_TASK``) must never call
  the bridge at all — "no solicitation outside a Kanban worker";
* a pre-existing durable grant (``has_instruction_edit_authorization``)
  short-circuits before any of this and must never reach the bridge either.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_live_channels(monkeypatch):
    """Force both the gateway-notify and CLI-callback paths to be absent so
    ``_present_protected_instruction_approval`` falls through to the
    Kanban-worker bridge branch under test."""
    from tools import approval

    monkeypatch.setattr(approval, "get_current_session_key", lambda: "no-such-session")
    with approval._lock:
        approval._gateway_notify_cbs.pop("no-such-session", None)

    import tools.terminal_tool as term

    monkeypatch.setattr(term, "_get_approval_callback", lambda: None)
    yield


def test_kanban_worker_bridge_approves(monkeypatch):
    from tools import file_tools as ft

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_bridge_ok")
    calls = []

    def _fake_request_decision(**kwargs):
        calls.append(kwargs)
        return {"resolved": True, "choice": "once", "reason": None}

    import tools.worker_approval as wa

    monkeypatch.setattr(wa, "request_decision", _fake_request_decision)

    result = ft._present_protected_instruction_approval(
        ["AGENTS.md"], ["/repo/AGENTS.md"],
    )

    assert result is None  # approved
    assert len(calls) == 1
    assert calls[0]["task_id"] == "t_bridge_ok"
    assert calls[0]["paths"] == ["/repo/AGENTS.md"]


def test_kanban_worker_bridge_denied(monkeypatch):
    from tools import file_tools as ft

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_bridge_deny")
    import tools.worker_approval as wa

    monkeypatch.setattr(
        wa, "request_decision",
        lambda **kwargs: {"resolved": True, "choice": "deny", "reason": None},
    )

    result = ft._present_protected_instruction_approval(
        ["AGENTS.md"], ["/repo/AGENTS.md"],
    )

    assert result is not None
    assert "was denied by the user" in result


def test_kanban_worker_bridge_timeout_denies_with_grant_hint(monkeypatch):
    """Silence is not consent: an unresolved bridge outcome BLOCKs and still
    surfaces the durable-grant hint (the fallback the rule explicitly keeps)."""
    from tools import file_tools as ft

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_bridge_timeout")
    import tools.worker_approval as wa

    monkeypatch.setattr(
        wa, "request_decision",
        lambda **kwargs: {
            "resolved": False, "choice": None,
            "reason": "no response within the approval window",
        },
    )

    result = ft._present_protected_instruction_approval(
        ["AGENTS.md"], ["/repo/AGENTS.md"],
    )

    assert result is not None
    assert "Silence is not consent" in result
    assert "authorize-instruction-edit t_bridge_timeout" in result


def test_no_kanban_task_never_calls_the_bridge(monkeypatch):
    """Outside a Kanban worker, the bridge must not be consulted at all —
    the plain no-channel refusal (with grant hint, task id absent) stands."""
    from tools import file_tools as ft

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    import tools.worker_approval as wa

    def _must_not_be_called(**kwargs):
        raise AssertionError("worker_approval.request_decision must not be called")

    monkeypatch.setattr(wa, "request_decision", _must_not_be_called)

    result = ft._present_protected_instruction_approval(
        ["AGENTS.md"], ["/repo/AGENTS.md"],
    )

    assert result is not None
    assert "no interactive user or gateway is present" in result


def test_preexisting_durable_grant_never_reaches_the_bridge(monkeypatch, tmp_path):
    """A durable ``authorize-instruction-edit`` grant is checked BEFORE any
    prompt is even considered — the bridge must never fire for it."""
    from tools import file_tools as ft
    import tools.worker_approval as wa

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_bridge_granted")

    def _must_not_be_called(**kwargs):
        raise AssertionError("worker_approval.request_decision must not be called")

    monkeypatch.setattr(wa, "request_decision", _must_not_be_called)

    target = tmp_path / "AGENTS.md"
    target.write_text("x")

    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(
        kb, "has_instruction_edit_authorization",
        lambda conn, task_id, path: task_id == "t_bridge_granted" and path == str(target),
    )
    monkeypatch.setattr(ft, "_resolve_path_for_task", lambda path, task_id: target)

    result = ft._request_protected_instruction_approval(
        ["AGENTS.md"], [str(target)], "t_bridge_granted",
    )

    assert result is None  # approved via the durable grant, no prompt at all
