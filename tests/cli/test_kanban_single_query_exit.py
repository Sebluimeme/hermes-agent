"""Regression tests for Kanban worker single-query exit codes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import cli as cli_mod
from cli import _single_query_exit_code
from hermes_cli.kanban_db import (
    KANBAN_GUARDRAIL_HALT_EXIT_CODE,
    KANBAN_RATE_LIMIT_EXIT_CODE,
)


def test_kanban_single_query_rate_limit_uses_tempfail(monkeypatch):
    """A failed Kanban worker provider limit must not look like rc=0."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_rate")

    assert _single_query_exit_code(
        {"failed": True, "failure_reason": "rate_limit"}
    ) == KANBAN_RATE_LIMIT_EXIT_CODE


def test_non_kanban_single_query_rate_limit_is_generic_failure(monkeypatch):
    """Only dispatcher-spawned workers use the Kanban sentinel."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    assert _single_query_exit_code(
        {"failed": True, "failure_reason": "rate_limit"}
    ) == 1


def test_successful_single_query_exits_zero(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_ok")

    assert _single_query_exit_code({"completed": True, "final_response": "ok"}) == 0


def test_kanban_guardrail_halt_uses_strategy_retry_sentinel(monkeypatch):
    """A controlled guardrail stop must not masquerade as a clean success."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_guardrail")

    assert _single_query_exit_code(
        {
            "failed": False,
            "final_response": "change strategy",
            "guardrail": {"code": "repeated_exact_failure_block"},
        }
    ) == KANBAN_GUARDRAIL_HALT_EXIT_CODE


def test_human_kanban_single_query_failed_turn_exits_nonzero(monkeypatch):
    """Human -q workers must not exit rc=0 after a failed raw turn."""
    calls = []

    class _Console:
        def print(self, *_args, **_kwargs):
            calls.append("query-label")

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.console = _Console()
            self.session_id = "single-query-session"
            self.agent = SimpleNamespace(
                session_id="single-query-session",
                platform="cli",
            )

        def _claim_active_session(self, surface, *, stderr=False):
            calls.append(("claim", surface, stderr))
            return True

        def _show_security_advisories(self):
            calls.append("advisories")

        def chat(self, query, images=None):
            calls.append(("chat", query, images))
            self._last_chat_result = {
                "failed": True,
                "failure_reason": "rate_limit",
                "error": "quota wall",
            }
            return ""

        def _print_exit_summary(self, clear_screen=True):
            calls.append(("summary", clear_screen))

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_rate")
    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cli_mod,
        "_finalize_single_query",
        lambda fake_cli: calls.append(("finalize", fake_cli.session_id)),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="work kanban task t_rate", quiet=False, toolsets="terminal")

    assert exc_info.value.code == KANBAN_RATE_LIMIT_EXIT_CODE
    assert calls == [
        ("claim", "cli", False),
        "query-label",
        "advisories",
        ("chat", "work kanban task t_rate", None),
        ("summary", False),
        ("finalize", "single-query-session"),
    ]
