"""Regression tests for Kanban worker single-query exit codes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import cli as cli_mod
from cli import _single_query_exit_code
from hermes_cli.kanban_db import (
    KANBAN_GUARDRAIL_HALT_EXIT_CODE,
    KANBAN_INTERRUPTED_EXIT_CODE,
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


def test_kanban_interrupted_turn_uses_neutral_resume_sentinel(monkeypatch):
    """An interrupted worker turn must not masquerade as clean success."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_interrupted")

    assert _single_query_exit_code(
        {
            "failed": False,
            "interrupted": True,
            "interrupt_message": "orchestrator continuation",
        }
    ) == KANBAN_INTERRUPTED_EXIT_CODE


def test_non_kanban_interrupted_single_query_keeps_normal_contract(monkeypatch):
    """Human one-shot interruptions are outside the Kanban exit protocol."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    assert _single_query_exit_code({"interrupted": True}) == 0


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


def test_kanban_goal_loop_rate_limit_stops_early_and_exits_tempfail(monkeypatch, tmp_path):
    """Regression for t_dbf31ad3 root cause 2.

    ``_run_turn`` used to drop ``result["failure_reason"]`` entirely, so a
    provider rate-limit/billing wall hit on turn 2+ of a Kanban goal-mode
    loop looked like "empty response, nothing to evaluate" to ``judge_goal``
    and the loop kept spending turns hitting the same 429 until the whole
    turn budget (12, then 40) was exhausted and the card landed in a sticky
    ``blocked_budget`` — instead of the bounded ``rate_limited`` exit
    (KANBAN_RATE_LIMIT_EXIT_CODE) the single-turn path already used, which
    the dispatcher's reap classifier and ``check_respawn_guard``'s
    ``rate_limit_cooldown`` check both rely on to avoid hammering an
    exhausted provider.
    """
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="do the thing",
            assignee="claude1",
            workspace_kind="scratch",
            goal_mode=True,
            goal_max_turns=12,
        )

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)

    call_count = {"n": 0}

    class FakeAgent:
        session_id = "sess-1"

        def run_conversation(self, *, user_message, conversation_history):
            call_count["n"] += 1
            return {
                "failed": True,
                "failure_reason": "rate_limit",
                "error": "429 rate limited, resets in 5m",
                "final_response": "",
            }

    class FakeCLI:
        def __init__(self):
            self.agent = FakeAgent()
            self.conversation_history = []
            self.session_id = "sess-1"

    fake_cli = FakeCLI()

    # judge_goal must not be the thing driving the early stop here (it would
    # normally treat an empty response as "continue" and keep looping) --
    # patch it to a scripted "continue" so any regression that stops relying
    # on the rate-limit signal is caught by the turn-budget assertion below
    # rather than by accident via a live/absent auxiliary judge call.
    monkeypatch.setattr(
        "hermes_cli.goals.judge_goal",
        lambda *a, **k: ("continue", "scripted", False, None, False),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_mod._run_kanban_goal_loop_q(fake_cli, "")

    assert exc_info.value.code == KANBAN_RATE_LIMIT_EXIT_CODE
    # The first turn's (empty) response was already passed in as
    # first_response; the loop must stop after the single rate-limited turn
    # 2 attempt, never spend the full 12-turn budget retrying the same 429.
    assert call_count["n"] == 1

    # The provider failure was captured as task-level evidence (feeds the
    # Coder relay / check_respawn_guard's rate_limit_cooldown), and the task
    # was never sticky-blocked -- a rate_limited exit must stay reclaimable.
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status in ("ready", "running")
        events = [
            r["kind"]
            for r in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
                (task_id,),
            )
        ]
    assert "claude_provider_reset" in events
    assert "blocked" not in events
