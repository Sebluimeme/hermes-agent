"""Regression coverage for the explicit Claude 2 → Claude 1 → GPT route."""

from unittest.mock import MagicMock, patch

from agent.error_classifier import FailoverReason
from run_agent import AIAgent


NOTICE = "Claude 2 et Claude 1 indisponibles : relais sur GPT."


def _client(base_url: str, api_key: str = "test-key"):
    client = MagicMock()
    client.base_url = base_url
    client.api_key = api_key
    return client


def _agent():
    chain = [
        {
            "provider": "custom",
            "model": "claude-sonnet-5",
            "base_url": "http://fake-claude1:18765",
            "fallback_on": ["rate_limit", "billing", "auth", "auth_permanent", "upstream_rate_limit", "overloaded"],
            "suppress_fallback_notice": True,
        },
        {
            "provider": "openai-codex",
            "model": "gpt-5.5",
            "base_url": "http://fake-gpt/v1",
            "fallback_on": ["rate_limit", "billing", "auth", "auth_permanent", "upstream_rate_limit", "overloaded"],
            "transition_notice": NOTICE,
        },
    ]
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="http://fake-claude2:18766",
            provider="custom",
            model="claude-sonnet-5",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=chain,  # type: ignore[arg-type]
        )
    agent.client = _client("http://fake-claude2:18766")
    return agent


def _activate(agent, resolver, reason=FailoverReason.rate_limit):
    with (
        patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolver),
        patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda model, _provider: model),
    ):
        return agent._try_activate_fallback(reason=reason)


def test_claude2_success_keeps_the_primary_route_without_notification():
    agent = _agent()
    emitted = []
    agent._emit_status = lambda message: emitted.append(message)

    assert agent.provider == "custom"
    assert agent.base_url == "http://fake-claude2:18766"
    assert agent.model == "claude-sonnet-5"
    assert agent._fallback_index == 0
    assert emitted == []
    assert not hasattr(agent, "_fallback_trace")


def test_claude2_unavailable_uses_claude1_without_notification():
    agent = _agent()
    emitted = []
    agent._emit_status = lambda message: emitted.append(message)

    assert _activate(agent, [(_client("http://fake-claude1:18765"), "claude-sonnet-5")])

    assert agent.provider == "custom"
    assert agent.base_url == "http://fake-claude1:18765"
    assert agent.model == "claude-sonnet-5"
    assert emitted == []
    assert not hasattr(agent, "_fallback_trace")


def test_both_claude_lanes_unavailable_uses_gpt_and_notifies_once():
    agent = _agent()
    emitted = []
    agent._emit_status = lambda message: emitted.append(message)

    assert _activate(agent, [(_client("http://fake-claude1:18765"), "claude-sonnet-5")])
    assert _activate(agent, [(_client("http://fake-gpt/v1"), "gpt-5.5")])

    assert agent.provider == "openai-codex"
    assert agent.model == "gpt-5.5"
    assert emitted == [NOTICE]
    assert agent._pending_fallback_notice is None
    assert not hasattr(agent, "_fallback_trace")


def test_non_quota_error_does_not_try_another_provider():
    agent = _agent()
    with patch("agent.auxiliary_client.resolve_provider_client") as resolver:
        assert not agent._try_activate_fallback(reason=FailoverReason.format_error)

    resolver.assert_not_called()
    assert agent._fallback_index == 1
    assert agent.provider == "custom"
    assert not hasattr(agent, "_fallback_trace")


def test_local_timeout_does_not_consume_fallback_chain_or_blame_provider():
    agent = _agent()
    emitted = []
    agent._emit_status = lambda message: emitted.append(message)
    with patch("agent.auxiliary_client.resolve_provider_client") as resolver:
        assert not agent._try_activate_fallback(reason=FailoverReason.timeout)

    resolver.assert_not_called()
    assert agent._fallback_index == 1
    assert agent.provider == "custom"
    assert agent.base_url == "http://fake-claude2:18766"
    assert emitted == []
