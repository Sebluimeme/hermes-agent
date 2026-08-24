"""Regression tests for pre-fallback API failure capture."""

from types import SimpleNamespace

from agent.agent_runtime_helpers import extract_api_error_context


class _ApiError(Exception):
    def __init__(self, *, status_code, body, headers):
        self.status_code = status_code
        self.body = body
        self.response = SimpleNamespace(status_code=status_code, headers=headers or {})


def _error(*, status_code=429, body=None, headers=None):
    return _ApiError(status_code=status_code, body=body, headers=headers)


def test_retry_after_is_anchored_to_observation_and_identifies_runtime():
    context = extract_api_error_context(
        _error(headers={"Retry-After": "90"}),
        provider="anthropic",
        model="claude-sonnet-4-6",
        observed_at=1_000.0,
    )

    assert context["status_code"] == 429
    assert context["provider"] == "anthropic"
    assert context["model"] == "claude-sonnet-4-6"
    assert context["reset_at"] == 1_090.0
    assert context["reset_source"] == "http.retry-after"
    assert "key" not in context
    assert "token" not in context


def test_explicit_api_reset_wins_over_duration():
    context = extract_api_error_context(
        _error(
            body={
                "error": {
                    "code": "rate_limit_exceeded",
                    "reset_at": "2026-08-21T06:00:00Z",
                    "retry_after": 90,
                }
            }
        ),
        observed_at=1_000.0,
    )

    assert context["reset_at"] == "2026-08-21T06:00:00Z"
    assert context["reset_source"] == "api.reset_at"


def test_no_reset_is_invented_without_provider_source():
    context = extract_api_error_context(
        _error(body={"error": {"code": "rate_limit_exceeded"}}),
        observed_at=1_000.0,
    )

    assert "reset_at" not in context
    assert "reset_source" not in context
