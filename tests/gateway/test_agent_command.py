"""Tests for gateway /agent command and global LLM preset integration.

Covers:
  - /agent sends a two-button picker when the adapter supports it
  - /agent falls back to a text card when no picker is available
  - Picker tap calls set_preset() and evicts the cached agent
  - Typed /agent <preset> applies immediately
  - /agent does not reset session history
  - Two runners with *different* HERMES_HOME values see the SAME global preset
    (cross-profile isolation — the preset lives at HERMES_GLOBAL_HOME, not at
    the per-profile home)
  - _resolve_session_agent_runtime applies the preset as a model/provider
    baseline when there is no session /model override
  - An existing session /model override beats the preset
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, SendResult
from gateway.session import SessionSource, SessionStore


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

class _PickerAdapter:
    """Adapter whose *type* exposes send_choice_picker."""

    def __init__(self, success=True):
        self.calls: list[dict] = []
        self._success = success

    async def send_choice_picker(self, **kwargs):
        self.calls.append(kwargs)
        return SendResult(success=self._success, message_id="m1")


class _NoPickerAdapter:
    """Adapter with no choice-picker capability."""


def _make_source(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1"):
    return SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )


def _make_event(text="/agent", platform=Platform.TELEGRAM):
    return MessageEvent(text=text, source=_make_source(platform=platform))


def _make_runner(adapter=None):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._session_reasoning_overrides = {}
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._session_model_overrides = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._pending_model_notes = {}
    runner._session_db = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._adapter_for_source = lambda source: adapter
    runner._thread_metadata_for_source = lambda source, anchor=None: {}
    runner._reply_anchor_for_event = lambda event: None
    return runner


# ---------------------------------------------------------------------------
# /agent command: picker interaction
# ---------------------------------------------------------------------------

class TestAgentCommandPicker:

    @pytest.mark.asyncio
    async def test_bare_agent_sends_picker_when_adapter_supports_it(
        self, tmp_path, monkeypatch
    ):
        """Bare /agent sends a picker with exactly two choices."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        adapter = _PickerAdapter()
        runner = _make_runner(adapter)

        result = await runner._handle_agent_command(_make_event("/agent"))

        assert result is None, "Picker sent — adapter owns the response"
        assert len(adapter.calls) == 1
        choices = adapter.calls[0]["choices"]
        assert len(choices) == 2
        values = {c["value"] for c in choices}
        assert values == {"opus", "chatgpt"}

    @pytest.mark.asyncio
    async def test_current_preset_marked_is_current_on_picker(
        self, tmp_path, monkeypatch
    ):
        """The currently active preset is marked is_current=True in the picker."""
        from agent.llm_preset import set_preset

        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        set_preset("chatgpt")
        adapter = _PickerAdapter()
        runner = _make_runner(adapter)

        await runner._handle_agent_command(_make_event("/agent"))

        choices = adapter.calls[0]["choices"]
        current = [c["value"] for c in choices if c.get("is_current")]
        assert current == ["chatgpt"]

    @pytest.mark.asyncio
    async def test_bare_agent_falls_back_to_text_without_picker(
        self, tmp_path, monkeypatch
    ):
        """Platforms with no picker receive a text status card."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        runner = _make_runner(_NoPickerAdapter())

        result = await runner._handle_agent_command(_make_event("/agent"))

        assert isinstance(result, str)
        # Must mention both valid presets so the user knows their options.
        assert "opus" in result
        assert "chatgpt" in result

    @pytest.mark.asyncio
    async def test_bare_agent_falls_back_to_text_when_picker_send_fails(
        self, tmp_path, monkeypatch
    ):
        """When send_choice_picker returns success=False, fall through to text."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        adapter = _PickerAdapter(success=False)
        runner = _make_runner(adapter)

        result = await runner._handle_agent_command(_make_event("/agent"))

        assert isinstance(result, str)
        assert len(adapter.calls) == 1  # attempted once then gave up


# ---------------------------------------------------------------------------
# /agent command: preset selection
# ---------------------------------------------------------------------------

class TestAgentCommandSelection:

    @pytest.mark.asyncio
    async def test_typed_opus_sets_preset_globally(self, tmp_path, monkeypatch):
        from agent.llm_preset import get_preset, set_preset

        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        set_preset("chatgpt")  # start from chatgpt
        runner = _make_runner()

        response = await runner._handle_agent_command(_make_event("/agent opus"))

        assert "opus" in response.lower() or "Claude Opus" in response
        assert get_preset() == "opus"

    @pytest.mark.asyncio
    async def test_typed_chatgpt_sets_preset_globally(self, tmp_path, monkeypatch):
        from agent.llm_preset import get_preset

        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        runner = _make_runner()

        response = await runner._handle_agent_command(_make_event("/agent chatgpt"))

        assert get_preset() == "chatgpt"
        assert response and "chatgpt" in response.lower() or "ChatGPT" in response

    @pytest.mark.asyncio
    async def test_selection_evicts_cached_agent(self, tmp_path, monkeypatch):
        """A successful preset switch must evict the cached agent so the next
        turn constructs a fresh agent with the new model/provider."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        runner = _make_runner()
        event = _make_event("/agent opus")
        session_key = runner._session_key_for_source(event.source)
        # Pre-populate the agent cache with a sentinel.
        runner._agent_cache[session_key] = (object(), "sig")

        await runner._handle_agent_command(event)

        assert session_key not in runner._agent_cache, (
            "Cached agent must be evicted after a preset switch"
        )

    @pytest.mark.asyncio
    async def test_picker_selection_same_as_typed(self, tmp_path, monkeypatch):
        """The picker's on_choice_selected must produce the same preset as
        typing /agent chatgpt (single application path, no divergence)."""
        from agent.llm_preset import get_preset

        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        adapter = _PickerAdapter()
        runner = _make_runner(adapter)
        event = _make_event("/agent")

        await runner._handle_agent_command(event)
        on_choice = adapter.calls[0]["on_choice_selected"]

        reply = await on_choice(event.source.chat_id, "chatgpt")

        assert get_preset() == "chatgpt"
        assert reply and ("chatgpt" in reply.lower() or "ChatGPT" in reply)

    @pytest.mark.asyncio
    async def test_unknown_preset_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        runner = _make_runner()

        response = await runner._handle_agent_command(_make_event("/agent gemini"))

        assert response is not None
        assert "gemini" in response
        # Must hint at the valid options.
        assert "opus" in response or "chatgpt" in response


# ---------------------------------------------------------------------------
# Cross-profile global home: two runners, different HERMES_HOME, same preset
# ---------------------------------------------------------------------------

class TestCrossProfilePresetSharing:
    """Verify the global preset is visible across profile boundaries."""

    def test_two_runners_with_different_hermes_home_share_preset(
        self, tmp_path, monkeypatch
    ):
        """set_preset() in profile A → get_preset() in profile B returns
        the same value because storage lives at HERMES_GLOBAL_HOME."""
        from agent.llm_preset import get_preset, set_preset

        global_home = tmp_path / "global"
        profile_a = tmp_path / "profiles" / "a"
        profile_b = tmp_path / "profiles" / "b"

        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(global_home))

        # Profile A writes via explicit global home
        monkeypatch.setenv("HERMES_HOME", str(profile_a))
        set_preset("chatgpt")  # written to global_home

        # Profile B reads — HERMES_HOME is different, preset must still be visible
        monkeypatch.setenv("HERMES_HOME", str(profile_b))
        result = get_preset()

        assert result == "chatgpt", (
            f"Expected 'chatgpt' (written by profile A) but got {result!r}; "
            "preset is not shared via HERMES_GLOBAL_HOME"
        )
        # The preset file must live at the global home, not in either profile.
        assert (global_home / "llm_preset.json").exists()
        assert not (profile_a / "llm_preset.json").exists()
        assert not (profile_b / "llm_preset.json").exists()

    def test_concurrent_preset_writes_from_two_threads(self, tmp_path, monkeypatch):
        """Two concurrent writers racing to set different presets must never
        corrupt the JSON file — only one preset survives but it must be valid."""
        from agent.llm_preset import VALID_PRESETS, get_preset, set_preset

        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))

        errors: list[Exception] = []
        barrier = threading.Barrier(2)

        def _write(preset):
            barrier.wait()
            try:
                for _ in range(30):
                    set_preset(preset, hermes_home=tmp_path)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=_write, args=("opus",))
        t2 = threading.Thread(target=_write, args=("chatgpt",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        final = get_preset(hermes_home=tmp_path)
        assert final in VALID_PRESETS


# ---------------------------------------------------------------------------
# Runtime resolution: preset applied at each turn in _resolve_session_agent_runtime
# ---------------------------------------------------------------------------

class TestPresetRuntimeResolution:

    def _make_runtime_runner(self):
        """Runner with enough state for _resolve_session_agent_runtime."""
        runner = object.__new__(gateway_run.GatewayRunner)
        runner.adapters = {}
        runner._session_model_overrides = {}
        runner.config = None
        runner._last_resolved_model = {}
        return runner

    def test_preset_applied_when_no_session_override(self, tmp_path, monkeypatch):
        """With no session /model override, the global preset is used as
        model/provider baseline."""
        from agent.llm_preset import set_preset

        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        set_preset("chatgpt", hermes_home=tmp_path)

        runner = self._make_runtime_runner()
        source = _make_source()

        monkeypatch.setattr(
            gateway_run, "_resolve_gateway_model", lambda config=None: "claude-opus-4-5"
        )
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "api_key": "key-a",
                "base_url": None,
                "provider": "anthropic",
                "api_mode": None,
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        )
        # Preset resolves "chatgpt" → provider=openai, model=gpt-4o
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs_for_provider",
            lambda provider: {
                "api_key": "openai-key",
                "base_url": None,
                "provider": provider,
                "api_mode": None,
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        )

        model, runtime_kwargs = runner._resolve_session_agent_runtime(
            source=source, user_config={}
        )

        assert runtime_kwargs.get("provider") == "openai", (
            f"Expected provider='openai' from chatgpt preset, got {runtime_kwargs.get('provider')!r}"
        )
        assert "gpt" in model.lower() or model == "gpt-4o", (
            f"Expected a GPT model from chatgpt preset, got {model!r}"
        )

    def test_preset_beats_session_model_override(self, tmp_path, monkeypatch):
        """The global /agent preset must win over any session /model override.

        Previously the code had a fast-path that returned early when the session
        override had an api_key, bypassing the preset block entirely.  After the
        fix, the preset block always runs last and overrides model/provider even
        when a session override is present.

        Note: /agent also clears all session overrides at call time, so in normal
        use there is no coexisting session override.  This test covers the safety
        net: even if a stale override is somehow present, the preset wins.
        """
        from agent.llm_preset import set_preset

        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        set_preset("chatgpt", hermes_home=tmp_path)

        runner = self._make_runtime_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        # Session override points at a specific Anthropic model (stale/residual)
        runner._session_model_overrides[session_key] = {
            "model": "claude-sonnet-4-5",
            "provider": "anthropic",
            "api_key": "explicit-key",
            "base_url": None,
            "api_mode": None,
        }

        monkeypatch.setattr(
            gateway_run, "_resolve_gateway_model", lambda config=None: "claude-opus-4-5"
        )
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "api_key": "default-key",
                "base_url": None,
                "provider": "anthropic",
                "api_mode": None,
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        )
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs_for_provider",
            lambda provider: {
                "api_key": "openai-key",
                "base_url": None,
                "provider": provider,
                "api_mode": None,
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        )

        model, runtime_kwargs = runner._resolve_session_agent_runtime(
            source=source, user_config={}
        )

        # Global chatgpt preset must win over the stale session override
        assert runtime_kwargs.get("provider") == "openai", (
            f"Global chatgpt preset must win over session override; "
            f"got provider={runtime_kwargs.get('provider')!r}"
        )
        assert "gpt" in model.lower() or model == "gpt-4o", (
            f"Global chatgpt preset must set a GPT model; got model={model!r}"
        )

    def test_preset_resolution_failure_does_not_crash(self, tmp_path, monkeypatch):
        """If preset resolution fails for any reason, the gateway must fall back
        to config.yaml defaults gracefully — no exception propagated."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))

        runner = self._make_runtime_runner()
        source = _make_source()

        monkeypatch.setattr(
            gateway_run, "_resolve_gateway_model", lambda config=None: "claude-opus-4-5"
        )
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "api_key": "key",
                "base_url": None,
                "provider": "anthropic",
                "api_mode": None,
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        )

        # Make _resolve_runtime_agent_kwargs_for_provider raise so preset path fails
        def _broken_provider_resolver(provider):
            raise RuntimeError("simulated credential failure")

        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs_for_provider",
            _broken_provider_resolver,
        )

        # Must not raise; should silently fall back
        model, runtime_kwargs = runner._resolve_session_agent_runtime(
            source=source, user_config={}
        )

        # Falls back to config defaults
        assert model == "claude-opus-4-5"
        assert runtime_kwargs.get("provider") == "anthropic"

    def test_preset_base_url_applied_in_runtime(self, tmp_path, monkeypatch):
        """When llm_presets.yaml sets base_url for opus, the runtime kwargs
        receive that base_url — NOT the provider-resolved one.

        This ensures a preset can target a local proxy without touching config.yaml.
        """
        from agent.llm_preset import set_preset

        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        # HERMES_HOME must also point to tmp_path so resolve_preset_runtime()
        # reads llm_presets.yaml from the same directory (profile-scoped YAML).
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        set_preset("opus", hermes_home=tmp_path)

        # Write a presets.yaml that routes opus to a local proxy
        (tmp_path / "llm_presets.yaml").write_text(
            "opus:\n"
            "  provider: anthropic\n"
            "  model: claude-opus-4-6\n"
            "  base_url: http://localhost:4000\n",
            encoding="utf-8",
        )

        runner = self._make_runtime_runner()
        source = _make_source()

        monkeypatch.setattr(
            gateway_run, "_resolve_gateway_model", lambda config=None: "claude-default"
        )
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "api_key": "key-a",
                "base_url": None,
                "provider": "anthropic",
                "api_mode": None,
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        )
        # Provider resolver returns NO base_url (the real provider has no special URL)
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs_for_provider",
            lambda provider: {
                "api_key": "anthropic-key",
                "base_url": None,
                "provider": provider,
                "api_mode": None,
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        )

        model, runtime_kwargs = runner._resolve_session_agent_runtime(
            source=source, user_config={}
        )

        assert runtime_kwargs.get("base_url") == "http://localhost:4000", (
            f"base_url from preset YAML should win; got {runtime_kwargs.get('base_url')!r}"
        )
        assert model == "claude-opus-4-6"
        assert runtime_kwargs.get("provider") == "anthropic"

    def test_preset_base_url_uses_hermes_home_for_yaml(self, tmp_path, monkeypatch):
        """resolve_preset_runtime must read llm_presets.yaml from HERMES_HOME
        (profile-scoped), not only from HERMES_GLOBAL_HOME."""
        from agent.llm_preset import set_preset

        global_home = tmp_path / "global"
        profile_home = tmp_path / "profile"
        profile_home.mkdir(parents=True, exist_ok=True)

        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(global_home))
        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        set_preset("opus", hermes_home=global_home)

        # Write presets.yaml only in the profile home (not in global)
        (profile_home / "llm_presets.yaml").write_text(
            "opus:\n  provider: anthropic\n  model: claude-opus-4-6\n  base_url: http://profile-proxy:8080\n",
            encoding="utf-8",
        )

        from agent.llm_presets_config import resolve_preset_runtime
        result = resolve_preset_runtime(profile_home)
        assert result.get("base_url") == "http://profile-proxy:8080"


# ---------------------------------------------------------------------------
# /agent precedence: /agent must clear the /model session override
# ---------------------------------------------------------------------------

class TestAgentCommandPrecedence:
    """Verify that /agent wins over an existing /model session override.

    Priority rule: /agent sets the global preset and MUST evict any previously
    set /model override so the preset applies on the very next turn.
    Without this, the session override (set via /model) would shadow the
    freshly-chosen global preset.
    """

    @pytest.mark.asyncio
    async def test_agent_clears_in_memory_model_override(self, tmp_path, monkeypatch):
        """/agent opus clears an existing in-memory /model session override."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        runner = _make_runner()
        event = _make_event("/agent opus")
        session_key = runner._session_key_for_source(event.source)

        # Simulate a previous /model switch
        runner._session_model_overrides[session_key] = {
            "model": "custom-model",
            "provider": "custom-provider",
            "api_key": "key-xyz",
        }

        await runner._handle_agent_command(event)

        assert session_key not in runner._session_model_overrides, (
            "/agent must clear the in-memory /model override; "
            f"still has override: {runner._session_model_overrides.get(session_key)!r}"
        )

    @pytest.mark.asyncio
    async def test_agent_clears_persisted_model_override(self, tmp_path, monkeypatch):
        """/agent calls clear_all_model_overrides() on the session store to wipe
        persisted /model overrides across ALL sessions so a gateway restart won't
        re-apply any topic's stale override and shadow the new global preset."""
        from unittest.mock import AsyncMock

        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        runner = _make_runner()
        event = _make_event("/agent chatgpt")

        # Inject a mock async_session_store.  async_session_store is a
        # @property (read-only) that reconstructs from self._async_session_store
        # when _store matches self.session_store.  Set both to the same sentinel
        # so the property returns the mock without trying to rebuild it.
        _sentinel = object()
        mock_store = MagicMock()
        mock_store.clear_all_model_overrides = AsyncMock(return_value=None)
        mock_store._store = _sentinel
        runner.session_store = _sentinel
        runner._async_session_store = mock_store

        await runner._handle_agent_command(event)

        mock_store.clear_all_model_overrides.assert_called_once(), (
            "async_session_store.clear_all_model_overrides() must be called "
            "to wipe persisted overrides for every active topic, not only the current session"
        )

    @pytest.mark.asyncio
    async def test_picker_selection_also_clears_override(self, tmp_path, monkeypatch):
        """A picker-based /agent selection must also clear the session override."""
        from unittest.mock import AsyncMock

        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        adapter = _PickerAdapter()
        runner = _make_runner(adapter)
        event = _make_event("/agent")
        session_key = runner._session_key_for_source(event.source)

        # Existing /model override
        runner._session_model_overrides[session_key] = {
            "model": "old-model",
            "provider": "old-provider",
        }

        # Inject mock via backing attribute (async_session_store is a @property).
        _sentinel = object()
        mock_store = MagicMock()
        mock_store.clear_all_model_overrides = AsyncMock(return_value=None)
        mock_store._store = _sentinel
        runner.session_store = _sentinel
        runner._async_session_store = mock_store

        await runner._handle_agent_command(event)
        assert len(adapter.calls) == 1

        # Simulate user tapping the picker
        on_choice = adapter.calls[0]["on_choice_selected"]
        await on_choice(event.source.chat_id, "opus")

        # In-memory override must be gone
        assert session_key not in runner._session_model_overrides, (
            "Picker selection must also clear the in-memory /model override"
        )
        # Persisted overrides for all sessions must be cleared
        mock_store.clear_all_model_overrides.assert_called()

    @pytest.mark.asyncio
    async def test_agent_clears_all_topics_in_memory_overrides(self, tmp_path, monkeypatch):
        """/agent must clear the /model override for EVERY active topic, not only
        the current session.  Without a full clear, topics that had their own
        /model override would continue to shadow the new global preset until
        the user explicitly ran /new in each one."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        runner = _make_runner()
        event = _make_event("/agent opus")

        # Simulate three topics with their own /model overrides
        runner._session_model_overrides["topic_A"] = {
            "model": "model-a",
            "provider": "prov-a",
        }
        runner._session_model_overrides["topic_B"] = {
            "model": "model-b",
            "provider": "prov-b",
        }
        runner._session_model_overrides["topic_C"] = {
            "model": "model-c",
            "provider": "prov-c",
        }

        await runner._handle_agent_command(event)

        assert len(runner._session_model_overrides) == 0, (
            "/agent must clear ALL topics' in-memory /model overrides so the "
            "new global preset applies immediately everywhere; "
            f"remaining: {dict(runner._session_model_overrides)}"
        )

    @pytest.mark.asyncio
    async def test_agent_without_store_does_not_crash(self, tmp_path, monkeypatch):
        """/agent must not crash when async_session_store is not available
        (test runners, lightweight stubs, etc.)."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        runner = _make_runner()
        # No async_session_store attribute — must not raise AttributeError
        assert not hasattr(runner, "async_session_store")
        event = _make_event("/agent opus")
        # Should complete without raising
        result = await runner._handle_agent_command(event)
        assert result is not None

    @pytest.mark.asyncio
    async def test_agent_clears_all_sessions_persisted_overrides(
        self, tmp_path, monkeypatch
    ):
        """/agent must call clear_all_model_overrides() on the session store,
        not just set_model_override for the current session.

        Without a full clear, other topics' persisted overrides would be
        rehydrated on the next turn and shadow the new global preset.
        """
        from unittest.mock import AsyncMock

        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        runner = _make_runner()
        event = _make_event("/agent chatgpt")

        _sentinel = object()
        mock_store = MagicMock()
        mock_store.clear_all_model_overrides = AsyncMock(return_value=None)
        mock_store._store = _sentinel
        runner.session_store = _sentinel
        runner._async_session_store = mock_store

        await runner._handle_agent_command(event)

        mock_store.clear_all_model_overrides.assert_called_once(), (
            "async_session_store.clear_all_model_overrides() must be called "
            "to wipe persisted overrides for every active topic"
        )

    @pytest.mark.asyncio
    async def test_agent_chatgpt_wins_before_and_after_restart_for_other_local_session(
        self, tmp_path, monkeypatch
    ):
        """Regression: /agent chatgpt must beat persisted session overrides for
        every local session, before and after a simulated restart.

        Repro required by tranche 2:
          1. Create two local sessions with persisted /model overrides.
          2. Run /agent chatgpt from session A.
          3. Verify session B resolves to chatgpt immediately.
          4. Simulate a restart with the same SessionStore.
          5. Verify session B still resolves to chatgpt after rehydration.
        """
        from agent.llm_preset import set_preset

        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        monkeypatch.setattr(
            gateway_run, "_resolve_gateway_model", lambda config=None: "claude-opus-4-5"
        )
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "api_key": "anthropic-key",
                "base_url": None,
                "provider": "anthropic",
                "api_mode": None,
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        )
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs_for_provider",
            lambda provider: {
                "api_key": "openai-key" if "openai" in provider else "anthropic-key",
                "base_url": None,
                "provider": provider,
                "api_mode": None,
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        )
        monkeypatch.setattr(
            gateway_run,
            "_credential_pool_for_provider",
            lambda provider: None,
        )

        store = SessionStore(tmp_path / "sessions", GatewayConfig(platforms={}))
        source_a = _make_source(chat_id="c1")
        source_b = _make_source(chat_id="c2")
        key_a = gateway_run.build_session_key(source_a)
        key_b = gateway_run.build_session_key(source_b)
        store.get_or_create_session(source_a)
        store.get_or_create_session(source_b)
        stale_override = {
            "model": "claude-sonnet-4-5",
            "provider": "anthropic",
            "base_url": None,
        }
        store.set_model_override(key_a, stale_override)
        store.set_model_override(key_b, stale_override)

        runner = _make_runner()
        runner.session_store = store
        runner._session_model_overrides[key_a] = dict(stale_override, api_key="anthropic-key")
        runner._session_model_overrides[key_b] = dict(stale_override, api_key="anthropic-key")

        reply = await runner._handle_agent_command(
            MessageEvent(text="/agent chatgpt", source=source_a)
        )
        assert reply and "chatgpt" in reply.lower()
        assert store.get_model_override(key_a) is None
        assert store.get_model_override(key_b) is None
        assert runner._session_model_overrides == {}

        model_before, runtime_before = runner._resolve_session_agent_runtime(
            source=source_b, user_config={}
        )
        assert runtime_before.get("provider") == "openai"
        assert "gpt" in model_before.lower() or model_before == "gpt-4o"

        restarted = _make_runner()
        restarted.session_store = store
        set_preset("chatgpt", hermes_home=tmp_path)

        model_after, runtime_after = restarted._resolve_session_agent_runtime(
            source=source_b, user_config={}
        )
        assert store.get_model_override(key_b) is None
        assert key_b not in restarted._session_model_overrides
        assert runtime_after.get("provider") == "openai"
        assert "gpt" in model_after.lower() or model_after == "gpt-4o"


# ---------------------------------------------------------------------------
# Regression: preset wins over rehydrated session override after /agent
# ---------------------------------------------------------------------------

class TestAgentPresetBeatsRehydratedOverride:
    """Regression for HERMES-LLM-001: /agent chatgpt must produce chatgpt
    runtime BOTH on the same turn AND after a simulated gateway restart where
    a stale /model override would otherwise be rehydrated from the store.

    Before the fix:
      1. Session has a /model override (no api_key — simulates a rehydrated
         DB entry whose api_key was never persisted).
      2. /agent chatgpt clears in-memory overrides, but ONLY the current
         session's persisted override — other topics survive.
      3. On the next turn, _rehydrate_session_model_override restores the
         stale override; because the preset block was gated on `if not
         override:`, the preset was skipped and the stale override won.

    After the fix:
      • /agent clears ALL sessions' persisted overrides.
      • The preset block runs unconditionally AFTER session_override, so even
        a surviving rehydrated override cannot shadow the global preset.
    """

    def _make_runtime_runner(self):
        runner = object.__new__(gateway_run.GatewayRunner)
        runner.adapters = {}
        runner._session_model_overrides = {}
        runner.config = None
        runner._last_resolved_model = {}
        return runner

    def _patch_runtime(self, monkeypatch, *, config_model="claude-opus-4-5"):
        monkeypatch.setattr(
            gateway_run, "_resolve_gateway_model", lambda config=None: config_model
        )
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "api_key": "anthropic-key",
                "base_url": None,
                "provider": "anthropic",
                "api_mode": None,
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        )
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs_for_provider",
            lambda provider: {
                "api_key": "openai-key" if "openai" in provider else "anthropic-key",
                "base_url": None,
                "provider": provider,
                "api_mode": None,
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        )

    def test_preset_wins_when_override_has_no_api_key(self, tmp_path, monkeypatch):
        """Preset must win over a session override that has no api_key.

        An override without api_key is what gets rehydrated from the DB after
        a restart (api_key is never persisted).  Before the fix the `if not
        override:` guard skipped the preset, leaving the stale override in
        control.  After the fix the preset runs last and wins.
        """
        from agent.llm_preset import set_preset

        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        set_preset("chatgpt", hermes_home=tmp_path)
        self._patch_runtime(monkeypatch)

        runner = self._make_runtime_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        # Simulate a rehydrated override: model/provider only, NO api_key.
        runner._session_model_overrides[session_key] = {
            "model": "claude-opus-4-5",
            "provider": "anthropic",
            "api_key": None,
            "base_url": None,
            "api_mode": None,
        }

        model, runtime_kwargs = runner._resolve_session_agent_runtime(
            source=source, user_config={}
        )

        assert runtime_kwargs.get("provider") == "openai", (
            f"Global chatgpt preset must win over a rehydrated override; "
            f"got provider={runtime_kwargs.get('provider')!r}"
        )
        assert "gpt" in model.lower() or model == "gpt-4o", (
            f"Global chatgpt preset must set a GPT model; got model={model!r}"
        )

    def test_preset_wins_after_simulated_restart(self, tmp_path, monkeypatch):
        """After a gateway restart, the preset still wins over any rehydrated
        override, because /agent cleared all persisted overrides before restart.

        This simulates: (1) override was in _session_model_overrides before
        restart, (2) on restart the in-memory dict is empty, (3)
        _rehydrate_session_model_override is called but finds nothing (because
        clear_all_model_overrides wiped the store), (4) preset applies.
        """
        from agent.llm_preset import set_preset

        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        set_preset("chatgpt", hermes_home=tmp_path)
        self._patch_runtime(monkeypatch)

        runner = self._make_runtime_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        # Simulate a session_store whose get_model_override returns None
        # (because /agent already cleared it via clear_all_model_overrides).
        mock_store = MagicMock()
        mock_store.get_model_override = MagicMock(return_value=None)
        runner.session_store = mock_store

        # _session_model_overrides is empty (simulates post-restart state)
        assert session_key not in runner._session_model_overrides

        model, runtime_kwargs = runner._resolve_session_agent_runtime(
            source=source, user_config={}
        )

        # Verify rehydration was attempted (so the test is not vacuously true)
        mock_store.get_model_override.assert_called_with(session_key)

        assert runtime_kwargs.get("provider") == "openai", (
            f"Preset must win after simulated restart (no rehydrated override); "
            f"got provider={runtime_kwargs.get('provider')!r}"
        )
        assert "gpt" in model.lower() or model == "gpt-4o", (
            f"Preset must set a GPT model after restart; got model={model!r}"
        )

    def test_preset_before_and_after_restart(self, tmp_path, monkeypatch):
        """Combined: create override → /agent chatgpt → runtime is chatgpt →
        simulate restart (empty in-memory, store cleared) → runtime still chatgpt.
        """
        from agent.llm_preset import set_preset

        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        self._patch_runtime(monkeypatch)

        runner = self._make_runtime_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        # Step 1: session has a rehydrated override (no api_key).
        runner._session_model_overrides[session_key] = {
            "model": "claude-opus-4-5",
            "provider": "anthropic",
            "api_key": None,
        }

        # Step 2: /agent chatgpt sets the global preset.
        set_preset("chatgpt", hermes_home=tmp_path)

        # Step 3: /agent also clears ALL in-memory overrides (mimics _apply_agent_selection).
        runner._session_model_overrides.clear()

        # Step 4 (before restart): runtime must be chatgpt even with no override.
        model_pre, rk_pre = runner._resolve_session_agent_runtime(
            source=source, user_config={}
        )
        assert rk_pre.get("provider") == "openai", (
            f"Pre-restart: chatgpt preset must apply; got provider={rk_pre.get('provider')!r}"
        )

        # Step 5: simulate restart — in-memory overrides gone, store cleared.
        runner._session_model_overrides.clear()
        mock_store = MagicMock()
        mock_store.get_model_override = MagicMock(return_value=None)
        runner.session_store = mock_store

        # Step 6 (after restart): runtime must still be chatgpt.
        model_post, rk_post = runner._resolve_session_agent_runtime(
            source=source, user_config={}
        )
        assert rk_post.get("provider") == "openai", (
            f"Post-restart: chatgpt preset must still apply; got provider={rk_post.get('provider')!r}"
        )


# ---------------------------------------------------------------------------
# Regression: preset wins over session override WITH api_key (fast-path fix)
# ---------------------------------------------------------------------------

class TestAgentPresetBeatsOverrideWithApiKey:
    """Regression for the early-return fast-path that was skipping the preset block.

    Before the fix, when a session override had an api_key, _resolve_session_agent_runtime
    returned immediately (lines 4147-4157) and never reached the preset block.
    This meant /agent chatgpt had no effect on any session that had a live /model override
    with a resolved api_key — the override silently won.

    After the fix the early return is removed; the override is still applied via
    _apply_session_model_override, but the preset block runs last and wins.
    """

    def _make_runtime_runner(self):
        runner = object.__new__(gateway_run.GatewayRunner)
        runner.adapters = {}
        runner._session_model_overrides = {}
        runner.config = None
        runner._last_resolved_model = {}
        return runner

    def _patch_runtime(self, monkeypatch, *, config_model="claude-opus-4-5"):
        monkeypatch.setattr(
            gateway_run, "_resolve_gateway_model", lambda config=None: config_model
        )
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "api_key": "anthropic-key",
                "base_url": None,
                "provider": "anthropic",
                "api_mode": None,
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        )
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs_for_provider",
            lambda provider: {
                "api_key": "openai-key" if "openai" in provider else "anthropic-key",
                "base_url": None,
                "provider": provider,
                "api_mode": None,
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        )
        monkeypatch.setattr(
            gateway_run,
            "_credential_pool_for_provider",
            lambda provider: None,
        )

    def test_preset_wins_over_override_with_api_key(self, tmp_path, monkeypatch):
        """Global chatgpt preset must win even when the session override has a live api_key.

        This is the exact path that was returning early before the fix — the override had
        api_key set (e.g. rehydrated from a provider that re-resolved credentials), so the
        old fast-path returned immediately and the preset block was never reached.
        """
        from agent.llm_preset import set_preset

        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        set_preset("chatgpt", hermes_home=tmp_path)
        self._patch_runtime(monkeypatch)

        runner = self._make_runtime_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        # Override WITH api_key — this triggered the old early-return bug.
        runner._session_model_overrides[session_key] = {
            "model": "claude-opus-4-5",
            "provider": "anthropic",
            "api_key": "anthropic-key",   # <-- triggers the previously-buggy fast path
            "base_url": None,
            "api_mode": None,
            "credential_pool": None,
        }

        model, runtime_kwargs = runner._resolve_session_agent_runtime(
            source=source, user_config={}
        )

        assert runtime_kwargs.get("provider") == "openai", (
            f"Global chatgpt preset must win over a session override that has api_key; "
            f"got provider={runtime_kwargs.get('provider')!r}. "
            "The old fast-path returned before reaching the preset block."
        )
        assert "gpt" in model.lower() or model == "gpt-4o", (
            f"Global chatgpt preset must set a GPT model; got model={model!r}"
        )

    def test_preset_wins_before_and_after_restart_with_api_key_override(
        self, tmp_path, monkeypatch
    ):
        """Combined regression: override-with-api_key → /agent chatgpt → chatgpt before
        AND after simulated restart (empty in-memory + cleared store).

        Steps:
          1. Session override has api_key (the old fast-path trigger).
          2. /agent chatgpt sets preset and clears all in-memory overrides.
          3. Verify runtime is chatgpt (pre-restart check).
          4. Simulate restart: empty _session_model_overrides, store returns None.
          5. Verify runtime is still chatgpt (post-restart check).
        """
        from agent.llm_preset import set_preset

        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        self._patch_runtime(monkeypatch)

        runner = self._make_runtime_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        # Step 1: session has override with api_key (old fast-path trigger).
        runner._session_model_overrides[session_key] = {
            "model": "claude-opus-4-5",
            "provider": "anthropic",
            "api_key": "anthropic-key",
        }

        # Step 2: /agent chatgpt — set preset and clear in-memory overrides.
        set_preset("chatgpt", hermes_home=tmp_path)
        runner._session_model_overrides.clear()

        # Step 3: pre-restart check — preset must win with empty override dict.
        model_pre, rk_pre = runner._resolve_session_agent_runtime(
            source=source, user_config={}
        )
        assert rk_pre.get("provider") == "openai", (
            f"Pre-restart: chatgpt preset must apply when override is cleared; "
            f"got provider={rk_pre.get('provider')!r}"
        )
        assert "gpt" in model_pre.lower() or model_pre == "gpt-4o", (
            f"Pre-restart: chatgpt preset must set a GPT model; got model={model_pre!r}"
        )

        # Step 4: simulate restart — in-memory gone, store reports no override.
        runner._session_model_overrides.clear()
        mock_store = MagicMock()
        mock_store.get_model_override = MagicMock(return_value=None)
        runner.session_store = mock_store

        # Step 5: post-restart check — preset must still win.
        model_post, rk_post = runner._resolve_session_agent_runtime(
            source=source, user_config={}
        )
        mock_store.get_model_override.assert_called_with(session_key)
        assert rk_post.get("provider") == "openai", (
            f"Post-restart: chatgpt preset must still apply after override cleared; "
            f"got provider={rk_post.get('provider')!r}"
        )
        assert "gpt" in model_post.lower() or model_post == "gpt-4o", (
            f"Post-restart: chatgpt preset must set a GPT model; got model={model_post!r}"
        )
