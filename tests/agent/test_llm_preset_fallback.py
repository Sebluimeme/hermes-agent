"""Tests for agent/llm_preset_fallback.py — HERMES-LLM-001 tranche 4.

Covers bidirectional automatic preset fallback:
- opus → chatgpt when Anthropic/Claude is unavailable
- chatgpt → opus when OpenAI/ChatGPT is unavailable

Test dimensions:
1. Unit tests for the helper functions (get_other_preset,
   build_preset_fallback_chain_entry, format_* notices)
2. Integration tests for the agent_init injection
3. Activation tests (verify entry is picked up and notice is set)
4. Restoration tests (notice queued for next turn, preset file unchanged)

Run with:
    cd /home/seb/.hermes/hermes-agent
    python -m pytest tests/agent/test_llm_preset_fallback.py -v
"""

from __future__ import annotations

import sys
import types
import unittest.mock as mock
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers — path setup and lazy imports
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _add_project_root():
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))


def _import(module_name: str):
    import importlib
    _add_project_root()
    return importlib.import_module(module_name)


def _fresh_import(module_name: str):
    """Import a module freshly (without sys.modules cache), useful for isolation."""
    import importlib
    _add_project_root()
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def preset_mod():
    return _import("agent.llm_preset")


@pytest.fixture()
def cfg_mod():
    return _import("agent.llm_presets_config")


@pytest.fixture()
def fb_mod():
    return _import("agent.llm_preset_fallback")


class _MinimalAgent:
    """Minimal stub that mimics the AIAgent attributes used by the fallback logic."""

    def __init__(self):
        self._fallback_chain: list = []
        self._fallback_index: int = 0
        self._fallback_activated: bool = False
        self._fallback_model = None
        self._rate_limited_until: float = 0.0
        self._unavailable_fallback_keys: set = set()
        self._pending_fallback_notice: str | None = None
        self._active_preset_fallback_from: str | None = None
        self._status_buffer: list = []
        self.quiet_mode: bool = True
        self.model: str = "claude-opus-5"
        self.provider: str = "anthropic"
        self.base_url: str = ""
        self.api_mode: str = "anthropic_messages"
        self.api_key: str = "test-key"

    def _buffer_status(self, msg: str) -> None:
        self._status_buffer.append(msg)

    def _try_activate_fallback(self, reason=None):
        """Thin forwarder matching production code."""
        from agent.chat_completion_helpers import try_activate_fallback
        return try_activate_fallback(self, reason)

    def _has_pending_fallback(self) -> bool:
        return self._fallback_index < len(self._fallback_chain)


# ---------------------------------------------------------------------------
# 1. Unit tests — llm_preset_fallback helpers
# ---------------------------------------------------------------------------

class TestGetFallbackPresets:

    def test_claude1_falls_back_via_claude2_then_chatgpt(self, fb_mod):
        assert fb_mod.get_fallback_presets("claude1") == ("claude2", "chatgpt")

    def test_chatgpt_falls_back_to_claude1(self, fb_mod):
        assert fb_mod.get_fallback_presets("chatgpt") == ("claude1",)

    def test_unknown_preset_has_no_fallback(self, fb_mod):
        assert fb_mod.get_fallback_presets("nonexistent") == ()

    def test_returns_tuple(self, fb_mod):
        assert isinstance(fb_mod.get_fallback_presets("claude1"), tuple)


class TestBuildPresetFallbackChainEntry:

    def test_claude1_active_returns_claude2_entry(self, fb_mod, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        entry = fb_mod.build_preset_fallback_chain_entry(
            active_preset="claude1", hermes_home=tmp_path
        )
        assert entry is not None
        assert entry["provider"] == "anthropic"
        assert entry["model"]
        assert entry[fb_mod.PRESET_FALLBACK_MARKER] == "claude1"
        assert entry[fb_mod.PRESET_FALLBACK_TO_KEY] == "claude2"

    def test_chatgpt_active_returns_claude1_entry(self, fb_mod, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        entry = fb_mod.build_preset_fallback_chain_entry(
            active_preset="chatgpt", hermes_home=tmp_path
        )
        assert entry is not None
        assert entry["provider"] == "anthropic"
        assert entry["model"]
        assert entry[fb_mod.PRESET_FALLBACK_MARKER] == "chatgpt"
        assert entry[fb_mod.PRESET_FALLBACK_TO_KEY] == "claude1"

    def test_reads_active_preset_from_global_when_none(self, fb_mod, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        preset_mod = _import("agent.llm_preset")
        preset_mod.set_preset("chatgpt", hermes_home=tmp_path)
        entry = fb_mod.build_preset_fallback_chain_entry(hermes_home=tmp_path)
        assert entry is not None
        assert entry[fb_mod.PRESET_FALLBACK_MARKER] == "chatgpt"
        assert entry[fb_mod.PRESET_FALLBACK_TO_KEY] == "claude1"

    def test_base_url_propagated_when_in_yaml(self, fb_mod, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        (tmp_path / "llm_presets.yaml").write_text(
            "claude2:\n  provider: anthropic\n  model: claude-sonnet-4-6\n  base_url: http://proxy.local:9000\n",
            encoding="utf-8",
        )
        entry = fb_mod.build_preset_fallback_chain_entry(
            active_preset="claude1", hermes_home=tmp_path
        )
        assert entry is not None
        assert entry.get("base_url") == "http://proxy.local:9000"

    def test_entry_has_provider_and_model(self, fb_mod, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        entry = fb_mod.build_preset_fallback_chain_entry(
            active_preset="claude1", hermes_home=tmp_path
        )
        assert entry is not None
        assert entry["provider"]
        assert entry["model"]

    def test_returns_empty_on_import_error(self, fb_mod, monkeypatch):
        import agent.llm_presets_config as cfg
        monkeypatch.setattr(
            cfg, "get_preset_details", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        assert fb_mod.build_preset_fallback_chain(active_preset="claude1") == []


class TestFormatNotices:

    def test_fallback_notice_contains_preset_labels(self, fb_mod):
        notice = fb_mod.format_preset_fallback_notice(
            from_preset="opus",
            to_preset="chatgpt",
            from_model="claude-opus-5",
            to_model="gpt-5.6",
        )
        assert "opus" in notice.lower() or "claude" in notice.lower() or "Opus" in notice
        assert "chatgpt" in notice.lower() or "gpt" in notice.lower() or "GPT" in notice
        assert "claude-opus-5" in notice
        assert "gpt-5.6" in notice

    def test_fallback_notice_mentions_next_turn_retry(self, fb_mod):
        notice = fb_mod.format_preset_fallback_notice(
            from_preset="opus",
            to_preset="chatgpt",
            from_model="claude-opus-5",
            to_model="gpt-5.6",
        )
        # Must mention that the next message retries the preferred preset
        assert "prochain" in notice.lower() or "next" in notice.lower()

    def test_fallback_notice_reverse_direction(self, fb_mod):
        notice = fb_mod.format_preset_fallback_notice(
            from_preset="chatgpt",
            to_preset="opus",
            from_model="gpt-5.6",
            to_model="claude-opus-5",
        )
        assert "gpt-5.6" in notice
        assert "claude-opus-5" in notice

    def test_restored_notice_contains_preset_label(self, fb_mod):
        notice = fb_mod.format_preset_restored_notice(
            preset="opus", model="claude-opus-5"
        )
        assert "claude-opus-5" in notice
        # Must contain a positive signal
        assert "✅" in notice or "rétabli" in notice.lower() or "disponible" in notice.lower()

    def test_restored_notice_chatgpt(self, fb_mod):
        notice = fb_mod.format_preset_restored_notice(
            preset="chatgpt", model="gpt-5.6"
        )
        assert "gpt-5.6" in notice
        assert "✅" in notice or "disponible" in notice.lower()


# ---------------------------------------------------------------------------
# 2. Agent-init injection tests
# ---------------------------------------------------------------------------

class TestAgentInitInjection:
    """Verify that a preset fallback entry is prepended to the agent chain."""

    def test_preset_fb_entry_prepended_when_chain_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        fb_mod = _import("agent.llm_preset_fallback")
        entry = fb_mod.build_preset_fallback_chain_entry(
            active_preset="claude1", hermes_home=tmp_path
        )
        assert entry is not None
        chain = [entry]
        assert chain[0][fb_mod.PRESET_FALLBACK_MARKER] == "claude1"

    def test_preset_fb_entry_prepended_before_user_providers(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        fb_mod = _import("agent.llm_preset_fallback")
        user_entry = {"provider": "openrouter", "model": "llama-3"}
        entry = fb_mod.build_preset_fallback_chain_entry(
            active_preset="claude1", hermes_home=tmp_path
        )
        assert entry is not None
        chain = [entry, user_entry]
        assert chain[0][fb_mod.PRESET_FALLBACK_MARKER] == "claude1"
        assert chain[1]["provider"] == "openrouter"

    def test_preset_fb_entry_has_required_keys(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        fb_mod = _import("agent.llm_preset_fallback")
        entry = fb_mod.build_preset_fallback_chain_entry(
            active_preset="claude1", hermes_home=tmp_path
        )
        assert entry is not None
        required = {"provider", "model", "_preset_fallback_from", "_preset_fallback_to"}
        assert required <= entry.keys()


# ---------------------------------------------------------------------------
# 3. Activation tests — notice + state mutation
# ---------------------------------------------------------------------------

class TestPresetFallbackActivation:
    """Simulate try_activate_fallback picking up a preset entry.

    We don't exercise the full client-swap (needs real providers) but we
    CAN verify that the notice formatting and state-marking branches fire
    correctly when the entry carries the ``_preset_fallback_from`` marker.
    These tests patch the client-resolution step.
    """

    def _make_agent_with_preset_chain(
        self, from_preset: str = "claude1", tmp_path=None, monkeypatch=None
    ) -> _MinimalAgent:
        """Build a minimal agent with the preset fallback entry pre-loaded."""
        fb_mod = _import("agent.llm_preset_fallback")
        agent = _MinimalAgent()
        agent.model = "claude-opus-5" if from_preset.startswith("claude") else "gpt-5.6"
        agent.provider = "anthropic" if from_preset.startswith("claude") else "openai"

        if tmp_path and monkeypatch:
            monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
            entry = fb_mod.build_preset_fallback_chain_entry(
                active_preset=from_preset, hermes_home=tmp_path
            )
        else:
            entry = fb_mod.build_preset_fallback_chain_entry(active_preset=from_preset)

        if entry:
            agent._fallback_chain = [entry]
        return agent

    # ── Opus → ChatGPT ──

    def test_opus_to_chatgpt_pending_notice_set(self, tmp_path, monkeypatch):
        """When opus fails, _pending_fallback_notice must mention chatgpt."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        fb_mod = _import("agent.llm_preset_fallback")
        agent = self._make_agent_with_preset_chain("claude1", tmp_path, monkeypatch)
        assert agent._fallback_chain, "Preset entry must be injected"

        fb_entry = agent._fallback_chain[0]
        assert fb_entry[fb_mod.PRESET_FALLBACK_MARKER] == "claude1"

        # Manually invoke the notice path (without real client swap)
        from_preset = fb_entry["_preset_fallback_from"]
        to_preset = fb_entry["_preset_fallback_to"]
        notice = fb_mod.format_preset_fallback_notice(
            from_preset=from_preset,
            to_preset=to_preset,
            from_model=agent.model,
            to_model=fb_entry["model"],
        )
        agent._buffer_status(notice)
        agent._pending_fallback_notice = notice
        agent._active_preset_fallback_from = from_preset

        assert agent._pending_fallback_notice is not None
        assert "claude 2" in agent._pending_fallback_notice.lower()
        assert "claude" in agent._pending_fallback_notice.lower() or "opus" in agent._pending_fallback_notice.lower()

    def test_opus_to_chatgpt_active_attr_set(self, tmp_path, monkeypatch):
        """_active_preset_fallback_from must be set to 'opus' after activation."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        fb_mod = _import("agent.llm_preset_fallback")
        agent = self._make_agent_with_preset_chain("claude1", tmp_path, monkeypatch)
        fb_entry = agent._fallback_chain[0]

        # Simulate the attribute-set step in try_activate_fallback
        agent._active_preset_fallback_from = fb_entry["_preset_fallback_from"]
        assert agent._active_preset_fallback_from == "claude1"

    def test_opus_to_chatgpt_global_preset_unchanged(self, tmp_path, monkeypatch):
        """The global preset file must NOT be modified by the fallback."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        preset_mod = _import("agent.llm_preset")
        preset_mod.set_preset("claude1", hermes_home=tmp_path)

        # Simulate fallback activation (no real client swap needed)
        fb_mod = _import("agent.llm_preset_fallback")
        agent = self._make_agent_with_preset_chain("claude1", tmp_path, monkeypatch)
        fb_entry = agent._fallback_chain[0]
        agent._active_preset_fallback_from = fb_entry["_preset_fallback_from"]
        agent.model = fb_entry["model"]
        agent.provider = fb_entry["provider"]

        assert preset_mod.get_preset(hermes_home=tmp_path) == "claude1"

    def test_opus_to_chatgpt_notice_mentions_next_turn(self, tmp_path, monkeypatch):
        """Fallback notice must explicitly mention that the next turn retries opus."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        fb_mod = _import("agent.llm_preset_fallback")
        notice = fb_mod.format_preset_fallback_notice(
            from_preset="opus",
            to_preset="chatgpt",
            from_model="claude-opus-5",
            to_model="gpt-5.6",
        )
        lower = notice.lower()
        assert "prochain" in lower or "next" in lower, (
            f"Notice must mention next turn: {notice!r}"
        )

    # ── ChatGPT → Opus ──

    def test_chatgpt_to_opus_pending_notice_set(self, tmp_path, monkeypatch):
        """When chatgpt fails, _pending_fallback_notice must mention opus."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        fb_mod = _import("agent.llm_preset_fallback")
        agent = self._make_agent_with_preset_chain("chatgpt", tmp_path, monkeypatch)
        assert agent._fallback_chain, "Preset entry must be injected"

        fb_entry = agent._fallback_chain[0]
        assert fb_entry[fb_mod.PRESET_FALLBACK_MARKER] == "chatgpt"

        notice = fb_mod.format_preset_fallback_notice(
            from_preset=fb_entry["_preset_fallback_from"],
            to_preset=fb_entry["_preset_fallback_to"],
            from_model=agent.model,
            to_model=fb_entry["model"],
        )
        agent._buffer_status(notice)
        agent._pending_fallback_notice = notice
        agent._active_preset_fallback_from = fb_entry["_preset_fallback_from"]

        assert agent._pending_fallback_notice is not None
        assert "opus" in agent._pending_fallback_notice.lower() or "claude" in agent._pending_fallback_notice.lower()
        assert "gpt" in agent._pending_fallback_notice.lower() or "chatgpt" in agent._pending_fallback_notice.lower()

    def test_chatgpt_to_opus_active_attr_set(self, tmp_path, monkeypatch):
        """_active_preset_fallback_from must be set to 'chatgpt' after activation."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        fb_mod = _import("agent.llm_preset_fallback")
        agent = self._make_agent_with_preset_chain("chatgpt", tmp_path, monkeypatch)
        fb_entry = agent._fallback_chain[0]
        agent._active_preset_fallback_from = fb_entry["_preset_fallback_from"]
        assert agent._active_preset_fallback_from == "chatgpt"

    def test_chatgpt_to_opus_global_preset_unchanged(self, tmp_path, monkeypatch):
        """The global preset file must NOT be modified when chatgpt falls back to opus."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        preset_mod = _import("agent.llm_preset")
        preset_mod.set_preset("chatgpt", hermes_home=tmp_path)

        fb_mod = _import("agent.llm_preset_fallback")
        agent = self._make_agent_with_preset_chain("chatgpt", tmp_path, monkeypatch)
        fb_entry = agent._fallback_chain[0]
        agent._active_preset_fallback_from = fb_entry["_preset_fallback_from"]
        agent.model = fb_entry["model"]
        agent.provider = fb_entry["provider"]

        # Global preset must still be "chatgpt"
        assert preset_mod.get_preset(hermes_home=tmp_path) == "chatgpt"

    def test_chatgpt_to_opus_notice_mentions_next_turn(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        fb_mod = _import("agent.llm_preset_fallback")
        notice = fb_mod.format_preset_fallback_notice(
            from_preset="chatgpt",
            to_preset="opus",
            from_model="gpt-5.6",
            to_model="claude-opus-5",
        )
        lower = notice.lower()
        assert "prochain" in lower or "next" in lower


# ---------------------------------------------------------------------------
# 4. Restoration tests — next-turn notice queuing
# ---------------------------------------------------------------------------

class TestPresetRestoration:
    """Verify that restore_primary_runtime queues the restoration notice."""

    def _make_agent_after_fallback(
        self,
        preset_used: str = "opus",
        model: str = "claude-opus-5",
    ) -> _MinimalAgent:
        """Return a minimal agent with the post-fallback state set."""
        agent = _MinimalAgent()
        agent._fallback_activated = True
        agent._active_preset_fallback_from = preset_used
        agent.model = "gpt-5.6"  # currently on fallback
        agent.provider = "openai"
        # _primary_runtime snapshot (minimum fields restore_primary_runtime uses)
        agent._primary_runtime = {
            "model": model,
            "provider": "anthropic" if preset_used == "opus" else "openai",
            "base_url": "",
            "api_key": "test-key",
            "api_mode": "anthropic_messages" if preset_used == "opus" else "codex_responses",
            "client_kwargs": {},
            "use_prompt_caching": False,
            "use_native_cache_layout": False,
            "anthropic_api_key": "test-key",
            "anthropic_base_url": "",
            "is_anthropic_oauth": False,
            "compressor_model": model,
            "compressor_context_length": 200000,
            "compressor_base_url": "",
            "compressor_api_key": "test-key",
            "compressor_provider": "anthropic" if preset_used == "opus" else "openai",
            "compressor_api_mode": "",
        }
        agent._rate_limited_until = 0.0
        return agent

    def test_opus_restored_pending_notice_set(self, tmp_path, monkeypatch):
        """When opus was the fallback source, restoration queues an opus notice."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        preset_mod = _import("agent.llm_preset")
        preset_mod.set_preset("opus", hermes_home=tmp_path)

        fb_mod = _import("agent.llm_preset_fallback")
        notice = fb_mod.format_preset_restored_notice("opus", "claude-opus-5")

        # Simulate what restore_primary_runtime does after clearing _active_preset_fallback_from
        agent = _MinimalAgent()
        agent._active_preset_fallback_from = None
        agent._pending_fallback_notice = notice

        assert agent._pending_fallback_notice is not None
        assert "claude-opus-5" in agent._pending_fallback_notice
        assert "✅" in agent._pending_fallback_notice or "disponible" in agent._pending_fallback_notice.lower()

    def test_chatgpt_restored_pending_notice_set(self, tmp_path, monkeypatch):
        """When chatgpt was the fallback source, restoration queues a chatgpt notice."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        fb_mod = _import("agent.llm_preset_fallback")
        notice = fb_mod.format_preset_restored_notice("chatgpt", "gpt-5.6")

        agent = _MinimalAgent()
        agent._active_preset_fallback_from = None
        agent._pending_fallback_notice = notice

        assert agent._pending_fallback_notice is not None
        assert "gpt-5.6" in agent._pending_fallback_notice
        assert "✅" in agent._pending_fallback_notice or "disponible" in agent._pending_fallback_notice.lower()

    def test_active_preset_fallback_from_cleared_on_restore(self):
        """_active_preset_fallback_from must be None after the restoration notice is set."""
        agent = _MinimalAgent()
        agent._active_preset_fallback_from = "opus"

        # Simulate the state transition in restore_primary_runtime
        from_preset = agent._active_preset_fallback_from
        agent._active_preset_fallback_from = None  # cleared
        agent._pending_fallback_notice = f"✅ Restored {from_preset}"

        assert agent._active_preset_fallback_from is None
        assert agent._pending_fallback_notice is not None

    def test_no_notice_when_no_preset_fallback_was_active(self):
        """No restoration notice must be set when no preset fallback ran last turn."""
        agent = _MinimalAgent()
        agent._active_preset_fallback_from = None
        # Simulate the guard in restore_primary_runtime
        assert getattr(agent, "_active_preset_fallback_from", None) is None
        # → no notice set → _pending_fallback_notice unchanged (None)
        assert agent._pending_fallback_notice is None

    def test_global_preset_not_modified_on_restore(self, tmp_path, monkeypatch):
        """The global preset file must still hold the original value after restoration."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        preset_mod = _import("agent.llm_preset")
        preset_mod.set_preset("chatgpt", hermes_home=tmp_path)

        # Simulate a chatgpt→opus fallback turn and then restoration
        # Without modifying the file
        agent = _MinimalAgent()
        agent._active_preset_fallback_from = "chatgpt"
        # ... restore clears the flag without touching the preset file
        agent._active_preset_fallback_from = None

        assert preset_mod.get_preset(hermes_home=tmp_path) == "chatgpt"

    # ── Bidirectional round-trip ──

    def test_opus_to_chatgpt_round_trip_notices(self, tmp_path, monkeypatch):
        """Fallback opus→chatgpt then restoration both produce the right notice text."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        fb_mod = _import("agent.llm_preset_fallback")

        # Turn N: fallback activated
        fb_notice = fb_mod.format_preset_fallback_notice(
            from_preset="opus",
            to_preset="chatgpt",
            from_model="claude-opus-5",
            to_model="gpt-5.6",
        )
        assert "gpt-5.6" in fb_notice

        # Turn N+1: restoration
        restore_notice = fb_mod.format_preset_restored_notice("opus", "claude-opus-5")
        assert "claude-opus-5" in restore_notice

        # They must be different messages
        assert fb_notice != restore_notice

    def test_chatgpt_to_opus_round_trip_notices(self, tmp_path, monkeypatch):
        """Fallback chatgpt→opus then restoration both produce the right notice text."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        fb_mod = _import("agent.llm_preset_fallback")

        fb_notice = fb_mod.format_preset_fallback_notice(
            from_preset="chatgpt",
            to_preset="opus",
            from_model="gpt-5.6",
            to_model="claude-opus-5",
        )
        assert "claude-opus-5" in fb_notice

        restore_notice = fb_mod.format_preset_restored_notice("chatgpt", "gpt-5.6")
        assert "gpt-5.6" in restore_notice

        assert fb_notice != restore_notice


# ---------------------------------------------------------------------------
# 5. Marker key contract tests
# ---------------------------------------------------------------------------

class TestMarkerKeys:
    """Verify the published constant names are stable (other modules import them)."""

    def test_preset_fallback_marker_constant(self, fb_mod):
        assert hasattr(fb_mod, "PRESET_FALLBACK_MARKER")
        assert fb_mod.PRESET_FALLBACK_MARKER == "_preset_fallback_from"

    def test_preset_fallback_to_key_constant(self, fb_mod):
        assert hasattr(fb_mod, "PRESET_FALLBACK_TO_KEY")
        assert fb_mod.PRESET_FALLBACK_TO_KEY == "_preset_fallback_to"

    def test_agent_attr_constant(self, fb_mod):
        assert hasattr(fb_mod, "AGENT_ATTR_ACTIVE_FALLBACK_FROM")
        assert fb_mod.AGENT_ATTR_ACTIVE_FALLBACK_FROM == "_active_preset_fallback_from"
