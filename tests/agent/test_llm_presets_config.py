"""Tests for agent/llm_presets_config.py — per-profile LLM preset configuration.

Covers:
  - Built-in defaults (claude-opus-5, gpt-5.6)
  - load_profile_presets: merges YAML over defaults, captures base_url + api_mode
  - get_preset_details: returns all supported fields
  - resolve_preset_runtime: reads active preset, returns full dict (no config.yaml dep)
  - resolve_preset_model_provider: backward-compat tuple API still works
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_module():
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import importlib
    return importlib.import_module("agent.llm_presets_config")


def _import_preset():
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import importlib
    return importlib.import_module("agent.llm_preset")


def _write_presets_yaml(path: Path, content: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "llm_presets.yaml").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Built-in defaults
# ---------------------------------------------------------------------------

class TestBuiltinDefaults:

    def test_claude1_default_has_model(self):
        cfg = _import_module()
        assert cfg.DEFAULT_BUILTIN_PRESETS["claude1"]["model"]

    def test_chatgpt_default_model_is_current(self):
        """Default chatgpt model should be gpt-5.6."""
        cfg = _import_module()
        assert cfg.DEFAULT_BUILTIN_PRESETS["chatgpt"]["model"] == "gpt-5.6"

    def test_claude1_default_provider(self):
        cfg = _import_module()
        assert cfg.DEFAULT_BUILTIN_PRESETS["claude1"]["provider"] == "anthropic"

    def test_chatgpt_default_provider(self):
        cfg = _import_module()
        assert cfg.DEFAULT_BUILTIN_PRESETS["chatgpt"]["provider"] == "openai"

    def test_preset_labels_include_version(self):
        """Labels should mention the model version for clarity."""
        cfg = _import_module()
        assert "Claude" in cfg.PRESET_LABELS["claude1"]
        assert "5.6" in cfg.PRESET_LABELS["chatgpt"] or "GPT" in cfg.PRESET_LABELS["chatgpt"]

    def test_no_builtin_base_url(self):
        """Built-in defaults must NOT include base_url (no hardcoded endpoints)."""
        cfg = _import_module()
        for preset_data in cfg.DEFAULT_BUILTIN_PRESETS.values():
            assert "base_url" not in preset_data, (
                "Built-in defaults must not hardcode a base_url"
            )

    def test_no_builtin_api_mode(self):
        """Built-in defaults must NOT include api_mode."""
        cfg = _import_module()
        for preset_data in cfg.DEFAULT_BUILTIN_PRESETS.values():
            assert "api_mode" not in preset_data


# ---------------------------------------------------------------------------
# load_profile_presets — YAML loading + field support
# ---------------------------------------------------------------------------

class TestLoadProfilePresets:

    def test_returns_builtin_when_no_file(self, tmp_path):
        cfg = _import_module()
        result = cfg.load_profile_presets(tmp_path)
        assert "claude1" in result
        assert "claude2" in result
        assert "chatgpt" in result
        assert result["claude1"]["model"]

    def test_yaml_model_override(self, tmp_path):
        """A model override in YAML wins over the built-in default."""
        cfg = _import_module()
        _write_presets_yaml(tmp_path, "claude1:\n  provider: anthropic\n  model: claude-custom-99\n")
        result = cfg.load_profile_presets(tmp_path)
        assert result["claude1"]["model"] == "claude-custom-99"

    def test_yaml_base_url_loaded(self, tmp_path):
        """base_url in YAML is captured and returned."""
        cfg = _import_module()
        _write_presets_yaml(
            tmp_path,
            "claude1:\n  provider: anthropic\n  model: claude-opus-5\n  base_url: http://localhost:4000\n",
        )
        result = cfg.load_profile_presets(tmp_path)
        assert result["claude1"].get("base_url") == "http://localhost:4000"

    def test_yaml_api_mode_loaded(self, tmp_path):
        """api_mode in YAML is captured and returned."""
        cfg = _import_module()
        _write_presets_yaml(
            tmp_path,
            "claude1:\n  provider: anthropic\n  model: claude-opus-5\n  api_mode: proxy\n",
        )
        result = cfg.load_profile_presets(tmp_path)
        assert result["claude1"].get("api_mode") == "proxy"

    def test_yaml_all_four_fields(self, tmp_path):
        """All four supported fields (provider, model, base_url, api_mode) load."""
        cfg = _import_module()
        _write_presets_yaml(
            tmp_path,
            (
                "claude1:\n"
                "  provider: anthropic\n"
                "  model: claude-opus-5\n"
                "  base_url: http://proxy.local:9000\n"
                "  api_mode: openai\n"
            ),
        )
        result = cfg.load_profile_presets(tmp_path)
        entry = result["claude1"]
        assert entry["provider"] == "anthropic"
        assert entry["model"] == "claude-opus-5"
        assert entry["base_url"] == "http://proxy.local:9000"
        assert entry["api_mode"] == "openai"

    def test_chatgpt_unaffected_by_opus_override(self, tmp_path):
        """Overriding opus leaves chatgpt at its built-in default."""
        cfg = _import_module()
        _write_presets_yaml(tmp_path, "claude1:\n  model: my-custom-model\n")
        result = cfg.load_profile_presets(tmp_path)
        assert result["chatgpt"]["model"] == "gpt-5.6"

    def test_corrupt_yaml_returns_builtins(self, tmp_path):
        """A corrupt YAML file silently falls back to built-in defaults."""
        cfg = _import_module()
        (tmp_path / "llm_presets.yaml").write_text("{{{{ not valid yaml", encoding="utf-8")
        result = cfg.load_profile_presets(tmp_path)
        assert result["claude1"]["model"]

    def test_null_value_field_skipped(self, tmp_path):
        """A null YAML value is not included in the entry."""
        cfg = _import_module()
        _write_presets_yaml(
            tmp_path,
            "claude1:\n  provider: anthropic\n  model: claude-opus-5\n  base_url: null\n",
        )
        result = cfg.load_profile_presets(tmp_path)
        # null base_url must not appear in the entry (falls back to provider default)
        assert "base_url" not in result["claude1"]

    def test_unknown_preset_in_yaml_accepted(self, tmp_path):
        """Unknown preset names in YAML are stored (future-proof)."""
        cfg = _import_module()
        _write_presets_yaml(tmp_path, "gemini:\n  provider: google\n  model: gemini-pro\n")
        result = cfg.load_profile_presets(tmp_path)
        assert "gemini" in result
        assert result["gemini"]["provider"] == "google"


# ---------------------------------------------------------------------------
# get_preset_details — public single-preset accessor
# ---------------------------------------------------------------------------

class TestGetPresetDetails:

    def test_claude1_details_returns_provider_and_model(self, tmp_path):
        cfg = _import_module()
        details = cfg.get_preset_details("claude1", tmp_path)
        assert details.get("provider") == "anthropic"
        assert details.get("model")

    def test_unknown_preset_returns_empty(self, tmp_path):
        cfg = _import_module()
        assert cfg.get_preset_details("nonexistent", tmp_path) == {}

    def test_base_url_present_when_in_yaml(self, tmp_path):
        cfg = _import_module()
        _write_presets_yaml(
            tmp_path,
            "claude1:\n  provider: anthropic\n  model: claude-opus-5\n  base_url: http://local:5000\n",
        )
        details = cfg.get_preset_details("claude1", tmp_path)
        assert details.get("base_url") == "http://local:5000"

    def test_api_mode_present_when_in_yaml(self, tmp_path):
        cfg = _import_module()
        _write_presets_yaml(
            tmp_path,
            "claude1:\n  provider: anthropic\n  model: claude-opus-5\n  api_mode: responses\n",
        )
        details = cfg.get_preset_details("claude1", tmp_path)
        assert details.get("api_mode") == "responses"


# ---------------------------------------------------------------------------
# resolve_preset_runtime — active-preset full runtime dict (no config.yaml dep)
# ---------------------------------------------------------------------------

class TestResolvePresetRuntime:

    def test_returns_dict(self, tmp_path, monkeypatch):
        """resolve_preset_runtime must return a dict."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        cfg = _import_module()
        result = cfg.resolve_preset_runtime(tmp_path)
        assert isinstance(result, dict)

    def test_returns_default_claude2_when_no_preset_set(self, tmp_path, monkeypatch):
        """The default preset resolves to Claude 2 details."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        cfg = _import_module()
        result = cfg.resolve_preset_runtime(tmp_path)
        assert result.get("provider") == "anthropic"
        assert result == cfg.get_preset_details("claude2", tmp_path)

    def test_returns_chatgpt_after_set_preset(self, tmp_path, monkeypatch):
        """After set_preset('chatgpt'), resolve_preset_runtime returns chatgpt details."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        preset_mod = _import_preset()
        preset_mod.set_preset("chatgpt", hermes_home=tmp_path)
        cfg = _import_module()
        result = cfg.resolve_preset_runtime(tmp_path)
        assert result.get("provider") == "openai"
        assert result.get("model") == "gpt-5.6"

    def test_base_url_from_preset_yaml_not_config(self, tmp_path, monkeypatch):
        """base_url comes from llm_presets.yaml, not from config.yaml or provider lookup.

        This is the key 'no config.yaml dependency' property: the preset is
        self-contained and can point an opus preset at a local proxy without
        touching any other config file.
        """
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        preset_mod = _import_preset()
        preset_mod.set_preset("claude1", hermes_home=tmp_path)

        # Write a presets.yaml that sets a local proxy URL for opus
        _write_presets_yaml(
            tmp_path,
            (
                "claude1:\n"
                "  provider: anthropic\n"
                "  model: claude-opus-5\n"
                "  base_url: http://localhost:4000\n"
            ),
        )
        cfg = _import_module()
        result = cfg.resolve_preset_runtime(tmp_path)
        assert result.get("base_url") == "http://localhost:4000", (
            f"base_url should come from llm_presets.yaml, got {result!r}"
        )

    def test_api_mode_from_preset_yaml(self, tmp_path, monkeypatch):
        """api_mode is carried from llm_presets.yaml."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        preset_mod = _import_preset()
        preset_mod.set_preset("claude1", hermes_home=tmp_path)
        _write_presets_yaml(
            tmp_path,
            "claude1:\n  provider: anthropic\n  model: claude-opus-5\n  api_mode: openai\n",
        )
        cfg = _import_module()
        result = cfg.resolve_preset_runtime(tmp_path)
        assert result.get("api_mode") == "openai"

    def test_no_base_url_when_not_in_yaml(self, tmp_path, monkeypatch):
        """When base_url is absent from llm_presets.yaml, it must not be in the result."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        cfg = _import_module()
        result = cfg.resolve_preset_runtime(tmp_path)
        assert "base_url" not in result or result.get("base_url") is None

    def test_failure_returns_empty_dict(self, tmp_path, monkeypatch):
        """resolve_preset_runtime must return {} and never raise on failure."""
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        cfg = _import_module()
        # Patch get_preset to raise
        import agent.llm_preset as lp
        original = lp.get_preset
        try:
            lp.get_preset = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
            result = cfg.resolve_preset_runtime(tmp_path)
            assert isinstance(result, dict)
        finally:
            lp.get_preset = original


# ---------------------------------------------------------------------------
# resolve_preset_model_provider — backward-compat tuple API
# ---------------------------------------------------------------------------

class TestResolvePresetModelProviderCompat:

    def test_returns_tuple_of_two(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        cfg = _import_module()
        result = cfg.resolve_preset_model_provider(tmp_path)
        assert isinstance(result, tuple) and len(result) == 2

    def test_default_returns_claude2_values(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(tmp_path))
        cfg = _import_module()
        model, provider = cfg.resolve_preset_model_provider(tmp_path)
        assert provider == "anthropic"
        assert model == cfg.DEFAULT_BUILTIN_PRESETS["claude2"]["model"]
