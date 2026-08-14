"""
Tests for agent/llm_preset.py — global inter-process LLM preset selector.

RED suite  (HERMES-LLM-001-RED)  : failures expected before implementation.
GREEN suite (HERMES-LLM-001-GREEN): all should pass with the implementation.

Run with:
    cd /home/seb/.hermes/hermes-agent
    python -m pytest tests/agent/test_llm_preset.py -v
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_module():
    """Import llm_preset, inserting project root into sys.path if needed."""
    import importlib
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return importlib.import_module("agent.llm_preset")


# ---------------------------------------------------------------------------
# RED: Import & contract tests (fail without the module)
# ---------------------------------------------------------------------------

class TestRED:
    """These tests define the required contract and fail if the module is absent."""

    def test_module_importable(self):
        """The module must exist and be importable."""
        mod = _import_module()
        assert mod is not None

    def test_valid_presets_defined(self):
        """VALID_PRESETS exposes both Claude accounts and ChatGPT."""
        mod = _import_module()
        assert hasattr(mod, "VALID_PRESETS"), "VALID_PRESETS missing"
        assert mod.VALID_PRESETS == frozenset({"claude1", "claude2", "chatgpt"})

    def test_default_preset_defined(self):
        """Claude 2 is the default workhorse preset."""
        mod = _import_module()
        assert hasattr(mod, "DEFAULT_PRESET"), "DEFAULT_PRESET missing"
        assert mod.DEFAULT_PRESET == "claude2"

    def test_get_preset_callable(self):
        """get_preset must be a callable."""
        mod = _import_module()
        assert callable(getattr(mod, "get_preset", None)), "get_preset not callable"

    def test_set_preset_callable(self):
        """set_preset must be a callable."""
        mod = _import_module()
        assert callable(getattr(mod, "set_preset", None)), "set_preset not callable"

    def test_preset_file_callable(self):
        """preset_file must be a callable returning a Path."""
        mod = _import_module()
        assert callable(getattr(mod, "preset_file", None)), "preset_file not callable"

    def test_get_preset_returns_default_when_no_file(self, tmp_path):
        """get_preset with no existing file returns DEFAULT_PRESET."""
        mod = _import_module()
        result = mod.get_preset(hermes_home=tmp_path)
        assert result == mod.DEFAULT_PRESET

    def test_set_preset_raises_on_invalid(self, tmp_path):
        """set_preset with an unknown preset must raise ValueError."""
        mod = _import_module()
        with pytest.raises(ValueError, match="Invalid preset"):
            mod.set_preset("gpt4", hermes_home=tmp_path)

    def test_set_then_get_roundtrip(self, tmp_path):
        """set_preset + get_preset must return the value that was set."""
        mod = _import_module()
        mod.set_preset("chatgpt", hermes_home=tmp_path)
        assert mod.get_preset(hermes_home=tmp_path) == "chatgpt"

    def test_preset_file_path_is_under_hermes_home(self, tmp_path):
        """preset_file(hermes_home) must return a Path inside hermes_home."""
        mod = _import_module()
        p = mod.preset_file(hermes_home=tmp_path)
        assert isinstance(p, Path)
        assert str(tmp_path) in str(p)


# ---------------------------------------------------------------------------
# GREEN: Behavioural tests (full coverage of the happy + edge paths)
# ---------------------------------------------------------------------------

class TestGREEN:
    """Behavioural tests; all must pass with the implementation in place."""

    @pytest.fixture(autouse=True)
    def mod(self):
        self.m = _import_module()

    # -- Basic state --------------------------------------------------------

    def test_default_is_claude2(self, tmp_path):
        assert self.m.get_preset(tmp_path) == "claude2"

    def test_set_claude1(self, tmp_path):
        self.m.set_preset("claude1", tmp_path)
        assert self.m.get_preset(tmp_path) == "claude1"

    def test_set_chatgpt(self, tmp_path):
        self.m.set_preset("chatgpt", tmp_path)
        assert self.m.get_preset(tmp_path) == "chatgpt"

    def test_overwrite_preset(self, tmp_path):
        self.m.set_preset("chatgpt", tmp_path)
        self.m.set_preset("claude1", tmp_path)
        assert self.m.get_preset(tmp_path) == "claude1"

    # -- File format --------------------------------------------------------

    def test_stored_as_json(self, tmp_path):
        self.m.set_preset("chatgpt", tmp_path)
        path = self.m.preset_file(tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data == {"preset": "chatgpt"}

    def test_lock_file_created(self, tmp_path):
        self.m.set_preset("claude1", tmp_path)
        lock = tmp_path / "llm_preset.lock"
        assert lock.exists(), "Lock file must be created alongside preset file"

    # -- Validation ---------------------------------------------------------

    def test_empty_string_raises(self, tmp_path):
        with pytest.raises(ValueError):
            self.m.set_preset("", tmp_path)

    def test_none_raises(self, tmp_path):
        with pytest.raises((ValueError, AttributeError)):
            self.m.set_preset(None, tmp_path)  # type: ignore[arg-type]

    def test_case_sensitive_invalid(self, tmp_path):
        with pytest.raises(ValueError):
            self.m.set_preset("Opus", tmp_path)

    # -- Resilience ---------------------------------------------------------

    def test_corrupt_json_falls_back_to_default(self, tmp_path):
        path = self.m.preset_file(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("NOT JSON {{{{", encoding="utf-8")
        # Must not raise, must return default
        result = self.m.get_preset(tmp_path)
        assert result == "claude2"

    def test_unknown_value_in_file_falls_back(self, tmp_path):
        path = self.m.preset_file(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"preset": "gemini"}), encoding="utf-8")
        assert self.m.get_preset(tmp_path) == "claude2"

    def test_hermes_home_created_if_missing(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        assert not deep.exists()
        self.m.set_preset("chatgpt", deep)
        assert self.m.get_preset(deep) == "chatgpt"

    # -- Thread safety (intra-process) --------------------------------------

    def test_concurrent_writes_no_corruption(self, tmp_path):
        """Multiple threads writing different presets must never corrupt the file."""
        errors: list[Exception] = []
        iterations = 50
        barrier = threading.Barrier(2)

        def writer(preset: str):
            barrier.wait()
            try:
                for _ in range(iterations):
                    self.m.set_preset(preset, tmp_path)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=writer, args=("claude1",))
        t2 = threading.Thread(target=writer, args=("chatgpt",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        # File must be valid JSON with a known preset
        final = self.m.get_preset(tmp_path)
        assert final in self.m.VALID_PRESETS

    # -- Cross-process isolation (subprocess writes, parent reads) ----------

    def test_subprocess_write_parent_reads(self, tmp_path):
        """A preset written by a subprocess must be visible to the parent."""
        import subprocess

        code = (
            "import sys; sys.path.insert(0, sys.argv[1]);"
            "from agent.llm_preset import set_preset;"
            f"set_preset('chatgpt', r'{tmp_path}')"
        )
        project_root = str(Path(__file__).parent.parent.parent)
        result = subprocess.run(
            [sys.executable, "-c", code, project_root],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, f"Subprocess stderr: {result.stderr}"
        assert self.m.get_preset(tmp_path) == "chatgpt"

    def test_valid_presets_are_immutable(self):
        """VALID_PRESETS must be a frozenset (immutable)."""
        assert isinstance(self.m.VALID_PRESETS, frozenset)


# ---------------------------------------------------------------------------
# Cross-profile global storage tests
# ---------------------------------------------------------------------------

class TestCrossProfileGlobalHome:
    """Tests that the preset storage is shared across profiles.

    The contract: set_preset()/get_preset() without an explicit hermes_home
    argument read/write to HERMES_GLOBAL_HOME (if set) or the platform-native
    default (~/.hermes on POSIX) — never to a per-profile HERMES_HOME.
    """

    @pytest.fixture(autouse=True)
    def mod(self):
        self.m = _import_module()

    def test_hermes_global_home_env_var_constant_exposed(self):
        """HERMES_GLOBAL_HOME_ENV_VAR must be a public string constant."""
        assert hasattr(self.m, "HERMES_GLOBAL_HOME_ENV_VAR")
        assert isinstance(self.m.HERMES_GLOBAL_HOME_ENV_VAR, str)
        assert self.m.HERMES_GLOBAL_HOME_ENV_VAR == "HERMES_GLOBAL_HOME"

    def test_hermes_global_home_env_honored(self, tmp_path, monkeypatch):
        """HERMES_GLOBAL_HOME env var is used as storage root when set."""
        global_home = tmp_path / "global"
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(global_home))

        self.m.set_preset("chatgpt")
        assert self.m.get_preset() == "chatgpt"
        assert (global_home / "llm_preset.json").exists()

    def test_two_profile_homes_share_global_preset(self, tmp_path, monkeypatch):
        """Two different HERMES_HOME profile dirs observe the same global preset.

        Simulates: profile 'default' writes the preset; profile
        'univers-arrosage' reads it.  Both see the same value because storage
        lives at HERMES_GLOBAL_HOME, not at the per-profile HERMES_HOME.
        """
        global_home = tmp_path / "global"
        profile_default = tmp_path / "profiles" / "default"
        profile_ua = tmp_path / "profiles" / "univers-arrosage"

        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(global_home))

        # --- Profile "default" sets the preset ---
        monkeypatch.setenv("HERMES_HOME", str(profile_default))
        self.m.set_preset("chatgpt")  # no hermes_home override → writes to global_home

        # --- Profile "univers-arrosage" reads the preset ---
        monkeypatch.setenv("HERMES_HOME", str(profile_ua))
        assert self.m.get_preset() == "chatgpt"  # reads from global_home, not profile_ua

        # The preset file must exist only at the global location.
        assert (global_home / "llm_preset.json").exists()
        assert not (profile_default / "llm_preset.json").exists()
        assert not (profile_ua / "llm_preset.json").exists()

    def test_explicit_hermes_home_overrides_global(self, tmp_path, monkeypatch):
        """Explicit hermes_home arg overrides HERMES_GLOBAL_HOME (test isolation)."""
        global_home = tmp_path / "global"
        test_home = tmp_path / "test"
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(global_home))

        # Write via explicit override
        self.m.set_preset("chatgpt", hermes_home=test_home)

        # Global home was NOT written → default remains Claude 2.
        assert self.m.get_preset() == "claude2"
        # Explicit override path has the value
        assert self.m.get_preset(hermes_home=test_home) == "chatgpt"

    def test_preset_file_reflects_global_home(self, tmp_path, monkeypatch):
        """preset_file() with no arg returns a path inside HERMES_GLOBAL_HOME."""
        global_home = tmp_path / "global"
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(global_home))

        p = self.m.preset_file()
        assert isinstance(p, Path)
        assert str(global_home) in str(p)

    def test_global_preset_isolated_from_hermes_home(self, tmp_path, monkeypatch):
        """Changing HERMES_HOME must not redirect the preset storage path."""
        global_home = tmp_path / "global"
        monkeypatch.setenv("HERMES_GLOBAL_HOME", str(global_home))

        # Write preset, then change HERMES_HOME to a completely different dir
        self.m.set_preset("chatgpt")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "some_other_profile"))

        # Preset still readable via global home
        assert self.m.get_preset() == "chatgpt"
