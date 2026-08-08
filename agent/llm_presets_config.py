"""Per-profile LLM preset configuration.

Maps the global preset names ("claude1", "claude2", "chatgpt") to provider/model details.
Provider/model values come from the profile's ``llm_presets.yaml`` — no ports
or secrets are hardcoded here.

File location
-------------
``$HERMES_HOME/llm_presets.yaml``  (profile-scoped; HERMES_HOME selects profile)

Example llm_presets.yaml::

    claude1:
      provider: anthropic
      model: claude-opus-5
    chatgpt:
      provider: openai
      model: gpt-4o

Built-in defaults are provided so the command works without a config file.
They can be overridden by writing the YAML.

Public API
----------
PRESET_LABELS           : dict[str, str]  — human-readable preset names
DEFAULT_BUILTIN_PRESETS : dict[str, dict] — fallback details per preset
get_preset_details(preset, hermes_home=None) -> dict  {"provider", "model", "base_url"?, "api_mode"?}
get_preset_label(preset) -> str
resolve_preset_runtime(hermes_home=None) -> dict   — full runtime details for active preset
resolve_preset_model_provider() -> tuple[str, str] | tuple[None, None]  — kept for compat
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PRESETS_FILENAME = "llm_presets.yaml"

# Human-readable labels.  Account names are intentional: the selector is
# about Claude subscriptions, not only model families.
PRESET_LABELS: dict[str, str] = {
    "claude1": "Claude 1 (bmsyoder)",
    "claude2": "Claude 2 (dev.hammer)",
    "chatgpt": "ChatGPT GPT-5.6",
}

# Built-in defaults: used when no llm_presets.yaml is present in the profile.
# These reference provider *names* only — no ports, keys, or base URLs.
# Supported fields: provider, model, base_url, api_mode.
# base_url allows a preset to target a local proxy without touching config.yaml.
DEFAULT_BUILTIN_PRESETS: dict[str, dict[str, str]] = {
    "claude1": {
        "provider": "anthropic",
        "model": "claude-opus-5",
    },
    "claude2": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
    },
    "chatgpt": {
        "provider": "openai",
        "model": "gpt-5.6",
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _profile_hermes_home() -> Path:
    """Return the active profile's Hermes home (HERMES_HOME or ~/.hermes)."""
    val = os.environ.get("HERMES_HOME", "").strip()
    if val:
        return Path(val)
    return Path.home() / ".hermes"


def _presets_file(hermes_home: Optional[Path | str] = None) -> Path:
    home = Path(hermes_home) if hermes_home is not None else _profile_hermes_home()
    return home / _PRESETS_FILENAME


def load_profile_presets(hermes_home: Optional[Path | str] = None) -> dict[str, dict[str, str]]:
    """Load preset definitions from the profile's llm_presets.yaml.

    Returns the built-in defaults merged with any profile overrides.
    Unrecognised keys in the YAML are ignored. Never raises.
    """
    result = {k: dict(v) for k, v in DEFAULT_BUILTIN_PRESETS.items()}
    path = _presets_file(hermes_home)
    try:
        if not path.exists():
            return result
        import yaml  # gateway already depends on yaml
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            return result
        for preset_name, details in raw.items():
            if not isinstance(preset_name, str) or not isinstance(details, dict):
                continue
            entry: dict[str, str] = {}
            for field in ("provider", "model", "base_url", "api_mode"):
                if field in details and details[field] is not None:
                    entry[field] = str(details[field]).strip()
            if entry:
                # A profile may override only one field (for example a local
                # proxy URL). Preserve the built-in provider/model fields.
                result[preset_name] = {**result.get(preset_name, {}), **entry}
    except Exception:
        logger.debug("llm_presets_config: failed to load %s", path, exc_info=True)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_preset_details(
    preset: str,
    hermes_home: Optional[Path | str] = None,
) -> dict[str, str]:
    """Return {provider, model, base_url?, api_mode?} for *preset*, or {} if unknown."""
    presets = load_profile_presets(hermes_home)
    return dict(presets.get(preset, {}))


def get_preset_label(preset: str) -> str:
    """Return the human-readable label for a preset ('Claude Opus', 'ChatGPT')."""
    return PRESET_LABELS.get(preset, preset)


def resolve_preset_runtime(
    hermes_home: Optional[Path | str] = None,
) -> dict[str, Optional[str]]:
    """Return the full runtime details dict for the active global preset.

    Reads the active global preset via ``agent.llm_preset.get_preset()`` then
    returns all details from the profile config (provider, model, base_url,
    api_mode).  Does NOT depend on the active model in config.yaml — the
    returned values come exclusively from llm_presets.yaml (or built-in
    defaults), so a preset can target a local proxy even in the default profile
    without touching config.yaml.

    Returns {} on failure (forward-compat guard).
    """
    try:
        from agent.llm_preset import get_preset
        preset = get_preset()
        return get_preset_details(preset, hermes_home)
    except Exception:
        logger.debug(
            "llm_presets_config: resolve_preset_runtime failed", exc_info=True
        )
        return {}


def resolve_preset_model_provider(
    hermes_home: Optional[Path | str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Read the active global preset and return (model, provider).

    Reads the global preset via ``agent.llm_preset.get_preset()`` then looks
    up its model/provider in the profile config.  Returns ``(None, None)``
    when the preset resolves to nothing useful (forward-compat guard).

    This is the runtime entry-point called by the gateway before each agent
    construction.
    """
    try:
        from agent.llm_preset import get_preset
        preset = get_preset()
        details = get_preset_details(preset, hermes_home)
        model = details.get("model") or None
        provider = details.get("provider") or None
        return model, provider
    except Exception:
        logger.debug(
            "llm_presets_config: resolve_preset_model_provider failed", exc_info=True
        )
        return None, None
