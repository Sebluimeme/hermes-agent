"""Preset-level automatic cross-provider fallback.

HERMES-LLM-001 tranche 4 — bidirectional visible preset failover.

When the active global preset encounters a recoverable error (quota, transient
auth, timeout, transport), this module provides its explicit fallback chain so
the agent can retry
in the same turn without permanently changing the persisted preset selection.

Key guarantees
--------------
- TRANSIENT only : the other preset is tried once per turn; the global preset
  file (llm_preset.json) is NEVER modified by this fallback mechanism.
- VISIBLE        : the agent emits a clear, user-facing notice when the
  fallback activates AND when the preferred preset is restored on the next
  turn (if the restored call succeeds).
- ROUTING        : Claude 1→Claude 2→ChatGPT; Claude 2→ChatGPT;
                   ChatGPT→Claude 1 (conversational fallback).
- SESSION-SAFE   : session history is preserved; only model/provider change
  per turn.  User services and configs are never touched.

Integration points (called by other modules)
--------------------------------------------
- ``build_preset_fallback_chain_entry()``  — called from agent_init.py to
  prepend the other-preset entry into the agent's ``_fallback_chain``.
- ``format_preset_fallback_notice()``      — called from chat_completion_helpers
  to produce a rich, preset-aware status notice replacing the generic one.
- ``format_preset_restored_notice()``      — called from agent_runtime_helpers
  to emit the "restored" one-shot notice at the start of the next successful
  turn.

Marker keys injected into the fallback chain entry
---------------------------------------------------
``_preset_fallback_from``  : str — the preset that failed (e.g. "opus")
``_preset_fallback_to``    : str — the preset being tried (e.g. "chatgpt")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Keys added to the fallback chain entry dict to identify a preset-level hop.
PRESET_FALLBACK_MARKER = "_preset_fallback_from"
PRESET_FALLBACK_TO_KEY = "_preset_fallback_to"

# Agent attribute set when a preset fallback was used this turn so the next
# turn's restore_primary_runtime can emit a "restored" notice.
AGENT_ATTR_ACTIVE_FALLBACK_FROM = "_active_preset_fallback_from"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

FALLBACK_ROUTES: dict[str, tuple[str, ...]] = {
    "claude2": ("claude1", "chatgpt"),
    "claude1": ("chatgpt",),
    "chatgpt": ("claude1",),
}


def get_fallback_presets(active_preset: str) -> tuple[str, ...]:
    """Return the ordered, deliberate fallback route for *active_preset*."""
    return FALLBACK_ROUTES.get(active_preset, ())


def build_preset_fallback_chain(
    active_preset: Optional[str] = None,
    hermes_home: Optional[Path | str] = None,
) -> list[dict]:
    """Build ordered fallback entries for the active preset.

    This entry is structured identically to the user-configured
    ``fallback_model`` entries (``{"provider": ..., "model": ...}``) but
    carries two extra marker keys (``_preset_fallback_from`` and
    ``_preset_fallback_to``) so the notice-formatting code can produce a
    preset-aware message instead of the generic one.

    Args:
        active_preset: The currently active preset name ("opus" or "chatgpt").
                       When None, reads the global preset via ``get_preset()``.
        hermes_home:   Override for the profile-preset YAML root (used by
                       tests for isolation via ``tmp_path``).

    Returns:
        A list for ``_fallback_chain``. Invalid configured steps are skipped.
    """
    try:
        from agent.llm_preset import get_preset
        from agent.llm_presets_config import get_preset_details

        if active_preset is None:
            active_preset = get_preset()

        entries: list[dict] = []
        for target_preset in get_fallback_presets(active_preset):
            details = get_preset_details(target_preset, hermes_home)
            provider = details.get("provider")
            model = details.get("model")
            if not provider or not model:
                logger.debug("llm_preset_fallback: preset %r has no provider/model", target_preset)
                continue
            entry: dict = {
                "provider": provider,
                "model": model,
                PRESET_FALLBACK_MARKER: active_preset,
                PRESET_FALLBACK_TO_KEY: target_preset,
            }
            for key in ("base_url", "api_mode"):
                if details.get(key):
                    entry[key] = details[key]
            entries.append(entry)
        return entries
    except Exception:
        logger.debug("llm_preset_fallback: build_preset_fallback_chain_entry failed", exc_info=True)
        return []


def build_preset_fallback_chain_entry(
    active_preset: Optional[str] = None,
    hermes_home: Optional[Path | str] = None,
) -> Optional[dict]:
    """Compatibility helper returning the first ordered fallback entry."""
    entries = build_preset_fallback_chain(active_preset, hermes_home)
    return entries[0] if entries else None


# ---------------------------------------------------------------------------
# Notice formatters
# ---------------------------------------------------------------------------

def _preset_label(preset: str) -> str:
    """Return the human-readable label for a preset, with graceful fallback."""
    try:
        from agent.llm_presets_config import get_preset_label
        return get_preset_label(preset)
    except Exception:
        return preset


def format_preset_fallback_notice(
    from_preset: str,
    to_preset: str,
    from_model: str,
    to_model: str,
    reset_at: object = None,
) -> str:
    """Return the one-shot notice emitted when a preset fallback activates.

    Clearly names both presets (not just provider/model slugs) so the user
    understands which configured preset is being used and why.

    Example output:
        ⚠️ Claude Opus 5 (claude-opus-5) indisponible ce tour —
        réponse via ChatGPT GPT-5.6 (gpt-5.6). Prochain message : réessaie Claude Opus 5.
    """
    from_label = _preset_label(from_preset)
    to_label = _preset_label(to_preset)
    notice = (
        f"⚠️ {from_label} ({from_model}) indisponible ce tour — "
        f"réponse via {to_label} ({to_model})."
    )
    if reset_at not in {None, ""}:
        notice += f" Retour {from_label} estimé vers {reset_at}."
    return notice


def format_preset_restored_notice(preset: str, model: str) -> str:
    """Return the one-shot notice emitted when the preferred preset is restored.

    Emitted on the NEXT turn's successful call, so the user knows the
    preferred preset is working again without requiring any action.

    Example output:
        ✅ Claude Opus 5 (claude-opus-5) de nouveau disponible.
    """
    label = _preset_label(preset)
    return f"✅ {label} ({model}) de nouveau disponible."
