"""
Global inter-process LLM preset selector.

Stores the active LLM preset ("claude1" | "claude2" | "chatgpt") as a single JSON file
under the **global** Hermes home, shared by all profiles.  Designed to be
imported from any gateway process without circular-import risk (no
agent-level dependencies).

Global vs profile storage
-------------------------
The preset is intentionally cross-profile: switching from "opus" to
"chatgpt" in the "default" profile must also be visible to the
"univers-arrosage" profile without a separate write.  The storage root is
therefore the *global* Hermes home (resolution below), not the per-profile
``HERMES_HOME`` directory.

Default resolution order (when ``hermes_home`` is NOT supplied explicitly):
    1. ``HERMES_GLOBAL_HOME`` env var — explicit, testable override.
    2. Platform-native default: ``~/.hermes`` on POSIX,
       ``%LOCALAPPDATA%\\hermes`` on Windows.

Pass an explicit ``hermes_home`` argument to any public function to
override this entirely (the primary use-case is test isolation via
``tmp_path``).

Concurrency model
-----------------
- Reads   : lock-free.  ``os.replace`` is atomic on POSIX; readers always
            see either the previous full file or the new full file.
- Writes  : acquire an exclusive ``fcntl.flock`` on a companion .lock file
            before writing so concurrent writers are serialised.  The lock
            is released as soon as ``atomic_json_write`` returns.

Public API
----------
VALID_PRESETS          : frozenset[str]   — {"claude1", "claude2", "chatgpt"}
DEFAULT_PRESET         : str              — "claude2"
HERMES_GLOBAL_HOME_ENV_VAR : str         — "HERMES_GLOBAL_HOME"
get_preset(hermes_home=None) -> str
set_preset(preset: str, hermes_home=None) -> None
preset_file(hermes_home=None) -> Path   (for testing / introspection)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

VALID_PRESETS: frozenset[str] = frozenset({"claude1", "claude2", "chatgpt"})
# Claude 2 is the normal workhorse.  Claude 1 is deliberately reserved for
# an explicit conversational selection and as ChatGPT's conversational peer.
DEFAULT_PRESET: str = "claude2"
HERMES_GLOBAL_HOME_ENV_VAR: str = "HERMES_GLOBAL_HOME"

_PRESET_FILENAME = "llm_preset.json"
_LOCK_FILENAME = "llm_preset.lock"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _native_hermes_root() -> Path:
    """Return the platform-native Hermes root (~/.hermes on POSIX).

    Deliberately ignores ``HERMES_HOME`` (which is per-profile) so the global
    preset always lands at the installation root, not inside a profile subdir.
    """
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
        return base / "hermes"
    return Path.home() / ".hermes"


def _global_hermes_home() -> Path:
    """Return the global Hermes home shared by all profiles.

    Resolution order:
        1. ``HERMES_GLOBAL_HOME`` env var — explicit, testable.
        2. Platform-native default (``~/.hermes`` on POSIX).

    This never follows ``HERMES_HOME`` so profile-scoped processes do not
    inadvertently write the preset into their profile sub-directory.
    """
    val = os.environ.get(HERMES_GLOBAL_HOME_ENV_VAR, "").strip()
    if val:
        return Path(val)
    return _native_hermes_root()


def _hermes_home(hermes_home: Optional[Path | str]) -> Path:
    if hermes_home is not None:
        # Explicit override: used by tests and direct callers that need a
        # specific directory rather than the global default.
        return Path(hermes_home)
    return _global_hermes_home()


def preset_file(hermes_home: Optional[Path | str] = None) -> Path:
    """Return the path of the preset JSON file (without creating it)."""
    return _hermes_home(hermes_home) / _PRESET_FILENAME


def _lock_file(hermes_home: Path) -> Path:
    return hermes_home / _LOCK_FILENAME


# ---------------------------------------------------------------------------
# Cross-platform exclusive lock context manager
# ---------------------------------------------------------------------------

class _ExclusiveLock:
    """Context manager that holds an exclusive file lock for the duration.

    Uses ``fcntl.flock`` on POSIX and ``msvcrt.locking`` on Windows.
    The lock file is created (but never written) if it does not exist.
    """

    def __init__(self, lock_path: Path) -> None:
        self._path = lock_path
        self._fd: Optional[int] = None

    def __enter__(self) -> "_ExclusiveLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(
            str(self._path),
            os.O_WRONLY | os.O_CREAT,
            0o600,
        )
        if sys.platform == "win32":
            import msvcrt  # noqa: PLC0415
            msvcrt.locking(self._fd, msvcrt.LK_LOCK, 1)
        else:
            import fcntl  # noqa: PLC0415
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_: object) -> None:
        if self._fd is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt  # noqa: PLC0415
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl  # noqa: PLC0415
                fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_preset(hermes_home: Optional[Path | str] = None) -> str:
    """Return the active LLM preset.

    Returns ``DEFAULT_PRESET`` ("claude2") when no preset file exists yet or
    when the stored value is unrecognised (forward-compat guard).
    Never raises for a missing / corrupt file.
    """
    path = preset_file(hermes_home)
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        preset = str(data.get("preset", "")).strip()
        # Migration for the original two-choice selector.  Existing sessions
        # which selected ``opus`` become Claude 1 without manual repair.
        if preset == "opus":
            return "claude1"
        if preset in VALID_PRESETS:
            return preset
        logger.warning(
            "llm_preset: unrecognised preset %r in %s; falling back to %r",
            preset,
            path,
            DEFAULT_PRESET,
        )
        return DEFAULT_PRESET
    except FileNotFoundError:
        return DEFAULT_PRESET
    except Exception:
        logger.exception("llm_preset: failed to read %s; using default", path)
        return DEFAULT_PRESET


def set_preset(
    preset: str,
    hermes_home: Optional[Path | str] = None,
) -> None:
    """Persist *preset* as the active LLM preset.

    Validates against ``VALID_PRESETS`` and raises ``ValueError`` for
    unrecognised values.  Writes atomically (temp-file + ``os.replace``)
    under an exclusive cross-process lock so concurrent callers from
    different gateway processes are serialised without data corruption.

    Args:
        preset:      One of the values in ``VALID_PRESETS``.
        hermes_home: Override the Hermes home directory (default: resolved
                     from ``hermes_constants.get_hermes_home()``).

    Raises:
        ValueError: If *preset* is not in ``VALID_PRESETS``.
    """
    preset = str(preset).strip()
    if preset == "opus":
        preset = "claude1"
    if preset not in VALID_PRESETS:
        raise ValueError(
            f"Invalid preset {preset!r}. Valid values: {sorted(VALID_PRESETS)}"
        )

    home = _hermes_home(hermes_home)
    home.mkdir(parents=True, exist_ok=True)
    path = home / _PRESET_FILENAME
    lock_path = _lock_file(home)

    with _ExclusiveLock(lock_path):
        # Import here to keep top-level free of agent dependencies.
        from utils import atomic_json_write  # noqa: PLC0415
        atomic_json_write(path, {"preset": preset})

    logger.debug("llm_preset: set to %r (written to %s)", preset, path)
