"""Regression tests for t_153f78d8 — a ready card assigned to a profile
name that no longer exists must never sit silently in ``ready`` forever.

Incident: the 2026-08-22 codex-worker -> spark rename left 13 open cards
pointing at a dead assignee. ``profile_exists()`` alone can't distinguish
that from an intentional non-profile control-plane lane (e.g. ``orion-cc``)
that is SUPPOSED to sit unspawned until a human terminal claims it — so the
fix has two parts, both covered here:

1. Dispatch time (``hermes_cli.kanban_db._block_orphaned_assignee``): a
   ready/review card whose assignee resolves via the rename ledger
   (``hermes_cli.profiles.resolve_renamed_profile``) to a real, currently
   installed profile gets explicitly BLOCKED with a readable reason instead
   of vanishing into ``skipped_nonspawnable``. A card whose assignee was
   never a real profile (a genuine control-plane lane) keeps the existing
   silent-skip behavior unchanged.

2. Rename time (``hermes_cli.profiles.rename_profile``): open kanban cards
   still pointing at the old name are migrated to the new name immediately,
   so a rename doesn't create new orphans going forward.
"""
from __future__ import annotations

import json
import sys
import tempfile
from unittest.mock import patch

import pytest


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    """Fresh HERMES_HOME + a clean re-import of the affected modules.

    Mirrors ``test_kanban_default_assignee.py``'s fixture: both
    ``kanban_db`` and ``profiles`` resolve paths dynamically off
    HERMES_HOME, but re-importing after the env var is set avoids any
    accidental cross-test module-level caching.
    """
    test_home = tempfile.mkdtemp(prefix="kanban_orphan_rename_test_")
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db, profiles
    yield kanban_db, profiles, test_home


def _fake_spawn(*args, **kwargs):
    return 12345


# ---------------------------------------------------------------------------
# 1. Dispatch-time orphan detection
# ---------------------------------------------------------------------------

def test_dead_renamed_profile_gets_blocked_not_left_ready(isolated_kanban_home):
    kb, profiles, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="orphaned card", assignee="codex-worker")

    # codex-worker was renamed to spark; spark exists, codex-worker doesn't.
    with patch.object(profiles, "profile_exists", lambda name: name == "spark"), \
         patch.object(profiles, "resolve_renamed_profile",
                       lambda name: "spark" if name == "codex-worker" else None):
        with kb.connect_closing() as conn:
            res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)

    assert task_id in res.auto_blocked
    assert task_id not in res.skipped_nonspawnable
    assert not res.spawned

    with kb.connect_closing() as conn:
        row = conn.execute(
            "SELECT status, block_kind FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
    assert row["status"] == "blocked"
    assert row["block_kind"] == "capability"

    with kb.connect_closing() as conn:
        evs = list(conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'blocked'",
            (task_id,),
        ))
    assert len(evs) == 1
    payload = json.loads(evs[0][0])
    assert payload["kind"] == "capability"
    assert "profil inconnu : codex-worker" in payload["reason"]
    assert "reassigner la carte" in payload["reason"]


def test_genuine_control_plane_lane_still_silently_skipped(isolated_kanban_home):
    """A lane name that was NEVER a real profile (e.g. ``orion-cc``) must
    keep the pre-existing "correctly idle" behavior — this is the
    regression guard against over-blocking a working feature."""
    kb, profiles, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="terminal lane card", assignee="orion-cc")

    with patch.object(profiles, "profile_exists", lambda name: False), \
         patch.object(profiles, "resolve_renamed_profile", lambda name: None):
        with kb.connect_closing() as conn:
            res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)

    assert task_id in res.skipped_nonspawnable
    assert task_id not in res.auto_blocked

    with kb.connect_closing() as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert row["status"] == "ready"


# ---------------------------------------------------------------------------
# 2. Rename-time migration + ledger resolution
# ---------------------------------------------------------------------------

def test_rename_profile_migrates_open_kanban_cards(isolated_kanban_home):
    kb, profiles, _home = isolated_kanban_home

    with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"):
        profiles.create_profile("codex-worker", no_alias=True)

    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        open_task = kb.create_task(conn, title="open card", assignee="codex-worker")

    with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"):
        profiles.rename_profile("codex-worker", "spark")

    with kb.connect_closing() as conn:
        row = conn.execute(
            "SELECT assignee FROM tasks WHERE id = ?", (open_task,),
        ).fetchone()
    assert row["assignee"] == "spark"


def test_resolve_renamed_profile_chain(isolated_kanban_home):
    """Unit-level check of the rename ledger: resolves a (possibly
    multi-hop) chain to the current name, but only if that name actually
    exists — never invents a destination."""
    kb, profiles, _home = isolated_kanban_home

    with patch.object(profiles, "_load_rename_log", lambda: [
        {"old": "codex-worker", "new": "spark-old", "at": 1},
        {"old": "spark-old", "new": "spark", "at": 2},
    ]):
        with patch.object(profiles, "profile_exists", lambda name: name == "spark"):
            assert profiles.resolve_renamed_profile("codex-worker") == "spark"
            # Never renamed: no verdict.
            assert profiles.resolve_renamed_profile("orion-cc") is None

        # Chain ends on a name that ALSO doesn't exist (e.g. deleted after
        # being renamed again) -- must not invent a destination.
        with patch.object(profiles, "profile_exists", lambda name: False):
            assert profiles.resolve_renamed_profile("codex-worker") is None
