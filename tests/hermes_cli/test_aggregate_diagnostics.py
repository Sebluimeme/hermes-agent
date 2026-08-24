"""Tests for hermes_cli.aggregate_diagnostics — the native read-only
aggregated diagnostic path for Kanban/session/context counters.

Covers: correct counters, date-window filtering, refusal of any free-form
query keyword, and that computing diagnostics never writes to any of the
three on-disk sources it reads.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hermes_cli import active_sessions
from hermes_cli import kanban_db as kb
from hermes_cli import aggregate_diagnostics as ad


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (same pattern as
    tests/hermes_cli/test_kanban_db.py)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _set_created_at(home: Path, task_id: str, ts: int) -> None:
    with kb.connect_closing() as conn:
        conn.execute("UPDATE tasks SET created_at = ? WHERE id = ?", (ts, task_id))
        conn.commit()


def _write_active_sessions(home: Path, entries: list[dict]) -> None:
    path = home / "runtime" / "active_sessions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")


def _write_context_handoff_log(home: Path, rows: list[dict]) -> Path:
    path = home / "state" / "context-handoff" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


# ---------------------------------------------------------------------------
# Kanban counters
# ---------------------------------------------------------------------------


def test_kanban_counts_by_status_matches_created_tasks(kanban_home):
    with kb.connect_closing() as conn:
        kb.create_task(conn, title="a")
        kb.create_task(conn, title="b")
        third = kb.create_task(conn, title="c")
        # create_task only ever produces ready/todo/blocked/triage at
        # creation time; "running" is a dispatch-time transition, so
        # simulate it directly to exercise the by-status counter.
        conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (third,))
        conn.commit()

    result = ad.run_aggregate_diagnostics(include_sessions=False, include_context=False)
    assert result["kanban"] == {"ready": 2, "running": 1}


def test_kanban_counts_exclude_archived(kanban_home):
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="a")
        conn.execute("UPDATE tasks SET status = 'archived' WHERE id = ?", (tid,))
        kb.create_task(conn, title="b")
        conn.commit()

    result = ad.run_aggregate_diagnostics(include_sessions=False, include_context=False)
    assert result["kanban"] == {"ready": 1}


def test_since_until_window_filters_tasks_by_created_at(kanban_home):
    with kb.connect_closing() as conn:
        old = kb.create_task(conn, title="old")
        mid = kb.create_task(conn, title="mid")
        new = kb.create_task(conn, title="new")
        conn.commit()
    _set_created_at(kanban_home, old, 1_000)
    _set_created_at(kanban_home, mid, 2_000)
    _set_created_at(kanban_home, new, 3_000)

    result = ad.run_aggregate_diagnostics(
        since=1_500, until=2_500, include_sessions=False, include_context=False,
    )
    assert result["kanban"] == {"ready": 1}
    assert result["window"] == {"since": 1500, "until": 2500}


def test_since_accepts_iso8601(kanban_home):
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="a")
        conn.commit()
    _set_created_at(kanban_home, tid, 1_700_000_000)

    result = ad.run_aggregate_diagnostics(
        since="2023-11-14T00:00:00Z", include_sessions=False, include_context=False,
    )
    assert result["kanban"] == {"ready": 1}


def test_invalid_since_raises(kanban_home):
    with pytest.raises(ad.DiagnosticsError):
        ad.run_aggregate_diagnostics(since="not-a-date")


def test_since_after_until_raises(kanban_home):
    with pytest.raises(ad.DiagnosticsError):
        ad.run_aggregate_diagnostics(since=2_000, until=1_000)


# ---------------------------------------------------------------------------
# Refusal of a free-form query surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(ad._FREE_FORM_QUERY_KEYS))
def test_rejects_free_form_query_keywords(kanban_home, key):
    with pytest.raises(ad.DiagnosticsError):
        ad.run_aggregate_diagnostics(**{key: "SELECT * FROM tasks"})


def test_rejects_unknown_keyword(kanban_home):
    with pytest.raises(ad.DiagnosticsError):
        ad.run_aggregate_diagnostics(unexpected_flag=True)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def test_session_counts_by_surface(kanban_home):
    _write_active_sessions(
        kanban_home,
        [
            {"pid": 999999991, "surface": "cli", "started_at": 5_000},
            {"pid": 999999992, "surface": "cli", "started_at": 5_100},
            {"pid": 999999993, "surface": "discord", "started_at": 5_200},
        ],
    )
    result = ad.session_counts()
    assert result["total"] == 0  # all pids are dead (not real processes)
    assert result["by_surface"] == {}


def test_session_counts_window_filters_on_started_at(kanban_home, monkeypatch):
    # Bypass liveness pruning to isolate the windowing logic under test.
    monkeypatch.setattr(active_sessions, "_prune_dead", lambda entries: entries)
    _write_active_sessions(
        kanban_home,
        [
            {"pid": 1, "surface": "cli", "started_at": 1_000},
            {"pid": 2, "surface": "cli", "started_at": 5_000},
        ],
    )
    result = ad.session_counts(since_ts=2_000)
    assert result["total"] == 1
    assert result["by_surface"] == {"cli": 1}


# ---------------------------------------------------------------------------
# Context handoffs
# ---------------------------------------------------------------------------


def test_context_handoff_counts_by_event(kanban_home):
    _write_context_handoff_log(
        kanban_home,
        [
            {"at": 1_000, "session_id": "s1", "event": "handoff_required"},
            {"at": 2_000, "session_id": "s2", "event": "handoff_required"},
            {"at": 3_000, "session_id": "s3", "event": "handoff_resumed"},
        ],
    )
    result = ad.context_handoff_counts()
    assert result == {
        "total": 3,
        "by_event": {"handoff_required": 2, "handoff_resumed": 1},
    }


def test_context_handoff_counts_window(kanban_home):
    _write_context_handoff_log(
        kanban_home,
        [
            {"at": 1_000, "event": "handoff_required"},
            {"at": 5_000, "event": "handoff_required"},
        ],
    )
    result = ad.context_handoff_counts(since_ts=2_000)
    assert result == {"total": 1, "by_event": {"handoff_required": 1}}


def test_context_handoff_skips_malformed_lines(kanban_home):
    path = _write_context_handoff_log(kanban_home, [{"at": 1_000, "event": "handoff_required"}])
    with path.open("a", encoding="utf-8") as fh:
        fh.write("not json at all\n")
        fh.write("[]\n")  # valid JSON, not an object
        fh.write(json.dumps({"session_id": "no-at-field", "event": "x"}) + "\n")
    result = ad.context_handoff_counts()
    assert result == {"total": 1, "by_event": {"handoff_required": 1}}


def test_context_handoff_missing_file_returns_zero(kanban_home):
    result = ad.context_handoff_counts()
    assert result == {"total": 0, "by_event": {}}


# ---------------------------------------------------------------------------
# No writes
# ---------------------------------------------------------------------------


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_running_diagnostics_does_not_modify_kanban_db(kanban_home):
    with kb.connect_closing() as conn:
        kb.create_task(conn, title="a")
        conn.commit()
    db_path = kb.kanban_db_path()
    # Force a WAL checkpoint so the on-disk main db file reflects the insert
    # before we snapshot it, then compare byte-for-byte after the read.
    with kb.connect_closing() as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    before = _hash_file(db_path)
    before_rows = None
    with kb.connect_closing() as conn:
        before_rows = conn.execute("SELECT id, status, created_at FROM tasks ORDER BY id").fetchall()

    ad.run_aggregate_diagnostics()
    ad.run_aggregate_diagnostics(since=0, until=9_999_999_999)

    with kb.connect_closing() as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    after = _hash_file(db_path)
    with kb.connect_closing() as conn:
        after_rows = conn.execute("SELECT id, status, created_at FROM tasks ORDER BY id").fetchall()

    assert [tuple(r) for r in before_rows] == [tuple(r) for r in after_rows]
    assert before == after


def test_running_diagnostics_does_not_modify_active_sessions_file(kanban_home):
    _write_active_sessions(kanban_home, [{"pid": 1, "surface": "cli", "started_at": 1}])
    path = kanban_home / "runtime" / "active_sessions.json"
    before = _hash_file(path)

    ad.session_counts()
    ad.run_aggregate_diagnostics()

    after = _hash_file(path)
    assert before == after


def test_running_diagnostics_does_not_modify_context_handoff_log(kanban_home):
    path = _write_context_handoff_log(kanban_home, [{"at": 1, "event": "handoff_required"}])
    before = _hash_file(path)

    ad.context_handoff_counts()
    ad.run_aggregate_diagnostics()

    after = _hash_file(path)
    assert before == after
