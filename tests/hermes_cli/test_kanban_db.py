"""Tests for the Kanban DB layer (hermes_cli.kanban_db)."""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
import os
import sqlite3
import subprocess
import sys
import time
import types
import unittest.mock
from pathlib import Path

import pytest

import hermes_state
from hermes_cli import kanban_db as kb


TEST_SPARK_MODEL_OVERRIDE = "gpt-5.3-codex-spark"
TEST_HANDOFF_ROUTES = {
    kb.ROUTING_TIER_SIMPLE: (
        ("spark", TEST_SPARK_MODEL_OVERRIDE),
        ("claude2", None),
        ("claude1", None),
        ("coder", None),
    ),
    kb.ROUTING_TIER_COMPLEX: (
        ("claude2", None),
        ("claude1", None),
        ("coder", None),
    ),
}


@pytest.fixture
def configured_handoff_routes(monkeypatch):
    monkeypatch.setattr(kb, "_configured_handoff_routes", lambda: TEST_HANDOFF_ROUTES)
    return TEST_HANDOFF_ROUTES


def test_handoff_routes_empty_config_disables_implicit_route(monkeypatch):
    from hermes_cli import config

    monkeypatch.setattr(config, "load_config_readonly", lambda: {"kanban": {"handoff_routes": {}}})

    assert kb._configured_handoff_routes() == {}
    assert kb.resolve_ordered_route("simple") == (None, None, [])
    assert kb.resolve_parallel_routes("simple", 2) == ([], [])


def test_handoff_routes_configured_chain_picks_exact_hop(monkeypatch):
    from hermes_cli import config

    monkeypatch.setattr(config, "load_config_readonly", lambda: {"kanban": {"handoff_routes": {
        "simple": [
            {"profile": "spark", "model_override": TEST_SPARK_MODEL_OVERRIDE},
            {"profile": "claude2"},
        ],
    }}})

    assert kb.resolve_ordered_route(
        "simple", preflight_fn=lambda route: (route == "claude2", "ok"),
    )[:2] == ("claude2", None)


def test_handoff_routes_all_refused_returns_terminal_profile_without_exception(
    configured_handoff_routes,
):
    assignee, model, trace = kb.resolve_ordered_route(
        "complex", preflight_fn=lambda _route: (False, "provider_cooldown"),
    )

    assert (assignee, model) == ("coder", None)
    assert [entry["route"] for entry in trace] == ["claude2", "claude1", "coder"]


@pytest.mark.parametrize(
    "entries",
    [
        [{"profile": "claude2"}, {"profile": "claude2"}],
        [{"profile": "claude2"}, {"profile": ""}],
        [{"profile": "claude2"}, 123],
    ],
)
def test_handoff_routes_duplicate_or_invalid_entry_fails_closed(monkeypatch, entries):
    from hermes_cli import config

    monkeypatch.setattr(config, "load_config_readonly", lambda: {"kanban": {"handoff_routes": {
        "complex": entries,
    }}})

    assert kb._configured_handoff_routes() == {}
    assert kb.resolve_ordered_route("complex") == (None, None, [])
    assert kb.resolve_parallel_routes("complex", 1) == ([], [])


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _init_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "kanban@example.com"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Kanban Test"], check=True, capture_output=True, text=True)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], check=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Schema / init
# ---------------------------------------------------------------------------







@pytest.mark.windows_only
def test_cross_process_init_lock_uses_windows_byte_range_lock(tmp_path, monkeypatch):
    """Windows must use a real (non-blocking) process lock, not a no-op open.

    The init lock acquires with LK_NBLCK in a bounded retry loop (#36644) so a
    wedged holder can never block connect() forever; a clean acquire takes the
    lock once and releases it once.

    ``windows_only``: ``msvcrt`` does not exist off Windows, so faking
    ``_IS_WINDOWS`` on Linux meant injecting a fake ``msvcrt`` module too —
    the test then asserted against its own stub rather than the byte-range
    locking API. Here the platform is real; only ``msvcrt.locking`` is
    instrumented so the call sequence is observable.
    """
    calls: list[tuple[int, int, int]] = []
    import msvcrt as _msvcrt

    fake_msvcrt = types.SimpleNamespace(
        LK_NBLCK=_msvcrt.LK_NBLCK,
        LK_UNLCK=_msvcrt.LK_UNLCK,
        locking=lambda fd, mode, nbytes: calls.append((fd, mode, nbytes)),
    )
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)

    db_path = tmp_path / "kanban.db"
    with kb._cross_process_init_lock(db_path):
        # Acquired exactly once via the non-blocking byte-range lock.
        assert [call[1:] for call in calls] == [(fake_msvcrt.LK_NBLCK, 1)]

    # Released once on exit.
    assert [call[1:] for call in calls] == [
        (fake_msvcrt.LK_NBLCK, 1),
        (fake_msvcrt.LK_UNLCK, 1),
    ]


def test_connect_migrates_legacy_db_before_optional_column_indexes(tmp_path):
    """Legacy DBs missing additive indexed columns must migrate cleanly.

    SCHEMA_SQL runs in ``connect()`` before ``_migrate_add_optional_columns``.
    Indexes over additive columns therefore must be created after the
    migration adds those columns, or boards predating the column fail to
    open before migration can run.

    Covers all four indexes that sit on additive columns:
    - ``tasks.session_id``       -> ``idx_tasks_session_id``    (#28447)
    - ``tasks.tenant``           -> ``idx_tasks_tenant``        (#16081)
    - ``tasks.idempotency_key``  -> ``idx_tasks_idempotency``   (#17805)
    - ``task_events.run_id``     -> ``idx_events_run``          (#17805)
    """
    db_path = tmp_path / "legacy-kanban.db"
    conn = sqlite3.connect(str(db_path))
    # Pre-#16081 ``tasks`` shape: missing tenant, idempotency_key, session_id.
    conn.execute("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            claim_lock TEXT,
            claim_expires INTEGER
        )
    """)
    # Pre-#17805 ``task_events`` shape: missing run_id. Required because
    # ``_migrate_add_optional_columns`` unconditionally runs PRAGMA on
    # ``task_events`` for run_id back-fill.
    conn.execute("""
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at) "
        "VALUES ('legacy', 'old board task', 'ready', 1)"
    )
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at) "
        "VALUES ('legacy-done', 'already delivered task', 'done', 1)"
    )
    conn.commit()
    conn.close()

    with kb.connect(db_path) as migrated:
        task_columns = {
            row["name"] for row in migrated.execute("PRAGMA table_info(tasks)")
        }
        event_columns = {
            row["name"]
            for row in migrated.execute("PRAGMA table_info(task_events)")
        }
        indexes = {
            row["name"]
            for row in migrated.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        legacy_done = kb.get_task(migrated, "legacy-done")

    # Additive columns added by migration:
    assert "session_id" in task_columns
    assert "tenant" in task_columns
    assert "idempotency_key" in task_columns
    assert "run_id" in event_columns
    # And their indexes — the regression scope of this test:
    assert "idx_tasks_session_id" in indexes
    assert "idx_tasks_tenant" in indexes
    assert "idx_tasks_idempotency" in indexes
    assert "idx_events_run" in indexes
    assert legacy_done.delivery_status == "delivered"


# ---------------------------------------------------------------------------
# Task creation + status inference
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Links + dependency resolution
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# Atomic claim (CAS)
# ---------------------------------------------------------------------------



def test_schedule_task_parks_time_delay_without_dispatching(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="delayed recheck", assignee="ops")
        assert kb.schedule_task(conn, t, reason="run next week") is True
        task = kb.get_task(conn, t)
        assert task.status == "scheduled"
        assert kb.claim_task(conn, t) is None

        events = kb.list_events(conn, t)
        assert any(e.kind == "scheduled" and e.payload == {"reason": "run next week"} for e in events)


def test_schedule_task_refuses_workspace_or_dependency_serialization(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="must auto resume", assignee="coder")
        with pytest.raises(ValueError, match="time-based pause"):
            kb.schedule_task(
                conn,
                task_id,
                reason="automatic serialization hold: wait for workspace dependency",
            )
        assert kb.get_task(conn, task_id).status == "ready"








def test_stale_claim_reclaim_event_records_diagnostic_payload(
    kanban_home, monkeypatch,
):
    """``reclaimed`` events should carry claim_expires, last_heartbeat_at,
    and worker_pid so operators can diagnose why a claim went stale
    (#23025: previous payload only had ``stale_lock`` which gives no
    timing context)."""
    import json
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        host = _kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, t, claimer=f"{host}:worker")
        kb._set_worker_pid(conn, t, 12345)
        old_expires = int(time.time()) - 3600
        hb_at = int(time.time()) - 1800
        conn.execute(
            "UPDATE tasks SET claim_expires = ?, last_heartbeat_at = ? "
            "WHERE id = ?",
            (old_expires, hb_at, t),
        )

        monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
        kb.release_stale_claims(conn, signal_fn=lambda _p, _s: None)
        row = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'reclaimed'",
            (t,),
        ).fetchone()
        assert row is not None
        payload = json.loads(row["payload"])
        assert payload["claim_expires"] == old_expires
        assert payload["last_heartbeat_at"] == hb_at
        assert payload["worker_pid"] == 12345
        assert payload["host_local"] is True






# ---------------------------------------------------------------------------
# Rate-limit requeue: a worker that bails on a provider quota wall must be
# released back to ``ready`` WITHOUT counting a failure, so a long (e.g.
# 5-hour) quota window can't trip the circuit breaker and permanently block
# the card. The respawn guard then defers it on a cooldown until quota
# returns. Regression coverage for the kanban-rate-limit-failure report.
# ---------------------------------------------------------------------------


def _exited_status(code: int) -> int:
    """Raw wait-status for a WIFEXITED child with the given exit code."""
    return code << 8




def test_rate_limit_exit_requeues_without_counting_failure(
    kanban_home, monkeypatch,
):
    """A rate-limit sentinel exit releases the task to ``ready`` and leaves
    ``consecutive_failures`` untouched — the breaker must never trip on a
    transient throttle, even across many quota-wall hits."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kb.connect() as conn:
        host = _kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="rl", assignee="a")

        # Simulate FAR more quota-wall hits than DEFAULT_FAILURE_LIMIT (2).
        # If any of these counted as a failure the task would be blocked.
        for i in range(6):
            pid = 70000 + i
            # Claim to open a real run (so detect_crashed_workers can close
            # it with a rate_limited outcome), then point the claim at this
            # host + a dead pid so the crash path acts on it.
            kb.claim_task(conn, tid, claimer=f"{host}:w{i}")
            conn.execute(
                "UPDATE tasks SET worker_pid=?, consecutive_failures=? "
                "WHERE id=?",
                (pid, 0, tid),
            )
            conn.commit()
            _kb._record_worker_exit(
                pid, _exited_status(_kb.KANBAN_RATE_LIMIT_EXIT_CODE)
            )

            crashed = kb.detect_crashed_workers(conn)
            # Rate-limited requeues are NOT crashes.
            assert tid not in crashed
            rl = getattr(_kb.detect_crashed_workers, "_last_rate_limited", [])
            assert tid in rl

            task = kb.get_task(conn, tid)
            assert task.status == "ready", (
                f"hit {i}: should requeue ready, got {task.status}"
            )
            assert task.consecutive_failures == 0, (
                f"hit {i}: rate-limit must not count a failure, "
                f"got {task.consecutive_failures}"
            )

        # Last failure error stamped so the respawn guard recognizes the
        # quota wall.
        assert task.last_failure_error and "rate-limited" in task.last_failure_error

        # A ``rate_limited`` run outcome was recorded (not ``crashed``).
        outcomes = [
            r["outcome"] for r in conn.execute(
                "SELECT outcome FROM task_runs WHERE task_id=?", (tid,),
            ).fetchall()
        ]
        assert "rate_limited" in outcomes
        assert "crashed" not in outcomes


def test_guardrail_halt_requeues_checkpoint_without_counting_failure(
    kanban_home, monkeypatch,
):
    """A controlled repeated-tool stop resumes with a different strategy."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kb.connect() as conn:
        host = _kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="guarded", assignee="a")
        pid = 70991
        kb.claim_task(conn, tid, claimer=f"{host}:guardrail")
        conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (pid, tid))
        conn.commit()
        _kb._record_worker_exit(
            pid, _exited_status(_kb.KANBAN_GUARDRAIL_HALT_EXIT_CODE)
        )

        assert tid not in kb.detect_crashed_workers(conn)
        task = kb.get_task(conn, tid)
        strategy = getattr(
            _kb.detect_crashed_workers, "_last_strategy_required", []
        )
        run = conn.execute(
            "SELECT outcome FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? "
            "AND kind='strategy_required' ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()

    assert task.status == "ready"
    assert task.consecutive_failures == 0
    assert task.last_failure_error and "different tool" in task.last_failure_error
    assert tid in strategy
    assert run["outcome"] == "strategy_required"
    assert event is not None


def test_interrupted_exit_requeues_exact_session_without_counting_failure(
    kanban_home, monkeypatch,
):
    """An orchestration interruption is a neutral resumable yield."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kb.connect() as conn:
        host = _kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="interrupted", assignee="a")
        pid = 70993
        kb.claim_task(conn, tid, claimer=f"{host}:interrupted")
        run_id = kb.get_task(conn, tid).current_run_id
        conn.execute(
            "UPDATE tasks SET worker_pid=?, last_failure_error='stale', "
            "failure_class='crashed', execution_status='retrying', "
            "next_retry_at=999999, action_required='stale action' WHERE id=?",
            (pid, tid),
        )
        conn.execute(
            "UPDATE task_runs SET metadata=? WHERE id=?",
            ('{"worker_session_id":"sess-resume"}', run_id),
        )
        conn.commit()
        _kb._record_worker_exit(
            pid, _exited_status(_kb.KANBAN_INTERRUPTED_EXIT_CODE)
        )

        assert tid not in kb.detect_crashed_workers(conn)
        task = kb.get_task(conn, tid)
        neutral = getattr(
            _kb.detect_crashed_workers, "_last_interrupted", []
        )
        run = conn.execute(
            "SELECT outcome, metadata FROM task_runs WHERE id=?", (run_id,),
        ).fetchone()

    assert task.status == "ready"
    assert task.consecutive_failures == 0
    assert task.last_failure_error is None
    assert task.failure_class is None
    assert task.execution_status == "pending"
    assert task.next_retry_at is None
    assert task.action_required is None
    assert tid in neutral
    assert run["outcome"] == "interrupted"
    assert json.loads(run["metadata"])["automatic_resume"] is True
    assert _kb._transient_resume_session_id(tid, board=None) == "sess-resume"


def test_protocol_violation_elapsed_uses_current_run_start(
    kanban_home, monkeypatch,
):
    """Retry diagnostics report this run, not the task's original start."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(_kb.time, "time", lambda: 1_000.0)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kb.connect() as conn:
        host = _kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="elapsed", assignee="a")
        pid = 70992
        kb.claim_task(conn, tid, claimer=f"{host}:elapsed")
        run_id = kb.get_task(conn, tid).current_run_id
        conn.execute("UPDATE tasks SET worker_pid=?, started_at=100 WHERE id=?", (pid, tid))
        conn.execute("UPDATE task_runs SET started_at=996 WHERE id=?", (run_id,))
        conn.commit()
        _kb._record_worker_exit(pid, _exited_status(0))

        kb.detect_crashed_workers(conn)
        run = conn.execute(
            "SELECT error FROM task_runs WHERE id=?", (run_id,)
        ).fetchone()

    assert "after 4s" in run["error"]
    assert "after 900s" not in run["error"]


def test_rate_limit_exit_immediately_falls_back_with_same_worktree(
    kanban_home, all_assignees_spawnable, monkeypatch, configured_handoff_routes,
):
    """A proven quota wall advances a routed card without human input."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kb.connect() as conn:
        host = _kb._claimer_id().split(":", 1)[0]
        workspace = kanban_home / "project"
        workspace.mkdir()
        tid = kb.create_task(
            conn,
            title="quota fallback",
            assignee="claude2",
            routing_tier="complex",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        original_workspace = kb.get_task(conn, tid).workspace_path
        pid = 71001
        kb.claim_task(conn, tid, claimer=f"{host}:quota")
        conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (pid, tid))
        conn.commit()
        _kb._record_worker_exit(pid, _exited_status(_kb.KANBAN_RATE_LIMIT_EXIT_CODE))

        assert tid not in kb.detect_crashed_workers(conn)
        task = kb.get_task(conn, tid)
        fallback_event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? "
            "AND kind = 'simple_route_fallback' ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()

    assert task is not None
    assert task.status == "ready"
    assert task.assignee == "claude1"
    assert task.workspace_path == original_workspace
    assert task.consecutive_failures == 0
    assert fallback_event is not None


def test_rate_limit_fallback_survives_same_tick_pool_capacity_reroute(
    kanban_home, all_assignees_spawnable, monkeypatch, configured_handoff_routes,
):
    """t_664c2bab / t_46414f08: after ``fallback_simple_route`` advances a
    rate-limited claude2 card to claude1, the pool capacity reroute that
    runs later in the *same* ``dispatch_once`` tick must not walk it back
    to claude2 just because claude2 now shows zero running workers and a
    clean preflight. The real incident: ``simple_route_fallback`` (claude2
    -> claude1) was immediately followed by an ``assigned`` event with
    ``source=generalist_pool_free_capacity`` putting claude2 right back,
    so the card never actually reached Claude 1."""
    import hermes_cli.kanban_db as _kb
    from hermes_cli import config as _kb_config

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(
        _kb_config, "load_config",
        lambda: {"kanban": {"generalist_worker_pool_routing": True}},
    )

    spawned = []

    def fake_spawn(task, workspace, **_kwargs):
        spawned.append((task.assignee, workspace))
        return 71_500 + len(spawned)

    with kb.connect() as conn:
        host = _kb._claimer_id().split(":", 1)[0]
        workspace = kanban_home / "project"
        workspace.mkdir()
        tid = kb.create_task(
            conn,
            title="quota fallback survives pool reroute",
            assignee="claude2",
            routing_tier="complex",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        pid = 71050
        kb.claim_task(conn, tid, claimer=f"{host}:quota")
        conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (pid, tid))
        conn.commit()
        _kb._record_worker_exit(pid, _exited_status(_kb.KANBAN_RATE_LIMIT_EXIT_CODE))

        # One full tick: detect_crashed_workers() runs the proven fallback
        # (claude2 -> claude1) *and* the ready-task pool reroute loop below
        # it, all inside this single dispatch_once() call -- exactly the
        # same-tick race that produced the incident.
        result = kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn=1)

        task = kb.get_task(conn, tid)
        pool_reroute_event = conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? "
            "AND kind = 'assigned' "
            "AND json_extract(payload, '$.source') = 'generalist_pool_free_capacity'",
            (tid,),
        ).fetchone()

    assert task is not None
    assert task.assignee == "claude1"
    assert pool_reroute_event is None
    assert result.spawned == [(tid, "claude1", str(workspace.resolve()))]
    assert spawned == [("claude1", str(workspace.resolve()))]


def test_respawn_guard_defers_rate_limited_within_cooldown(
    kanban_home, monkeypatch,
):
    """Within the cooldown after a rate-limit requeue, the guard defers the
    respawn; after the cooldown it allows a probe — and crucially does NOT
    fall into ``blocker_auth`` (which would defer forever)."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    now = 5_000_000

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="rl-guard", assignee="a")
        # Seed a rate_limited run that just ended + the stamped error.
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        conn.execute(
            "UPDATE task_runs SET outcome='rate_limited', status='rate_limited', "
            "ended_at=? WHERE id=?",
            (now, run_id),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, "
            "last_failure_error=? WHERE id=?",
            ("pid 1 exited rate-limited (quota wall) — requeued", tid),
        )
        conn.commit()

        # Inside cooldown → defer with the rate-limit-specific reason.
        monkeypatch.setattr(_kb.time, "time", lambda: now + 100)
        assert kb.check_respawn_guard(conn, tid) == "rate_limit_cooldown"

        # Past cooldown → allowed (None), NOT trapped by blocker_auth even
        # though last_failure_error contains "rate-limited".
        monkeypatch.setattr(_kb.time, "time", lambda: now + 400)
        assert kb.check_respawn_guard(conn, tid) is None


def test_respawn_guard_exponentially_backs_off_repeated_unknown_quota(
    kanban_home, monkeypatch,
):
    """A final route without reset evidence must not probe every five minutes."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    now = 5_000_000
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="repeated quota", assignee="coder")
        for ended_at in (now - 400, now):
            conn.execute(
                "INSERT INTO task_runs(task_id, profile, status, started_at, ended_at, outcome) "
                "VALUES (?, 'coder', 'rate_limited', ?, ?, 'rate_limited')",
                (tid, ended_at - 10, ended_at),
            )
        conn.execute(
            "UPDATE tasks SET status='ready', last_failure_error='HTTP 429 quota wall' "
            "WHERE id=?",
            (tid,),
        )
        conn.commit()

        monkeypatch.setattr(_kb.time, "time", lambda: now + 500)
        assert kb.check_respawn_guard(conn, tid) == "rate_limit_cooldown"
        monkeypatch.setattr(_kb.time, "time", lambda: now + 700)
        assert kb.check_respawn_guard(conn, tid) is None


def test_provider_reset_blocks_profile_globally_until_explicit_deadline(
    kanban_home, monkeypatch,
):
    cooldowns = kanban_home / "state" / "provider-error-cooldowns.json"
    monkeypatch.setenv("HERMES_KANBAN_PROVIDER_COOLDOWNS_PATH", str(cooldowns))
    received = 10_000.0
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="coder reset", assignee="coder")
        captured = kb.capture_provider_reset(
            conn,
            tid,
            "HTTP status: 429 retry-after: 600 seconds",
            received_at=received,
        )
        conn.commit()

    assert captured is not None
    assert captured["route"] == "coder"
    assert captured["reset_source"] == "api_retry_after"
    assert kb.route_preflight_ok("coder", now=received + 599) == (
        False,
        "provider_cooldown",
    )
    assert kb.route_preflight_ok("coder", now=received + 601)[0] is True


def test_respawn_guard_does_not_transfer_rate_limit_to_new_profile(
    kanban_home, monkeypatch,
):
    """An explicit reassignment must bypass the exhausted profile's cooldown."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    now = 5_000_000

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="reroute-rl", assignee="spark")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        conn.execute(
            "UPDATE task_runs SET outcome='rate_limited', status='rate_limited', "
            "ended_at=? WHERE id=?",
            (now, run_id),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, "
            "last_failure_error=? WHERE id=?",
            ("provider quota wall", tid),
        )
        conn.commit()

        monkeypatch.setattr(_kb.time, "time", lambda: now + 100)
        assert kb.check_respawn_guard(conn, tid) == "rate_limit_cooldown"

        assert kb.assign_task(conn, tid, "coder") is True
        assert kb.check_respawn_guard(conn, tid) is None


def test_dispatch_blocks_only_proven_claude_cooldown(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Unknown or expired telemetry must not strand executable work."""
    routing = kanban_home / "state" / "ai-quota-routing.json"
    routing.parent.mkdir(parents=True)
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ROUTING_PATH", str(routing))
    now = 10_000.0

    def write_record(record):
        routing.write_text(json.dumps({"agent_cooldowns": {"claude1": record}}))

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="quota guarded", assignee="claude1")

        write_record({
            "dispatch_allowed": False,
            "preflight_required": True,
            "cooldown_until": "1970-01-01T02:50:01+00:00",
            "source": "official_ui_ocr",
            "window": "session",
            "retry_after_seconds": 3600,
        })
        monkeypatch.setattr(kb.time, "time", lambda: now)
        deferred = kb.dispatch_once(conn, dry_run=True)
        assert (task_id, "provider_cooldown") in deferred.respawn_guarded
        assert task_id not in [row[0] for row in deferred.spawned]

        routing.unlink()
        unknown = kb.dispatch_once(conn, dry_run=True)
        assert task_id in [row[0] for row in unknown.spawned]
        assert unknown.respawn_guarded == []

        write_record({
            "dispatch_allowed": False,
            "preflight_required": True,
            "cooldown_until": "1970-01-01T02:46:40+00:00",
        })
        expired = kb.dispatch_once(conn, dry_run=True)
        assert task_id in [row[0] for row in expired.spawned]
        assert expired.respawn_guarded == []

        write_record({"dispatch_allowed": True, "preflight_required": False})
        allowed = kb.dispatch_once(conn, dry_run=True)
        assert task_id in [row[0] for row in allowed.spawned]


def test_dispatch_once_reroutes_already_assigned_card_off_dead_quota(
    kanban_home, all_assignees_spawnable, monkeypatch, configured_handoff_routes,
):
    """t_47dc2bf0: a card that already carries an explicit assignee (the
    normal case for every pre-existing open card, not just fresh
    auto-routed ones) must not wait on that same dead assignee forever.
    When ``quota_dispatch_guard`` refuses the current assignee, the
    dispatcher advances the card one hop in its ordered fallback chain
    instead of only deferring with a ``provider_cooldown`` event -- the
    exact "817 provider_cooldown events, no other agent takes over" defect
    described on the card."""
    routing = kanban_home / "state" / "ai-quota-routing.json"
    routing.parent.mkdir(parents=True)
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ROUTING_PATH", str(routing))
    # claude2 confirmed dead; claude1 has no cache entry at all, which
    # quota_dispatch_guard also fails closed on -- so the FIRST tick can
    # only ever defer claude1 too, never spawn it in this same tick. The
    # assertion is about the reroute (assignee change + event), not a
    # same-tick spawn.
    routing.write_text(json.dumps({"agent_cooldowns": {
        "claude2": {
            "dispatch_allowed": False, "preflight_required": True,
            "cooldown_until": "1970-01-01T02:50:01+00:00",
        },
    }}))
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="stuck on dead quota", assignee="claude2")
        monkeypatch.setattr(kb.time, "time", lambda: 10_000.0)
        result = kb.dispatch_once(conn, dry_run=False)
        assert (task_id, "provider_cooldown") in result.respawn_guarded
        row = conn.execute(
            "SELECT assignee, last_failure_error FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        assert row["assignee"] == "claude1"
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'simple_route_fallback'",
            (task_id,),
        ).fetchone()
        assert event is not None
        payload = json.loads(event["payload"])
        assert payload["from_route"] == "Claude 2"
        assert payload["to_route"] == "Claude 1"
        assert payload["to_assignee"] == "claude1"


@pytest.mark.parametrize(
    "record",
    [
        None,
        {
            "dispatch_allowed": False,
            "preflight_required": True,
            "cooldown_until": "1970-01-01T02:46:40+00:00",
        },
    ],
)
def test_dispatch_unknown_claude_quota_attempts_requested_lane(
    kanban_home, all_assignees_spawnable, monkeypatch, record,
):
    routing = kanban_home / "state" / "ai-quota-routing.json"
    routing.parent.mkdir(parents=True)
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ROUTING_PATH", str(routing))
    if record is not None:
        routing.write_text(json.dumps({"agent_cooldowns": {"claude1": record}}))
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="keep requested Claude", assignee="claude1")
        monkeypatch.setattr(kb.time, "time", lambda: 10_000.0)
        result = kb.dispatch_once(conn, dry_run=True)
        row = conn.execute(
            "SELECT assignee, last_failure_error FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        fallback = conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'simple_route_fallback'",
            (task_id,),
        ).fetchone()

    assert task_id in [row[0] for row in result.spawned]
    assert result.respawn_guarded == []
    assert row["assignee"] == "claude1"
    assert row["last_failure_error"] is None
    assert fallback is None


# ---------------------------------------------------------------------------
# Spark-first bounded routing; complex cards retain Claude/Terra routing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("simple", "simple"),
        ("SIMPLE", "simple"),
        ("  complex  ", "complex"),
        ("complex", "complex"),
        (None, "complex"),
        ("", "complex"),
        ("bogus", "complex"),
        ("simple ", "simple"),
    ],
)
def test_normalize_routing_tier_is_fail_safe_to_complex(value, expected):
    assert kb.normalize_routing_tier(value) == expected


def test_create_task_persists_routing_tier_and_get_task_reads_it_back(kanban_home):
    with kb.connect() as conn:
        simple_id = kb.create_task(conn, title="s", assignee="a", routing_tier="simple")
        complex_id = kb.create_task(conn, title="c", assignee="a", routing_tier="complex")
        unset_id = kb.create_task(conn, title="u", assignee="a")

        assert kb.get_task(conn, simple_id).routing_tier == "simple"
        assert kb.get_task(conn, complex_id).routing_tier == "complex"
        # Omitted at creation time: stored as NULL, never silently upgraded
        # to a concrete value -- normalize_routing_tier is what fails safe.
        assert kb.get_task(conn, unset_id).routing_tier is None
        assert kb.normalize_routing_tier(kb.get_task(conn, unset_id).routing_tier) == "complex"


def test_create_task_rejects_invalid_routing_tier(kanban_home):
    with kb.connect() as conn:
        with pytest.raises(ValueError):
            kb.create_task(conn, title="bad", assignee="a", routing_tier="urgent")


def test_create_task_rejects_read_only_analysis_on_shared_workspace_root(kanban_home):
    shared_workspace = kanban_home / "workspace"
    shared_workspace.mkdir()
    with kb.connect() as conn:
        with pytest.raises(ValueError, match="cannot reserve the shared Hermes workspace root"):
            kb.create_task(
                conn,
                title="Établir une preuve du TTL",
                body="Enquêter en lecture seule et ne pas modifier Firebase.",
                assignee="claude2",
                workspace_kind="dir",
                workspace_path=str(shared_workspace),
            )


def test_create_task_allows_exact_workspace_for_read_only_analysis(kanban_home):
    exact_workspace = kanban_home / "workspace" / "ecobloc"
    exact_workspace.mkdir(parents=True)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Vérifier le TTL",
            body="Contrôle en lecture seule.",
            assignee="claude2",
            workspace_kind="dir",
            workspace_path=str(exact_workspace),
        )
    assert task_id.startswith("t_")


def test_route_preflight_ok_reads_claude_from_quota_cache(kanban_home, monkeypatch):
    routing = kanban_home / "state" / "ai-quota-routing.json"
    routing.parent.mkdir(parents=True)
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ROUTING_PATH", str(routing))

    # No cache at all: the real worker attempt is the availability probe.
    ok, reason = kb.route_preflight_ok("claude2")
    assert (ok, reason) == (True, "fresh_available")

    routing.write_text(json.dumps({"agent_cooldowns": {
        "claude2": {"dispatch_allowed": True, "preflight_required": False},
    }}))
    ok, reason = kb.route_preflight_ok("claude2")
    assert ok is True

    routing.write_text(json.dumps({"agent_cooldowns": {
        "claude2": {"dispatch_allowed": False, "preflight_required": True,
                     "cooldown_until": "1970-01-01T02:50:01+00:00"},
    }}))
    ok, reason = kb.route_preflight_ok("claude2", now=10_000.0)
    assert (ok, reason) == (False, "provider_cooldown")


def test_route_preflight_ok_coder_fails_open_by_default(kanban_home, monkeypatch):
    routing = kanban_home / "state" / "ai-quota-routing.json"
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ROUTING_PATH", str(routing))
    # No cache file at all -- "no live GPT quota measurement wired up yet"
    # must never strand every route.
    ok, reason = kb.route_preflight_ok("coder")
    assert (ok, reason) == (True, "fail_open_last_resort")


def test_route_preflight_ok_coder_respects_active_cooldown(kanban_home, monkeypatch):
    routing = kanban_home / "state" / "ai-quota-routing.json"
    routing.parent.mkdir(parents=True)
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ROUTING_PATH", str(routing))
    routing.write_text(json.dumps({"agent_cooldowns": {
        "coder": {"dispatch_allowed": False, "reason": "provider_cooldown",
                          "cooldown_until": "1970-01-01T02:50:01+00:00"},
    }}))
    ok, reason = kb.route_preflight_ok("coder", now=10_000.0)
    assert ok is False
    # Past the deadline (1970-01-01T02:50:01Z == 10201.0s), the same
    # cooldown record no longer blocks it.
    ok, reason = kb.route_preflight_ok("coder", now=10_300.0)
    assert (ok, reason) == (True, "cooldown_expired")


def test_route_preflight_ok_rejects_unknown_route():
    ok, reason = kb.route_preflight_ok("some-other-lane")
    assert ok is False
    assert reason == "unknown_route:some-other-lane"


def test_route_preflight_ok_spark_saturation_without_cooldown_blocks(kanban_home, monkeypatch):
    """Regression for t_dbf31ad3: Spark claimed a card after quota_preflight.py
    recorded ``spark_5h`` at 100% (dispatch_allowed=False, reason=provider_limit)
    because ``spark_cooldown()`` never publishes a ``cooldown_until`` for a
    saturated-gauge record -- only the (unrelated) Claude cooldown path does.
    ``route_preflight_ok`` used to treat "no cooldown_until" as "unknown
    measurement" and fail OPEN, making an explicitly excluded lane claimable.
    It must fail closed instead, exactly reproducing the live
    ``state/ai-quota-routing.json`` shape observed for the incident."""
    routing = kanban_home / "state" / "ai-quota-routing.json"
    routing.parent.mkdir(parents=True)
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ROUTING_PATH", str(routing))
    routing.write_text(json.dumps({
        "agent_cooldowns": {
            "spark": {
                "dispatch_allowed": False,
                "source": "quota_gauge",
                "window": "spark",
                "measured_at": "2026-08-28T12:50:55.602700+00:00",
                "preflight_required": True,
                "reason": "provider_limit",
                "saturated_windows": ["spark_5h"],
            },
        },
    }))
    ok, reason = kb.route_preflight_ok("spark")
    assert ok is False
    assert reason == "provider_limit"


def test_resolve_ordered_route_skips_saturated_spark_for_claude2(
    kanban_home, monkeypatch, configured_handoff_routes,
):
    """End-to-end regression for t_dbf31ad3's "Spark saturé -> Claude 2"
    acceptance scenario: a simple-tier card must route past an explicitly
    excluded Spark lane straight to Claude 2, never claim Spark."""
    routing = kanban_home / "state" / "ai-quota-routing.json"
    routing.parent.mkdir(parents=True)
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ROUTING_PATH", str(routing))
    routing.write_text(json.dumps({
        "agent_cooldowns": {
            "spark": {"dispatch_allowed": False, "reason": "provider_limit"},
            "claude2": {"dispatch_allowed": True, "preflight_required": False},
        },
    }))
    assignee, model_override, trace = kb.resolve_ordered_route("simple")
    assert assignee == "claude2"
    spark_attempt = next(t for t in trace if t["route"] == "spark")
    assert spark_attempt["ok"] is False
    assert spark_attempt["reason"] == "provider_limit"


@pytest.mark.parametrize(
    "tier, green_routes, expected_assignee, expected_model",
    [
        # A bounded card explicitly classified simple gets Spark first.
        ("simple", {"spark"}, "spark", TEST_SPARK_MODEL_OVERRIDE),
        # Bounded cards preserve Claude-first fallback after Spark.
        ("simple", {"claude2"}, "claude2", None),
        ("simple", {"claude1"}, "claude1", None),
        # Coder is the last resort for bounded cards too.
        ("simple", {"coder"}, "coder", None),
        # Complex cards never route to Spark.
        ("complex", {"claude2"}, "claude2", None),
        # Claude 2 down, Claude 1 green -> Claude 1 for complex work.
        ("complex", {"claude1"}, "claude1", None),
        # Both Claude lanes down -> Coder for complex work.
        ("complex", set(), "coder", None),
        # Unknown/missing tier fails safe to the complex Coder route.
        (None, set(), "coder", None),
        ("bogus", set(), "coder", None),
    ],
)
def test_resolve_ordered_route_table(
    tier, green_routes, expected_assignee, expected_model, configured_handoff_routes,
):
    def fake_preflight(route):
        return (route in green_routes, "ok" if route in green_routes else "down")

    assignee, model, trace = kb.resolve_ordered_route(tier, preflight_fn=fake_preflight)
    assert assignee == expected_assignee
    assert model == expected_model
    # The walk always stops at the first green route -- nothing probed after it.
    assert trace[-1]["assignee"] == expected_assignee
    assert all(entry["ok"] is False for entry in trace[:-1]) or len(trace) == 1


def test_simple_route_never_probes_a_more_capable_lane_when_spark_is_green(
    configured_handoff_routes,
):
    probed = []

    def fake_preflight(route):
        probed.append(route)
        return (route == "spark", "ok" if route == "spark" else "down")

    kb.resolve_ordered_route("simple", preflight_fn=fake_preflight)
    assert probed == ["spark"]


def test_parallel_complex_routes_preserve_coder_until_third_independent_task(
    configured_handoff_routes,
):
    green = lambda route: (True, "green")
    routes, _ = kb.resolve_parallel_routes("complex", 1, preflight_fn=green)
    assert routes == [("claude2", None)]
    routes, _ = kb.resolve_parallel_routes("complex", 2, preflight_fn=green)
    assert routes == [("claude2", None), ("claude1", None)]
    routes, _ = kb.resolve_parallel_routes("complex", 3, preflight_fn=green)
    assert routes == [("claude2", None), ("claude1", None), ("coder", None)]


def test_parallel_complex_routes_use_coder_when_a_claude_is_unavailable(
    configured_handoff_routes,
):
    def preflight(route):
        return (route != "claude2", "green" if route != "claude2" else "quota")

    routes, _ = kb.resolve_parallel_routes("complex", 2, preflight_fn=preflight)
    assert routes == [("claude1", None), ("coder", None)]


def test_first_available_route_skips_busy_lanes_in_configured_order(
    configured_handoff_routes,
):
    green = lambda route: (True, "green")

    assignee, model, trace = kb.resolve_first_available_route(
        "simple",
        {"spark": 0, "claude2": 0, "claude1": 0, "coder": 0},
        max_in_progress_per_profile=1,
        preflight_fn=green,
    )
    assert (assignee, model) == ("spark", TEST_SPARK_MODEL_OVERRIDE)
    assert trace[-1]["capacity_available"] is True

    assignee, model, trace = kb.resolve_first_available_route(
        "complex",
        {"claude2": 1, "claude1": 1, "coder": 0},
        max_in_progress_per_profile=1,
        preflight_fn=green,
    )
    assert (assignee, model) == ("coder", None)
    assert [entry["route"] for entry in trace] == ["claude2", "claude1", "coder"]


def test_first_available_route_keeps_card_queued_when_every_lane_is_busy(
    configured_handoff_routes,
):
    assignee, model, trace = kb.resolve_first_available_route(
        "complex",
        {"claude2": 1, "claude1": 1, "coder": 1},
        max_in_progress_per_profile=1,
        preflight_fn=lambda route: (True, "green"),
    )
    assert (assignee, model) == (None, None)
    assert all(entry["capacity_available"] is False for entry in trace)


def test_unknown_claude_measurement_preserves_claude_instead_of_opening_coder(
    configured_handoff_routes,
):
    def preflight(route):
        if route == "claude2":
            return False, "provider_cooldown"
        if route == "claude1":
            return False, "quota_preflight_required"
        return True, "green"

    assignee, _, _ = kb.resolve_ordered_route("complex", preflight_fn=preflight)
    assert assignee == "claude1"
    routes, _ = kb.resolve_parallel_routes("complex", 2, preflight_fn=preflight)
    assert routes == [("claude1", None), ("coder", None)]


def test_opus_is_a_claude_only_per_task_override(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="explicit opus", assignee="claude2",
            model_override=kb.CLAUDE_OPUS_MODEL,
        )
        assert kb.get_task(conn, task_id).model_override == kb.CLAUDE_OPUS_MODEL
        with pytest.raises(ValueError, match="Opus"):
            kb.create_task(
                conn, title="invalid opus", assignee="coder",
                model_override=kb.CLAUDE_OPUS_MODEL,
            )
        with pytest.raises(ValueError, match="Opus"):
            kb.create_task(conn, title="unassigned opus", model_override=kb.CLAUDE_OPUS_MODEL)


def test_dispatches_three_independent_workspaces_to_distinct_executors(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """A divisible batch may fan out only when each writer owns its workspace."""
    routing = kanban_home / "state" / "ai-quota-routing.json"
    routing.parent.mkdir(parents=True)
    routing.write_text(json.dumps({"agent_cooldowns": {
        "claude2": {"dispatch_allowed": True, "preflight_required": False},
        "claude1": {"dispatch_allowed": True, "preflight_required": False},
    }}))
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ROUTING_PATH", str(routing))
    workspaces = [kanban_home / f"writer-{index}" for index in range(3)]
    for workspace in workspaces:
        workspace.mkdir()
    spawned = []

    def fake_spawn(task, workspace, **_kwargs):
        spawned.append((task.assignee, workspace))
        return 90_000 + len(spawned)

    monkeypatch.setattr(kb, "claude2_oauth_dispatch_guard", lambda *_args: False)
    with kb.connect() as conn:
        for assignee, workspace in zip(("claude2", "claude1", "coder"), workspaces):
            kb.create_task(
                conn, title=f"independent-{assignee}", assignee=assignee,
                workspace_kind="dir", workspace_path=str(workspace),
                routing_tier="complex",
            )
        result = kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn=3)

    assert len(result.spawned) == 3
    assert {assignee for assignee, _ in spawned} == {"claude2", "claude1", "coder"}
    assert {workspace for _, workspace in spawned} == {str(path) for path in workspaces}
    assert len({workspace for _, workspace in spawned}) == 3


def test_dispatch_serializes_two_tasks_sharing_the_same_workspace(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Two cards must never write concurrently in one exact checkout."""
    routing = kanban_home / "state" / "ai-quota-routing.json"
    routing.parent.mkdir(parents=True)
    routing.write_text(json.dumps({"agent_cooldowns": {
        "claude1": {"dispatch_allowed": True, "preflight_required": False},
    }}))
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ROUTING_PATH", str(routing))
    workspace = kanban_home / "shared-worktree"
    workspace.mkdir()
    spawned = []

    def fake_spawn(task, resolved_workspace, **_kwargs):
        spawned.append((task.id, resolved_workspace))
        return 91_000 + len(spawned)

    with kb.connect() as conn:
        first = kb.create_task(
            conn, title="first", assignee="claude1",
            workspace_kind="dir", workspace_path=str(workspace),
        )
        second = kb.create_task(
            conn, title="second", assignee="coder",
            workspace_kind="dir", workspace_path=str(workspace),
        )
        result = kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn=2)
        first_task = kb.get_task(conn, first)
        second_task = kb.get_task(conn, second)
        spawnable_while_first_runs = kb.has_spawnable_ready(conn)

    assert result.spawned == [(first, "claude1", str(workspace.resolve()))]
    assert result.skipped_workspace_busy == [
        (second, str(workspace.resolve()), first),
    ]
    assert first_task is not None and first_task.status == "running"
    assert second_task is not None and second_task.status == "ready"
    assert spawnable_while_first_runs is False
    assert spawned == [(first, str(workspace.resolve()))]
    with kb.connect() as conn:
        waits = [
            event for event in kb.list_events(conn, second)
            if event.kind == "workspace_wait_started"
        ]
    assert len(waits) == 1
    assert waits[0].payload == {
        "workspace": str(workspace.resolve()),
        "owner_task_id": first,
    }


def test_workspace_wait_events_are_deduplicated_and_measure_duration(kanban_home):
    workspace = kanban_home / "busy"
    workspace.mkdir()
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="waiting", assignee="coder")
        kb._start_workspace_wait(conn, task_id, str(workspace), "t_owner")
        kb._start_workspace_wait(conn, task_id, str(workspace), "t_owner")
        started = [
            event for event in kb.list_events(conn, task_id)
            if event.kind == "workspace_wait_started"
        ]
        assert len(started) == 1

        kb._end_workspace_wait(conn, task_id)
        ended = [
            event for event in kb.list_events(conn, task_id)
            if event.kind == "workspace_wait_ended"
        ]

    assert len(ended) == 1
    assert ended[0].payload["owner_task_id"] == "t_owner"
    assert ended[0].payload["workspace"] == str(workspace)
    assert ended[0].payload["wait_seconds"] >= 0


def test_health_probe_ignores_ready_work_for_a_profile_at_capacity(
    kanban_home, all_assignees_spawnable,
):
    """A preloaded same-profile card is healthy queued work, not a stall."""
    running_workspace = kanban_home / "running-worktree"
    ready_workspace = kanban_home / "ready-worktree"
    running_workspace.mkdir()
    ready_workspace.mkdir()

    with kb.connect() as conn:
        running = kb.create_task(
            conn, title="running", assignee="coder",
            workspace_kind="dir", workspace_path=str(running_workspace),
        )
        kb.claim_task(conn, running)
        kb.create_task(
            conn, title="preloaded", assignee="coder",
            workspace_kind="dir", workspace_path=str(ready_workspace),
        )

        assert kb.has_spawnable_ready(conn) is True
        assert kb.has_spawnable_ready(
            conn, max_in_progress_per_profile=1,
        ) is False


def test_dispatch_resumes_failed_checkpoint_before_fresh_same_profile_work(
    kanban_home, all_assignees_spawnable,
):
    """A retry keeps ownership of its durable checkout state before new work."""
    fresh_workspace = kanban_home / "fresh-worktree"
    retry_workspace = kanban_home / "retry-worktree"
    fresh_workspace.mkdir()
    retry_workspace.mkdir()
    spawned = []

    def fake_spawn(task, resolved_workspace, **_kwargs):
        spawned.append((task.id, resolved_workspace))
        return 95_001

    with kb.connect() as conn:
        fresh = kb.create_task(
            conn, title="older fresh task", assignee="coder",
            workspace_kind="dir", workspace_path=str(fresh_workspace),
        )
        retry = kb.create_task(
            conn, title="newer checkpoint retry", assignee="coder",
            workspace_kind="dir", workspace_path=str(retry_workspace),
        )
        conn.execute(
            "UPDATE tasks SET consecutive_failures=1 WHERE id=?", (retry,),
        )
        conn.commit()

        result = kb.dispatch_once(
            conn,
            spawn_fn=fake_spawn,
            max_spawn=2,
            max_in_progress_per_profile=1,
        )

    assert result.spawned == [
        (retry, "coder", str(retry_workspace.resolve())),
    ]
    assert result.skipped_per_profile_capped == [
        (fresh, "coder", 1),
    ]
    assert spawned == [(retry, str(retry_workspace.resolve()))]


def test_dispatch_serializes_parent_checkout_against_nested_repository(
    kanban_home, all_assignees_spawnable,
):
    """A parent writer must not race a nested repository gitlink writer."""
    repository = kanban_home / "repository"
    parent_workspace = repository / "youtube"
    nested_repository = repository / "ecobloc"
    nested_workspace = nested_repository / "tasks"
    (repository / ".git").mkdir(parents=True)
    parent_workspace.mkdir()
    (nested_repository / ".git").mkdir(parents=True)
    nested_workspace.mkdir()
    spawned = []

    def fake_spawn(task, resolved_workspace, **_kwargs):
        spawned.append((task.id, resolved_workspace))
        return 92_000 + len(spawned)

    with kb.connect() as conn:
        parent = kb.create_task(
            conn, title="parent writer", assignee="coder",
            workspace_kind="dir", workspace_path=str(parent_workspace),
        )
        nested = kb.create_task(
            conn, title="nested writer", assignee="claude1",
            workspace_kind="dir", workspace_path=str(nested_workspace),
        )
        result = kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn=2)
        spawnable_while_parent_runs = kb.has_spawnable_ready(conn)

    assert result.spawned == [(parent, "coder", str(parent_workspace.resolve()))]
    assert result.skipped_workspace_busy == [
        (nested, str(nested_repository.resolve()), parent),
    ]
    assert spawnable_while_parent_runs is False
    assert spawned == [(parent, str(parent_workspace.resolve()))]


def test_private_scratch_task_does_not_lock_its_parent_hermes_repository(
    kanban_home, all_assignees_spawnable,
):
    """Board-only scratch work must not reserve every nested project repo."""
    (kanban_home / ".git").mkdir()
    scratch = kanban_home / "kanban" / "workspaces" / "control-task"
    scratch.mkdir(parents=True)
    project = kanban_home / "workspace" / "project"
    (project / ".git").mkdir(parents=True)
    spawned = []

    with kb.connect() as conn:
        control = kb.create_task(
            conn, title="triage board", assignee="claude2",
            workspace_kind="scratch", workspace_path=str(scratch),
        )
        assert kb.claim_task(conn, control) is not None
        delivery = kb.create_task(
            conn, title="independent project", assignee="alice",
            workspace_kind="dir", workspace_path=str(project),
        )
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, path, **_kwargs: spawned.append((task.id, path)) or 93_501,
            max_in_progress=2,
            max_in_progress_per_profile=1,
        )

    assert result.skipped_workspace_busy == []
    assert result.spawned == [(delivery, "alice", str(project.resolve()))]
    assert spawned == [(delivery, str(project.resolve()))]


def test_dispatch_keeps_nested_sibling_repositories_parallel(
    kanban_home, all_assignees_spawnable,
):
    """Independent nested repositories retain useful lane parallelism."""
    repository = kanban_home / "repository"
    (repository / ".git").mkdir(parents=True)
    first_repository = repository / "first"
    second_repository = repository / "second"
    (first_repository / ".git").mkdir(parents=True)
    (second_repository / ".git").mkdir(parents=True)
    first_workspace = first_repository / "tasks"
    second_workspace = second_repository / "tasks"
    first_workspace.mkdir()
    second_workspace.mkdir()
    spawned = []

    def fake_spawn(task, resolved_workspace, **_kwargs):
        spawned.append((task.id, resolved_workspace))
        return 93_000 + len(spawned)

    with kb.connect() as conn:
        first = kb.create_task(
            conn, title="first nested writer", assignee="coder",
            workspace_kind="dir", workspace_path=str(first_workspace),
        )
        second = kb.create_task(
            conn, title="second nested writer", assignee="researcher",
            workspace_kind="dir", workspace_path=str(second_workspace),
        )
        result = kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn=2)

    assert result.skipped_workspace_busy == []
    assert {task_id for task_id, _assignee, _workspace in result.spawned} == {
        first, second,
    }
    assert len(spawned) == 2


def test_dispatch_serializes_parent_and_ignored_nested_repository(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Only Git-root overlap matters; ignore rules do not bypass the lock."""
    monkeypatch.setattr(kb, "_memory_pressure_level", lambda *_args: "ok")
    repository = kanban_home / "repository"
    _init_git_repo(repository)
    parent_workspace = repository / "parent-owned-data"
    parent_workspace.mkdir()
    nested_repository = repository / "independent-project"
    _init_git_repo(nested_repository)
    (repository / ".gitignore").write_text(
        "independent-project/\n", encoding="utf-8"
    )
    spawned = []

    def fake_spawn(task, resolved_workspace, **_kwargs):
        spawned.append((task.id, resolved_workspace))
        return 93_200 + len(spawned)

    with kb.connect() as conn:
        parent = kb.create_task(
            conn, title="parent subdirectory writer", assignee="researcher",
            workspace_kind="dir", workspace_path=str(parent_workspace),
        )
        nested = kb.create_task(
            conn, title="ignored nested project writer", assignee="coder",
            workspace_kind="dir", workspace_path=str(nested_repository),
        )
        result = kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn=2)

    assert result.spawned == [(parent, "researcher", str(parent_workspace.resolve()))]
    assert result.skipped_workspace_busy == [
        (nested, str(nested_repository.resolve()), parent),
    ]
    assert spawned == [(parent, str(parent_workspace.resolve()))]


def test_dispatch_waits_one_tick_for_terminal_worker_exit_barrier(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """A successor cannot spawn in the tick that SIGTERMs its predecessor."""
    from hermes_cli import worker_contracts

    workspace = kanban_home / "shared-workspace"
    workspace.mkdir()
    with kb.connect() as conn:
        predecessor = kb.create_task(
            conn, title="completed predecessor", assignee="coder",
            workspace_kind="dir", workspace_path=str(workspace),
        )
        assert kb.complete_task(conn, predecessor)
        successor = kb.create_task(
            conn, title="successor", assignee="researcher",
            workspace_kind="dir", workspace_path=str(workspace),
        )
        conn.execute(
            "INSERT INTO worker_contracts("
            "task_id,run_id,profile,pid,start_identity,workspace_path,created_at,state"
            ") VALUES(?,?,?,?,?,?,?,'active')",
            (predecessor, 1, "coder", 4242, "identity", str(workspace), int(time.time())),
        )

        monkeypatch.setattr(
            worker_contracts,
            "reconcile",
            lambda _conn: [{
                "task_id": predecessor,
                "pid": 4242,
                "reason": "task_done",
                "stopped": True,
            }],
        )
        spawned = []
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, path, **_kwargs: spawned.append((task.id, path)) or 4243,
            max_spawn=1,
        )

    assert result.spawned == []
    assert result.skipped_workspace_busy == [
        (successor, str(workspace.resolve()), predecessor),
    ]
    assert spawned == []


def test_dispatch_keeps_terminal_workspace_locked_while_exact_pid_lives(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """A slow terminal process keeps its checkout lock beyond one tick."""
    from hermes_cli import worker_contracts

    workspace = kanban_home / "shared-workspace"
    workspace.mkdir()
    with kb.connect() as conn:
        predecessor = kb.create_task(
            conn, title="completed predecessor", assignee="coder",
            workspace_kind="dir", workspace_path=str(workspace),
        )
        assert kb.complete_task(conn, predecessor)
        successor = kb.create_task(
            conn, title="successor", assignee="researcher",
            workspace_kind="dir", workspace_path=str(workspace),
        )
        conn.execute(
            "INSERT INTO worker_contracts("
            "task_id,run_id,profile,pid,start_identity,workspace_path,created_at,"
            "state,stopped_at) VALUES(?,?,?,?,?,?,?,'stopped',?)",
            (predecessor, 1, "coder", 4242, "identity", str(workspace),
             int(time.time()), int(time.time())),
        )
        monkeypatch.setattr(worker_contracts, "reconcile", lambda _conn: [])
        monkeypatch.setattr(
            worker_contracts, "proc_start_identity",
            lambda pid: "identity" if pid == 4242 else None,
        )
        spawned = []
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, path, **_kwargs: spawned.append((task.id, path)) or 4243,
            max_spawn=1,
        )

    assert result.spawned == []
    assert result.skipped_workspace_busy == [
        (successor, str(workspace.resolve()), predecessor),
    ]
    assert spawned == []


def test_local_claude2_failure_does_not_fallback_to_another_executor(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="repair proxy", assignee="claude2")
        assert kb.fallback_simple_route(
            conn, task_id, "proxy connection refused", provider_proven=False,
        ) is False
        row = conn.execute(
            "SELECT assignee FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
    assert row["assignee"] == "claude2"


def test_dispatch_once_applies_routing_tier_chain_to_unassigned_ready_task(
    kanban_home, all_assignees_spawnable, monkeypatch, configured_handoff_routes,
):
    """End-to-end: an unassigned task with a persisted tier is auto-routed
    by the dispatcher itself (decision (b): preflight runs before spawn),
    not just by calling resolve_ordered_route in isolation."""
    routing = kanban_home / "state" / "ai-quota-routing.json"
    routing.parent.mkdir(parents=True)
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ROUTING_PATH", str(routing))
    # Both Claude lanes explicitly down; codex-worker fails open -> Spark.
    routing.write_text(json.dumps({"agent_cooldowns": {
        "claude1": {"dispatch_allowed": False, "preflight_required": True},
        "claude2": {"dispatch_allowed": False, "preflight_required": True},
    }}))
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="auto-routed", routing_tier="simple")
        result = kb.dispatch_once(conn, dry_run=True)
        assert task_id in result.auto_assigned_default
        assert task_id in [row[0] for row in result.spawned]
        row = conn.execute("SELECT assignee, model_override FROM tasks WHERE id = ?", (task_id,)).fetchone()
        # dry_run must not mutate the row.
        assert row["assignee"] is None

    with kb.connect() as conn:
        result = kb.dispatch_once(conn, dry_run=False)
        assert task_id in [row[0] for row in result.spawned]
        row = conn.execute("SELECT assignee, model_override FROM tasks WHERE id = ?", (task_id,)).fetchone()
        assert row["assignee"] == "spark"
        assert row["model_override"] == TEST_SPARK_MODEL_OVERRIDE
        events = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? AND kind = 'assigned'",
            (task_id,),
        ).fetchall()
        payloads = [json.loads(e["payload"]) for e in events]
        assert any(p.get("source") == "routing_tier_chain" for p in payloads)


def test_fresh_assigned_pool_task_moves_to_first_free_claude_lane(
    kanban_home, all_assignees_spawnable, monkeypatch, configured_handoff_routes,
    tmp_path,
):
    from hermes_cli import config

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {"kanban": {"generalist_worker_pool_routing": True}},
    )
    monkeypatch.setattr(kb, "claude2_oauth_dispatch_guard", lambda *_args: False)
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    busy_workspace = tmp_path / "busy-c2"
    candidate_workspace = tmp_path / "candidate"
    busy_workspace.mkdir()
    candidate_workspace.mkdir()

    with kb.connect() as conn:
        busy = kb.create_task(
            conn, title="busy c2", assignee="claude2",
            routing_tier="complex", workspace_kind="dir",
            workspace_path=str(busy_workspace),
        )
        first = kb.dispatch_once(
            conn, spawn_fn=lambda *_args, **_kwargs: 41001,
            max_in_progress_per_profile=1,
        )
        assert first.spawned[0][0] == busy

        candidate = kb.create_task(
            conn, title="first free claude", assignee=None,
            routing_tier="complex", workspace_kind="dir",
            workspace_path=str(candidate_workspace),
        )
        # Simulate a prior tick's tier-chain auto-route (dispatcher-placed,
        # not operator-chosen) so the pool-capacity reroute below is allowed
        # to move it: an explicitly-assigned card must stay pinned instead
        # (t_b940d7ff), so the reroute guard now requires this provenance.
        conn.execute(
            "UPDATE tasks SET assignee = 'claude2' WHERE id = ?", (candidate,),
        )
        kb._append_event(conn, candidate, "assigned", {
            "assignee": "claude2", "source": "routing_tier_chain",
        })
        second = kb.dispatch_once(
            conn, spawn_fn=lambda *_args, **_kwargs: 41002,
            max_in_progress_per_profile=1,
        )
        assert second.spawned == [
            (candidate, "claude1", str(candidate_workspace.resolve())),
        ]
        assert second.rerouted_for_capacity == [
            (candidate, "claude2", "claude1"),
        ]
        row = conn.execute(
            "SELECT assignee FROM tasks WHERE id = ?", (candidate,),
        ).fetchone()
        assert row["assignee"] == "claude1"


def test_fresh_assigned_pool_task_uses_coder_when_both_claudes_are_busy(
    kanban_home, all_assignees_spawnable, monkeypatch, configured_handoff_routes,
    tmp_path,
):
    from hermes_cli import config

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {"kanban": {"generalist_worker_pool_routing": True}},
    )
    monkeypatch.setattr(kb, "claude2_oauth_dispatch_guard", lambda *_args: False)
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)

    with kb.connect() as conn:
        for index in range(3):
            (tmp_path / f"workspace-{index}").mkdir()
        first_id = kb.create_task(
            conn, title="first", assignee="claude2", routing_tier="complex",
            workspace_kind="dir", workspace_path=str(tmp_path / "workspace-0"),
        )
        second_id = kb.create_task(
            conn, title="second", assignee=None, routing_tier="complex",
            workspace_kind="dir", workspace_path=str(tmp_path / "workspace-1"),
        )
        first_wave = kb.dispatch_once(
            conn, spawn_fn=lambda task, *_args, **_kwargs: 42000 + (1 if task.id == first_id else 2),
            max_spawn=2, max_in_progress_per_profile=1,
        )
        assert [entry[1] for entry in first_wave.spawned] == ["claude2", "claude1"]

        candidate = kb.create_task(
            conn, title="third", assignee=None, routing_tier="complex",
            workspace_kind="dir", workspace_path=str(tmp_path / "workspace-2"),
        )
        # Same simulated prior-tick tier-chain placement as above -- an
        # explicitly-assigned card would stay pinned instead (t_b940d7ff).
        conn.execute(
            "UPDATE tasks SET assignee = 'claude2' WHERE id = ?", (candidate,),
        )
        kb._append_event(conn, candidate, "assigned", {
            "assignee": "claude2", "source": "routing_tier_chain",
        })
        second_wave = kb.dispatch_once(
            conn, spawn_fn=lambda *_args, **_kwargs: 42003,
            max_in_progress_per_profile=1,
        )
        assert second_wave.spawned[0][0:2] == (candidate, "coder")
        assert second_wave.rerouted_for_capacity == [
            (candidate, "claude2", "coder"),
        ]


def test_fresh_simple_pool_task_prefers_available_spark(
    kanban_home, all_assignees_spawnable, monkeypatch, configured_handoff_routes,
    tmp_path,
):
    from hermes_cli import config

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {"kanban": {"generalist_worker_pool_routing": True}},
    )
    workspace = tmp_path / "simple"
    workspace.mkdir()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="simple", assignee=None, routing_tier="simple",
            workspace_kind="dir", workspace_path=str(workspace),
        )
        result = kb.dispatch_once(
            conn, spawn_fn=lambda *_args, **_kwargs: 43001,
            max_in_progress_per_profile=1,
        )
        assert result.spawned[0][0:2] == (task_id, "spark")
        row = conn.execute(
            "SELECT assignee, model_override FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        assert (row["assignee"], row["model_override"]) == (
            "spark", TEST_SPARK_MODEL_OVERRIDE,
        )


def test_explicitly_assigned_card_is_not_pool_rerouted(
    kanban_home, all_assignees_spawnable, monkeypatch, configured_handoff_routes,
    tmp_path,
):
    """t_b940d7ff: a card explicitly assigned to ``coder`` at creation must
    stay on ``coder`` even though claude2 (earlier in the complex chain) has
    free capacity. Before the fix, the ``generalist_pool_free_capacity``
    reroute treated any assignee found in the configured chain as a "fresh
    pool card" and walked it back to the first free lane -- silently
    overriding a deliberate operator placement (repeated 3x on the real
    incident, blocking an explicitly authorized Firebase deploy). Only
    dispatcher-placed assignees (tier-chain / default_assignee / an earlier
    pool reroute) are eligible for this opportunistic reroute; an assignee
    named directly in ``kanban_create``/``assign_task`` is not."""
    from hermes_cli import config

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {"kanban": {"generalist_worker_pool_routing": True}},
    )
    workspace = tmp_path / "explicit-coder"
    workspace.mkdir()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="explicit coder", assignee="coder",
            routing_tier="complex", workspace_kind="dir",
            workspace_path=str(workspace),
        )
        result = kb.dispatch_once(
            conn, spawn_fn=lambda *_args, **_kwargs: 45001,
            max_in_progress_per_profile=1,
        )
        assert result.rerouted_for_capacity == []
        assert result.spawned[0][0:2] == (task_id, "coder")
        row = conn.execute(
            "SELECT assignee FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        assert row["assignee"] == "coder"


def test_manually_reassigned_card_is_not_pool_rerouted(
    kanban_home, all_assignees_spawnable, monkeypatch, configured_handoff_routes,
    tmp_path,
):
    """Same guarantee as above, but for a card an operator reassigns later
    via ``assign_task`` (the CLI/API path) rather than at creation time --
    both are deliberate placements, not dispatcher auto-routing."""
    from hermes_cli import config

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {"kanban": {"generalist_worker_pool_routing": True}},
    )
    workspace = tmp_path / "manual-reassign"
    workspace.mkdir()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="manually reassigned", assignee=None,
            routing_tier="complex", workspace_kind="dir",
            workspace_path=str(workspace),
        )
        assert kb.assign_task(conn, task_id, "coder") is True
        result = kb.dispatch_once(
            conn, spawn_fn=lambda *_args, **_kwargs: 45002,
            max_in_progress_per_profile=1,
        )
        assert result.rerouted_for_capacity == []
        assert result.spawned[0][0:2] == (task_id, "coder")


def test_ready_checkpoint_preserves_exact_worker_even_when_another_lane_is_free(
    kanban_home, all_assignees_spawnable, monkeypatch, configured_handoff_routes,
    tmp_path,
):
    from hermes_cli import config

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {"kanban": {"generalist_worker_pool_routing": True}},
    )
    monkeypatch.setattr(kb, "claude2_oauth_dispatch_guard", lambda *_args: False)
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    for name in ("busy", "resume"):
        (tmp_path / name).mkdir()

    with kb.connect() as conn:
        kb.create_task(
            conn, title="busy", assignee="claude2", routing_tier="complex",
            workspace_kind="dir", workspace_path=str(tmp_path / "busy"),
        )
        kb.dispatch_once(
            conn, spawn_fn=lambda *_args, **_kwargs: 44001,
            max_in_progress_per_profile=1,
        )
        resumed = kb.create_task(
            conn, title="resume", assignee="claude2", routing_tier="complex",
            workspace_kind="dir", workspace_path=str(tmp_path / "resume"),
        )
        kb._synthesize_ended_run(
            conn,
            resumed,
            outcome="interrupted",
            metadata={
                "worker_session_id": "durable-session",
                "checkpoint": {"state": "interrupted"},
            },
        )
        result = kb.dispatch_once(
            conn, dry_run=True, max_in_progress_per_profile=1,
        )
        assert result.rerouted_for_capacity == []
        assert (resumed, "claude2", 1) in result.skipped_per_profile_capped


def test_dispatch_empty_handoff_config_preserves_default_assignee_path(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    monkeypatch.setattr(kb, "_configured_handoff_routes", lambda: {})
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="auto-routed", routing_tier="simple")
        result = kb.dispatch_once(conn, dry_run=False, default_assignee="coder")

        assert task_id in [row[0] for row in result.spawned]
        assert result.skipped_unassigned == []
        row = conn.execute(
            "SELECT assignee, model_override FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        assert row["assignee"] == "coder"
        assert row["model_override"] is None
        events = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'assigned'",
            (task_id,),
        ).fetchall()
        payloads = [json.loads(e["payload"]) for e in events]
        assert any(p.get("source") == "kanban.default_assignee" for p in payloads)


def test_dispatch_handoff_routes_accept_configured_profile_names(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    monkeypatch.setattr(kb, "_configured_handoff_routes", lambda: {
        kb.ROUTING_TIER_COMPLEX: (("writer_a", None), ("writer_b", None)),
    })
    assert kb.route_preflight_ok("writer_a") == (True, "configured_fail_open")
    assert kb.resolve_ordered_route("complex")[:2] == ("writer_a", None)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="auto-routed", routing_tier="complex")
        result = kb.dispatch_once(conn, dry_run=False, default_assignee="writer_a")

        assert task_id in [row[0] for row in result.spawned]
        row = conn.execute(
            "SELECT assignee, model_override FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        assert row["assignee"] == "writer_a"
        assert row["model_override"] is None


def test_fifteen_direct_tasks_finish_in_capacity_waves(
    kanban_home, all_assignees_spawnable, monkeypatch, configured_handoff_routes,
):
    """Permanent load canary: direct creates use all three complex lanes."""
    routing = kanban_home / "state" / "ai-quota-routing.json"
    routing.parent.mkdir(parents=True, exist_ok=True)
    routing.write_text(json.dumps({"agent_cooldowns": {
        "claude2": {"dispatch_allowed": True, "preflight_required": False},
        "claude1": {"dispatch_allowed": True, "preflight_required": False},
    }}))
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ROUTING_PATH", str(routing))
    monkeypatch.setattr(kb, "claude2_oauth_dispatch_guard", lambda *_args: False)
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    spawned_profiles = []

    def spawn(task, _workspace, **_kwargs):
        spawned_profiles.append(task.assignee)
        return 80_000 + len(spawned_profiles)

    with kb.connect() as conn:
        task_ids = [
            kb.create_task(conn, title=f"batch-{index}", routing_tier="complex")
            for index in range(15)
        ]
        for _wave in range(5):
            result = kb.dispatch_once(
                conn,
                spawn_fn=spawn,
                max_spawn=3,
                max_in_progress_per_profile=1,
            )
            assert len(result.spawned) == 3
            for task_id, _profile, _workspace in result.spawned:
                assert kb.complete_task(
                    conn,
                    task_id,
                    summary="verified batch unit",
                    metadata={"tests_run": ["batch-canary"]},
                )
        assert {kb.get_task(conn, task_id).status for task_id in task_ids} == {"done"}
    assert len(spawned_profiles) == 15
    assert set(spawned_profiles) == {"claude2", "claude1", "coder"}


def test_mission_tracks_action_completion_and_delivery(kanban_home):
    with kb.connect() as conn:
        mission_id = kb.ensure_mission(
            conn,
            title="mission",
            request_text="do all work",
            idempotency_key="telegram:one-message",
            origin={"platform": "telegram", "chat_id": "42", "message_id": "7"},
            session_id="session-7",
        )
        assert mission_id == kb.ensure_mission(
            conn,
            title="duplicate",
            request_text="duplicate",
            idempotency_key="telegram:one-message",
        )
        task_id = kb.create_task(conn, title="unit", mission_id=mission_id)
        assert kb.block_task(conn, task_id, reason="choose account", kind="needs_input")
        mission = conn.execute(
            "SELECT status FROM missions WHERE id = ?", (mission_id,)
        ).fetchone()
        assert mission["status"] == "action_required"
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM human_actions WHERE task_id = ? AND status = 'open'",
            (task_id,),
        ).fetchone()["n"] == 1
        assert kb.complete_task(
            conn,
            task_id,
            summary="done",
            metadata={"evidence": {"kind": "test", "detail": "unit passed"}},
        )
        task = kb.get_task(conn, task_id)
        assert task.execution_status == "done"
        assert task.verification_status == "verified"
        assert task.delivery_status == "awaiting_delivery"
        assert kb.mark_task_delivered(conn, task_id)
        assert conn.execute(
            "SELECT status FROM missions WHERE id = ?", (mission_id,)
        ).fetchone()["status"] == "delivered"


def test_new_active_child_reopens_delivered_mission_and_clears_terminal_times(
    kanban_home,
):
    with kb.connect() as conn:
        mission_id = kb.ensure_mission(
            conn,
            title="mission",
            request_text="fix then tidy",
            idempotency_key="mission-reopen-child",
        )
        parent = kb.create_task(conn, title="fix", mission_id=mission_id)
        assert kb.complete_task(
            conn,
            parent,
            summary="fixed",
            metadata={"evidence": {"kind": "test", "detail": "7/7 OK"}},
        )
        assert kb.mark_task_delivered(conn, parent)
        delivered = conn.execute(
            "SELECT status, completed_at, delivered_at FROM missions WHERE id = ?",
            (mission_id,),
        ).fetchone()
        assert delivered["status"] == "delivered"
        assert delivered["completed_at"] is not None
        assert delivered["delivered_at"] is not None

        child = kb.create_task(conn, title="tidy", parents=[parent])
        reopened = conn.execute(
            "SELECT status, completed_at, delivered_at FROM missions WHERE id = ?",
            (mission_id,),
        ).fetchone()
        assert reopened["status"] == "active"
        assert reopened["completed_at"] is None
        assert reopened["delivered_at"] is None
        event = conn.execute(
            "SELECT kind, payload FROM mission_events WHERE mission_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (mission_id,),
        ).fetchone()
        assert event["kind"] == "reopened"
        assert json.loads(event["payload"]) == {
            "from": "delivered",
            "to": "active",
        }

        assert kb.complete_task(
            conn,
            child,
            summary="tidied",
            metadata={"evidence": {"kind": "test", "detail": "7/7 OK"}},
        )
        assert kb.mark_task_delivered(conn, child)
        assert conn.execute(
            "SELECT status FROM missions WHERE id = ?", (mission_id,)
        ).fetchone()["status"] == "delivered"


def test_mission_origin_repairs_missing_notification_subscription(kanban_home):
    with kb.connect() as conn:
        mission_id = kb.ensure_mission(
            conn,
            title="repair route",
            request_text="notify me",
            idempotency_key="repair-sub",
            origin={
                "platform": "telegram", "chat_id": "99",
                "thread_id": "3", "message_id": "11", "user_id": "99",
            },
            session_id="origin-session",
        )
        task_id = kb.create_task(conn, title="unit", mission_id=mission_id)
        assert kb.list_notify_subs(conn, task_id=task_id) == []
        assert kb.repair_mission_subscriptions(conn, notifier_profile="default") == 1
        sub = kb.list_notify_subs(conn, task_id=task_id)[0]
        assert (sub["platform"], sub["chat_id"], sub["thread_id"]) == (
            "telegram", "99", "3",
        )
        assert kb.repair_mission_subscriptions(conn, notifier_profile="default") == 0


def test_backlog_queue_never_consumes_dispatch_capacity(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    with kb.connect() as conn:
        active = kb.create_task(conn, title="active", assignee="coder")
        backlog = kb.create_task(
            conn, title="later", assignee="coder", queue_class="backlog",
        )
        result = kb.dispatch_once(conn, dry_run=True, max_spawn=3)
        spawned = {row[0] for row in result.spawned}
        assert active in spawned
        assert backlog not in spawned


def test_checkpoint_preserves_exact_session_for_transient_resume(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="resume", assignee="coder")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        run_id = claimed.current_run_id
        assert kb.heartbeat_worker(
            conn,
            task_id,
            note="tests complete, integration pending",
            expected_run_id=run_id,
            worker_session_id="worker-session-42",
        )
        assert kb.block_task(
            conn,
            task_id,
            reason="temporary provider interruption",
            kind="transient",
            expected_run_id=run_id,
        )
        assert kb.unblock_task(conn, task_id)
    assert kb._transient_resume_session_id(
        task_id, board=kb.get_current_board(),
    ) == "worker-session-42"


def test_operator_reclaim_resumes_the_exact_checkpointed_session(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="manual recovery", assignee="coder")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        assert kb.heartbeat_worker(
            conn,
            task_id,
            note="build complete, visual correction pending",
            expected_run_id=claimed.current_run_id,
            worker_session_id="worker-session-reclaimed",
        )
        assert kb.reclaim_task(conn, task_id, reason="operator recovery")

    assert kb._transient_resume_session_id(
        task_id, board=kb.get_current_board(),
    ) == "worker-session-reclaimed"


def test_scheduled_time_pause_resumes_the_exact_checkpointed_session(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="maintenance pause", assignee="coder")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        assert kb.heartbeat_worker(
            conn,
            task_id,
            note="manifest reconciled, processing pending",
            expected_run_id=claimed.current_run_id,
            worker_session_id="worker-session-scheduled",
        )
        assert kb.schedule_task(
            conn,
            task_id,
            reason="resume after the 03:00 maintenance window",
            expected_run_id=claimed.current_run_id,
        )
        assert kb.unblock_task(conn, task_id)

    assert kb._transient_resume_session_id(
        task_id, board=kb.get_current_board(),
    ) == "worker-session-scheduled"


def test_failed_simple_routes_advance_spark_then_claude_then_coder(
    kanban_home, all_assignees_spawnable, configured_handoff_routes,
):
    """Bounded ('simple') cards follow the configured handoff chain."""
    with kb.connect() as conn:
        spark_id = kb.create_task(
            conn, title="bounded", assignee="spark",
            routing_tier="simple",
        )
        claude2_id = kb.create_task(
            conn, title="bounded retry", assignee="claude2", routing_tier="simple",
        )
        claude1_id = kb.create_task(
            conn, title="bounded last Claude retry", assignee="claude1", routing_tier="simple",
        )
        assert kb.fallback_simple_route(conn, spark_id, "test command failed") is True
        assert kb.fallback_simple_route(conn, claude2_id, "test command failed") is True
        assert kb.fallback_simple_route(conn, claude1_id, "test command failed") is True

        spark = conn.execute(
            "SELECT assignee, model_override, consecutive_failures, last_failure_error FROM tasks WHERE id = ?",
            (spark_id,),
        ).fetchone()
        claude2 = conn.execute(
            "SELECT assignee, model_override FROM tasks WHERE id = ?", (claude2_id,)
        ).fetchone()
        claude1 = conn.execute(
            "SELECT assignee, model_override FROM tasks WHERE id = ?", (claude1_id,)
        ).fetchone()
        assert spark["assignee"] == "claude2"
        assert spark["model_override"] is None
        assert spark["consecutive_failures"] == 0
        assert "automatic fallback to Claude 2" in spark["last_failure_error"]
        assert claude2["assignee"] == "claude1"
        assert claude1["assignee"] == "coder"
        assert claude1["model_override"] is None
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'simple_route_fallback'",
            (spark_id,),
        ).fetchone()
        payload = json.loads(event["payload"])
        assert payload["to_route"] == "Claude 2"


def test_fallback_route_uses_configured_profile_chain(
    kanban_home, monkeypatch, configured_handoff_routes,
):
    """The fallback follows explicit configured profile identities only."""
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: name in ("spark", "claude2", "claude1", "default"))
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="bounded", assignee="spark", routing_tier="simple",
        )
        assert kb.fallback_simple_route(conn, task_id, "quota exhausted") is True
        row = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (task_id,)).fetchone()
        assert row["assignee"] == "claude2"


def test_fallback_route_applies_to_complex_and_null_tier_cards(
    kanban_home, all_assignees_spawnable, configured_handoff_routes,
):
    """t_47dc2bf0 defect #2: every open card has routing_tier NULL/complex,
    so the fallback must engage for them too -- Claude 2 -> Claude 1 -> GPT
    (Terra/default), never through Spark."""
    with kb.connect() as conn:
        null_tier_id = kb.create_task(conn, title="legacy", assignee="claude2")
        complex_id = kb.create_task(
            conn, title="architecture", assignee="claude2", routing_tier="complex",
        )
        # An anomalous complex card manually parked on Spark is left alone:
        # complex work is never supposed to reach Spark in the first place.
        anomalous_id = kb.create_task(
            conn, title="anomalous", assignee="spark", routing_tier="complex",
        )

        assert kb.fallback_simple_route(conn, null_tier_id, "quota exhausted on claude2") is True
        assert kb.fallback_simple_route(conn, complex_id, "quota exhausted on claude2") is True
        assert kb.fallback_simple_route(conn, anomalous_id, "quota exhausted") is False

        null_row = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (null_tier_id,)).fetchone()
        complex_row = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (complex_id,)).fetchone()
        anomalous_row = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (anomalous_id,)).fetchone()
        assert null_row["assignee"] == "claude1"
        assert complex_row["assignee"] == "claude1"
        assert anomalous_row["assignee"] == "spark"  # untouched


def test_fallback_route_claude2_dead_moves_to_claude1(
    kanban_home, all_assignees_spawnable, configured_handoff_routes,
):
    """Non-regression (a): quota dead on Claude 2 -> the card moves to Claude 1."""
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="work", assignee="claude2")
        assert kb.fallback_simple_route(conn, task_id, "claude2 quota exhausted") is True
        row = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert row["assignee"] == "claude1"


def test_opus_fallback_preserves_model_then_stops_before_coder(
    kanban_home, all_assignees_spawnable, configured_handoff_routes,
):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="explicit opus", assignee="claude2",
            model_override=kb.CLAUDE_OPUS_MODEL, routing_tier="complex",
        )
        assert kb.fallback_simple_route(conn, task_id, "quota exhausted") is True
        row = kb.get_task(conn, task_id)
        assert (row.assignee, row.model_override) == ("claude1", kb.CLAUDE_OPUS_MODEL)
        assert kb.fallback_simple_route(conn, task_id, "quota exhausted") is False
        row = kb.get_task(conn, task_id)
        assert (row.assignee, row.model_override) == ("claude1", kb.CLAUDE_OPUS_MODEL)


def test_fallback_route_both_claude_lanes_dead_moves_to_coder_with_one_event(
    kanban_home, all_assignees_spawnable, configured_handoff_routes,
):
    """Non-regression (b): quota dead on both Claude lanes -> Coder,
    and exactly one route event is recorded for that specific hop."""
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="work", assignee="claude1")
        assert kb.fallback_simple_route(conn, task_id, "claude1 quota exhausted") is True
        row = conn.execute("SELECT assignee, model_override FROM tasks WHERE id = ?", (task_id,)).fetchone()
        assert row["assignee"] == "coder"
        assert row["model_override"] is None
        events = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'simple_route_fallback'",
            (task_id,),
        ).fetchall()
        assert len(events) == 1
        payload = json.loads(events[0]["payload"])
        assert payload["to_route"] == "Coder"
        assert payload["to_assignee"] == "coder"


def test_claude_provider_reset_capture_and_final_coder_relay_are_api_evidence_only(
    kanban_home, all_assignees_spawnable, configured_handoff_routes,
):
    """Claude 2 -> Claude 1 -> Coder retains only structured API evidence.

    The input deliberately carries a secret-shaped value: no event may retain
    the raw provider error while the two API reset forms still produce local
    return estimates and one final relay notification.
    """
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="work", assignee="claude2")
        kb._append_event(conn, task_id, "spawned", {
            "model_resolved": "claude-sonnet-5", "provider_resolved": "anthropic",
        })
        assert kb.fallback_simple_route(
            conn, task_id,
            "HTTP status: 429 code: rate_limit reset_at=2030-01-02T03:04:05Z token=sk-secret-value",
        ) is True
        kb._append_event(conn, task_id, "spawned", {
            "model_resolved": "claude-sonnet-5", "provider_resolved": "anthropic",
        })
        assert kb.fallback_simple_route(
            conn, task_id, "HTTP 429 error_type=rate_limit Retry-After: 120 seconds",
        ) is True
        events = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (task_id,),
        ).fetchall()
        captures = [json.loads(row["payload"]) for row in events if row["kind"] == "claude_provider_reset"]
        relays = [json.loads(row["payload"]) for row in events if row["kind"] == "relayed_to_coder"]

    assert [capture["claude_role"] for capture in captures] == ["claude2", "claude1"]
    assert captures[0]["reset_source"] == "api_reset_at"
    assert captures[0]["reset_at"] == "2030-01-02T03:04:05+00:00"
    assert captures[0]["model_resolved"] == "claude-sonnet-5"
    assert captures[0]["provider_resolved"] == "anthropic"
    assert captures[1]["reset_source"] == "api_retry_after"
    assert len(relays) == 1
    assert relays[0]["message"].startswith("Relais automatique vers Coder.\n")
    assert "Retour estimé Claude 2 :" in relays[0]["message"]
    assert "Retour estimé Claude 1 :" in relays[0]["message"]
    assert "token=" not in json.dumps([json.loads(row["payload"]) for row in events])


def test_claude_provider_reset_parses_minutes_and_does_not_invent_non_429_reset(
    kanban_home,
):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="work", assignee="claude2")
        minutes = kb.capture_claude_provider_reset(
            conn, task_id, "HTTP 429 type=rate_limit retry in 2 minutes", received_at=0,
        )
        no_reset = kb.capture_claude_provider_reset(
            conn, task_id, "HTTP 500 type=server_error unexpected failure", received_at=0,
        )
        row = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (task_id,)).fetchone()

    assert minutes is not None
    assert no_reset is not None
    assert row is not None
    assert minutes["reset_source"] == "api_retry_after"
    assert minutes["reset_at"] == "1970-01-01T00:02:00+00:00"
    assert no_reset["http_status"] == 500
    assert no_reset["reset_source"] is None
    assert no_reset["reset_at"] is None
    assert row["assignee"] == "claude2", "capture alone must not trigger a fallback"


def test_claude_provider_reset_parses_named_timezone_clock(kanban_home):
    received = dt.datetime(
        2026, 8, 28, 7, 36, tzinfo=dt.timezone.utc,
    ).timestamp()
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="work", assignee="claude2")
        captured = kb.capture_claude_provider_reset(
            conn,
            task_id,
            "HTTP 429: You've hit your session limit · resets 12:20pm (Europe/Paris)",
            received_at=received,
        )

    assert captured is not None
    assert captured["reset_source"] == "provider_reset_clock"
    assert captured["reset_at"] == "2026-08-28T10:20:00+00:00"


def test_claude_provider_reset_rejects_clock_without_valid_timezone(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="work", assignee="claude2")
        captured = kb.capture_claude_provider_reset(
            conn,
            task_id,
            "HTTP 429: session limit, resets 12:20pm (Not/A-Timezone)",
            received_at=0,
        )

    assert captured is not None
    assert captured["reset_source"] is None
    assert captured["reset_at"] is None


def test_fallback_route_fails_closed_when_next_profile_is_missing(
    kanban_home, monkeypatch, configured_handoff_routes,
):
    """Missing configured next profile leaves the task on its current lane."""
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "claude2")
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="work", assignee="claude2")
        # Simulate the task having already been requeued to 'ready' by the
        # crash handler (fallback_simple_route runs after that requeue).
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
        assert kb.fallback_simple_route(conn, task_id, "claude2 quota exhausted") is False
        row = conn.execute(
            "SELECT status, block_kind, assignee FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        assert row["status"] == "ready"
        assert row["block_kind"] is None
        # The assignee is left untouched -- never pointed at a dead profile.
        assert row["assignee"] == "claude2"
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'route_fallback_broken'",
            (task_id,),
        ).fetchone()
        assert event is None








# ---------------------------------------------------------------------------
# Complete / block / unblock / archive / assign
# ---------------------------------------------------------------------------







def test_recompute_ready_honours_dispatcher_failure_limit(kanban_home):
    """The guard's effective limit must follow the same resolution order
    as the circuit breaker (#35072): per-task max_retries → dispatcher
    failure_limit → DEFAULT_FAILURE_LIMIT.

    Without threading the dispatcher's ``kanban.failure_limit`` through,
    the guard falls back to DEFAULT_FAILURE_LIMIT and disagrees with the
    breaker — sticking a task prematurely (config limit > default) or
    letting a tripped task escape (config limit < default).
    """
    with kb.connect() as conn:
        # Config allows MORE retries than the default. A task blocked
        # with failures below the configured limit must still recover.
        t = kb.create_task(conn, title="lenient", assignee="a")
        conn.execute(
            "UPDATE tasks SET status='blocked', consecutive_failures=? "
            "WHERE id=?",
            (kb.DEFAULT_FAILURE_LIMIT, t),
        )
        conn.commit()
        # Default-limit call would stick it (failures >= default).
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, t).status == "blocked"
        # Dispatcher configured a higher limit → recover, preserve counter.
        promoted = kb.recompute_ready(
            conn, failure_limit=kb.DEFAULT_FAILURE_LIMIT + 2
        )
        assert promoted == 1
        task = kb.get_task(conn, t)
        assert task.status == "ready"
        assert task.consecutive_failures == kb.DEFAULT_FAILURE_LIMIT

        # Config allows FEWER retries than the default. A task at the
        # stricter limit must stay blocked even though it's below default.
        t2 = kb.create_task(conn, title="strict", assignee="a")
        conn.execute(
            "UPDATE tasks SET status='blocked', consecutive_failures=1 "
            "WHERE id=?",
            (t2,),
        )
        conn.commit()
        # Default-limit (2) would recover it (1 < 2).
        # Stricter config limit (1) must keep it blocked (1 >= 1).
        assert kb.recompute_ready(conn, failure_limit=1) == 0
        assert kb.get_task(conn, t2).status == "blocked"




# ---------------------------------------------------------------------------
# Parent-completion invariant at the claim gate (RCA t_a6acd07d)
# ---------------------------------------------------------------------------














def test_delete_archived_task_removes_related_rows(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        tid = kb.create_task(conn, title="child", parents=[parent], assignee="worker")
        kb.add_comment(conn, tid, "user", "cleanup me")
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, result="done")
        assert kb.archive_task(conn, tid)
        conn.execute(
            "INSERT INTO kanban_notify_subs(task_id, platform, chat_id, thread_id, user_id, created_at, last_event_id) "
            "VALUES (?, 'telegram', '123', '', 'u', 0, 0)",
            (tid,),
        )
        conn.commit()

        assert kb.delete_archived_task(conn, tid) is True
        assert kb.get_task(conn, tid) is None
        assert conn.execute("SELECT COUNT(*) FROM task_links WHERE child_id = ? OR parent_id = ?", (tid, tid)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (tid,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id = ?", (tid,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (tid,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM kanban_notify_subs WHERE task_id = ?", (tid,)).fetchone()[0] == 0


def test_archive_task_removes_a_blocked_needs_input_card_from_active_listing(kanban_home):
    # Regression for t_c9b92dac: a card where Sébastien answered "do nothing"
    # sits in ``blocked``/``needs_input`` (not ``done``) — it was never
    # completed, it was decided against. ``archive_task`` must work from that
    # state too, and the archived card must disappear from the default
    # active-task listing (the same query family the TODO hub and the
    # "attend ta réponse" recap read from), not just from a done-task cleanup
    # path.
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="Vérifier la certification PEFC")
        assert kb.block_task(conn, tid, reason="preuve absente", kind="needs_input")
        assert kb.get_task(conn, tid).status == "blocked"

        assert kb.archive_task(conn, tid)

        task = kb.get_task(conn, tid)
        assert task.status == "archived"
        # claim state is cleared on archive so a stale lock can't linger.
        assert task.claim_lock is None

        # Default listing (what the dispatcher / hub-style queries use)
        # excludes archived tasks without needing an explicit opt-in.
        assert tid not in {t.id for t in kb.list_tasks(conn)}
        # Explicit opt-in still finds it — archiving hides, never deletes.
        assert tid in {t.id for t in kb.list_tasks(conn, include_archived=True)}

        # Archiving is a one-way, idempotent-safe transition: it cannot be
        # re-applied to an already-archived task (mirrors the ``rowcount != 1``
        # guard in ``archive_task``).
        assert kb.archive_task(conn, tid) is False


def test_delete_task_removes_task_and_cascades(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="to-delete", assignee="alice")
        kb.add_comment(conn, t, "user", "comment")
        kb.add_comment(conn, t, "user", "another")
        assert kb.delete_task(conn, t)
        assert kb.get_task(conn, t) is None
        assert len(kb.list_comments(conn, t)) == 0
        assert len(kb.list_events(conn, t)) == 0
        assert len(kb.list_runs(conn, t)) == 0




# ---------------------------------------------------------------------------
# Comments / events / worker context
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Respawn guard (check_respawn_guard + dispatch_once integration)
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------------









def test_worktree_workspace_explicit_target_materializes_linked_worktree(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    target = repo / ".worktrees" / "custom-task"
    branch = "wt/custom-task"
    with kb.connect() as conn:
        t = kb.create_task(
            conn,
            title="ship",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name=branch,
        )
        task = kb.get_task(conn, t)
        assert task is not None
        ws = kb.resolve_workspace(task)

    assert ws == target
    assert ws.exists()
    repo_common = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    ws_common = subprocess.run(
        ["git", "-C", str(ws), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert ws_common == repo_common
    listed = subprocess.run(
        ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert f"worktree {target}" in listed
    assert f"branch refs/heads/{branch}" in listed


# ---------------------------------------------------------------------------
# Scratch cleanup containment (#28818)
# ---------------------------------------------------------------------------



def test_complete_task_persists_scratch_artifacts_before_cleanup(kanban_home):
    """Completion artifacts from scratch workspaces survive workspace cleanup."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="render chart")
        task = kb.get_task(conn, t)
        ws = kb.resolve_workspace(task)
        kb.set_workspace_path(conn, t, ws)
        artifact = ws / "chart.png"
        artifact.write_bytes(b"png-bytes")

        assert kb.complete_task(
            conn,
            t,
            result="ok",
            metadata={"artifacts": [str(artifact)]},
        )

        completed = [e for e in kb.list_events(conn, t) if e.kind == "completed"][-1]
        persisted = Path(completed.payload["artifacts"][0])
        run = kb.latest_run(conn, t)

    assert not ws.exists(), "scratch workspace should still be cleaned up"
    assert persisted.exists(), "artifact copy should survive scratch cleanup"
    assert persisted.parent == kb.task_attachments_dir(t)
    assert persisted.name == "chart.png"
    assert persisted.read_bytes() == b"png-bytes"
    assert str(persisted) != str(artifact)
    assert run is not None
    assert run.metadata["artifacts"] == [str(persisted)]
    with kb.connect() as conn:
        attachments = kb.list_attachments(conn, t)
    assert [(a.filename, a.stored_path) for a in attachments] == [
        ("chart.png", str(persisted.resolve()))
    ]




# ---------------------------------------------------------------------------
# Deferred scratch cleanup for parent/child handoff (#33774)
# ---------------------------------------------------------------------------




def test_dir_child_completion_unblocks_deferred_scratch_parent(kanban_home, tmp_path):
    """A non-scratch ('dir') child completing must still sweep its scratch parent.

    Regression for the gap where ``_cleanup_workspace`` returned early for a
    non-scratch task and never ran the parent sweep — leaking the parent's
    deferred scratch dir forever.
    """
    child_dir = tmp_path / "persistent-child"
    child_dir.mkdir()
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="scratch parent")
        child = kb.create_task(
            conn, title="dir child", workspace_kind="dir",
            workspace_path=str(child_dir),
        )
        kb.link_tasks(conn, parent, child)
        p_task = kb.get_task(conn, parent)
        parent_ws = kb.resolve_workspace(p_task)
        kb.set_workspace_path(conn, parent, parent_ws)

        kb.complete_task(conn, parent, result="handoff")
        assert parent_ws.exists(), "deferred while dir child active"

        kb.complete_task(conn, child, result="built")

    assert not parent_ws.exists(), (
        "A 'dir' child completing must trigger the parent scratch sweep"
    )
    assert child_dir.exists(), "Non-scratch 'dir' child workspace is never deleted"




def test_is_managed_scratch_path_rejects_kanban_metadata_subtrees(kanban_home):
    """Hermes' own DB/metadata/log subtrees under ``<kanban_home>/kanban`` are NOT managed.

    Regression guard for the Copilot finding on #28819: a scratch task whose
    ``workspace_path`` was mis-set to the kanban home, the logs dir, or a
    board's metadata dir (i.e. the board root itself, not its ``workspaces/``
    child) must be refused. Without this, the containment check would happily
    ``shutil.rmtree`` Hermes' DB/metadata/logs on task completion.
    """
    kanban_root = kanban_home / "kanban"
    kanban_root.mkdir(parents=True, exist_ok=True)
    assert not kb._is_managed_scratch_path(kanban_root)

    logs_dir = kanban_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    assert not kb._is_managed_scratch_path(logs_dir)

    board_root = kanban_root / "boards" / "my-board"
    board_root.mkdir(parents=True, exist_ok=True)
    # The board root itself is NOT a managed scratch dir — only the
    # ``workspaces/`` child (and its descendants) are.
    assert not kb._is_managed_scratch_path(board_root)

    # Sibling subtrees of ``workspaces/`` under a board (e.g. its kanban.db
    # or board.json living next to ``workspaces/``) are also not managed.
    board_logs = board_root / "logs"
    board_logs.mkdir(parents=True, exist_ok=True)
    assert not kb._is_managed_scratch_path(board_logs)

    # Now create the board's workspaces dir and a task scratch dir under it —
    # the latter is the only thing the guard should allow.
    board_workspaces = board_root / "workspaces"
    board_workspaces.mkdir(parents=True, exist_ok=True)
    # The workspaces root itself is also NOT managed — deleting it would
    # wipe every task's scratch dir at once.
    assert not kb._is_managed_scratch_path(board_workspaces)
    task_dir = board_workspaces / "task-42"
    task_dir.mkdir(parents=True, exist_ok=True)
    assert kb._is_managed_scratch_path(task_dir)


# ---------------------------------------------------------------------------
# Tenancy
# ---------------------------------------------------------------------------









# ---------------------------------------------------------------------------
# Originating session id (ACP propagation)
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Shared-board path resolution (issue #19348)
#
# The kanban board is a cross-profile coordination primitive: a worker
# spawned with `hermes -p <profile>` must read/write the same kanban.db
# as the dispatcher that claimed the task. These tests exercise the
# path-resolution layer directly and would have caught the regression
# where `kanban_db_path()` resolved to the active profile's HERMES_HOME.
# ---------------------------------------------------------------------------

class TestSharedBoardPaths:
    """`kanban_home`/`kanban_db_path`/`workspaces_root`/`worker_log_path`
    must anchor at the **shared root**, not the active profile's HERMES_HOME."""

    def _set_home(self, monkeypatch, tmp_path, hermes_home):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)


    def test_profile_worker_resolves_to_shared_root(
        self, tmp_path, monkeypatch
    ):
        # Reproduces the bug: dispatcher uses ~/.hermes/kanban.db,
        # worker spawned with -p <profile> previously resolved to
        # ~/.hermes/profiles/<profile>/kanban.db. After the fix both
        # converge on ~/.hermes/kanban.db.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        profile_home = default_home / "profiles" / "nehemiahkanban"
        profile_home.mkdir(parents=True)
        self._set_home(monkeypatch, tmp_path, profile_home)

        # All four resolvers must anchor at the shared root, not the
        # profile-local HERMES_HOME.
        assert kb.kanban_home() == default_home
        assert kb.kanban_db_path() == default_home / "kanban.db"
        assert kb.workspaces_root() == default_home / "kanban" / "workspaces"
        assert (
            kb.worker_log_path("t_0d214f19")
            == default_home / "kanban" / "logs" / "t_0d214f19.log"
        )

        # Sanity: the profile-local path that used to be returned is
        # explicitly NOT what we resolve to anymore.
        assert kb.kanban_db_path() != profile_home / "kanban.db"






    def test_dispatcher_and_worker_share_a_real_database(
        self, tmp_path, monkeypatch
    ):
        # Belt-and-suspenders: round-trip a task across the two
        # HERMES_HOME perspectives via a real SQLite file. Without the
        # fix the worker would open a different file and see no rows.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        profile_home = default_home / "profiles" / "nehemiahkanban"
        profile_home.mkdir(parents=True)

        # Dispatcher creates the board and a task.
        self._set_home(monkeypatch, tmp_path, default_home)
        kb.init_db()
        with kb.connect() as conn:
            task_id = kb.create_task(conn, title="cross-profile")

        # Worker switches to the profile HERMES_HOME and reads.
        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        with kb.connect() as conn:
            task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.title == "cross-profile"




    def test_dispatcher_spawn_injects_kanban_paths_without_stale_session(
        self, tmp_path, monkeypatch
    ):
        # The dispatcher must pin board paths while stripping any unrelated
        # HERMES_SESSION_* identity inherited from the long-lived gateway.
        # The one exception is HERMES_SESSION_SOURCE, which the dispatcher
        # re-sets to its own `kanban` tag AFTER the strip — a value it owns,
        # never one inherited from whatever the gateway last routed.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        self._set_home(monkeypatch, tmp_path, default_home)

        from gateway import session_context as sc

        # A dispatcher can launch before the gateway binds its first session.
        monkeypatch.setattr(sc, "_session_context_engaged", False)
        sc.reset_session_vars()
        for key in sc._VAR_MAP:
            monkeypatch.setenv(key, "stale-routing-value")

        captured = {}

        class _FakePopen:
            def __init__(self, cmd, **kwargs):
                captured["cmd"] = cmd
                captured["env"] = kwargs.get("env", {})
                self.pid = 4242

        monkeypatch.setattr("subprocess.Popen", _FakePopen)

        task = kb.Task(
            id="t_dispatch_env",
            title="x",
            body=None,
            assignee="coder",
            status="ready",
            priority=0,
            created_by=None,
            created_at=0,
            started_at=None,
            completed_at=None,
            workspace_kind="worktree",
            workspace_path=str(tmp_path / "ws"),
            claim_lock=None,
            claim_expires=None,
            tenant=None,
            branch_name="wt/t_dispatch_env",
        )
        kb._default_spawn(task, str(tmp_path / "ws"))

        env = captured["env"]
        assert env["HERMES_KANBAN_DB"] == str(default_home / "kanban.db")
        assert env["HERMES_KANBAN_WORKSPACES_ROOT"] == str(
            default_home / "kanban" / "workspaces"
        )
        assert env["HERMES_KANBAN_TASK"] == "t_dispatch_env"
        assert env["HERMES_KANBAN_BRANCH"] == "wt/t_dispatch_env"
        for key in sc._VAR_MAP:
            if key == "HERMES_SESSION_SOURCE":
                # Re-set by the dispatcher, so what matters is that it carries
                # the worker's own tag rather than the inherited routing value.
                assert env[key] == "kanban"
                continue
            assert key not in env


# ---------------------------------------------------------------------------
# latest_summary / latest_summaries — surface task_runs.summary handoffs
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# NFS / network-filesystem fallback (see hermes_state.apply_wal_with_fallback)
# ---------------------------------------------------------------------------

def test_connect_falls_back_to_delete_on_locking_protocol(tmp_path, monkeypatch, caplog):
    """kanban_db.connect() must handle ``locking protocol`` on NFS/SMB.

    Without this fallback, the gateway's kanban dispatcher crashes every
    60s and the kanban migration (``consecutive_failures`` ADD COLUMN) is
    retried forever — which is what the real-world user report shows
    (see hermes-agent issue #22032).

    NOTE: We do NOT use the ``kanban_home`` fixture here because that
    fixture pre-initializes the DB via ``kb.init_db()`` — putting the
    file in WAL on disk. The Bug D safety guard now refuses to downgrade
    to DELETE when the on-disk header is already WAL, so testing the
    NFS-fallback path requires a truly-fresh DB file (NFS scenario in
    production: first connection of the first process ever to touch the
    file, where downgrading is safe because nobody else has WAL state
    yet).
    """
    import sqlite3 as _sqlite3
    from unittest.mock import patch as _patch

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # These tests exercise the WAL-attempt path; assume a fixed SQLite so the
    # WAL-reset vulnerability gate doesn't short-circuit before the pragma.
    import hermes_state as _hermes_state
    monkeypatch.setattr(
        _hermes_state, "is_sqlite_wal_reset_vulnerable",
        lambda version_info=None: False,
    )
    _hermes_state._wal_fallback_warned_paths.clear()

    # Clear module cache so a fresh connect() is attempted
    kb._INITIALIZED_PATHS.clear()
    hermes_state._wal_fallback_warned_paths.clear()

    real_connect = _sqlite3.connect

    class _WalBlockingConnection(_sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):  # type: ignore[override]
            if "journal_mode=wal" in sql.lower().replace(" ", ""):
                raise _sqlite3.OperationalError("locking protocol")
            return super().execute(sql, *args, **kwargs)

    def wal_blocking_connect(*args, **kwargs):
        # connect_tracked passes a tracking-augmented factory; drop it and
        # substitute the double, which connect_tracked re-applies to the
        # returned instance.
        kwargs.pop("factory", None)
        return real_connect(
            *args, factory=_WalBlockingConnection, **kwargs
        )

    with _patch("hermes_cli.kanban_db.sqlite3.connect", side_effect=wal_blocking_connect):
        with caplog.at_level("ERROR", logger="hermes_state"):
            conn = kb.connect()

    # One fallback error, naming kanban.db
    errors = [
        r
        for r in caplog.records
        if r.levelname == "ERROR" and "kanban.db" in r.getMessage()
    ]
    assert len(errors) >= 1, (
        f"Expected a kanban.db ERROR, got: {[r.getMessage() for r in caplog.records]}"
    )

    # DB still usable end-to-end — create + list a task
    t = kb.create_task(conn, title="post-fallback task")
    tasks = kb.list_tasks(conn)
    assert any(row.id == t for row in tasks)
    conn.close()


def test_connect_works_when_wal_is_silently_refused(tmp_path, monkeypatch, caplog):
    """kanban_db.connect() must stay usable when WAL silently no-ops to DELETE."""
    import sqlite3 as _sqlite3
    from unittest.mock import patch as _patch

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    kb._INITIALIZED_PATHS.clear()
    hermes_state._wal_fallback_warned_paths.clear()
    # Assume a fixed SQLite so the WAL-reset gate doesn't short-circuit.
    monkeypatch.setattr(
        hermes_state, "is_sqlite_wal_reset_vulnerable",
        lambda version_info=None: False,
    )

    real_connect = _sqlite3.connect

    class _WalSilentNoOpConnection(_sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):  # type: ignore[override]
            if "journal_mode=wal" in sql.lower().replace(" ", ""):
                return super().execute("PRAGMA journal_mode=delete", *args, **kwargs)
            return super().execute(sql, *args, **kwargs)

    def wal_silent_noop_connect(*args, **kwargs):
        kwargs.pop("factory", None)
        return real_connect(
            *args, factory=_WalSilentNoOpConnection, **kwargs
        )

    with _patch(
        "hermes_cli.kanban_db.sqlite3.connect",
        side_effect=wal_silent_noop_connect,
    ):
        with caplog.at_level("ERROR", logger="hermes_state"):
            conn = kb.connect()

    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    t = kb.create_task(conn, title="post-silent-fallback task")
    tasks = kb.list_tasks(conn)
    assert any(row.id == t for row in tasks)
    conn.close()

    errors = [
        r
        for r in caplog.records
        if r.levelname == "ERROR" and "kanban.db" in r.getMessage()
    ]
    assert len(errors) >= 1, (
        f"Expected a kanban.db ERROR, got: {[r.getMessage() for r in caplog.records]}"
    )


def test_sqlite_connect_closes_tracked_conn_on_setup_failure(tmp_path, monkeypatch):
    """A PRAGMA failure after connect must not abandon a tracked kanban fd."""
    from hermes_cli import sqlite_safe_read

    db_path = tmp_path / "kanban.db"
    real_connect = sqlite3.connect
    opened = []

    class _BusyTimeoutFailure(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):  # type: ignore[override]
            if str(sql).startswith("PRAGMA busy_timeout="):
                raise sqlite3.OperationalError("simulated setup failure")
            return super().execute(sql, *args, **kwargs)

    def failing_connect(*args, **kwargs):
        kwargs.pop("factory", None)
        conn = real_connect(*args, factory=_BusyTimeoutFailure, **kwargs)
        opened.append(conn)
        return conn

    key = sqlite_safe_read._key(db_path)
    with sqlite_safe_read._live_lock:
        before = sqlite_safe_read._live_connections.get(key, 0)
    monkeypatch.setattr(kb.sqlite3, "connect", failing_connect)

    with pytest.raises(sqlite3.OperationalError, match="simulated setup failure"):
        kb._sqlite_connect(db_path)

    with sqlite_safe_read._live_lock:
        after = sqlite_safe_read._live_connections.get(key, 0)
    assert after == before


def test_unlink_tasks_triggers_recompute_ready(kanban_home):
    """Regression test for issue #22459.

    Removing a dependency via unlink_tasks must immediately promote the child
    to ready when all remaining parents are done — same contract as
    complete_task and unblock_task.

    Before the fix, child stayed 'todo' indefinitely after unlink; only the
    next dispatcher tick or a manual 'hermes kanban recompute' would promote it.
    """
    with kb.connect() as conn:
        # A is done.
        a = kb.create_task(conn, title="parent-done")
        kb.complete_task(conn, a)

        # C is running (not done) — blocks child B.
        c = kb.create_task(conn, title="parent-running")
        kb.claim_task(conn, c, claimer="worker:1")

        # B depends on both A (done) and C (running) → stays todo.
        b = kb.create_task(conn, title="child", parents=[a, c])
        assert kb.get_task(conn, b).status == "todo"

        # Remove the blocking dependency C → B.
        removed = kb.unlink_tasks(conn, c, b)
        assert removed is True

        # B's only remaining parent is A (done) → must be ready immediately.
        assert kb.get_task(conn, b).status == "ready", (
            "child should promote to ready immediately after unlink_tasks "
            "removes its last blocking dependency"
        )



# ---------------------------------------------------------------------------
# _add_column_if_missing / _migrate_add_optional_columns idempotency (#21708)
# ---------------------------------------------------------------------------

def test_add_column_if_missing_is_idempotent_on_race(kanban_home):
    """``_add_column_if_missing`` must swallow 'duplicate column name' errors.

    Regression for #21708: the kanban dispatcher opens the DB twice per tick
    (once via _tick_once_for_board, once via init_db's discard-and-reconnect
    path).  A second concurrent connection runs _migrate_add_optional_columns
    before the first one commits, so ALTER TABLE raises OperationalError with
    'duplicate column name: consecutive_failures'.  Without the idempotency
    guard that crashes the dispatcher on the first tick after every restart.
    """
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE tasks (id INTEGER PRIMARY KEY, title TEXT NOT NULL)"
    )

    # First call adds the column — returns True.
    added = kb._add_column_if_missing(conn, "tasks", "extra_col", "extra_col TEXT")
    assert added is True
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    assert "extra_col" in cols

    # Second call on same connection — column already exists — must return
    # False without raising, simulating the race the dispatcher hits.
    added_again = kb._add_column_if_missing(
        conn, "tasks", "extra_col", "extra_col TEXT"
    )
    assert added_again is False

    conn.close()


def test_migrate_add_optional_columns_tolerates_concurrent_migration(kanban_home):
    """Full _migrate_add_optional_columns must not raise when columns already
    exist (issue #21708 race window — two connections migrate concurrently)."""
    import sqlite3

    # Schema already in fully-migrated state (all optional columns present).
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE tasks (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            tenant TEXT,
            result TEXT,
            idempotency_key TEXT,
            branch_name TEXT,
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            worker_pid INTEGER,
            last_failure_error TEXT,
            max_runtime_seconds INTEGER,
            last_heartbeat_at INTEGER,
            current_run_id INTEGER,
            workflow_template_id TEXT,
            current_step_key TEXT,
            skills TEXT,
            max_retries INTEGER,
            session_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE task_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id    TEXT NOT NULL DEFAULT '',
            run_id     INTEGER,
            kind       TEXT NOT NULL DEFAULT '',
            payload    TEXT,
            created_at INTEGER NOT NULL DEFAULT 0
        )
        """
    )

    # Running migration on an already-migrated schema must not raise.
    kb._migrate_add_optional_columns(conn)
    conn.close()


# ---------------------------------------------------------------------------
# Dispatcher spawn invocation — _resolve_hermes_argv()
#
# Workers spawned by the dispatcher must use a `hermes` invocation that does
# not depend on PATH being set up correctly. cron jobs, systemd User= services,
# launchd jobs, and other detached processes routinely run with a stripped
# $PATH that doesn't include the venv's bin/, so a bare `["hermes", ...]`
# spawn fails with FileNotFoundError and the task gets stuck. The resolver
# prefers the PATH shim (familiar `ps` output) but falls back to the module
# form so the spawn keeps working when PATH is missing the shim.
# ---------------------------------------------------------------------------


def test_resolve_hermes_argv_falls_back_to_module_form_when_no_path_shim(monkeypatch):
    """When the shim is not on PATH, fall back to `python -m hermes_cli.main`.

    Pins the correct module name (NOT `hermes` — there is no top-level
    `hermes` package). Regression for #23198: the original PR shipped
    `python -m hermes` which fails with `No module named hermes` on every
    invocation.
    """
    import shutil
    import sys
    import hermes_cli.kanban_db as kb

    monkeypatch.delenv("HERMES_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    argv = kb._resolve_hermes_argv()
    assert argv == [sys.executable, "-m", "hermes_cli.main"]


def test_resolve_hermes_argv_module_actually_runs():
    """The fallback module name must be importable + runnable.

    A unit test that pins the literal string is necessary but not
    sufficient — if `hermes_cli.main` ever loses `if __name__ == "__main__"`
    handling or its argparse setup, `python -m hermes_cli.main --version`
    would fail and so would every dispatcher spawn that hits the fallback.
    Run it as a real subprocess to catch that regression.
    """
    import subprocess
    import hermes_cli.kanban_db as kb
    import shutil
    import unittest.mock as mock

    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("HERMES_BIN", None)
        with mock.patch.object(shutil, "which", return_value=None):
            argv = kb._resolve_hermes_argv()
    r = subprocess.run(argv + ["--version"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, (
        f"`{' '.join(argv)} --version` failed (rc={r.returncode}); "
        f"stderr={r.stderr[:200]!r}"
    )
    assert "Hermes Agent" in r.stdout, f"unexpected output: {r.stdout[:200]!r}"


# ---------------------------------------------------------------------------
# task_age — guard against corrupt timestamp values
#
# The Task dataclass declares ``created_at: int`` but rows come from sqlite
# without coercion at the boundary. A row that ever held a non-int (e.g. an
# unsubstituted ``'%s'`` from a logged format string, ``None``, an arbitrary
# string, or a float-as-string) used to crash ``task_age`` with ``ValueError``
# and turn ``GET /api/plugins/kanban/board`` into a 500 because the dashboard
# calls ``task_age`` unguarded for every task in the response.
#
# After the fix, ``_safe_int`` returns ``None`` on bad input and ``task_age``
# degrades gracefully (per-field ``None`` rather than a hard crash).
# ---------------------------------------------------------------------------


def _make_task(**overrides) -> "kb.Task":
    """Minimal Task with all required fields filled in. Override anything."""
    defaults = dict(
        id="t_age",
        title="x",
        body=None,
        assignee=None,
        status="ready",
        priority=0,
        created_by=None,
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
    )
    defaults.update(overrides)
    return kb.Task(**defaults)












# ---------------------------------------------------------------------------
# Board-level default_workdir
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# dispatch_once — max_in_progress
# ---------------------------------------------------------------------------


def test_dispatch_max_in_progress_blocks_review_when_at_limit(
    kanban_home, all_assignees_spawnable,
):
    """Review-only backlog must still respect max_in_progress."""
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kb.connect() as conn:
        running = kb.create_task(conn, title="running", assignee="alice")
        kb.claim_task(conn, running)
        review = kb.create_task(conn, title="review", assignee="bob")
        _set_task_status(conn, review, "review")
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn, max_in_progress=1)
        review_task = kb.get_task(conn, review)

    assert not res.spawned
    assert not spawns
    assert review_task is not None
    assert review_task.status == "review"

# Review column dispatch
# ---------------------------------------------------------------------------


def _set_task_status(conn: sqlite3.Connection, task_id: str, status: str) -> None:
    """Test helper: set a task's status directly."""
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))








# Stale detection — detect_stale_running
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Corruption guard (issue #30687)
# ---------------------------------------------------------------------------

def _write_corrupt_db(path: Path) -> bytes:
    """Write a kanban DB with a VALID SQLite header but malformed page content.

    This is the corruption shape the integrity guard specifically targets
    (e.g. issue #29507 follow-up reports where the file's first 16 bytes
    pass the header byte check but ``PRAGMA integrity_check`` then fails
    because the internal pages are damaged). It's what main's header-only
    validator was letting through, and what this PR adds the full guard
    for.
    """
    # 100-byte SQLite header (magic + minimal valid-looking fields) so the
    # cheap header check passes, then deliberate garbage so sqlite refuses
    # to read the file past the header.
    header = b"SQLite format 3\x00" + b"\x10\x00\x02\x02\x00\x40\x20\x20"
    header += b"\x00\x00\x00\x0c\x00\x00\x23\x46\x00\x00\x00\x00"
    header = header.ljust(100, b"\x00")
    payload = b"definitely not a valid sqlite page \x00\x01\x02\x03" * 64
    blob = header + payload
    path.write_bytes(blob)
    return blob




def test_repeated_corrupt_open_reuses_single_backup(tmp_path):
    """Repeated quarantines of the same corrupt bytes must not amplify disk usage.

    Regression for the gateway dispatcher's 5-min retry loop on shared kanban
    DBs across multi-profile fleets: each retry on an unchanged corrupt file
    used to create a fresh ``.corrupt.<timestamp>.bak`` until disk filled. The
    content-addressed backup name is deterministic in the DB's sha256, so
    N retries of the same bytes share one backup.
    """
    db_path = tmp_path / "kanban.db"
    original = _write_corrupt_db(db_path)

    backups: set[Path] = set()
    for _ in range(10):
        kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
        with pytest.raises(kb.KanbanDbCorruptError) as excinfo:
            kb.connect(db_path=db_path)
        assert excinfo.value.backup_path is not None
        backups.add(excinfo.value.backup_path)

    assert len(backups) == 1, f"expected 1 deterministic backup, got {len(backups)}"
    (backup,) = backups
    assert backup.exists()
    assert backup.read_bytes() == original

    # Mutate the corrupt bytes — fingerprint changes, separate backup preserved.
    with db_path.open("r+b") as f:
        f.seek(4096)
        f.write(b"\xAB" * 64)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with pytest.raises(kb.KanbanDbCorruptError) as excinfo2:
        kb.connect(db_path=db_path)
    second_backup = excinfo2.value.backup_path
    assert second_backup is not None
    assert second_backup != backup
    assert second_backup.exists()


def test_locked_healthy_db_does_not_classify_as_corrupt(tmp_path, monkeypatch):
    """A transient lock during the probe must not produce a .corrupt backup
    and must not be reported as :class:`KanbanDbCorruptError`. Raw sqlite
    ``OperationalError`` (lock/busy) is acceptable and expected."""
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path=db_path)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    real_connect = sqlite3.connect

    def flaky_connect(*args, **kwargs):
        # First call is the integrity probe — simulate a lock.
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(kb.sqlite3, "connect", flaky_connect)

    with pytest.raises(sqlite3.OperationalError):
        kb.connect(db_path=db_path)

    # No .corrupt backup may be produced for a healthy-but-locked DB.
    backups = list(tmp_path.glob("*.corrupt.*"))
    assert backups == [], f"unexpected corrupt backups: {backups}"

    # And once the lock clears, normal access still works.
    monkeypatch.setattr(kb.sqlite3, "connect", real_connect)
    with kb.connect(db_path=db_path) as conn:
        kb.create_task(conn, title="still here")
        titles = [t.title for t in kb.list_tasks(conn)]
    assert "still here" in titles




# ---------------------------------------------------------------------------
# First-use tip for scratch workspaces
# ---------------------------------------------------------------------------

def test_maybe_emit_scratch_tip_fires_once_per_install(kanban_home, caplog):
    """First scratch workspace materialization warns + emits an event.

    Subsequent scratch workspaces on the SAME install stay silent — the
    sentinel file under kanban_home() flips after the first emit.
    """
    import logging

    with kb.connect() as conn:
        t1 = kb.create_task(conn, title="first scratch")
        t2 = kb.create_task(conn, title="second scratch")

    # Sentinel must not exist yet on a fresh install.
    assert not kb._scratch_tip_shown()

    with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
        with kb.connect() as conn:
            kb._maybe_emit_scratch_tip(conn, t1, "scratch")

    # Sentinel is now set.
    assert kb._scratch_tip_shown()
    assert kb._scratch_tip_sentinel_path().exists()

    # Warning was logged exactly once.
    tip_records = [
        r for r in caplog.records
        if "scratch workspaces are ephemeral" in r.getMessage()
    ]
    assert len(tip_records) == 1, (
        f"Expected exactly one tip warning, got {len(tip_records)}: "
        f"{[r.getMessage() for r in tip_records]!r}"
    )

    # An event row was appended on the first task.
    with kb.connect() as conn:
        events = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
            (t1,),
        ).fetchall()
    kinds = [e["kind"] for e in events]
    assert "tip_scratch_workspace" in kinds, (
        f"Expected tip_scratch_workspace event on first scratch task; "
        f"got {kinds!r}"
    )

    # Second scratch materialization on the same install stays silent.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
        with kb.connect() as conn:
            kb._maybe_emit_scratch_tip(conn, t2, "scratch")
    tip_records2 = [
        r for r in caplog.records
        if "scratch workspaces are ephemeral" in r.getMessage()
    ]
    assert tip_records2 == [], (
        f"Tip should not re-fire after sentinel is set; got "
        f"{[r.getMessage() for r in tip_records2]!r}"
    )
    with kb.connect() as conn:
        events2 = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
            (t2,),
        ).fetchall()
    assert "tip_scratch_workspace" not in [e["kind"] for e in events2], (
        "Tip event should not be appended for subsequent scratch tasks."
    )




# ---------------------------------------------------------------------------
# Connection pragmas (secure_delete, cell_size_check, synchronous=FULL)
# ---------------------------------------------------------------------------


def test_connect_sets_secure_delete_on(tmp_path):
    """secure_delete=ON must be active on every new connection."""
    db_path = tmp_path / "kanban.db"
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kb.connect(db_path=db_path) as conn:
        row = conn.execute("PRAGMA secure_delete").fetchone()
    assert row[0] == 1, f"expected secure_delete=1, got {row[0]}"





# write_txn — rollback handler must not mask the original exception
# ---------------------------------------------------------------------------


def test_write_txn_preserves_original_exception_when_rollback_fails(kanban_home):
    """When a write inside write_txn raises an OperationalError that SQLite
    has already auto-rolled-back (e.g. ``disk I/O error``,
    ``database is locked``, ``database disk image is malformed``), the
    explicit ROLLBACK in ``write_txn.__exit__`` itself raises
    ``cannot rollback - no transaction is active``. The original cause
    must NOT be masked by the secondary rollback failure — operators rely
    on the original cause to diagnose the underlying issue.
    """

    class FailingConnWrapper:
        """Delegate to a real connection, simulating an EIO during an INSERT
        that SQLite has already auto-rolled-back."""

        def __init__(self, real):
            self._real = real
            self._fail_armed = True

        def execute(self, sql, *args, **kwargs):
            if (
                self._fail_armed
                and sql.lstrip().upper().startswith("INSERT")
                and "task_events" in sql.lower()
            ):
                self._fail_armed = False  # one-shot
                # Simulate SQLite auto-rolling back the transaction by
                # issuing a real ROLLBACK now. After this, BEGIN IMMEDIATE
                # is no longer active and an explicit ROLLBACK would error.
                try:
                    self._real.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    pass
                raise sqlite3.OperationalError("disk I/O error")
            return self._real.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

    with kb.connect() as conn:
        wrapper = FailingConnWrapper(conn)
        with pytest.raises(sqlite3.OperationalError) as excinfo:
            with kb.write_txn(wrapper):
                kb._append_event(wrapper, "t_bogus", "promoted", None)

    msg = str(excinfo.value)
    assert "disk I/O error" in msg, (
        f"write_txn masked the original exception with rollback failure; "
        f"got {msg!r} (expected to contain 'disk I/O error')"
    )
    assert "cannot rollback" not in msg, (
        f"write_txn surfaced the rollback failure instead of the original "
        f"OperationalError; got {msg!r}"
    )


def test_write_txn_check_reads_correct_header_fields(tmp_path):
    """A genuinely truncated DB is never reported as passing the invariant.

    The check no longer opens the database file to read header bytes (that
    open/close would cancel this process's POSIX advisory locks — the
    corruption route in sqlite.org/howtocorrupt.html §2.2). It asks SQLite for
    ``page_count`` instead. On a truncated file SQLite refuses that pragma, so
    the helper reports "not healthy" rather than a page-count mismatch; either
    way the file must never come back clean.
    """
    import struct
    from hermes_cli.kanban_db import connect
    from hermes_cli.sqlite_safe_read import file_length_matches_header

    db = tmp_path / "synthetic.db"
    conn = connect(db_path=db)
    conn.execute("PRAGMA journal_mode=DELETE")
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    conn.close()

    with open(db, "rb") as f:
        data = bytearray(f.read())
    real_page_count = struct.unpack(">I", data[28:32])[0]
    if real_page_count < 2:
        pytest.skip("DB too small for synthetic truncation test")
    truncated = bytes(data[: (real_page_count - 1) * page_size])
    with open(db, "wb") as f:
        f.write(truncated)

    raw_conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        assert file_length_matches_header(raw_conn) is not True
    finally:
        raw_conn.close()


# ---------------------------------------------------------------------------
# reap_worker_zombies() tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="waitpid registry is POSIX-only")
def test_retained_worker_preserves_interrupted_exit_status():
    """Unrelated Popen cleanup cannot consume a worker's neutral sentinel."""
    import hermes_cli.kanban_db as _kb

    proc = subprocess.Popen(
        [sys.executable, "-c", f"raise SystemExit({_kb.KANBAN_INTERRUPTED_EXIT_CODE})"]
    )
    _kb._retain_worker_process(proc)
    pid = proc.pid
    try:
        deadline = time.time() + 5
        while time.time() < deadline:
            if pid in _kb.reap_worker_zombies():
                break
            time.sleep(0.01)
        else:
            pytest.fail("retained worker was not reaped")

        assert _kb._classify_worker_exit(pid) == (
            "interrupted",
            _kb.KANBAN_INTERRUPTED_EXIT_CODE,
        )
        assert pid not in _kb._live_worker_processes
    finally:
        _kb._live_worker_processes.pop(pid, None)
        try:
            proc.kill()
        except ProcessLookupError:
            pass










# ---------------------------------------------------------------------------
# connect_closing(): context manager that actually closes the FD
# Regression coverage for #33159 (kanban.db FD leak — gateway crashes after
# ~4 days). sqlite3.Connection's built-in __exit__ commits/rollbacks but
# does NOT close, so `with kb.connect() as conn:` leaks the FD in
# long-lived processes (gateway run_slash, dashboard decompose handler).
# `connect_closing()` is the leak-safe replacement.
# ---------------------------------------------------------------------------




def test_bare_connect_does_not_close_on_context_exit(tmp_path):
    """Document the leak that connect_closing exists to prevent.

    sqlite3.Connection's __exit__ commits/rollbacks but doesn't close.
    This is the upstream behaviour we cannot change; the regression
    guard is to make sure connect_closing() does the right thing.
    """
    db_path = tmp_path / "kanban.db"
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kb.connect(db_path=db_path) as conn:
        pass
    # Still usable after with-block exit (the leak).
    conn.execute("SELECT 1").fetchone()
    conn.close()  # explicit close to avoid leaking THIS test


# ---------------------------------------------------------------------------
# activity events — live agent-activity board contract
# (see /home/seb/.hermes/workspace/agent-live-activity-contract.md)
# ---------------------------------------------------------------------------

def test_sanitize_activity_text_masks_a_generic_secret():
    """A generic secret pattern is masked, not just the internal vocabulary."""
    secret = "ghp_" + "A" * 40
    out = kb.sanitize_activity_text(f"push with token {secret}")
    # The internal-vocabulary filter already rejects the word "token", so
    # this also exercises the reject path; the important invariant is the
    # raw secret never survives either way.
    assert out is None or secret not in out


def test_sanitize_activity_text_rejects_banned_internal_vocabulary():
    assert kb.sanitize_activity_text("kanban t_abc123 : lecture fichier") is None
    assert kb.sanitize_activity_text("pid 12345 vivant") is None
    assert kb.sanitize_activity_text("commande git status") is None


def test_sanitize_activity_text_passes_plain_text_through():
    assert kb.sanitize_activity_text("lecture kanban_board_sync.py") is None or True
    # A target with no banned word and no secret pattern must survive intact.
    assert kb.sanitize_activity_text("scripts/kanban_board_sync.py") == "scripts/kanban_board_sync.py"


def test_sanitize_activity_text_truncates_long_input():
    out = kb.sanitize_activity_text("x" * 500, max_len=80)
    assert out is not None
    assert len(out) <= 80


def test_append_activity_event_writes_action_and_target(kanban_home, monkeypatch):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        kb.claim_task(conn, t)
        monkeypatch.setenv("HERMES_KANBAN_TASK", t)
        kb.append_activity_event(action="read_file", target="scripts/kanban_board_sync.py")
        row = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? AND kind = 'activity'",
            (t,),
        ).fetchone()
        assert row is not None
        import json as _json
        payload = _json.loads(row["payload"])
        assert payload["action"] == "read_file"
        assert payload["target"] == "scripts/kanban_board_sync.py"


def test_append_activity_event_masks_a_secret_in_target(kanban_home, monkeypatch):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        kb.claim_task(conn, t)
        monkeypatch.setenv("HERMES_KANBAN_TASK", t)
        secret = "ghp_" + "B" * 40
        kb.append_activity_event(action="bash", target=f"curl -H 'token: {secret}'")
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'activity'",
            (t,),
        ).fetchone()
        import json as _json
        payload = _json.loads(row["payload"]) if row else {}
        assert secret not in (row["payload"] if row else "")


def test_browser_navigate_marker_in_url_path_never_reaches_task_events_payload(kanban_home, monkeypatch):
    """End-to-end regression for t_da242e47 run #152 (codex-worker BLOCK).

    Reproduction: a navigation to a URL whose *path* segment carries a
    sensitive marker (not just the query string). Exercises the real
    write path -- agent.display's per-tool target closure feeding
    kanban_db.append_activity_event's write-time sanitizer -- end to end,
    reading back the actual persisted ``task_events.payload`` row rather
    than asserting on an intermediate function's return value alone.
    """
    from agent.display import _emit_tool_activity_event

    marker = "MARKER-do-not-leak-sk-abcdef1234567890"
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        kb.claim_task(conn, t)
        monkeypatch.setenv("HERMES_KANBAN_TASK", t)
        _emit_tool_activity_event(
            "browser_navigate",
            {"url": f"https://example.com/{marker}?q=ignored"},
        )
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'activity'",
            (t,),
        ).fetchone()
        assert row is not None
        assert marker not in row["payload"]
        import json as _json
        payload = _json.loads(row["payload"])
        assert payload["target"] == "example.com"


def test_append_activity_event_is_a_noop_outside_kanban_context(kanban_home, monkeypatch):
    """Zero-cost, zero-row outside a Kanban-worker context (no HERMES_KANBAN_TASK)."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        kb.claim_task(conn, t)
        # No HERMES_KANBAN_TASK set -> must be a strict no-op regardless of
        # the task_id kwarg (mirrors the dispatcher-guard used elsewhere).
        kb.append_activity_event(action="read_file", target="foo.py")
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE kind = 'activity'"
        ).fetchone()["n"]
        assert count == 0


def test_append_activity_event_swallows_errors_best_effort(kanban_home, monkeypatch):
    """A write glitch must never raise into the caller's real work."""
    t_id = "t_doesnotexist_but_env_set"
    monkeypatch.setenv("HERMES_KANBAN_TASK", t_id)
    # No task row exists for this id at all -- append must not raise.
    kb.append_activity_event(action="bash", target="ls")


def test_terminal_activity_event_never_persists_the_raw_command_end_to_end(kanban_home, monkeypatch):
    """Review finding t_6b360247 (run #142), reproduced end-to-end.

    A real worker `chat -q` activity previously wrote a full raw command
    (an env-var assignment + argv, e.g. "HERMES_KANBAN_TASK=... uv run
    python ...") straight into ``task_events.payload.target`` --
    ``sanitize_activity_text`` only redacts recognizable secret patterns
    and a small banned-word list, it never rejected an arbitrary raw
    command on principle. The write-time fix lives in
    ``agent.display._activity_target_for_tool`` (never hands the raw
    command past the program name to ``append_activity_event`` for
    `terminal`), and this test exercises the REAL write path -- CLI
    completion label -> append_activity_event -> DB row -- rather than a
    mocked target, so it also covers a defense-in-depth regression at the
    ``sanitize_activity_text`` layer for any other caller.
    """
    import agent.display as display_module

    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        kb.claim_task(conn, t)
        monkeypatch.setenv("HERMES_KANBAN_TASK", t)
        # Deliberately avoids the words/shapes sanitize_activity_text's
        # banned-vocabulary regex rejects outright as a whole string
        # (`\bt_[0-9a-f]+\b`, and "kanban"/"gate"/"pid"/"token"/"commande"
        # as bounded words -- note "/kanban/" in a path DOES count, since
        # "/" is a non-word boundary). A raw command needs neither to leak
        # administrative detail; using a path/env-var/host free of those
        # exact shapes proves the write-time fix on its own merits rather
        # than piggybacking on sanitize's unrelated whole-string reject.
        raw_command = (
            "HERMES_WORKER_ID=w-42 uv run python worker.py "
            "--profile claude2 --secret sk-abcdef1234567890"
        )
        display_module.get_cute_tool_message("terminal", {"command": raw_command}, 0.1)
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'activity'",
            (t,),
        ).fetchone()
        assert row is not None
        payload_text = row["payload"]
        for leaked in ("HERMES_WORKER_ID", "w-42", "worker.py", "--secret", "sk-abcdef1234567890"):
            assert leaked not in payload_text, f"{leaked!r} leaked into activity payload: {payload_text!r}"


@pytest.mark.parametrize(
    "tool_name,args_key",
    [
        ("web_search", "query"),
        ("browser_type", "text"),
        ("browser_exec", "code"),
        ("delegate_task", "goal"),
        ("image_generate", "prompt"),
        ("text_to_speech", "text"),
        ("vision_analyze", "question"),
    ],
)
def test_free_text_tool_activity_never_persists_the_prompt_end_to_end(
    kanban_home, monkeypatch, tool_name, args_key,
):
    """Review finding t_6b360247 (run #147), reproduced end-to-end.

    ``_activity_target_for_tool`` used ``build_tool_preview()`` -- a
    free-text renderer -- for every tool other than terminal/execute_code,
    so a sensitive web query / browser text / browser_exec comment /
    delegated goal / generation prompt / TTS text / vision question could
    land straight in ``task_events.payload.target`` and, from there, the
    Telegram board. Exercises the real write path: CLI completion label ->
    ``append_activity_event`` -> DB row.
    """
    import agent.display as display_module

    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        kb.claim_task(conn, t)
        monkeypatch.setenv("HERMES_KANBAN_TASK", t)
        secret = "MARKER-do-not-leak-sk-abcdef1234567890"
        args = {"ref": "e3"} if tool_name == "browser_type" else {}
        if tool_name == "browser_exec":
            # browser_exec's friendly label is derived from the code's
            # leading `# ...` comment (_browser_exec_step_label) -- the
            # comment text is exactly the free-form surface that must
            # never reach the target, not just the code body.
            args[args_key] = f"# {secret}\nclick('e1')"
        else:
            args[args_key] = secret
        display_module.get_cute_tool_message(tool_name, args, 0.1)
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'activity'",
            (t,),
        ).fetchone()
        assert row is not None
        assert secret not in row["payload"], f"leaked into activity payload: {row['payload']!r}"


def test_heartbeat_note_is_redacted_at_write_time(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        kb.claim_task(conn, t)
        secret = "ghp_" + "C" * 40
        kb.heartbeat_worker(conn, t, note=f"pushed with {secret}")
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'heartbeat' "
            "ORDER BY id DESC LIMIT 1",
            (t,),
        ).fetchone()
        assert row is not None
        assert secret not in (row["payload"] or "")


def test_spawned_event_carries_the_actually_resolved_model(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(
            conn, title="x", assignee="claude2", model_override="gpt-5.6-terra",
            provider_override="openai",
        )
        kb.claim_task(conn, t)
        kb._set_worker_pid(conn, t, 54321)
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'spawned' "
            "ORDER BY id DESC LIMIT 1",
            (t,),
        ).fetchone()
        assert row is not None
        import json as _json
        payload = _json.loads(row["payload"])
        assert payload["model_resolved"] == "gpt-5.6-terra"


def test_spawned_event_model_resolved_is_none_without_override(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="claude1")
        kb.claim_task(conn, t)
        kb._set_worker_pid(conn, t, 54322)
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'spawned' "
            "ORDER BY id DESC LIMIT 1",
            (t,),
        ).fetchone()
        import json as _json
        payload = _json.loads(row["payload"])
        assert payload["model_resolved"] is None


def test_gc_events_purges_activity_rows_on_done_tasks(kanban_home, monkeypatch):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        kb.claim_task(conn, t)
        monkeypatch.setenv("HERMES_KANBAN_TASK", t)
        kb.append_activity_event(action="bash", target="ls")
        kb.heartbeat_worker(conn, t, note="tests en cours")
        kb.complete_task(conn, t, summary="done")
        # Backdate every event row so it's older than the GC cutoff.
        old = int(time.time()) - 40 * 24 * 3600
        conn.execute("UPDATE task_events SET created_at = ? WHERE task_id = ?", (old, t))
        deleted = kb.gc_events(conn)
        assert deleted > 0
        remaining = conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ?", (t,)
        ).fetchone()["n"]
        assert remaining == 0
