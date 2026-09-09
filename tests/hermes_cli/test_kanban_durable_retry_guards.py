"""Regression tests for durable, no-new-fact Kanban retry boundaries."""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _event_count(conn, task_id: str) -> int:
    return int(conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,),
    ).fetchone()[0])


def _run_count(conn, task_id: str) -> int:
    return int(conn.execute(
        "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,),
    ).fetchone()[0])


def test_direct_claims_are_noops_before_durable_retry_deadline(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 10_000
    monkeypatch.setattr(kb.time, "time", lambda: float(now))
    with kb.connect() as conn:
        ready_id = kb.create_task(conn, title="ready later", assignee="worker")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET next_retry_at = ? WHERE id = ?",
                (now + 100, ready_id),
            )
        ready_before = (_event_count(conn, ready_id), _run_count(conn, ready_id))
        assert kb.claim_task(conn, ready_id, claimer="worker:early") is None
        assert (_event_count(conn, ready_id), _run_count(conn, ready_id)) == ready_before
        ready = kb.get_task(conn, ready_id)
        assert ready is not None
        assert (ready.status, ready.next_retry_at) == ("ready", now + 100)

        review_id = kb.create_task(conn, title="review later", assignee="builder")
        implementation = kb.claim_task(conn, review_id, claimer="builder:1")
        assert implementation is not None
        assert kb.request_review(
            conn,
            review_id,
            summary="candidate ready",
            reviewer="reviewer",
            expected_run_id=implementation.current_run_id,
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET next_retry_at = ? WHERE id = ?",
                (now + 100, review_id),
            )
        review_before = (_event_count(conn, review_id), _run_count(conn, review_id))
        assert kb.claim_review_task(conn, review_id, claimer="reviewer:early") is None
        assert (_event_count(conn, review_id), _run_count(conn, review_id)) == review_before
        review = kb.get_task(conn, review_id)
        assert review is not None
        assert (review.status, review.next_retry_at) == ("review", now + 100)


def test_identical_review_deferral_is_idempotent_then_backs_off_after_due(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 20_000}
    monkeypatch.setattr(kb.time, "time", lambda: float(clock["now"]))
    reason = "Gemini visual review unavailable: HTTP 429"

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="review retry", assignee="builder")
        implementation = kb.claim_task(conn, task_id, claimer="builder:1")
        assert implementation is not None
        assert kb.request_review(
            conn,
            task_id,
            summary="same candidate",
            reviewer="reviewer",
            expected_run_id=implementation.current_run_id,
            metadata={"commit": "abc1234"},
        )
        review = kb.claim_review_task(conn, task_id, claimer="reviewer:1")
        assert review is not None
        assert kb.heartbeat_worker(
            conn,
            task_id,
            expected_run_id=review.current_run_id,
            worker_session_id="review-session",
        )
        assert kb.defer_review_task(
            conn,
            task_id,
            reason=reason,
            retry_at=clock["now"] + 100,
            expected_run_id=review.current_run_id,
            metadata={"worker_session_id": "review-session"},
        ) == (True, None)
        first = kb.get_task(conn, task_id)
        assert first is not None and first.next_retry_at == 20_100
        counts = (_event_count(conn, task_id), _run_count(conn, task_id))

        # A duplicated terminal tool call before the boundary is acknowledged
        # without rewriting the task, event stream, or run history.
        assert kb.defer_review_task(
            conn,
            task_id,
            reason=reason,
            retry_at=20_100,
            expected_run_id=review.current_run_id,
            metadata={"worker_session_id": "review-session"},
        ) == (True, None)
        assert (_event_count(conn, task_id), _run_count(conn, task_id)) == counts

        clock["now"] = 20_101
        resumed = kb.claim_review_task(conn, task_id, claimer="reviewer:2")
        assert resumed is not None
        assert kb._transient_resume_session_id(
            task_id, board=kb.get_current_board(),
        ) == "review-session"
        assert kb.defer_review_task(
            conn,
            task_id,
            reason=reason,
            retry_at=clock["now"] + 100,
            expected_run_id=resumed.current_run_id,
            metadata={"worker_session_id": "review-session"},
        ) == (True, None)

        second = kb.get_task(conn, task_id)
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? "
            "AND kind = 'visual_review_deferred' ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        payload = json.loads(event["payload"])

        second_counts = (_event_count(conn, task_id), _run_count(conn, task_id))
        clock["now"] = 20_250  # requested time passed; effective backoff has not
        assert kb.defer_review_task(
            conn,
            task_id,
            reason=reason,
            retry_at=20_201,
            expected_run_id=resumed.current_run_id,
            metadata={"worker_session_id": "review-session"},
        ) == (True, None)
        assert (_event_count(conn, task_id), _run_count(conn, task_id)) == second_counts
        assert kb.get_task(conn, task_id).next_retry_at == 20_301

    assert second is not None and second.next_retry_at == 20_301
    assert payload["candidate_fingerprint"] == "abc1234"
    assert payload["gate"] == "visual_review"
    assert payload["failure_signature"] == "http:429"
    assert payload["recurrence"] == 2
    assert payload["backoff_seconds"] == 200


def test_dependency_block_requires_a_real_unfinished_parent(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        no_parent = kb.create_task(conn, title="no parent", assignee="worker")
        claimed = kb.claim_task(conn, no_parent)
        assert claimed is not None
        with pytest.raises(ValueError, match="link_tasks"):
            kb.block_task(
                conn,
                no_parent,
                reason="wait for something",
                kind="dependency",
                expected_run_id=claimed.current_run_id,
            )
        assert kb.get_task(conn, no_parent).status == "running"

        done_parent = kb.create_task(conn, title="done parent", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (done_parent,))
        done_child = kb.create_task(conn, title="done child", assignee="worker")
        done_claim = kb.claim_task(conn, done_child)
        assert done_claim is not None
        kb.link_tasks(conn, done_parent, done_child)
        with pytest.raises(ValueError, match="linked, unfinished parent"):
            kb.block_task(
                conn,
                done_child,
                reason="already satisfied",
                kind="dependency",
                expected_run_id=done_claim.current_run_id,
            )

        parent = kb.create_task(conn, title="real parent", assignee="worker")
        child = kb.create_task(conn, title="real child", assignee="worker")
        child_claim = kb.claim_task(conn, child)
        assert child_claim is not None
        kb.link_tasks(conn, parent, child)
        assert kb.block_task(
            conn,
            child,
            reason="waiting for real parent",
            kind="dependency",
            expected_run_id=child_claim.current_run_id,
        )
        assert kb.get_task(conn, child).status == "todo"


def test_strategy_required_resumes_the_exact_worker_session(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    with kb.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        task_id = kb.create_task(conn, title="change strategy", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer=f"{host}:guard")
        assert claimed is not None
        assert kb.heartbeat_worker(
            conn,
            task_id,
            expected_run_id=claimed.current_run_id,
            worker_session_id="same-strategy-session",
        )
        pid = 79991
        conn.execute("UPDATE tasks SET worker_pid = ? WHERE id = ?", (pid, task_id))
        conn.commit()
        kb._record_worker_exit(
            pid, kb.KANBAN_GUARDRAIL_HALT_EXIT_CODE << 8,
        )
        kb.detect_crashed_workers(conn)
        resumed = kb.claim_task(conn, task_id, claimer=f"{host}:resume")
        assert resumed is not None

    assert kb._transient_resume_session_id(
        task_id, board=kb.get_current_board(),
    ) == "same-strategy-session"


def test_unified_quota_deadline_persists_and_prefers_latest_source(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = int(time.time())
    routing_deadline = now + 120
    provider_deadline = now + 240
    routing = kanban_home / "state" / "ai-quota-routing.json"
    routing.parent.mkdir(parents=True, exist_ok=True)
    routing.write_text(json.dumps({
        "agent_cooldowns": {
            "claude1": {
                "dispatch_allowed": False,
                "cooldown_until_epoch": routing_deadline,
                "cooldown_until": dt.datetime.fromtimestamp(
                    routing_deadline, tz=dt.timezone.utc,
                ).isoformat(),
            },
        },
    }))
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ROUTING_PATH", str(routing))
    observed = dt.datetime.fromtimestamp(now, tz=dt.timezone.utc)
    reset = dt.datetime.fromtimestamp(provider_deadline, tz=dt.timezone.utc)
    kb._persist_provider_error_cooldown(
        "claude1", reset_at=reset, observed_at=observed, source="test",
    )

    assert kb.provider_cooldown_retry_at("claude1", now=now) == provider_deadline
    persisted = json.loads(
        (kanban_home / "state" / "provider-error-cooldowns.json").read_text()
    )["profiles"]["claude1"]
    assert persisted["cooldown_until_epoch"] == provider_deadline


def test_dispatch_tick_before_quota_deadline_does_not_mutate_ready_or_review(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    all_assignees_spawnable,
) -> None:
    now = int(time.time())
    deadline = now + 600
    routing = kanban_home / "state" / "ai-quota-routing.json"
    routing.parent.mkdir(parents=True, exist_ok=True)
    routing.write_text(json.dumps({
        "agent_cooldowns": {
            "claude1": {
                "dispatch_allowed": False,
                "cooldown_until_epoch": deadline,
                "cooldown_until": dt.datetime.fromtimestamp(
                    deadline, tz=dt.timezone.utc,
                ).isoformat(),
            },
        },
    }))
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ROUTING_PATH", str(routing))
    monkeypatch.setattr(kb, "_configured_handoff_routes", lambda: {})

    with kb.connect() as conn:
        ready_id = kb.create_task(conn, title="ready quota", assignee="claude1")
        review_id = kb.create_task(conn, title="review quota", assignee="builder")
        implementation = kb.claim_task(conn, review_id)
        assert implementation is not None
        assert kb.request_review(
            conn,
            review_id,
            summary="ready",
            reviewer="claude1",
            expected_run_id=implementation.current_run_id,
        )

        first = kb.dispatch_once(conn, dry_run=False)
        assert (ready_id, "provider_cooldown") in first.respawn_guarded
        assert (review_id, "provider_cooldown") in first.respawn_guarded
        for task_id, status in ((ready_id, "ready"), (review_id, "review")):
            task = kb.get_task(conn, task_id)
            assert task is not None
            assert (task.status, task.next_retry_at) == (status, deadline)
        before = {
            task_id: (
                _event_count(conn, task_id),
                _run_count(conn, task_id),
                kb.get_task(conn, task_id).next_retry_at,
            )
            for task_id in (ready_id, review_id)
        }

        second = kb.dispatch_once(conn, dry_run=False)
        assert second.respawn_guarded == []
        after = {
            task_id: (
                _event_count(conn, task_id),
                _run_count(conn, task_id),
                kb.get_task(conn, task_id).next_retry_at,
            )
            for task_id in (ready_id, review_id)
        }

    assert after == before
