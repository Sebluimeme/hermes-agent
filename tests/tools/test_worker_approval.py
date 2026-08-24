"""Kanban worker → gateway approval bridge (t_f98bc92d).

A Kanban worker is a separate OS process from the gateway, so it cannot use
``tools.approval``'s in-memory ``_gateway_queues`` / notify-callback
machinery. ``tools/worker_approval.py`` closes that gap with a durable file
transport. These tests cover the contract the task explicitly calls out:

* durable dedup — a retry of the SAME demand reattaches to one in-flight
  request instead of soliciting twice, even across process restarts;
* silence is never consent — a worker-side timeout resolves to refusal, and
  a decision written after that point can never flip it to approval;
* the gateway-side claim is atomic/idempotent, so two dispatch ticks (or two
  gateway processes) can never both prompt for the same request.
"""

from __future__ import annotations

import threading
import time

import pytest


@pytest.fixture(autouse=True)
def _isolated_state_dir(tmp_path, monkeypatch):
    """Point the worker-approval file store at a throwaway directory."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def test_request_id_is_stable_for_the_same_demand():
    from tools import worker_approval as wa

    a = wa._request_id("t_abc123", ["/repo/AGENTS.md", "/repo/CLAUDE.md"])
    b = wa._request_id("t_abc123", ["/repo/CLAUDE.md", "/repo/AGENTS.md"])  # order-independent
    c = wa._request_id("t_abc123", ["/repo/AGENTS.md"])
    d = wa._request_id("t_other", ["/repo/AGENTS.md", "/repo/CLAUDE.md"])

    assert a == b
    assert a != c
    assert a != d


def test_retry_reattaches_to_the_same_pending_request_without_a_second_solicitation():
    """A worker crash/respawn mid-wait must not double-prompt the operator."""
    from tools import worker_approval as wa

    task_id = "t_retry001"
    paths = ["/repo/AGENTS.md"]

    # First call: leader, blocks until resolved (long timeout so it's still
    # pending when the "retry" call arrives).
    results: dict[str, dict] = {}

    def _leader():
        results["leader"] = wa.request_decision(
            task_id=task_id, title="t", description="d", paths=paths,
            timeout_seconds=10,
        )

    leader_thread = threading.Thread(target=_leader)
    leader_thread.start()

    # Wait for the request file to exist (leader has written it).
    request_id = wa._request_id(task_id, paths)
    path = wa._request_path(request_id)
    for _ in range(200):
        if path.exists():
            break
        time.sleep(0.01)
    assert path.exists()
    first_snapshot = wa._read(path)

    # Simulate a worker retry for the exact same demand: it must reattach,
    # not create a competing request or reset the clock.
    def _retry():
        results["retry"] = wa.request_decision(
            task_id=task_id, title="t", description="d", paths=paths,
            timeout_seconds=10,
        )

    retry_thread = threading.Thread(target=_retry)
    retry_thread.start()
    time.sleep(0.1)

    second_snapshot = wa._read(path)
    assert second_snapshot["request_id"] == first_snapshot["request_id"]
    assert second_snapshot["created_at"] == first_snapshot["created_at"]
    assert second_snapshot["expires_at"] == first_snapshot["expires_at"]

    # A single decision resolves BOTH waiters — one solicitation, one answer.
    assert wa.write_decision(request_id, "once") is True
    leader_thread.join(timeout=5)
    retry_thread.join(timeout=5)

    assert results["leader"] == {"resolved": True, "choice": "once", "reason": None}
    assert results["retry"] == {"resolved": True, "choice": "once", "reason": None}


def test_worker_side_timeout_denies_and_a_late_decision_cannot_override_it():
    from tools import worker_approval as wa

    task_id = "t_timeout001"
    paths = ["/repo/AGENTS.md"]

    outcome = wa.request_decision(
        task_id=task_id, title="t", description="d", paths=paths,
        timeout_seconds=0.05,
    )
    assert outcome["resolved"] is False
    assert outcome["choice"] is None
    assert "Silence is not consent" in outcome["reason"] or "silence is not consent" in outcome["reason"]

    # A tardy human click (or a bug in the bridge) must never flip a closed
    # request to an approval after the caller already treated it as denied.
    request_id = wa._request_id(task_id, paths)
    assert wa.write_decision(request_id, "once") is False
    final = wa._read(wa._request_path(request_id))
    assert final["decision"] == "timeout"


def test_missing_task_id_or_paths_is_never_a_solicitation():
    from tools import worker_approval as wa

    assert wa.request_decision(
        task_id="", title="t", description="d", paths=["/repo/AGENTS.md"],
    ) == {"resolved": False, "choice": None, "reason": "missing task_id/paths"}
    assert wa.request_decision(
        task_id="t_x", title="t", description="d", paths=[],
    ) == {"resolved": False, "choice": None, "reason": "missing task_id/paths"}
    assert list(wa._state_dir().glob("*.json")) == []


def test_claim_for_dispatch_is_atomic_and_single_use():
    from tools import worker_approval as wa

    task_id = "t_claim001"
    paths = ["/repo/AGENTS.md"]

    def _leader():
        wa.request_decision(
            task_id=task_id, title="t", description="d", paths=paths,
            timeout_seconds=10,
        )

    thread = threading.Thread(target=_leader)
    thread.start()
    request_id = wa._request_id(task_id, paths)
    for _ in range(200):
        if wa._request_path(request_id).exists():
            break
        time.sleep(0.01)

    pending = wa.list_pending_requests()
    assert [p["request_id"] for p in pending] == [request_id]

    # First claim wins; a second, concurrent claim (another tick or another
    # gateway process racing the same pending file) must lose.
    assert wa.claim_for_dispatch(request_id) is True
    assert wa.claim_for_dispatch(request_id) is False

    # Claimed requests drop out of the pending list — no re-dispatch.
    assert wa.list_pending_requests() == []

    wa.write_decision(request_id, "deny")
    thread.join(timeout=5)


def test_second_ask_after_resolution_creates_a_new_request(monkeypatch):
    """Each write needs fresh consent — dedup only covers a demand still in
    flight, not a brand-new ask that happens to share task+paths."""
    from tools import worker_approval as wa

    task_id = "t_fresh002"
    paths = ["/repo/AGENTS.md"]
    request_id = wa._request_id(task_id, paths)

    def _first():
        return wa.request_decision(
            task_id=task_id, title="t", description="first", paths=paths,
            timeout_seconds=10,
        )

    thread = threading.Thread(target=lambda: _first())
    thread.start()
    for _ in range(200):
        if wa._request_path(request_id).exists():
            break
        time.sleep(0.01)
    wa.claim_for_dispatch(request_id)
    wa.write_decision(request_id, "deny")
    thread.join(timeout=5)

    first_snapshot = wa._read(wa._request_path(request_id))
    assert first_snapshot["decision"] == "deny"
    # The dispatch marker from the first ask must not silently block the
    # second ask's delivery.
    assert wa._dispatch_marker(request_id).exists()

    second = wa.request_decision(
        task_id=task_id, title="t", description="second", paths=paths,
        timeout_seconds=0.05,
    )
    second_snapshot = wa._read(wa._request_path(request_id))
    assert second_snapshot["description"] == "second"
    assert second_snapshot["created_at"] > first_snapshot["created_at"]
    assert not wa._dispatch_marker(request_id).exists()
    assert second["resolved"] is False  # nobody answered the new ask either
