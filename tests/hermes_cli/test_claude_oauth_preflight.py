from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

import hermes_cli.kanban_db as kb
from hermes_cli import claude_oauth_preflight as oauth


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_oauth_500_requires_reconnection_without_exposing_profile_data(tmp_path, monkeypatch):
    config_dir = tmp_path / ".claude2"
    config_dir.mkdir()
    (config_dir / ".claude.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(oauth, "CLAUDE2_CONFIG_DIR", config_dir)
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return CompletedProcess(
            command,
            1,
            stdout="HTTP 500: OAuth session expired and could not be refreshed",
            stderr="",
        )

    result = oauth.probe_claude2_oauth(run_fn=fake_run, environ={"PATH": "/bin"})

    assert not result.ok
    assert result.reconnect_required
    assert result.reason == "OAuth Claude 2 expiré ou déconnecté"
    assert captured["command"] == oauth.canary_command()
    assert captured["env"]["CLAUDE_CONFIG_DIR"] == str(config_dir)
    assert "PONG" in captured["command"][-1]


def test_dispatch_blocks_claude2_before_worker_spawn_on_expired_oauth(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    routing = kanban_home / "state" / "ai-quota-routing.json"
    routing.parent.mkdir(parents=True)
    routing.write_text(json.dumps({
        "agent_cooldowns": {
            "claude2": {"dispatch_allowed": True, "preflight_required": False},
        },
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ROUTING_PATH", str(routing))
    monkeypatch.setattr(
        "hermes_cli.claude_oauth_preflight.probe_claude2_oauth",
        lambda: oauth.ProbeResult(False, True, "OAuth Claude 2 expiré ou déconnecté"),
    )
    spawned = []

    def should_not_spawn(*args, **kwargs):
        spawned.append((args, kwargs))
        return 1234

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="OAuth gate", assignee="claude2")
        outcome = kb.dispatch_once(conn, spawn_fn=should_not_spawn)
        task = kb.get_task(conn, task_id)
        run = conn.execute(
            "SELECT summary FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()

    assert outcome.oauth_blocked == [task_id]
    assert spawned == []
    assert task is not None
    assert task.status == "blocked"
    assert run["summary"] == (
        "reconnexion Claude 2 requise — OAuth expiré ou déconnecté; aucun worker n’a été lancé."
    )


def test_noncredential_canary_failure_reroutes_without_human_block(
    kanban_home, all_assignees_spawnable, configured_handoff_routes, monkeypatch,
):
    routing = kanban_home / "state" / "ai-quota-routing.json"
    routing.parent.mkdir(parents=True)
    routing.write_text(json.dumps({
        "agent_cooldowns": {
            "claude2": {"dispatch_allowed": True, "preflight_required": False},
            "claude1": {"dispatch_allowed": True, "preflight_required": False},
        },
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ROUTING_PATH", str(routing))
    monkeypatch.setattr(
        "hermes_cli.claude_oauth_preflight.probe_claude2_oauth",
        lambda: oauth.ProbeResult(False, False, "canari OAuth Claude 2 échoué (code 1)"),
    )
    spawned = []

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="provider fallback", assignee="claude2", routing_tier="complex",
        )
        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            ("provider rate-limited (quota wall)", task_id),
        )
        conn.commit()
        outcome = kb.dispatch_once(
            conn, spawn_fn=lambda *args, **kwargs: spawned.append((args, kwargs)),
        )
        task = kb.get_task(conn, task_id)
        runs = conn.execute(
            "SELECT COUNT(*) AS count FROM task_runs WHERE task_id = ?", (task_id,),
        ).fetchone()["count"]

    assert outcome.oauth_rerouted == [task_id]
    assert outcome.oauth_blocked == []
    assert spawned == []
    assert task is not None
    assert task.status == "ready"
    assert task.assignee == "claude1"
    assert runs == 0


def test_fresh_noncredential_canary_failure_reroutes_without_prior_worker_error(
    kanban_home, all_assignees_spawnable, configured_handoff_routes, monkeypatch,
):
    """A failed live canary is enough proof to leave Claude 2.

    Regression for t_c9b92dac run 1240: after a human unblocked a card, the
    Claude 2 canary failed before a worker existed. The old path demanded an
    earlier worker error too and converted the card to block_loop_detected
    instead of routing it to Claude 1.
    """
    routing = kanban_home / "state" / "ai-quota-routing.json"
    routing.parent.mkdir(parents=True)
    routing.write_text(json.dumps({
        "agent_cooldowns": {
            "claude2": {"dispatch_allowed": True, "preflight_required": False},
            "claude1": {"dispatch_allowed": True, "preflight_required": False},
        },
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ROUTING_PATH", str(routing))
    monkeypatch.setattr(
        "hermes_cli.claude_oauth_preflight.probe_claude2_oauth",
        lambda: oauth.ProbeResult(False, False, "canari OAuth Claude 2 échoué (code 1)"),
    )

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="fresh provider fallback", assignee="claude2", routing_tier="complex",
        )
        outcome = kb.dispatch_once(conn, spawn_fn=lambda *args, **kwargs: 4242)
        task = kb.get_task(conn, task_id)

    assert outcome.oauth_rerouted == [task_id]
    assert outcome.oauth_blocked == []
    assert task is not None
    assert task.status == "ready"
    assert task.assignee == "claude1"


# --- Transition-gated canary (t_b0bc4445 LOT 2) ----------------------------
# Table tests for the pure decision function, then integration tests proving
# the paid canary is skipped on steady-state dispatch and re-fires on each
# of the four fixed transitions.

_OK_FINGERPRINT = {
    "last_status": "ok",
    "credentials_mtime": 100.0,
    "model_override": None,
    "provider_override": None,
}


@pytest.mark.parametrize(
    "stored, model_override, provider_override, credentials_mtime, task_failure_error, "
    "expected_required, expected_trigger",
    [
        # (a) route newly activated -- nothing persisted yet.
        (None, None, None, 100.0, None, True, "route_newly_activated"),
        # Steady state: identical fingerprint, last check was ok -> skip.
        (_OK_FINGERPRINT, None, None, 100.0, None, False, "steady_state"),
        # (c) previous provider error recorded at the route level.
        (
            {**_OK_FINGERPRINT, "last_status": "error"},
            None, None, 100.0, None, True, "previous_provider_error",
        ),
        # (b) OAuth reconnected -- credentials mtime moved.
        (_OK_FINGERPRINT, None, None, 200.0, None, True, "oauth_reconnected"),
        # (d) model changed.
        (_OK_FINGERPRINT, "claude-opus-4-6", None, 100.0, None, True, "model_or_provider_changed"),
        # (d) provider changed.
        (_OK_FINGERPRINT, None, "anthropic", 100.0, None, True, "model_or_provider_changed"),
        # (c) previous provider error surfaced on THIS task's own last failure,
        # even though the route-level fingerprint still reads ok/unchanged.
        (_OK_FINGERPRINT, None, None, 100.0, "429 rate limit exceeded", True, "previous_provider_error"),
        # An unrelated task failure (e.g. a timeout) must NOT force a canary.
        (_OK_FINGERPRINT, None, None, 100.0, "worker timed out after 300s", False, "steady_state"),
    ],
)
def test_oauth_canary_transition_required_table(
    stored, model_override, provider_override, credentials_mtime,
    task_failure_error, expected_required, expected_trigger,
):
    required, trigger = kb.oauth_canary_transition_required(
        "claude2",
        model_override=model_override,
        provider_override=provider_override,
        credentials_mtime=credentials_mtime,
        stored=stored,
        task_failure_error=task_failure_error,
    )
    assert (required, trigger) == (expected_required, expected_trigger)


def _seed_quota_routing_allows_claude2(kanban_home, monkeypatch):
    routing = kanban_home / "state" / "ai-quota-routing.json"
    routing.parent.mkdir(parents=True, exist_ok=True)
    routing.write_text(json.dumps({
        "agent_cooldowns": {
            "claude2": {"dispatch_allowed": True, "preflight_required": False},
        },
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ROUTING_PATH", str(routing))


def test_oauth_canary_skipped_on_steady_state_dispatch(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """A second normal dispatch tick must NOT spend another paid canary."""
    _seed_quota_routing_allows_claude2(kanban_home, monkeypatch)
    monkeypatch.setattr(oauth, "credentials_fingerprint", lambda: 111.0)
    calls = []

    def fake_probe():
        calls.append(1)
        return oauth.ProbeResult(True, False, "canari PONG réussi")

    monkeypatch.setattr(
        "hermes_cli.claude_oauth_preflight.probe_claude2_oauth", fake_probe,
    )
    spawned = []

    def fake_spawn(*args, **kwargs):
        spawned.append(1)
        return 4321

    with kb.connect() as conn:
        kb.create_task(conn, title="first", assignee="claude2")
        kb.dispatch_once(conn, spawn_fn=fake_spawn)
        kb.create_task(conn, title="second", assignee="claude2")
        kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert len(calls) == 1, "canary must only run once (route newly activated), not on every tick"
    assert len(spawned) == 2


def test_oauth_canary_rechecked_when_model_override_changes(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    _seed_quota_routing_allows_claude2(kanban_home, monkeypatch)
    monkeypatch.setattr(oauth, "credentials_fingerprint", lambda: 111.0)
    calls = []

    def fake_probe():
        calls.append(1)
        return oauth.ProbeResult(True, False, "canari PONG réussi")

    monkeypatch.setattr(
        "hermes_cli.claude_oauth_preflight.probe_claude2_oauth", fake_probe,
    )

    with kb.connect() as conn:
        kb.create_task(conn, title="first", assignee="claude2")
        kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 1)
        kb.create_task(
            conn, title="second", assignee="claude2",
            model_override="claude-opus-4-6",
        )
        kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 1)

    assert len(calls) == 2, "a model_override change must force a fresh canary"


def test_oauth_canary_rechecked_after_credentials_reconnect(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    _seed_quota_routing_allows_claude2(kanban_home, monkeypatch)
    mtime = {"value": 111.0}
    monkeypatch.setattr(oauth, "credentials_fingerprint", lambda: mtime["value"])
    calls = []

    def fake_probe():
        calls.append(1)
        return oauth.ProbeResult(True, False, "canari PONG réussi")

    monkeypatch.setattr(
        "hermes_cli.claude_oauth_preflight.probe_claude2_oauth", fake_probe,
    )

    with kb.connect() as conn:
        kb.create_task(conn, title="first", assignee="claude2")
        kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 1)
        mtime["value"] = 222.0  # simulates a human re-login
        kb.create_task(conn, title="second", assignee="claude2")
        kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 1)

    assert len(calls) == 2, "a credentials mtime change (reconnect) must force a fresh canary"


def test_oauth_canary_rechecked_after_previous_route_error_then_recovers(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """A failed canary blocks without looping, then a human unblock is the
    next transition (recovering from a previous provider error) -- exactly
    one extra canary is spent to confirm recovery, not one per tick."""
    _seed_quota_routing_allows_claude2(kanban_home, monkeypatch)
    monkeypatch.setattr(oauth, "credentials_fingerprint", lambda: 111.0)
    outcomes = iter([
        oauth.ProbeResult(False, True, "OAuth Claude 2 expiré ou déconnecté"),
        oauth.ProbeResult(True, False, "canari PONG réussi"),
    ])
    calls = []

    def fake_probe():
        calls.append(1)
        return next(outcomes)

    monkeypatch.setattr(
        "hermes_cli.claude_oauth_preflight.probe_claude2_oauth", fake_probe,
    )
    spawned = []

    def fake_spawn(*args, **kwargs):
        spawned.append(1)
        return 1

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="first", assignee="claude2")
        out1 = kb.dispatch_once(conn, spawn_fn=fake_spawn)
        assert out1.oauth_blocked == [task_id]

        # Re-dispatching a still-blocked task must not spend a canary --
        # it's simply not in the ready/review lanes.
        out_noop = kb.dispatch_once(conn, spawn_fn=fake_spawn)
        assert out_noop.oauth_blocked == []
        assert len(calls) == 1

        assert kb.unblock_task(conn, task_id)
        out2 = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert out2.oauth_blocked == []
    assert spawned == [1]
    assert len(calls) == 2
