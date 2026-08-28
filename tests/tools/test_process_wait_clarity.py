"""Tests for process wait timeout-result clarity (not-an-error semantics)."""

import time

import pytest

from tools.process_registry import ProcessRegistry


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    return ProcessRegistry()


def _spawn_sleeper(registry, notify=False):
    session = registry.spawn_local("sleep 30", cwd="/tmp", task_id="t-waitclar")
    session.notify_on_complete = notify
    return session.id


class TestWaitTimeoutClarity:
    def test_wait_timeout_marks_process_running(self, registry):
        sid = _spawn_sleeper(registry)
        try:
            r = registry.wait(sid, timeout=1)
            assert r["status"] == "timeout"
            assert r["process_running"] is True
            assert "not an error" in r["timeout_note"]
            assert "Uptime" in r["timeout_note"]
        finally:
            registry.kill_process(sid)

    def test_wait_timeout_suggests_notify_when_unset(self, registry):
        sid = _spawn_sleeper(registry, notify=False)
        try:
            r = registry.wait(sid, timeout=1)
            assert "notify_on_complete=true" in r["timeout_note"]
        finally:
            registry.kill_process(sid)

    def test_wait_timeout_defers_to_notify_when_set(self, registry):
        sid = _spawn_sleeper(registry, notify=True)
        try:
            r = registry.wait(sid, timeout=1)
            assert "you will be notified on exit" in r["timeout_note"]
        finally:
            registry.kill_process(sid)

    def test_clamped_wait_keeps_clamp_note_and_running_semantics(self, registry, monkeypatch):
        monkeypatch.setenv("TERMINAL_TIMEOUT", "1")
        sid = _spawn_sleeper(registry)
        try:
            r = registry.wait(sid, timeout=600)
            assert r["status"] == "timeout"
            assert "clamped" in r["timeout_note"]
            assert "not an error" in r["timeout_note"]
            assert r["process_running"] is True
        finally:
            registry.kill_process(sid)

    def test_exited_process_unaffected(self, registry):
        session = registry.spawn_local("true", cwd="/tmp", task_id="t-waitclar")
        r = registry.wait(session.id, timeout=10)
        assert r["status"] == "exited"
        assert "process_running" not in r

    def test_kanban_wait_is_coalesced_and_keeps_claim_alive(
        self, registry, monkeypatch
    ):
        """A worker's short model wait must not cause another agent turn."""
        monkeypatch.setenv("TERMINAL_TIMEOUT", "1")
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t-worker")
        monkeypatch.setenv("HERMES_KANBAN_PROCESS_WAIT_SECONDS", "2")
        heartbeats = []
        monkeypatch.setattr(
            "tools.kanban_tools.heartbeat_current_worker_from_env",
            lambda: heartbeats.append(time.monotonic()) or True,
        )
        session = registry.spawn_local("sleep 1.2", cwd="/tmp", task_id="t-worker")
        try:
            r = registry.wait(session.id, timeout=1)
            assert r["status"] == "exited"
            assert heartbeats
        finally:
            registry.kill_process(session.id)

    def test_kanban_coalesced_wait_remains_interruptible(
        self, registry, monkeypatch
    ):
        monkeypatch.setenv("TERMINAL_TIMEOUT", "1")
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t-worker")
        monkeypatch.setenv("HERMES_KANBAN_PROCESS_WAIT_SECONDS", "30")
        monkeypatch.setattr(
            "tools.kanban_tools.heartbeat_current_worker_from_env", lambda: True
        )
        interrupt_polls = []

        def interrupt_after_first_poll():
            interrupt_polls.append(True)
            return len(interrupt_polls) > 1

        monkeypatch.setattr(
            "tools.interrupt.is_interrupted", interrupt_after_first_poll
        )
        session = registry.spawn_local("sleep 30", cwd="/tmp", task_id="t-worker")
        started = time.monotonic()
        try:
            r = registry.wait(session.id, timeout=1)
            assert r["status"] == "interrupted"
            assert time.monotonic() - started < 3
        finally:
            registry.kill_process(session.id)
