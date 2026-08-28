from __future__ import annotations

import sqlite3
import subprocess
import time
import unittest
from unittest.mock import patch

import pytest

from hermes_cli import worker_contracts as wc


class WorkerContractsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, status TEXT, current_run_id INTEGER)")
        self.conn.execute("CREATE TABLE task_events (id INTEGER PRIMARY KEY, task_id TEXT, kind TEXT, payload TEXT, created_at INTEGER)")
        wc.ensure_schema(self.conn)
        self.conn.execute("INSERT INTO tasks VALUES ('t1','running',7)")
        self.kills: list[tuple[int, int]] = []

    def tearDown(self) -> None:
        self.conn.close()

    def register(self, pid: int = 42) -> None:
        with patch.object(wc, "proc_start_identity", return_value="start-42"), patch.object(wc, "process_group", return_value=42):
            self.assertTrue(wc.register(
                self.conn, task_id="t1", run_id=7, profile="claude1", model="model-x",
                pid=pid, workspace_path="/tmp/work", max_runtime_seconds=300,
                max_retries=2, now=100,
            ))

    def test_legitimate_worker_with_descriptive_checkpoint_is_preserved(self) -> None:
        self.register()
        self.conn.execute("INSERT INTO task_events VALUES (1,'t1','heartbeat','{\"note\":\"tests en cours\"}',95)")
        with patch.object(wc, "proc_start_identity", return_value="start-42"):
            self.assertEqual(wc.reconcile(self.conn, now=100), [])

    def test_new_worker_gets_bounded_grace_before_first_checkpoint(self) -> None:
        self.register()
        with patch.object(wc, "proc_start_identity", return_value="start-42"):
            # Uncapped Kanban work is required to heartbeat at least hourly;
            # a ten-minute descriptive-checkpoint guard used to SIGTERM a
            # healthy long-running worker before that contract elapsed.
            self.assertEqual(wc.CHECKPOINT_STALE_SECONDS, 3600)
            self.assertEqual(wc.reconcile(self.conn, now=100 + 601), [])
            self.assertEqual(wc.reconcile(self.conn, now=100 + wc.CHECKPOINT_STALE_SECONDS), [])

    def test_terminal_card_stops_only_recorded_worker(self) -> None:
        self.register()
        self.conn.execute("UPDATE tasks SET status='done' WHERE id='t1'")
        with patch.object(wc, "proc_start_identity", return_value="start-42"), patch.object(wc, "process_group", return_value=42), patch.object(wc, "process_alive", return_value=True):
            with patch.object(wc.os, "kill", side_effect=lambda pid, sig: self.kills.append((pid, sig))):
                actions = wc.reconcile(self.conn, now=101)
        self.assertEqual(actions, [{"task_id": "t1", "pid": 42, "reason": "task_done", "stopped": True}])
        self.assertEqual(self.kills[0][0], -42)

    def test_review_handoff_stops_implementer_contract(self) -> None:
        self.register()
        self.conn.execute("UPDATE tasks SET status='review' WHERE id='t1'")
        with patch.object(wc, "proc_start_identity", return_value="start-42"), patch.object(wc, "process_group", return_value=42), patch.object(wc, "process_alive", return_value=True):
            with patch.object(wc.os, "kill", side_effect=lambda pid, sig: self.kills.append((pid, sig))):
                actions = wc.reconcile(self.conn, now=101)
        self.assertEqual(actions, [{"task_id": "t1", "pid": 42, "reason": "task_review", "stopped": True}])
        self.assertEqual(self.kills, [(-42, wc.signal.SIGTERM)])

    def test_exit_barrier_persists_and_force_stops_after_grace(self) -> None:
        self.register()
        self.conn.execute(
            "UPDATE worker_contracts SET state='stopped', stopped_at=101 WHERE task_id='t1'"
        )
        with patch.object(wc, "proc_start_identity", return_value="start-42"), patch.object(wc, "process_group", return_value=42):
            with patch.object(wc.os, "kill", side_effect=lambda pid, sig: self.kills.append((pid, sig))):
                fresh = wc.live_exit_barriers(self.conn, now=101 + wc.EXIT_GRACE_SECONDS - 1)
                expired = wc.live_exit_barriers(self.conn, now=101 + wc.EXIT_GRACE_SECONDS)
        self.assertEqual(fresh[0]["forced"], False)
        self.assertEqual(expired[0]["forced"], True)
        self.assertEqual(self.kills, [(-42, wc.signal.SIGKILL)])

    def test_exit_barrier_releases_only_after_exact_identity_is_gone(self) -> None:
        self.register()
        self.conn.execute(
            "UPDATE worker_contracts SET state='stopped', stopped_at=101 WHERE task_id='t1'"
        )
        with patch.object(wc, "proc_start_identity", return_value=None):
            self.assertEqual(wc.live_exit_barriers(self.conn, now=102), [])

    def test_pid_reuse_identity_mismatch_is_never_killed(self) -> None:
        self.register()
        with patch.object(wc, "proc_start_identity", return_value="new-process"):
            actions = wc.reconcile(self.conn, now=101)
        self.assertEqual(actions[0]["reason"], "pid_identity_mismatch")
        self.assertFalse(actions[0]["stopped"])
        self.assertEqual(self.kills, [])

    def test_missing_descriptive_checkpoint_stops_once_and_is_durable(self) -> None:
        self.register()
        with patch.object(wc, "proc_start_identity", return_value="start-42"), patch.object(wc, "process_group", return_value=42), patch.object(wc, "process_alive", return_value=True):
            with patch.object(wc.os, "kill", side_effect=lambda pid, sig: self.kills.append((pid, sig))):
                first = wc.reconcile(self.conn, now=100 + wc.CHECKPOINT_STALE_SECONDS + 1)
                second = wc.reconcile(self.conn, now=100 + wc.CHECKPOINT_STALE_SECONDS + 2)
        self.assertEqual(first[0]["reason"], "checkpoint_stale")
        self.assertTrue(first[0]["stopped"])
        self.assertEqual(second, [])
        self.assertEqual(len(self.kills), 1)

    def test_unknown_manual_session_is_not_discovered_or_touched(self) -> None:
        self.assertEqual(wc.reconcile(self.conn, now=100), [])
        self.assertEqual(self.kills, [])

    def test_contract_persists_attempt_bounds_and_identity(self) -> None:
        self.register()
        row = self.conn.execute("SELECT model,start_identity,max_runtime_seconds,max_retries,state FROM worker_contracts").fetchone()
        self.assertEqual(tuple(row), ("model-x", "start-42", 300, 2, "active"))

    @pytest.mark.live_system_guard_bypass
    def test_controlled_orphan_process_is_stopped_without_scanning_external_pids(self) -> None:
        worker = subprocess.Popen(["sleep", "30"], start_new_session=True)
        self.addCleanup(lambda: worker.poll() is None and worker.kill())
        self.assertTrue(wc.register(
            self.conn, task_id="t1", run_id=7, profile="claude1", model=None,
            pid=worker.pid, workspace_path="/tmp/work", max_runtime_seconds=300,
            max_retries=2, now=int(time.time()),
        ))
        self.conn.execute("UPDATE tasks SET status='done' WHERE id='t1'")
        actions = wc.reconcile(self.conn, now=int(time.time()))
        worker.wait(timeout=3)
        self.assertEqual(actions[0]["reason"], "task_done")
        self.assertTrue(actions[0]["stopped"])


if __name__ == "__main__":
    unittest.main()
