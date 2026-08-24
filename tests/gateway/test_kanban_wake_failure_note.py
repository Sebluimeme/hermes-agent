import unittest
from types import SimpleNamespace

from gateway.kanban_watchers import _failure_note


class FakeKb:
    def __init__(self, run): self._run = run
    def get_run(self, conn, run_id): return self._run


class FailureNoteTests(unittest.TestCase):
    def test_carries_duration_and_error(self):
        run = SimpleNamespace(
            started_at=1000, ended_at=1612, profile="claude2",
            error=("worker exited cleanly (rc=0) without calling "
                   "kanban_complete or kanban_block — protocol violation."),
        )
        ev = SimpleNamespace(kind="crashed", run_id=7,
                             payload={"incident_category": "workflow_bug"})
        note = _failure_note(FakeKb(run), None, [ev])
        self.assertIn("ran 612s", note)          # the timeout wall is visible
        self.assertIn("profile=claude2", note)
        self.assertIn("protocol violation", note)

    def test_no_failure_event_yields_empty(self):
        ev = SimpleNamespace(kind="completed", run_id=1, payload={})
        self.assertEqual(_failure_note(FakeKb(None), None, [ev]), "")

    def test_never_raises_on_broken_input(self):
        class Boom:
            def get_run(self, *a): raise RuntimeError("db gone")
        ev = SimpleNamespace(kind="crashed", run_id=1, payload={})
        self.assertEqual(_failure_note(Boom(), None, [ev]), "")

    def test_uses_last_failure_event(self):
        run = SimpleNamespace(started_at=0, ended_at=20, profile="claude1",
                              error="second failure")
        evs = [SimpleNamespace(kind="crashed", run_id=1, payload={}),
               SimpleNamespace(kind="timed_out", run_id=2, payload={})]
        self.assertIn("second failure", _failure_note(FakeKb(run), None, evs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
