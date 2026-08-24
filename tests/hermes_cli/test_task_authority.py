"""Tests for the LOT 4 authority/contradiction gate (hermes_cli/task_authority.py).

Pure-function tests only — no DB, no filesystem. Confirms Kanban SQLite and
TASKS.md stay two independent authorities and that a mismatch between them
is always named, never silently resolved by picking one side.
"""
from __future__ import annotations

import pytest

from hermes_cli.task_authority import (
    KANBAN_STATUSES,
    TASKS_MD_STATUSES,
    check_authority_contradiction,
)


class TestNoContradiction:
    def test_both_unknown_is_not_a_contradiction(self):
        assert check_authority_contradiction(kanban_status=None, tasks_md_status=None) is None

    def test_kanban_known_tasks_md_unknown_is_not_a_contradiction(self):
        # Only one authority present: nothing to compare against.
        assert (
            check_authority_contradiction(kanban_status="done", tasks_md_status=None)
            is None
        )

    def test_tasks_md_known_kanban_unknown_is_not_a_contradiction(self):
        assert (
            check_authority_contradiction(kanban_status=None, tasks_md_status="DONE")
            is None
        )

    @pytest.mark.parametrize(
        "kanban_status,tasks_md_status",
        [
            ("done", "DONE"),
            ("running", "IN_PROGRESS"),
            ("todo", "TODO"),
            ("blocked", "BLOCKED"),
            ("review", "NEEDS_DECISION"),
            ("changes_requested", "TODO"),
        ],
    )
    def test_agreeing_open_or_closed_states_are_not_contradictions(
        self, kanban_status, tasks_md_status
    ):
        assert (
            check_authority_contradiction(
                kanban_status=kanban_status, tasks_md_status=tasks_md_status
            )
            is None
        )


class TestContradiction:
    def test_kanban_done_but_tasks_md_still_open_is_flagged(self):
        result = check_authority_contradiction(kanban_status="done", tasks_md_status="IN_PROGRESS")
        assert result is not None
        assert result.kanban_status == "done"
        assert result.tasks_md_status == "IN_PROGRESS"
        assert "TASKS.md" in result.message

    @pytest.mark.parametrize("open_status", ["TODO", "IN_PROGRESS", "BLOCKED", "NEEDS_DECISION"])
    def test_kanban_done_vs_every_open_tasks_md_status_is_flagged(self, open_status):
        result = check_authority_contradiction(kanban_status="done", tasks_md_status=open_status)
        assert result is not None

    def test_kanban_archived_but_tasks_md_still_open_is_flagged(self):
        # archived is terminal too, even though it isn't a success outcome.
        result = check_authority_contradiction(kanban_status="archived", tasks_md_status="TODO")
        assert result is not None

    def test_tasks_md_done_but_kanban_still_open_is_flagged(self):
        result = check_authority_contradiction(kanban_status="running", tasks_md_status="DONE")
        assert result is not None
        assert "Kanban" in result.message

    @pytest.mark.parametrize(
        "open_kanban_status",
        ["triage", "todo", "scheduled", "ready", "running", "review", "changes_requested", "blocked"],
    )
    def test_tasks_md_done_vs_every_open_kanban_status_is_flagged(self, open_kanban_status):
        result = check_authority_contradiction(
            kanban_status=open_kanban_status, tasks_md_status="DONE"
        )
        assert result is not None

    def test_contradiction_never_declares_a_winner(self):
        # The message must not tell the caller which side to trust — only
        # that they disagree and need explicit reconciliation.
        result = check_authority_contradiction(kanban_status="done", tasks_md_status="TODO")
        assert result is not None
        lowered = result.message.lower()
        assert "do not assume" in lowered


class TestUnknownStatusesRaise:
    def test_unknown_kanban_status_raises(self):
        with pytest.raises(ValueError):
            check_authority_contradiction(kanban_status="deleted", tasks_md_status="DONE")

    def test_unknown_tasks_md_status_raises(self):
        with pytest.raises(ValueError):
            check_authority_contradiction(kanban_status="done", tasks_md_status="ARCHIVED")


def test_status_vocabularies_match_the_documented_lifecycles():
    # Regression guard: if kanban_db.py or hermes_next_task.py ever grow a
    # new status, this module's tables must be updated deliberately rather
    # than silently drifting out of sync.
    assert set(KANBAN_STATUSES) == {
        "triage", "todo", "scheduled", "ready", "running", "blocked",
        "review", "changes_requested", "done", "archived",
    }
    assert set(TASKS_MD_STATUSES) == {
        "TODO", "IN_PROGRESS", "DONE", "BLOCKED", "NEEDS_DECISION",
    }
