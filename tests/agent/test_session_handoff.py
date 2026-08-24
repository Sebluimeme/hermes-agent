"""Tests for the deterministic session-relay mechanism (LOT 3).

Covers:
  - the summary-vs-reset-vs-none decision table,
  - checkpoint writing to TASKS.md (creation, append, idempotency),
  - Topic/session identity preservation across summary and reset relays,
  - the invariant that a reset relay is rejected if it tries to move the
    Topic identity (the exact failure mode this module exists to prevent).
"""

from __future__ import annotations

import pytest

from agent.session_handoff import (
    DURABLE_TRANSITIONS,
    SUMMARY_ELIGIBLE_TRANSITIONS,
    RelayResult,
    SessionIdentity,
    decide_relay_action,
    perform_session_relay,
    write_relay_checkpoint,
)


# ---------------------------------------------------------------------------
# Decision table: summary vs reset vs none
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "context_tokens,threshold_tokens,transition_reason,expected",
    [
        # No threshold crossed, no transition -> nothing happens.
        (1_000, 100_000, None, "none"),
        (0, 100_000, None, "none"),
        # Threshold crossed, no durable transition -> in-place summary.
        (100_000, 100_000, None, "summary"),
        (250_000, 100_000, None, "summary"),
        # Durable transitions always force a reset, even with an empty
        # context (small context does not make a stale model/provider or
        # corrupted state safe).
        (0, 100_000, "model_change", "reset"),
        (0, 100_000, "provider_change", "reset"),
        (0, 100_000, "corrupted_context", "reset"),
        # Durable transition wins even when the threshold is ALSO crossed.
        (999_999, 100_000, "model_change", "reset"),
        # Summary-eligible transitions get a checkpointed summary, never a
        # reset, even under threshold.
        (0, 100_000, "explicit_new_topic_request", "summary"),
        (0, 100_000, "idle_boundary", "summary"),
        # Unrecognized transition reason with no threshold crossed -> none
        # (unknown reasons are not silently promoted to durable).
        (0, 100_000, "some_unknown_reason", "none"),
        # threshold_tokens <= 0 disables the threshold branch entirely.
        (10_000_000, 0, None, "none"),
        (10_000_000, -1, None, "none"),
    ],
)
def test_decide_relay_action_table(context_tokens, threshold_tokens, transition_reason, expected):
    assert (
        decide_relay_action(
            context_tokens=context_tokens,
            threshold_tokens=threshold_tokens,
            transition_reason=transition_reason,
        )
        == expected
    )


def test_durable_and_summary_eligible_sets_are_disjoint():
    assert DURABLE_TRANSITIONS.isdisjoint(SUMMARY_ELIGIBLE_TRANSITIONS)


# ---------------------------------------------------------------------------
# Checkpoint writing
# ---------------------------------------------------------------------------

def test_write_relay_checkpoint_creates_file_and_heading(tmp_path):
    tasks_md = tmp_path / "TASKS.md"
    written = write_relay_checkpoint(
        tasks_md, task_label="T-42", note="context threshold crossed", action="summary"
    )
    assert written is True
    content = tasks_md.read_text(encoding="utf-8")
    assert "## Checkpoints" in content
    assert "T-42" in content
    assert "résumé technique" in content


def test_write_relay_checkpoint_appends_under_existing_content(tmp_path):
    tasks_md = tmp_path / "TASKS.md"
    tasks_md.write_text("# Project tasks\n\n- [ ] T-1 do the thing\n", encoding="utf-8")

    write_relay_checkpoint(tasks_md, task_label="T-1", note="first checkpoint", action="summary")
    content = tasks_md.read_text(encoding="utf-8")
    assert "# Project tasks" in content
    assert "- [ ] T-1 do the thing" in content
    assert "## Checkpoints" in content
    assert "first checkpoint" in content


def test_write_relay_checkpoint_second_entry_appends_after_first(tmp_path):
    tasks_md = tmp_path / "TASKS.md"
    write_relay_checkpoint(tasks_md, task_label="T-1", note="alpha", action="summary")
    write_relay_checkpoint(tasks_md, task_label="T-1", note="beta", action="reset")

    content = tasks_md.read_text(encoding="utf-8")
    assert content.index("alpha") < content.index("beta")
    assert content.count("## Checkpoints") == 1


def test_write_relay_checkpoint_is_idempotent_for_identical_note(tmp_path):
    tasks_md = tmp_path / "TASKS.md"
    write_relay_checkpoint(tasks_md, task_label="T-1", note="same note", action="summary")
    written_again = write_relay_checkpoint(
        tasks_md, task_label="T-1", note="same note", action="summary"
    )
    assert written_again is False
    content = tasks_md.read_text(encoding="utf-8")
    assert content.count("same note") == 1


def test_write_relay_checkpoint_not_idempotent_across_different_actions(tmp_path):
    tasks_md = tmp_path / "TASKS.md"
    write_relay_checkpoint(tasks_md, task_label="T-1", note="same note", action="summary")
    written_again = write_relay_checkpoint(
        tasks_md, task_label="T-1", note="same note", action="reset"
    )
    # Different action -> different rendered line -> written again.
    assert written_again is True


# ---------------------------------------------------------------------------
# perform_session_relay: identity preservation
# ---------------------------------------------------------------------------

def test_perform_relay_none_does_nothing(tmp_path):
    identity = SessionIdentity(topic_key="telegram:123:456", session_id="sess-1")
    calls = []
    result = perform_session_relay(
        identity=identity,
        action="none",
        tasks_md_path=tmp_path / "TASKS.md",
        task_label="T-1",
        note="irrelevant",
        compress_fn=lambda: calls.append("compress"),
        reset_fn=lambda: calls.append("reset") or identity,
    )
    assert result == RelayResult(
        action="none", checkpoint_written=False, identity_before=identity, identity_after=identity
    )
    assert calls == []
    assert not (tmp_path / "TASKS.md").exists()


def test_perform_relay_summary_preserves_identity_and_calls_compress(tmp_path):
    identity = SessionIdentity(topic_key="telegram:123:456", session_id="sess-1")
    calls = []
    result = perform_session_relay(
        identity=identity,
        action="summary",
        tasks_md_path=tmp_path / "TASKS.md",
        task_label="T-1",
        note="context threshold crossed",
        compress_fn=lambda: calls.append("compress"),
        reset_fn=lambda: calls.append("reset") or identity,
    )
    assert calls == ["compress"]
    assert result.action == "summary"
    assert result.checkpoint_written is True
    # Summary NEVER rotates session_id or topic_key.
    assert result.identity_after == identity
    assert (tmp_path / "TASKS.md").exists()


def test_perform_relay_reset_preserves_topic_but_may_rotate_session_id(tmp_path):
    identity = SessionIdentity(topic_key="telegram:123:456", session_id="sess-1")
    rotated = SessionIdentity(topic_key="telegram:123:456", session_id="sess-2")
    calls = []
    result = perform_session_relay(
        identity=identity,
        action="reset",
        tasks_md_path=tmp_path / "TASKS.md",
        task_label="T-1",
        note="model_change",
        compress_fn=lambda: calls.append("compress"),
        reset_fn=lambda: calls.append("reset") or rotated,
    )
    assert calls == ["reset"]
    assert result.action == "reset"
    assert result.checkpoint_written is True
    # Topic identity (what the user sees as "the conversation") is
    # unchanged even though the internal session id rotated.
    assert result.identity_after.topic_key == identity.topic_key
    assert result.identity_after.session_id == "sess-2"


def test_perform_relay_reset_that_moves_topic_is_rejected(tmp_path):
    """The exact failure mode this module exists to prevent: a reset must
    never surface as a brand-new visible conversation/Topic."""
    identity = SessionIdentity(topic_key="telegram:123:456", session_id="sess-1")
    moved = SessionIdentity(topic_key="telegram:999:000", session_id="sess-2")

    with pytest.raises(ValueError, match="must not move the Topic"):
        perform_session_relay(
            identity=identity,
            action="reset",
            tasks_md_path=tmp_path / "TASKS.md",
            task_label="T-1",
            note="model_change",
            reset_fn=lambda: moved,
        )


def test_perform_relay_reset_without_reset_fn_keeps_identity(tmp_path):
    identity = SessionIdentity(topic_key="telegram:123:456", session_id="sess-1")
    result = perform_session_relay(
        identity=identity,
        action="reset",
        tasks_md_path=tmp_path / "TASKS.md",
        task_label="T-1",
        note="model_change",
    )
    assert result.identity_after == identity


def test_perform_relay_writes_checkpoint_before_callback_runs(tmp_path):
    """Checkpoint-then-act ordering: if the callback inspects TASKS.md, the
    checkpoint must already be there (the AGENTS.md "handoff before reset"
    rule, enforced as code rather than trusted-by-convention)."""
    tasks_md = tmp_path / "TASKS.md"
    seen = {}

    def compress_fn():
        seen["content_at_call_time"] = tasks_md.read_text(encoding="utf-8")

    identity = SessionIdentity(topic_key="t:1", session_id="s-1")
    perform_session_relay(
        identity=identity,
        action="summary",
        tasks_md_path=tasks_md,
        task_label="T-1",
        note="threshold",
        compress_fn=compress_fn,
    )
    assert "T-1" in seen["content_at_call_time"]
