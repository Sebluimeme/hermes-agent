from types import SimpleNamespace

from hermes_cli.kanban_db import adaptive_worker_turn_budget, _worker_retry_strategy_hint


def _run(*, outcome="crashed", summary=None, metadata=None, error="boom"):
    return SimpleNamespace(
        outcome=outcome,
        summary=summary,
        metadata=metadata,
        error=error,
        ended_at=1,
    )


def test_adaptive_budget_starts_small_and_extends_only_on_durable_progress():
    simple = SimpleNamespace(routing_tier="simple")
    complex_task = SimpleNamespace(routing_tier="complex")
    automatic_checkpoint = _run(metadata={"checkpoint": {"state": "crashed"}})
    real_checkpoint = _run(metadata={"checkpoint": {"note": "parsed 4/10 inputs"}})

    assert adaptive_worker_turn_budget(simple, []) == 12
    assert adaptive_worker_turn_budget(complex_task, []) == 30
    assert adaptive_worker_turn_budget(simple, [automatic_checkpoint]) == 12
    assert adaptive_worker_turn_budget(simple, [automatic_checkpoint, real_checkpoint]) == 22


def test_adaptive_budget_has_absolute_ceiling():
    task = SimpleNamespace(routing_tier="complex")
    runs = [_run(summary=f"progress {index}") for index in range(20)]
    assert adaptive_worker_turn_budget(task, runs) == 90


def test_three_identical_failures_force_strategy_change_hint():
    same = [_run(error="terminal failed: permission denied") for _ in range(3)]
    assert "different strategy" in _worker_retry_strategy_hint(same)
    mixed = same[:2] + [_run(error="network timeout")]
    assert _worker_retry_strategy_hint(mixed) == ""


def test_neutral_resume_outcomes_never_force_strategy_change_hint():
    neutral_outcomes = (
        "interrupted",
        "rate_limited",
        "reclaimed",
        "stale",
        "strategy_required",
        "review_deferred",
        "scheduled",
        "blocked",
    )
    for outcome in neutral_outcomes:
        runs = [_run(outcome=outcome, error="same resumable event") for _ in range(3)]
        assert _worker_retry_strategy_hint(runs) == ""


def test_neutral_yields_do_not_hide_three_real_identical_failures():
    runs = [
        _run(error="terminal failed: permission denied"),
        _run(outcome="interrupted", error="neutral orchestration yield"),
        _run(error="terminal failed: permission denied"),
        _run(outcome="reclaimed", error="operator restart"),
        _run(error="terminal failed: permission denied"),
    ]
    assert "different strategy" in _worker_retry_strategy_hint(runs)
