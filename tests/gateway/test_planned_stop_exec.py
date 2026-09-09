"""Regression tests for clean direct systemd gateway stops/restarts."""

from gateway import run as gateway_run
from gateway import planned_stop
from hermes_cli import gateway as gateway_cli


def test_planned_stop_helper_writes_marker_for_valid_pid(monkeypatch):
    calls: list[int] = []

    monkeypatch.setattr(
        "gateway.status.write_planned_stop_marker",
        lambda pid: calls.append(pid) or True,
    )

    assert planned_stop.main(["12345"]) == 0
    assert calls == [12345]


def test_planned_stop_helper_rejects_invalid_pid_before_marker_import(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        "gateway.status.write_planned_stop_marker",
        lambda pid: calls.append(pid) or True,
    )

    assert planned_stop.main([]) == 2
    assert planned_stop.main(["not-a-pid"]) == 2
    assert planned_stop.main(["0"]) == 2
    assert calls == []


def test_generated_user_systemd_unit_marks_stop_before_sigterm():
    unit = gateway_cli.generate_systemd_unit(system=False)

    exec_stop = next(line for line in unit.splitlines() if line.startswith("ExecStop="))
    assert exec_stop.endswith(" -m gateway.planned_stop $MAINPID")
    assert "KillSignal=SIGTERM" in unit


def test_generated_system_systemd_unit_marks_stop_before_sigterm(monkeypatch):
    monkeypatch.setattr(
        gateway_cli,
        "_system_service_identity",
        lambda run_as_user=None: ("alice", "alice", "/home/alice"),
    )
    monkeypatch.setattr(
        gateway_cli,
        "_build_user_local_paths",
        lambda home, existing: [],
    )
    unit = gateway_cli.generate_systemd_unit(system=True, run_as_user="alice")

    exec_stop = next(line for line in unit.splitlines() if line.startswith("ExecStop="))
    assert exec_stop.endswith(" -m gateway.planned_stop $MAINPID")
    assert "KillSignal=SIGTERM" in unit


def test_watcher_callback_planned_verdict_survives_followup_sigterm():
    """watcher(None) consumes the marker; the queued SIGTERM cannot degrade it."""
    kind, process_first = gateway_run._claim_gateway_shutdown_kind(
        None,
        planned_takeover=False,
        planned_stop=True,
    )
    repeated_kind, process_second = gateway_run._claim_gateway_shutdown_kind(
        kind,
        planned_takeover=False,
        planned_stop=False,
    )

    assert (kind, process_first) == ("planned_stop", True)
    assert (repeated_kind, process_second) == ("planned_stop", False)


def test_first_external_sigterm_remains_unexpected_if_marker_appears_later():
    """A late marker cannot mask a genuine first external SIGTERM."""
    kind, process_first = gateway_run._claim_gateway_shutdown_kind(
        None,
        planned_takeover=False,
        planned_stop=False,
    )
    repeated_kind, process_second = gateway_run._claim_gateway_shutdown_kind(
        kind,
        planned_takeover=False,
        planned_stop=True,
    )

    assert (kind, process_first) == ("unexpected", True)
    assert (repeated_kind, process_second) == ("unexpected", False)
