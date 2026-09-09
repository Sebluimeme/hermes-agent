"""Regression tests for clean direct systemd gateway stops/restarts."""

from gateway import run as gateway_run
from gateway import planned_stop
from hermes_cli import gateway as gateway_cli


def test_planned_stop_helper_writes_marker_for_valid_pid(monkeypatch):
    calls: list[int] = []

    monkeypatch.setenv("MAINPID", "99999")
    monkeypatch.setattr(
        "gateway.status.write_planned_stop_marker",
        lambda pid: calls.append(pid) or True,
    )

    assert planned_stop.main(["12345"]) == 0
    assert calls == [12345]


def test_planned_stop_helper_uses_systemd_mainpid_when_pid_is_omitted(monkeypatch):
    calls: list[int] = []
    monkeypatch.setenv("MAINPID", "24680")
    monkeypatch.setattr(
        "gateway.status.write_planned_stop_marker",
        lambda pid: calls.append(pid) or True,
    )

    assert planned_stop.main([]) == 0
    assert calls == [24680]


def test_planned_stop_helper_is_clean_noop_without_systemd_mainpid(monkeypatch):
    calls: list[int] = []
    monkeypatch.delenv("MAINPID", raising=False)
    monkeypatch.setattr(
        "gateway.status.write_planned_stop_marker",
        lambda pid: calls.append(pid) or True,
    )

    assert planned_stop.main([]) == 0
    assert calls == []


def test_planned_stop_helper_never_falls_back_to_replacement_gateway(monkeypatch):
    """A late ExecStop from gateway A must never discover and mark gateway B."""
    calls: list[int] = []
    monkeypatch.delenv("MAINPID", raising=False)
    monkeypatch.setattr(
        "gateway.status.get_running_pid",
        lambda: (_ for _ in ()).throw(AssertionError("pidfile fallback is unsafe")),
    )
    monkeypatch.setattr(
        "gateway.status.write_planned_stop_marker",
        lambda pid: calls.append(pid) or True,
    )

    assert planned_stop.main([]) == 0
    assert calls == []


def test_planned_stop_helper_rejects_invalid_explicit_pid(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        "gateway.status.write_planned_stop_marker",
        lambda pid: calls.append(pid) or True,
    )

    assert planned_stop.main(["not-a-pid"]) == 2
    assert planned_stop.main(["0"]) == 2
    assert planned_stop.main(["123", "456"]) == 2
    assert calls == []


def test_planned_stop_helper_rejects_invalid_systemd_mainpid(monkeypatch):
    calls: list[int] = []
    monkeypatch.setenv("MAINPID", "not-a-pid")
    monkeypatch.setattr(
        "gateway.status.write_planned_stop_marker",
        lambda pid: calls.append(pid) or True,
    )

    assert planned_stop.main([]) == 2
    assert calls == []


def test_generated_user_systemd_unit_marks_stop_before_sigterm():
    unit = gateway_cli.generate_systemd_unit(system=False)

    exec_stop = next(line for line in unit.splitlines() if line.startswith("ExecStop="))
    assert exec_stop.endswith(" -m gateway.planned_stop")
    assert "$MAINPID" not in exec_stop
    assert "KillSignal=SIGTERM" in unit
    assert "SuccessExitStatus=75" in unit
    assert "RestartForceExitStatus=75" in unit


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
    assert exec_stop.endswith(" -m gateway.planned_stop")
    assert "$MAINPID" not in exec_stop
    assert "KillSignal=SIGTERM" in unit
    assert "SuccessExitStatus=75" in unit
    assert "RestartForceExitStatus=75" in unit


def test_systemd_unit_without_planned_restart_success_status_is_stale(
    tmp_path, monkeypatch
):
    expected = (
        "[Service]\n"
        "Restart=always\n"
        "SuccessExitStatus=75\n"
        "RestartForceExitStatus=75\n"
    )
    installed = expected.replace("SuccessExitStatus=75\n", "")
    unit_path = tmp_path / "hermes-gateway.service"
    unit_path.write_text(installed, encoding="utf-8")

    monkeypatch.setattr(
        gateway_cli,
        "get_systemd_unit_path",
        lambda system=False: unit_path,
    )
    monkeypatch.setattr(
        gateway_cli,
        "generate_systemd_unit",
        lambda system=False, run_as_user=None: expected,
    )

    assert gateway_cli.systemd_unit_is_current(system=False) is False


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
