"""Regression tests for bounded Kanban incident categories."""

from hermes_cli.kanban_db import classify_worker_incident


def test_worker_incident_categories_are_explicit_and_secret_free():
    assert classify_worker_incident("provider returned 429") == "provider_limit"
    assert classify_worker_incident("approval prompt timed out") == "approval_expired"
    assert classify_worker_incident("security guard blocked command") == "security_guard"
    assert classify_worker_incident("ignored", protocol_violation=True) == "workflow_bug"
    assert classify_worker_incident("exit code 2") == "normal_failure"
