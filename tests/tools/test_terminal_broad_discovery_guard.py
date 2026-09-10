"""Regression tests for bounded source discovery in Kanban workers."""

import json

from tools.code_execution_tool import execute_code
from tools.terminal_tool import _broad_recursive_discovery_guidance


def test_worker_rejects_recursive_scan_of_shared_hermes_root(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_scan")

    message = _broad_recursive_discovery_guidance(
        f"python3 -c \"from pathlib import Path; "
        f"print(list(Path('{home}').rglob('*')))\""
    )

    assert message is not None
    assert "rg --files" in message


def test_worker_rejects_recursive_scan_of_multi_project_workspace(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    workspace = home / "workspace"
    workspace.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_scan")

    message = _broad_recursive_discovery_guidance(
        f"python3 - <<'PY'\nimport os\nfor root in ['{workspace}']:\n"
        "    for path, dirs, files in os.walk(root):\n        pass\nPY"
    )

    assert message is not None
    assert "Do not retry" in message


def test_worker_allows_recursive_scan_inside_one_explicit_project(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    project = home / "workspace" / "poivre-et-sale"
    project.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_scan")

    assert _broad_recursive_discovery_guidance(
        f"python3 -c \"from pathlib import Path; "
        f"print(list(Path('{project}').rglob('*.py')))\""
    ) is None


def test_direct_interactive_terminal_is_not_subject_to_worker_guard(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    assert _broad_recursive_discovery_guidance(
        f"python3 -c \"from pathlib import Path; "
        f"print(list(Path('{home}').rglob('*')))\""
    ) is None


def test_worker_rg_discovery_remains_allowed(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_scan")

    assert _broad_recursive_discovery_guidance(
        f"rg --files {home}/workspace {home}/scripts | rg -i 'poivre|ga4'"
    ) is None


def test_worker_execute_code_rejects_shared_workspace_rglob(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    workspace = home / "workspace"
    workspace.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_scan")

    result = json.loads(execute_code(
        "from pathlib import Path\n"
        f"for path in Path('{workspace}').rglob('*'):\n    pass\n"
    ))

    assert result["status"] == "error"
    assert result["duration_seconds"] == 0
    assert "rg --files" in result["error"]


def test_worker_execute_code_allows_project_scoped_rglob(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    project = home / "workspace" / "ecobloc"
    project.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_scan")

    assert _broad_recursive_discovery_guidance(
        "from pathlib import Path\n"
        f"print(list(Path('{project}').rglob('*.py')))\n"
    ) is None
