from __future__ import annotations

import subprocess


def _make_task(kb, *, assignee: str):
    return kb.Task(
        id="t_spawn_tools",
        title="spawn tools",
        body=None,
        assignee=assignee,
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
    )


def test_default_spawn_pins_assignee_profile_cli_toolsets(monkeypatch, tmp_path):
    """Manual profile assignment should keep that profile's CLI tools.

    Regression guard for dispatcher-spawned workers that boot with
    HERMES_KANBAN_TASK: the worker must not collapse to only kanban lifecycle
    tools when the assigned profile's top-level ``toolsets`` is the default
    composite. The spawned CLI gets an explicit --toolsets pin resolved from
    platform_toolsets.cli; model_tools appends task-scoped kanban tools later.
    """
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text(
        """
platform_toolsets:
  cli:
    - clarify
    - code_execution
    - delegation
    - file
    - memory
    - session_search
    - skills
    - terminal
    - web
toolsets:
  - hermes-cli
agent:
  disabled_toolsets: []
""".lstrip(),
        encoding="utf-8",
    )
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        captured["cwd"] = kwargs.get("cwd")
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pid = kb._default_spawn(_make_task(kb, assignee="elias"), str(workspace))

    assert pid == 4242
    assert captured["env"]["HERMES_HOME"] == str(profile)
    assert captured["env"]["HERMES_KANBAN_TASK"] == "t_spawn_tools"
    assert "--toolsets" in captured["cmd"]
    pinned = captured["cmd"][captured["cmd"].index("--toolsets") + 1].split(",")
    for required in ("terminal", "web", "file", "skills", "code_execution", "delegation"):
        assert required in pinned


def test_default_spawn_isolates_live_gateway_worker_in_memory_scope(
    monkeypatch, tmp_path,
):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    root.joinpath("config.yaml").write_text(
        "kanban:\n  worker_memory_max_mb: 128\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("_HERMES_GATEWAY", "1")

    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(
        "tools.process_registry._systemd_run_user_scope_available",
        lambda: True,
    )
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
    captured = {}

    class FakeProc:
        pid = 4243

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    kb._default_spawn(_make_task(kb, assignee="elias"), str(workspace))

    assert captured["cmd"][0] == "/usr/bin/systemd-run"
    assert "--scope" in captured["cmd"]
    memory_prop = next(x for x in captured["cmd"] if x.startswith("MemoryMax="))
    assert memory_prop == "MemoryMax=536870912"
    assert captured["env"]["HERMES_KANBAN_SYSTEMD_UNIT"].startswith(
        "hermes-worker-kanban-t_spawn_tools-"
    )
    assert "hermes" in captured["cmd"][captured["cmd"].index("--") + 1:]


def test_kanban_worker_memory_config_preserves_legacy_clamps(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    root.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    cases = [
        ("{}\n", 6144),
        ("kanban:\n  worker_memory_max_mb: not-an-int\n", 6144),
        ("kanban:\n  worker_memory_max_mb: 128\n", 512),
        ("kanban:\n  worker_memory_max_mb: 99999\n", 8192),
    ]
    for content, expected in cases:
        root.joinpath("config.yaml").write_text(content, encoding="utf-8")
        assert kb._configured_kanban_worker_memory_max_mb() == expected


def test_default_spawn_model_override_survives_real_cli_parse(monkeypatch, tmp_path):
    """The dispatcher's pre-``chat`` model flag must reach ``args.model``.

    This is an integration contract between Kanban's worker argv builder and
    the real CLI parser. A parser default once erased the explicit override,
    silently sending the worker to its profile default or fallback instead.
    """
    root = tmp_path / ".hermes"
    (root / "profiles" / "elias").mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb
    from hermes_cli._parser import build_top_level_parser

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4244

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = _make_task(kb, assignee="elias")
    task.model_override = "gpt-5.6-sol"
    kb._default_spawn(task, str(workspace))

    parser, _subparsers, _chat_parser = build_top_level_parser()
    # Profile selection is attached by the outer CLI bootstrap rather than
    # build_top_level_parser(); remove that already-validated prefix and parse
    # the worker flags/subcommand through the real shared parser.
    assert captured["cmd"][1:3] == ["-p", "elias"]
    args = parser.parse_args(captured["cmd"][3:])

    assert args.command == "chat"
    assert args.model == "gpt-5.6-sol"
    assert args.query == "work kanban task t_spawn_tools"


def test_default_spawn_resumes_only_unblocked_transient_worker(monkeypatch, tmp_path):
    """Safe tool retry keeps one worker session instead of spawning a new one."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "elias").mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb
    from hermes_cli._parser import build_top_level_parser

    kb.init_db()
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4245

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs["env"])
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="retry a read", assignee="elias")
        kb.claim_task(conn, tid)
        kb.block_task(
            conn, tid, reason="read-only guard", kind="transient",
            metadata={"worker_session_id": "worker-session-42"},
        )
        assert kb.unblock_task(conn, tid)
        assert kb.claim_task(conn, tid) is not None
        task = kb.get_task(conn, tid)
        assert task is not None

    kb._default_spawn(task, str(workspace))

    assert captured["env"]["HERMES_KANBAN_RESUME_SESSION_ID"] == "worker-session-42"
    assert captured["cmd"][captured["cmd"].index("--resume") + 1] == "worker-session-42"
    parser, _subparsers, _chat_parser = build_top_level_parser()
    parsed = parser.parse_args(captured["cmd"][3:])
    assert parsed.resume == "worker-session-42"
    # The resumed command references the existing worker session; it does not
    # allocate a second session identity for this retry.
    assert len({"worker-session-42", parsed.resume}) == 1
    assert captured["cmd"][-1].startswith(f"continue kanban task {tid}")


def test_default_spawn_keeps_human_block_fresh_and_resumes_crash(monkeypatch, tmp_path):
    """Human decisions start fresh; transient crashes resume the checkpoint."""
    monkeypatch.delenv("HERMES_KANBAN_RESUME_SESSION_ID", raising=False)
    root = tmp_path / ".hermes"
    (root / "profiles" / "elias").mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    kb.init_db()
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4246

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs["env"])
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="needs decision", assignee="elias")
        kb.claim_task(conn, tid)
        kb.block_task(
            conn, tid, reason="need human input", kind="needs_input",
            metadata={"worker_session_id": "must-not-resume"},
        )
        assert kb.unblock_task(conn, tid)
        assert kb.claim_task(conn, tid) is not None
        task = kb.get_task(conn, tid)
        assert task is not None

    kb._default_spawn(task, str(workspace))

    assert "--resume" not in captured["cmd"]
    assert "HERMES_KANBAN_RESUME_SESSION_ID" not in captured["env"]
    assert captured["cmd"][-1] == f"work kanban task {tid}"

    with kb.connect() as conn:
        crash_tid = kb.create_task(conn, title="crashed retry", assignee="elias")
        kb.claim_task(conn, crash_tid)
        kb.block_task(
            conn, crash_tid, reason="temporary", kind="transient",
            metadata={"worker_session_id": "crashed-session"},
        )
        conn.execute(
            "UPDATE task_runs SET outcome = 'crashed' WHERE task_id = ?",
            (crash_tid,),
        )
        assert kb.unblock_task(conn, crash_tid)
        assert kb.claim_task(conn, crash_tid) is not None
        crashed_task = kb.get_task(conn, crash_tid)
        assert crashed_task is not None

    kb._default_spawn(crashed_task, str(workspace))
    assert captured["env"]["HERMES_KANBAN_RESUME_SESSION_ID"] == "crashed-session"
    assert captured["cmd"][captured["cmd"].index("--resume") + 1] == "crashed-session"
    assert captured["cmd"][-1].startswith(f"continue kanban task {crash_tid}")


def test_resolve_worker_cli_toolsets_uses_profile_home_not_parent_config(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    root.joinpath("config.yaml").write_text("platform_toolsets:\n  cli:\n    - kanban\n", encoding="utf-8")
    profile.joinpath("config.yaml").write_text(
        """
platform_toolsets:
  cli:
    - terminal
    - web
toolsets:
  - hermes-cli
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    resolved = kb._resolve_worker_cli_toolsets(str(profile))

    assert resolved is not None
    assert "terminal" in resolved
    assert "web" in resolved
    assert "kanban" in resolved  # recovered worker lifecycle surface
    assert resolved != ["kanban"]
