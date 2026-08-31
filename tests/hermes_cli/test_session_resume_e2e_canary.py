"""Real, non-mocked E2E canary for the transient-block session-resume mechanism.

``t_e087dcd1`` asked for proof — not a mock — that:

  * transient block -> unblock -> respawn reuses the *exact* worker session
    (same session id, real message continuity in the real session store).
  * needs_input / capability / deferred blocks -> unblock -> respawn always
    start a fresh session (no --resume, no env leak, no continuity).
  * a crash never resumes either, even when a *prior* transient block left
    ``block_kind="transient"`` on the row (the crashed run's outcome is
    ``"crashed"``, not ``"blocked"`` — the exact guard in
    ``_transient_resume_session_id``).

Nothing here is mocked at the boundary under test: ``subprocess.Popen`` runs
for real (only ``_resolve_hermes_argv`` is redirected to a tiny real Python
worker stub instead of the full ``hermes`` binary, so the canary doesn't pay
for a real LLM turn per run). That stub touches a real, fully isolated
``state.db`` under pytest's ``tmp_path`` exactly the way a real dispatcher
worker would — same schema, same ``SessionDB`` API, same file. The identity
proof is an independent re-open of that file from the test process, not a
value captured in-memory from the child.

Every board/profile/session lives under ``tmp_path`` (via an isolated
``HERMES_HOME``) and is discarded by pytest when the test ends — this never
touches the user's real ``~/.hermes``, kanban board, or Telegram board.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

# .../hermes-agent (repo root — where hermes_state.py and hermes_cli/ live)
REPO_ROOT = Path(__file__).resolve().parents[2]

_STUB_SOURCE = """\
import json, os, sys
sys.path.insert(0, {repo_root!r})
from pathlib import Path
from hermes_state import SessionDB

home = Path(os.environ["HERMES_HOME"])
state_db_path = home / "state.db"
resume_id = os.environ.get("HERMES_KANBAN_RESUME_SESSION_ID")
argv = sys.argv[1:]
argv_resume = argv[argv.index("--resume") + 1] if "--resume" in argv else None

db = SessionDB(state_db_path)
try:
    if resume_id:
        session_id = resume_id
        is_new_session = False
        db.append_message(
            session_id, role="assistant", content="resumed-continuation-probe"
        )
    else:
        session_id = "canary-fresh-" + os.urandom(6).hex()
        is_new_session = True
        db.create_session(session_id, source="kanban")
        db.append_message(
            session_id, role="assistant", content="fresh-session-probe"
        )
    msg_count = db.message_count(session_id)
finally:
    db.close()

result = {{
    "pid": os.getpid(),
    "env_resume_session_id": resume_id,
    "argv_resume_session_id": argv_resume,
    "session_id_used": session_id,
    "is_new_session": is_new_session,
    "message_count_after": msg_count,
    "task_id": os.environ.get("HERMES_KANBAN_TASK"),
}}
with open(os.environ["HERMES_TEST_CANARY_RESULT"], "w") as f:
    json.dump(result, f)
"""


def _write_stub(tmp_path: Path) -> Path:
    stub = tmp_path / "worker_stub.py"
    stub.write_text(_STUB_SOURCE.format(repo_root=str(REPO_ROOT)), encoding="utf-8")
    return stub


def _await_result(result_path: Path, *, timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if result_path.exists():
            try:
                return json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        time.sleep(0.05)
    raise AssertionError(f"canary worker did not report a result within {timeout}s")


@pytest.fixture
def canary_env(monkeypatch, tmp_path):
    """Fully isolated kanban board + profile home; every DB write is real."""
    root = tmp_path / ".hermes"
    profile_home = root / "profiles" / "elias"
    profile_home.mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)

    from hermes_cli import kanban_db as kb

    kb.init_db()

    stub = _write_stub(tmp_path)
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: [sys.executable, str(stub)])

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    return {
        "root": root,
        "profile_home": profile_home,
        "kb": kb,
        "workspace": workspace,
    }


def _spawn_and_wait(kb, task, workspace, tmp_path, monkeypatch) -> dict:
    result_path = tmp_path / f"result-{uuid.uuid4().hex}.json"
    monkeypatch.setenv("HERMES_TEST_CANARY_RESULT", str(result_path))
    pid = kb._default_spawn(task, str(workspace))
    assert pid and pid > 0, "real subprocess spawn must return a live pid"
    result = _await_result(result_path)
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    return result


def _seed_original_session(profile_home: Path) -> str:
    from hermes_state import SessionDB

    session_id = "canary-original-" + uuid.uuid4().hex[:12]
    db = SessionDB(profile_home / "state.db")
    try:
        db.create_session(session_id, source="kanban")
        db.append_message(session_id, role="user", content="do the thing")
        db.append_message(session_id, role="assistant", content="working on it")
    finally:
        db.close()
    return session_id


def _message_count(db_path: Path, session_id: str) -> int:
    """Independent re-open — not the SessionDB instance the child wrote
    through — so the count really comes from disk, not from memory."""
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
    finally:
        conn.close()


def _session_ids(db_path: Path) -> list[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return [r[0] for r in conn.execute("SELECT id FROM sessions").fetchall()]
    finally:
        conn.close()


class TestTransientResumeRealCanary:
    def test_transient_block_unblock_resumes_exact_session(
        self, canary_env, monkeypatch, tmp_path
    ):
        kb = canary_env["kb"]
        profile_home = canary_env["profile_home"]
        workspace = canary_env["workspace"]
        state_db_path = profile_home / "state.db"

        original_session_id = _seed_original_session(profile_home)
        count_before = _message_count(state_db_path, original_session_id)
        assert count_before == 2

        with kb.connect() as conn:
            tid = kb.create_task(conn, title="canary transient", assignee="elias")
            kb.claim_task(conn, tid)
            # The real tool-layer stamp: this is exactly what a worker's
            # kanban_block(kind="transient") call does with HERMES_SESSION_ID.
            kb.block_task(
                conn, tid,
                reason="read-only guard; safe retry",
                kind="transient",
                metadata={"worker_session_id": original_session_id},
            )
            assert kb.unblock_task(conn, tid)
            assert kb.claim_task(conn, tid) is not None
            task = kb.get_task(conn, tid)

        result = _spawn_and_wait(kb, task, workspace, tmp_path, monkeypatch)

        assert result["env_resume_session_id"] == original_session_id
        assert result["argv_resume_session_id"] == original_session_id
        assert result["session_id_used"] == original_session_id
        assert result["is_new_session"] is False

        # Independent proof, re-read from disk: the same session row grew
        # by exactly the one message the resumed child wrote, and no
        # second session row was ever created.
        assert _message_count(state_db_path, original_session_id) == count_before + 1
        assert _session_ids(state_db_path) == [original_session_id]


class TestNonResumingBlocksRealCanary:
    @pytest.mark.parametrize("kind", ["needs_input", "capability", "deferred"])
    def test_human_or_capability_block_starts_fresh_session(
        self, kind, canary_env, monkeypatch, tmp_path
    ):
        kb = canary_env["kb"]
        profile_home = canary_env["profile_home"]
        workspace = canary_env["workspace"]
        state_db_path = profile_home / "state.db"

        original_session_id = _seed_original_session(profile_home)
        count_before = _message_count(state_db_path, original_session_id)

        reasons = {
            "needs_input": "needs a human decision",
            "capability": "missing capability",
            "deferred": "human already decided, parked until explicit resume",
        }
        with kb.connect() as conn:
            tid = kb.create_task(conn, title=f"canary {kind}", assignee="elias")
            kb.claim_task(conn, tid)
            kb.block_task(
                conn, tid,
                reason=reasons[kind],
                kind=kind,
                # A worker calling kanban_block would still stamp this for
                # needs_input/capability too (only kind="transient" is
                # actually treated as resumable) — proves the gate is on
                # block *kind*, not on whether metadata happens to carry a
                # session id.
                metadata={"worker_session_id": original_session_id},
            )
            assert kb.unblock_task(conn, tid)
            assert kb.claim_task(conn, tid) is not None
            task = kb.get_task(conn, tid)

        result = _spawn_and_wait(kb, task, workspace, tmp_path, monkeypatch)

        assert result["env_resume_session_id"] is None
        assert result["argv_resume_session_id"] is None
        assert result["session_id_used"] != original_session_id
        assert result["is_new_session"] is True

        # The original session was never touched by the second run.
        assert _message_count(state_db_path, original_session_id) == count_before
        ids = _session_ids(state_db_path)
        assert original_session_id in ids
        assert result["session_id_used"] in ids
        assert len(ids) == 2


class TestCrashNeverResumesRealCanary:
    def test_crash_after_transient_history_does_not_resume(
        self, canary_env, monkeypatch, tmp_path
    ):
        kb = canary_env["kb"]
        profile_home = canary_env["profile_home"]
        workspace = canary_env["workspace"]
        state_db_path = profile_home / "state.db"

        original_session_id = _seed_original_session(profile_home)

        # Skip the real-time crash-detection grace window instead of
        # sleeping for it.
        monkeypatch.setattr(kb, "_resolve_crash_grace_seconds", lambda: 0)

        with kb.connect() as conn:
            tid = kb.create_task(conn, title="canary crash history", assignee="elias")
            kb.claim_task(conn, tid)
            # Run 1: a genuine safe transient block + unblock — same as the
            # positive case. This is what leaves block_kind="transient" on
            # the row permanently (unblock_task deliberately never clears
            # it — see its docstring).
            kb.block_task(
                conn, tid,
                reason="read-only guard; safe retry",
                kind="transient",
                metadata={"worker_session_id": original_session_id},
            )
            assert kb.unblock_task(conn, tid)
            assert kb.claim_task(conn, tid) is not None
            task = kb.get_task(conn, tid)
            assert task.block_kind == "transient"

        # Run 2: instead of blocking again or completing, this worker just
        # dies. A real, short-lived OS process, reaped for real via wait()
        # so _pid_alive() genuinely reports it gone — no fabricated state.
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait(timeout=5)
        dead_pid = dead.pid

        with kb.connect() as conn:
            kb._set_worker_pid(conn, tid, dead_pid)
            crashed_ids = kb.detect_crashed_workers(conn)
            assert tid in crashed_ids
            task = kb.get_task(conn, tid)
            # block_kind survives the crash reclaim untouched...
            assert task.block_kind == "transient"
            assert kb.claim_task(conn, tid) is not None
            task = kb.get_task(conn, tid)

        # Run 3: ...but the next spawn must NOT treat the crash as a safe
        # transient continuation, because the immediately preceding run's
        # outcome is "crashed", not "blocked" (the exact guard inside
        # _transient_resume_session_id).
        result = _spawn_and_wait(kb, task, workspace, tmp_path, monkeypatch)

        assert result["env_resume_session_id"] is None
        assert result["argv_resume_session_id"] is None
        assert result["session_id_used"] != original_session_id
        assert result["is_new_session"] is True
