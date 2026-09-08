import subprocess

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def _repo_with_remote(tmp_path):
    repo = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    subprocess.check_call(["git", "init", "-q", "-b", "main", str(repo)])
    subprocess.check_call(["git", "init", "-q", "--bare", str(remote)])
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "result.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "result.txt")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "-u", "origin", "main")
    _git(repo, "remote", "set-head", "origin", "main")
    _git(repo, "checkout", "-q", "-b", "candidate")
    (repo / "result.txt").write_text("candidate\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "candidate")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_unmerged_candidate_is_kept_in_review(kanban_home, tmp_path):
    repo, candidate = _repo_with_remote(tmp_path)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Deliver candidate",
            body="Commit and push to origin/main before completion.",
            assignee="coder",
            workspace_kind="worktree",
            workspace_path=str(repo),
        )
        kb.claim_task(conn, task_id)

        with pytest.raises(kb.CompletionValidationError, match="awaiting_integration"):
            kb.complete_task(
                conn,
                task_id,
                summary="candidate complete",
                metadata={
                    "evidence": {"kind": "test", "detail": "tests passed"},
                    "integration": {
                        "required": True,
                        "repo_path": str(repo),
                        "target_remote": "origin",
                        "target_branch": "main",
                        "commit": candidate,
                    },
                },
            )

        task = kb.get_task(conn, task_id)
        assert task.status == "review"
        assert task.integration_status == "awaiting_integration"
        assert any(event.kind == "awaiting_integration" for event in kb.list_events(conn, task_id))


def test_integrated_candidate_can_complete(kanban_home, tmp_path):
    repo, candidate = _repo_with_remote(tmp_path)
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "candidate")
    _git(repo, "push", "-q", "origin", "main")
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Deliver candidate",
            body="Commit and push to origin/main before completion.",
            assignee="coder",
            workspace_kind="worktree",
            workspace_path=str(repo),
        )
        kb.claim_task(conn, task_id)

        assert kb.complete_task(
            conn,
            task_id,
            summary="integrated candidate",
            metadata={
                "evidence": {"kind": "test", "detail": "tests passed"},
                "integration": {
                    "required": True,
                    "repo_path": str(repo),
                    "target_remote": "origin",
                    "target_branch": "main",
                    "commit": candidate,
                },
            },
        )

        task = kb.get_task(conn, task_id)
        assert task.status == "done"
        assert task.integration_status == "integrated"


def test_scratch_docs_do_not_require_integration(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Production documentation [NO-INTEGRATION]",
            body="Describe the deployment process without changing code.",
            assignee="writer",
            workspace_kind="scratch",
        )
        kb.claim_task(conn, task_id)
        assert kb.complete_task(
            conn,
            task_id,
            summary="documentation written",
            metadata={"evidence": {"kind": "test", "detail": "spellcheck passed"}},
        )


@pytest.mark.parametrize(
    "instruction",
    [
        "Ne pas commit/push.",
        "Sans commit ni push.",
        "Do not commit/push.",
        "No commit/push.",
    ],
)
def test_explicit_no_commit_instruction_disables_inferred_integration(
    kanban_home, instruction
):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Verify a local candidate",
            body=f"{instruction} Vérifier le déploiement local uniquement.",
            assignee="claude2",
            workspace_kind="dir",
            workspace_path=str(kanban_home),
        )
        kb.claim_task(conn, task_id)
        assert kb.complete_task(
            conn,
            task_id,
            summary="local verification complete",
            metadata={"evidence": {"kind": "test", "detail": "tests passed"}},
        )
        task = kb.get_task(conn, task_id)
        assert task.status == "done"
        assert task.integration_status != "awaiting_integration"
        assert not any(
            event.kind == "awaiting_integration"
            for event in kb.list_events(conn, task_id)
        )
