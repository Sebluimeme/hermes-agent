import json
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


@pytest.mark.parametrize(
    ("delivery_target", "forbids_remote", "requires_remote"),
    [
        ("working_tree", True, False),
        ("commit", False, False),
        ("push", False, True),
        ("deploy", False, True),
    ],
)
def test_durable_delivery_target_controls_remote_integration_semantics(
    kanban_home, delivery_target, forbids_remote, requires_remote,
):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Explicit delivery contract",
            body="This text deliberately mentions deploy and no commit.",
            assignee="coder",
            workspace_kind="dir",
            workspace_path=str(kanban_home),
            delivery_target=delivery_target,
        )
        task = kb.get_task(conn, task_id)

    assert task is not None
    assert kb.completion_explicitly_forbids_integration(task, {}) is forbids_remote
    assert kb._completion_requires_integration(task, {}) is requires_remote


def test_push_target_does_not_infer_deployment_proof_from_prose(
    kanban_home, tmp_path,
):
    repo, candidate = _repo_with_remote(tmp_path)
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "candidate")
    _git(repo, "push", "-q", "origin", "main")
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Push production deployment notes",
            body="Deploy is mentioned, but the durable target is only push.",
            assignee="coder",
            workspace_kind="worktree",
            workspace_path=str(repo),
            delivery_target="push",
        )
        task = kb.get_task(conn, task_id)

    error, projected = kb._integration_delivery_projection(task, {
        "integration": {
            "repo_path": str(repo),
            "target_remote": "origin",
            "target_branch": "main",
            "commit": candidate,
        },
    })
    assert error is None
    assert projected["integration"]["delivery_target"] == "push"


def test_deploy_target_requires_production_proof_even_without_deploy_prose(
    kanban_home, tmp_path,
):
    repo, candidate = _repo_with_remote(tmp_path)
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "candidate")
    _git(repo, "push", "-q", "origin", "main")
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Ship exact candidate",
            body="No lexical production marker is needed.",
            assignee="coder",
            workspace_kind="worktree",
            workspace_path=str(repo),
            delivery_target="deploy",
        )
        task = kb.get_task(conn, task_id)

    error, _ = kb._integration_delivery_projection(task, {
        "integration": {
            "repo_path": str(repo),
            "target_remote": "origin",
            "target_branch": "main",
            "commit": candidate,
        },
    })
    assert error == "deployment required but production_proof is missing"


def test_stale_completion_cannot_park_or_close_successor_run(
    kanban_home, tmp_path,
):
    repo, candidate = _repo_with_remote(tmp_path)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Push exact candidate",
            assignee="coder",
            workspace_kind="worktree",
            workspace_path=str(repo),
            delivery_target="push",
        )
        first = kb.claim_task(conn, task_id, claimer="first")
        assert first is not None and first.current_run_id is not None
        assert kb.reclaim_task(conn, task_id, reason="start successor")
        second = kb.claim_task(conn, task_id, claimer="second")
        assert second is not None and second.current_run_id is not None

        # This metadata deliberately fails the remote-integration gate. The
        # obsolete worker must still be a strict no-op: before the CAS fix it
        # parked the successor in review and closed its run.
        assert not kb.complete_task(
            conn,
            task_id,
            expected_run_id=first.current_run_id,
            metadata={
                "integration": {
                    "repo_path": str(repo),
                    "target_remote": "origin",
                    "target_branch": "main",
                    "commit": candidate,
                },
            },
        )

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "running"
        assert task.current_run_id == second.current_run_id
        assert kb.latest_run(conn, task_id).ended_at is None
        assert not any(
            event.kind == "awaiting_integration"
            for event in kb.list_events(conn, task_id)
        )


def test_commit_target_is_verified_locally_without_remote_or_global_dirty_gate(
    kanban_home, tmp_path,
):
    repo, _candidate = _repo_with_remote(tmp_path)
    (repo / "unrelated.txt").write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", "unrelated.txt")
    _git(repo, "commit", "-q", "-m", "local baseline")
    baseline = _git(repo, "rev-parse", "HEAD")
    (repo / "result.txt").write_text("delivered locally\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "local delivery")
    delivered = _git(repo, "rev-parse", "HEAD")
    # User-owned tracked dirt outside the declared delivery scope must not
    # recreate the old whole-worktree development-guard false wait.
    (repo / "unrelated.txt").write_text("pre-existing user edit\n", encoding="utf-8")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Keep a verified local commit",
            assignee="coder",
            workspace_kind="dir",
            workspace_path=str(repo),
            delivery_target="commit",
        )
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        assert kb.complete_task(
            conn,
            task_id,
            expected_run_id=claimed.current_run_id,
            metadata={
                "evidence": {"kind": "test", "detail": "focused tests passed"},
                "integration": {
                    "repo_path": str(repo),
                    "base_commit": baseline,
                    "commit": delivered,
                    "changed_files": ["result.txt"],
                },
            },
        )
        completed = kb.get_task(conn, task_id)
        assert completed is not None and completed.status == "done"

    remote_head = _git(repo, "ls-remote", "origin", "refs/heads/main").split()[0]
    assert remote_head != delivered


def test_commit_target_rejects_missing_or_incoherent_commit(kanban_home, tmp_path):
    repo, delivered = _repo_with_remote(tmp_path)
    baseline = _git(repo, "rev-parse", "HEAD~1")
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Require a real local commit",
            assignee="coder",
            workspace_kind="dir",
            workspace_path=str(repo),
            delivery_target="commit",
        )
        task = kb.get_task(conn, task_id)

    error, _ = kb._integration_delivery_projection(task, {})
    assert error == "commit required but no delivered commit was declared"

    error, _ = kb._integration_delivery_projection(task, {
        "integration": {
            "repo_path": str(repo),
            "base_commit": delivered,
            "commit": delivered,
        },
    })
    assert error == "delivered commit does not advance the declared base_commit"

    # A pre-existing HEAD is not task delivery evidence by itself.
    error, _ = kb._integration_delivery_projection(task, {
        "integration": {"repo_path": str(repo), "commit": delivered},
    })
    assert error is not None and "commit proof is incomplete" in error

    # Without a persisted baseline, exact changed paths form the compatible
    # proof: every path must be clean and actually occur in the HEAD commit.
    error, projected = kb._integration_delivery_projection(task, {
        "integration": {
            "repo_path": str(repo),
            "commit": delivered,
            "changed_files": ["result.txt"],
        },
    })
    assert error is None
    assert projected["integration"]["status"] == "committed"

    error, _ = kb._integration_delivery_projection(task, {
        "integration": {
            "repo_path": str(repo),
            "commit": delivered,
            "changed_files": ["not-in-delivery.txt"],
        },
    })
    assert error is not None and "absent from the delivered commit diff" in error

    error, projected = kb._integration_delivery_projection(task, {
        "integration": {
            "repo_path": str(repo),
            "base_commit": baseline,
            "commit": delivered,
        },
    })
    assert error is None
    assert projected["integration"]["status"] == "committed"

    (repo / "result.txt").write_text("new uncommitted edit\n", encoding="utf-8")
    error, _ = kb._integration_delivery_projection(task, {
        "integration": {
            "repo_path": str(repo),
            "commit": delivered,
            "changed_files": ["result.txt"],
        },
    })
    assert error == "declared changed_files still contain uncommitted changes"


def test_deploy_rejects_free_form_commit_or_deployment_assertions(
    kanban_home, tmp_path,
):
    repo, candidate = _repo_with_remote(tmp_path)
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "candidate")
    _git(repo, "push", "-q", "origin", "main")
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Deploy exact candidate",
            assignee="coder",
            workspace_kind="worktree",
            workspace_path=str(repo),
            delivery_target="deploy",
        )
        task = kb.get_task(conn, task_id)

    integration = {
        "repo_path": str(repo),
        "target_remote": "origin",
        "target_branch": "main",
        "commit": candidate,
    }
    error, _ = kb._integration_delivery_projection(task, {
        "integration": integration,
        "production_proof": {"commit": candidate},
    })
    assert error is not None and "evidence_path is missing" in error

    error, _ = kb._integration_delivery_projection(task, {
        "integration": integration,
        "production_proof": {
            "commit": candidate,
            "deployment_id": "deploy-123",
            "verdict": "SUCCESS",
        },
    })
    assert error is not None and "evidence_path is missing" in error


def test_deploy_accepts_successful_schema_evidence_tied_to_commit(
    kanban_home, tmp_path,
):
    repo, candidate = _repo_with_remote(tmp_path)
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "candidate")
    _git(repo, "push", "-q", "origin", "main")
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Deploy with evidence",
            assignee="coder",
            workspace_kind="worktree",
            workspace_path=str(repo),
            delivery_target="deploy",
        )
        task = kb.get_task(conn, task_id)

    proof_path = tmp_path / "production-proof.json"
    proof_path.write_text(json.dumps({
        "schema": "hermes.production-proof.v1",
        "task_id": task_id,
        "commit": candidate,
        "url": "https://example.invalid/release",
        "verdict": "OK",
    }), encoding="utf-8")
    error, _ = kb._integration_delivery_projection(task, {
        "integration": {
            "repo_path": str(repo),
            "target_remote": "origin",
            "target_branch": "main",
            "commit": candidate,
        },
        "production_proof": {
            "commit": candidate,
            "evidence_path": str(proof_path),
        },
    })
    assert error is None

    proof_path.write_text(json.dumps({
        "schema": "hermes.production-proof.v1",
        "task_id": task_id,
        "commit": "0" * 40,
        "url": "https://example.invalid/release",
        "verdict": "OK",
    }), encoding="utf-8")
    error, _ = kb._integration_delivery_projection(task, {
        "integration": {
            "repo_path": str(repo),
            "target_remote": "origin",
            "target_branch": "main",
            "commit": candidate,
        },
        "production_proof": {
            "commit": candidate,
            "evidence_path": str(proof_path),
        },
    })
    assert error == "production_proof evidence does not match the delivered commit"
