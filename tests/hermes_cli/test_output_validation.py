from hermes_cli import kanban_db as kb
from hermes_cli.output_validation import (
    OUTPUT_VALIDATION_RETRY_SIGNAL,
    SAFE_UNVERIFIED_RESPONSE,
    apply_output_validation_fallback,
    completed_kanban_task_id,
    output_validation_recovery_prompt,
    requests_output_validation_retry,
    verified_kanban_completion_message,
)


def test_private_retry_protocol_never_uses_public_guard_text():
    assert requests_output_validation_retry({"output_validation_retry": True})
    assert not requests_output_validation_retry({"output_validation_retry": False})
    assert "output-proof-guard" not in OUTPUT_VALIDATION_RETRY_SIGNAL
    assert "output-proof-guard" not in SAFE_UNVERIFIED_RESPONSE


def test_verified_kanban_completion_is_rendered_from_database(tmp_path):
    db_path = tmp_path / "kanban.db"
    with kb.connect(db_path) as conn:
        mission_id = kb.ensure_mission(
            conn, title="mission", request_text="corriger", idempotency_key="one"
        )
        task_id = kb.create_task(conn, title="corriger", mission_id=mission_id)
        assert kb.complete_task(
            conn,
            task_id,
            summary="Le correctif est rangé proprement",
            metadata={
                "evidence": {
                    "kind": "test",
                    "detail": "python -m unittest -v — 7/7 OK, commit d47423d",
                }
            },
        )
        assert kb.mark_task_delivered(conn, task_id)

    inbound = f"[kanban] Task {task_id} completed.\nResult: internal"
    rendered = verified_kanban_completion_message(inbound, db_path=db_path)
    assert rendered is not None
    assert "7/7 OK" in rendered
    assert "d47423d" in rendered
    assert "Le correctif est rangé proprement" in rendered
    assert "1/1 cartes terminées et livrées" in rendered
    assert "output-proof-guard" not in rendered


def test_verified_kanban_completion_reports_the_whole_delivered_mission(tmp_path):
    db_path = tmp_path / "kanban.db"
    with kb.connect(db_path) as conn:
        mission_id = kb.ensure_mission(
            conn,
            title="Corriger puis ranger",
            request_text="faire tout",
            idempotency_key="all",
        )
        parent = kb.create_task(conn, title="corriger", mission_id=mission_id)
        assert kb.complete_task(
            conn,
            parent,
            summary="Correctif fonctionnel",
            metadata={"evidence": {"kind": "test", "detail": "7/7 tests OK"}},
        )
        assert kb.mark_task_delivered(conn, parent)
        child = kb.create_task(conn, title="ranger", parents=[parent])
        assert kb.complete_task(
            conn,
            child,
            summary="Rangement et commit propres",
            metadata={"evidence": {"kind": "commit", "detail": "commit d47423d"}},
        )
        assert kb.mark_task_delivered(conn, child)

    rendered = verified_kanban_completion_message(
        f"[kanban] Task {child} completed.", db_path=db_path
    )
    assert rendered is not None
    assert "Corriger puis ranger" in rendered
    assert "2/2 cartes terminées et livrées" in rendered
    assert "Rangement et commit propres" in rendered
    assert "7/7 tests OK" in rendered
    assert "commit d47423d" in rendered
    assert rendered.count(".") == 1


def test_kanban_fallback_refuses_a_reopened_mission(tmp_path):
    db_path = tmp_path / "kanban.db"
    with kb.connect(db_path) as conn:
        mission_id = kb.ensure_mission(
            conn, title="mission", request_text="corriger", idempotency_key="two"
        )
        parent = kb.create_task(conn, title="corriger", mission_id=mission_id)
        assert kb.complete_task(
            conn,
            parent,
            summary="correctif vérifié",
            metadata={"evidence": {"kind": "test", "detail": "7/7 OK"}},
        )
        assert kb.mark_task_delivered(conn, parent)
        kb.create_task(conn, title="ranger", parents=[parent])

    inbound = f"[kanban] Task {parent} completed."
    assert verified_kanban_completion_message(inbound, db_path=db_path) is None


def test_recovery_prompt_is_private_and_carries_durable_proof():
    proof = "C’est terminé, preuve vérifiée : 7/7 OK, commit d47423d."
    prompt = output_validation_recovery_prompt(
        rejected_response="C’est terminé.", verified_fallback=proof
    )
    assert prompt.startswith("[HERMES_RECOVERY_OUTPUT_VALIDATION]")
    assert proof in prompt
    assert "Ne mentionne jamais le garde interne" in prompt


def test_apply_output_validation_fallback_uses_structured_kanban_state(tmp_path):
    db_path = tmp_path / "kanban.db"
    with kb.connect(db_path) as conn:
        task_id = kb.create_task(conn, title="corriger")
        assert kb.complete_task(
            conn,
            task_id,
            summary="Correctif fonctionnel",
            metadata={"evidence": {"kind": "test", "detail": "Ran 7 tests, OK"}},
        )
        assert kb.mark_task_delivered(conn, task_id)

    result = {"output_validation_retry": True, "final_response": "safe"}

    assert apply_output_validation_fallback(
        result, f"[kanban] Task {task_id} completed.", db_path=db_path
    )
    assert result["output_validation_retry"] is False
    assert result["output_validation_structured_fallback"] is True
    assert "Correctif fonctionnel" in result["final_response"]
    assert "Ran 7 tests, OK" in result["final_response"]


def test_apply_output_validation_fallback_stops_after_unproven_repair():
    result = {"output_validation_retry": True, "final_response": "C'est fini."}

    assert apply_output_validation_fallback(result, "message ordinaire")
    assert result["output_validation_retry"] is False
    assert result["final_response"] == SAFE_UNVERIFIED_RESPONSE


def test_completed_task_id_only_accepts_internal_notification_shape():
    assert completed_kanban_task_id("[kanban] Task t_7c465190 completed.") == "t_7c465190"
    assert completed_kanban_task_id("la tâche t_7c465190 est terminée") is None
