from hermes_cli.closure_evidence import classify_closure_evidence


def test_classifies_explicit_metadata_evidence():
    evidence = classify_closure_evidence(
        metadata={"evidence": {"kind": "test", "detail": "pytest -q — 7 passed"}}
    )

    assert evidence.satisfied is True
    assert evidence.kind == "test"
    assert evidence.detail == "pytest -q — 7 passed"


def test_classifies_artifact_metadata_without_reintroducing_gate_policy():
    evidence = classify_closure_evidence(metadata={"artifacts": ["/tmp/report.pdf"]})

    assert evidence.satisfied is True
    assert evidence.kind == "artifacts"
    assert "1 artifact" in evidence.detail


def test_reviewer_checks_are_normalized_as_closure_evidence():
    evidence = classify_closure_evidence(
        metadata={
            "reviewer_checks": [
                "GA4 live API: devis_form_submit custom=True",
                "origin/main == 0dc5047",
            ]
        }
    )

    assert evidence.satisfied is True
    assert evidence.kind == "review"
    assert "GA4 live API" in evidence.detail
    assert "0dc5047" in evidence.detail


def test_classifies_legacy_lot_c_verification_metadata():
    evidence = classify_closure_evidence(
        metadata={
            "file": "/home/seb/.hermes/kanban/workspaces/t_bde04777/lot_c_validation.txt",
            "profile_resolved": "coder",
            "validation_string": "lot C validé",
            "verification": {
                "read_file": "ligne 1: Profil réellement résolu : coder; ligne 2: Chaîne de validation : lot C validé",
                "wc": "73 octets",
            },
        }
    )

    assert evidence.satisfied is True
    assert evidence.kind == "verification"
    assert "lot C validé" in evidence.detail
    assert "73 octets" in evidence.detail


def test_classifies_legacy_lot_d_proof_metadata():
    evidence = classify_closure_evidence(
        metadata={
            "file": "/home/seb/.hermes/kanban/workspaces/t_90cac273/lot_d_validation.txt",
            "proof": "read_file a confirmé les lignes: profil réellement résolu coder2 et « lot D validé ».",
        }
    )

    assert evidence.satisfied is True
    assert evidence.kind == "proof"
    assert "lot D validé" in evidence.detail


def test_file_path_alone_is_not_closure_evidence():
    evidence = classify_closure_evidence(
        metadata={"file": "/home/seb/.hermes/kanban/workspaces/t_x/file.txt"}
    )

    assert evidence.satisfied is False
    assert evidence.kind == ""


def test_refuses_empty_metadata():
    evidence = classify_closure_evidence(prior_status="review", metadata={})

    assert evidence.satisfied is False
    assert evidence.kind == ""
    assert evidence.detail == ""
