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


def test_refuses_empty_metadata():
    evidence = classify_closure_evidence(prior_status="review", metadata={})

    assert evidence.satisfied is False
    assert evidence.kind == ""
    assert evidence.detail == ""
