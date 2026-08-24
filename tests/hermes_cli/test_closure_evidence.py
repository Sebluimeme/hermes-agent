"""Tests for the LOT 4 closure evidence gate (hermes_cli/closure_evidence.py).

Pure-function tests only — no DB, no CLI. Wiring into `hermes kanban
complete` / `kanban_complete` is covered separately in
test_kanban_closure_gate.py so a failure here isolates the decision table
from the call sites.
"""
from __future__ import annotations

import pytest

from hermes_cli.closure_evidence import classify_closure_evidence, closure_gate


class TestClassifyClosureEvidence:
    def test_no_metadata_no_prior_review_is_unsatisfied(self):
        ev = classify_closure_evidence(prior_status="running", metadata=None)
        assert ev.kind is None
        assert not ev.satisfied

    def test_prior_review_status_satisfies_via_review(self):
        ev = classify_closure_evidence(prior_status="review", metadata=None)
        assert ev.kind == "review"
        assert ev.satisfied

    def test_prior_review_wins_even_with_no_metadata_at_all(self):
        # Landing in review already required a verifiable summary; approving
        # from there is proof on its own, metadata is not required.
        ev = classify_closure_evidence(prior_status="review", metadata={})
        assert ev.kind == "review"

    def test_nonempty_artifacts_list_satisfies_via_artifact(self):
        ev = classify_closure_evidence(
            prior_status="running",
            metadata={"artifacts": ["/tmp/out.log"]},
        )
        assert ev.kind == "artifact"
        assert "/tmp/out.log" in ev.detail

    def test_empty_artifacts_list_does_not_satisfy(self):
        ev = classify_closure_evidence(
            prior_status="running", metadata={"artifacts": []}
        )
        assert ev.kind is None

    def test_artifacts_list_of_blank_strings_does_not_satisfy(self):
        ev = classify_closure_evidence(
            prior_status="running", metadata={"artifacts": ["  ", ""]}
        )
        assert ev.kind is None

    def test_evidence_kind_test_with_detail_satisfies(self):
        ev = classify_closure_evidence(
            prior_status="running",
            metadata={"evidence": {"kind": "test", "detail": "pytest -q: 12 passed"}},
        )
        assert ev.kind == "test"
        assert ev.detail == "pytest -q: 12 passed"

    def test_evidence_kind_canary_with_detail_satisfies(self):
        ev = classify_closure_evidence(
            prior_status="running",
            metadata={"evidence": {"kind": "canary", "detail": "probe_claude2_oauth: OK"}},
        )
        assert ev.kind == "canary"

    def test_evidence_kind_test_without_detail_does_not_satisfy(self):
        ev = classify_closure_evidence(
            prior_status="running",
            metadata={"evidence": {"kind": "test", "detail": ""}},
        )
        assert ev.kind is None

    def test_evidence_unknown_kind_does_not_satisfy(self):
        ev = classify_closure_evidence(
            prior_status="running",
            metadata={"evidence": {"kind": "vibes", "detail": "trust me"}},
        )
        assert ev.kind is None

    def test_evidence_kind_review_via_metadata_alone_does_not_satisfy(self):
        # "review" is only granted via prior_status (an actual reviewer
        # workflow), never by a worker self-declaring it in metadata.
        ev = classify_closure_evidence(
            prior_status="running",
            metadata={"evidence": {"kind": "review", "detail": "looks good"}},
        )
        assert ev.kind is None

    def test_metadata_not_a_dict_is_ignored_safely(self):
        ev = classify_closure_evidence(prior_status="running", metadata="oops")  # type: ignore[arg-type]
        assert ev.kind is None

    def test_evidence_not_a_dict_is_ignored_safely(self):
        ev = classify_closure_evidence(
            prior_status="running", metadata={"evidence": "trust me"}
        )
        assert ev.kind is None


class TestClosureGate:
    def test_gate_blocks_bare_completion(self):
        msg = closure_gate(prior_status="running", metadata=None)
        assert msg is not None
        assert "structured closure evidence" in msg

    def test_gate_names_all_accepted_kinds_in_rejection(self):
        msg = closure_gate(prior_status="running", metadata=None)
        for kind in ("test", "canary", "kanban_request_review"):
            assert kind in msg

    @pytest.mark.parametrize(
        "prior_status,metadata",
        [
            ("review", None),
            ("running", {"artifacts": ["/tmp/a.txt"]}),
            ("ready", {"evidence": {"kind": "test", "detail": "12 passed"}}),
            ("blocked", {"evidence": {"kind": "canary", "detail": "probe OK"}}),
        ],
    )
    def test_gate_allows_any_structured_evidence_kind(self, prior_status, metadata):
        assert closure_gate(prior_status=prior_status, metadata=metadata) is None
