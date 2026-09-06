"""Compatibility helpers for external Kanban closure-proof detectors.

Core Kanban completion policy now lives in plugin validators.  This module is a
small read-only classifier kept for external ``transform_llm_output`` guards
that need to recognize already-completed Kanban runs carrying structured proof.
It does not veto completion and does not contain any textual output gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ClosureEvidence:
    """Normalized, read-only closure proof classification."""

    satisfied: bool
    kind: str = ""
    detail: str = ""


def classify_closure_evidence(
    *,
    prior_status: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ClosureEvidence:
    """Classify durable completion evidence without enforcing a gate.

    ``prior_status`` is accepted for source compatibility with the former
    runtime gate API.  The current implementation only reads metadata already
    attached to a completed run: explicit ``metadata.evidence`` first, then
    durable artifacts, verification/proof facts, then review handoff facts and
    reviewer checks.
    """
    data = metadata if isinstance(metadata, Mapping) else {}
    evidence = data.get("evidence")
    if isinstance(evidence, Mapping):
        detail = _clean(evidence.get("detail"))
        kind = _clean(evidence.get("kind")) or "evidence"
        if detail:
            return ClosureEvidence(True, kind=kind, detail=detail)

    artifacts = data.get("artifacts")
    if isinstance(artifacts, (list, tuple)) and artifacts:
        return ClosureEvidence(
            True,
            kind="artifacts",
            detail=f"{len(artifacts)} artifact(s) declared",
        )

    verification = _verification_detail(data.get("verification"))
    if verification:
        return ClosureEvidence(True, kind="verification", detail=verification)

    proof = _clean(data.get("proof"))
    if proof:
        return ClosureEvidence(True, kind="proof", detail=proof)

    review = data.get("review")
    if isinstance(review, Mapping):
        detail = _clean(review.get("summary") or review.get("detail"))
        if detail:
            return ClosureEvidence(True, kind="review", detail=detail)

    reviewer_checks = data.get("reviewer_checks")
    if isinstance(reviewer_checks, (list, tuple)):
        checks = [_clean(item) for item in reviewer_checks]
        checks = [item for item in checks if item]
        if checks:
            return ClosureEvidence(
                True,
                kind="review",
                detail=" | ".join(checks),
            )

    visual = data.get("visual_review")
    if isinstance(visual, Mapping):
        screenshots = visual.get("screenshots")
        if isinstance(screenshots, Mapping) and screenshots:
            return ClosureEvidence(
                True,
                kind="visual_review",
                detail=f"{len(screenshots)} screenshot(s) declared",
            )

    return ClosureEvidence(False)


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _verification_detail(value: object) -> str:
    """Normalize legacy structured verification metadata.

    Some completed Kanban workers persisted proof under a top-level
    ``verification`` key rather than the newer ``evidence``/``artifacts``
    shapes.  Treat only non-empty structured verification values as proof;
    a bare output file path still does not satisfy the classifier.
    """
    if isinstance(value, Mapping):
        parts = [f"{key}: {_clean(item)}" for key, item in value.items() if _clean(item)]
        return " | ".join(parts)
    if isinstance(value, (list, tuple)):
        parts = [_clean(item) for item in value if _clean(item)]
        return " | ".join(parts)
    return _clean(value)
