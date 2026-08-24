"""Closure evidence gate — LOT 4 (authority & closure proof).

A Kanban card must not become ``done`` on bare assertion. Every completion
attempt routed through the CLI (`hermes kanban complete`) or the
`kanban_complete` tool must carry at least one *structured* evidence kind:

- ``review``   — the card was sitting in the ``review`` lane (reached via
                  :func:`hermes_cli.kanban_db.request_review`) and is now
                  being approved. Landing in ``review`` already required a
                  worker-authored summary describing verification; a human
                  or reviewer profile closing it from there IS the proof.
- ``artifact`` — ``metadata["artifacts"]`` is a non-empty list of real
                  paths (the same field `kanban_complete` already uploads
                  as native attachments — see tools/kanban_tools.py).
- ``test``     — ``metadata["evidence"] == {"kind": "test", "detail": ...}``,
                  naming the concrete test command/run and its outcome.
- ``canary``   — ``metadata["evidence"] == {"kind": "canary", "detail": ...}``,
                  naming a live probe result (e.g. the LOT 2 OAuth canary).

Deliberately NOT accepted as evidence: free-form ``summary``/``result``
prose. Sniffing prose for words like "tests passed" is exactly the silent,
unverifiable shortcut this gate exists to close off — every claim must be
backed by a structured field the next reader (human or worker) can check
without re-deriving trust from adjectives.

Pure functions only: no DB access, no I/O. The caller (``hermes_cli.kanban``
/ ``tools.kanban_tools``) is responsible for reading the card's current
status before the completion write lands and passing it in as
``prior_status``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# Ordered for messages only; every kind here is equally sufficient.
STRUCTURED_EVIDENCE_KINDS = ("review", "artifact", "test", "canary")

# metadata["evidence"]["kind"] values accepted in addition to the
# artifact/review shortcuts above.
_FREEFORM_EVIDENCE_KINDS = ("test", "canary")

REJECTION_MESSAGE = (
    "no structured closure evidence found — a card can only become 'done' "
    "with one of: metadata.evidence={{'kind': 'test'|'canary', 'detail': "
    "'<concrete proof>'}}, a non-empty metadata.artifacts=[...], or by being "
    "approved from the review lane (kanban_request_review, then "
    "kanban_complete). Provide one, or call kanban_request_review / "
    "kanban_block instead of asserting completion."
)


@dataclass(frozen=True)
class ClosureEvidence:
    """The evidence kind found for a completion attempt, if any."""

    kind: Optional[str]
    detail: Optional[str] = None

    @property
    def satisfied(self) -> bool:
        return self.kind is not None


def classify_closure_evidence(
    *,
    prior_status: Optional[str],
    metadata: Optional[dict[str, Any]],
) -> ClosureEvidence:
    """Determine what structured evidence (if any) backs this completion.

    ``prior_status`` is the card's status *before* this completion call
    (fetched by the caller via ``kb.get_task`` prior to ``kb.complete_task``).
    """
    if prior_status == "review":
        return ClosureEvidence(kind="review", detail="approved from the review lane")

    meta = metadata if isinstance(metadata, dict) else {}

    artifacts = meta.get("artifacts")
    if isinstance(artifacts, (list, tuple)):
        real = [str(a).strip() for a in artifacts if str(a).strip()]
        if real:
            return ClosureEvidence(
                kind="artifact", detail=f"{len(real)} artifact(s): {', '.join(real[:3])}"
            )

    evidence = meta.get("evidence")
    if isinstance(evidence, dict):
        kind = evidence.get("kind")
        detail = evidence.get("detail")
        if kind in _FREEFORM_EVIDENCE_KINDS and isinstance(detail, str) and detail.strip():
            return ClosureEvidence(kind=kind, detail=detail.strip())

    return ClosureEvidence(kind=None, detail=None)


def closure_gate(
    *,
    prior_status: Optional[str],
    metadata: Optional[dict[str, Any]],
) -> Optional[str]:
    """Return a rejection message, or ``None`` if completion may proceed."""
    evidence = classify_closure_evidence(prior_status=prior_status, metadata=metadata)
    if evidence.satisfied:
        return None
    return REJECTION_MESSAGE
