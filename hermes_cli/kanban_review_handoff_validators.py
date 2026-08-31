"""Plugin-facing Kanban review-handoff projection seam.

This module owns the pre-request-review extension point used by Kanban core,
the CLI/manual surface, and worker tools.  Validators run after the built-in
visual-review handoff normalisation but before any durable review mutation;
they may veto the handoff or project normalized metadata/reviewer values.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect
import logging
import threading
from types import MappingProxyType
from typing import Any, Mapping

logger = logging.getLogger(__name__)

HOOK_NAME = "validate_kanban_review_handoff"
_validation_depth = threading.local()


@dataclass(frozen=True)
class KanbanReviewHandoffContext:
    """Typed, minimal context exposed to review-handoff plugins."""

    task_id: str
    title: str | None
    body: str | None
    created_by: str | None
    board: str | None
    prior_status: str | None
    assignee: str | None
    run_id: int | None
    summary: str | None
    metadata: Mapping[str, Any]
    reviewer: str | None
    source: str
    surface: str


@dataclass(frozen=True)
class KanbanReviewHandoffResult:
    """Normalized review-handoff validator outcome."""

    accepted: bool
    reason: str | None = None
    code: str | None = None
    metadata: Mapping[str, Any] | None = None
    reviewer: str | None = None

    @classmethod
    def accept(
        cls,
        metadata: Mapping[str, Any] | None = None,
        *,
        reviewer: str | None = None,
    ) -> "KanbanReviewHandoffResult":
        return cls(accepted=True, metadata=metadata, reviewer=reviewer)

    @classmethod
    def veto(
        cls,
        reason: str,
        *,
        code: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        reviewer: str | None = None,
    ) -> "KanbanReviewHandoffResult":
        return cls(
            accepted=False,
            reason=reason,
            code=code,
            metadata=metadata,
            reviewer=reviewer,
        )


class KanbanReviewHandoffValidationError(ValueError):
    """Raised when a validator blocks or the seam must fail closed."""

    def __init__(self, reason: str, *, code: str = "validator_rejected"):
        self.reason = reason
        self.code = code
        super().__init__(reason)


def project_kanban_review_handoff(
    context: KanbanReviewHandoffContext,
) -> tuple[dict[str, Any], str | None, list[KanbanReviewHandoffResult]]:
    """Run review-handoff validators and return projected metadata/reviewer.

    No registered validators is an accept: the seam is optional. Once a
    validator is present, malformed returns, explicit vetoes, exceptions, and
    recursive validation attempts fail closed before any Kanban mutation.
    """
    depth = int(getattr(_validation_depth, "value", 0) or 0)
    if depth > 0:
        error = KanbanReviewHandoffValidationError(
            "recursive Kanban review handoff validation is not allowed",
            code="validator_reentrant",
        )
        _validation_depth.reentrant_error = error
        raise error
    _validation_depth.reentrant_error = None

    try:
        from hermes_cli.plugins import get_plugin_manager, has_hook
    except Exception as exc:  # pragma: no cover - import failure is fail-closed by contract
        raise KanbanReviewHandoffValidationError(
            f"review handoff validator discovery failed: {exc}",
            code="validator_discovery_error",
        ) from exc

    try:
        if not has_hook(HOOK_NAME):
            return dict(context.metadata), context.reviewer, []
    except Exception as exc:
        raise KanbanReviewHandoffValidationError(
            f"review handoff validator discovery failed: {exc}",
            code="validator_discovery_error",
        ) from exc

    try:
        manager = get_plugin_manager()
        callbacks = tuple(getattr(manager, "iter_hook_callbacks")(HOOK_NAME))
    except Exception as exc:
        raise KanbanReviewHandoffValidationError(
            f"review handoff validator discovery failed: {exc}",
            code="validator_discovery_error",
        ) from exc
    metadata = dict(context.metadata)
    reviewer = context.reviewer
    outcomes: list[KanbanReviewHandoffResult] = []
    if not callbacks:
        return metadata, reviewer, outcomes

    _validation_depth.value = depth + 1
    try:
        for callback in callbacks:
            callback_name = getattr(callback, "__name__", repr(callback))
            current_context = KanbanReviewHandoffContext(
                task_id=context.task_id,
                title=context.title,
                body=context.body,
                created_by=context.created_by,
                board=context.board,
                prior_status=context.prior_status,
                assignee=context.assignee,
                run_id=context.run_id,
                summary=context.summary,
                metadata=MappingProxyType(dict(metadata)),
                reviewer=reviewer,
                source=context.source,
                surface=context.surface,
            )
            try:
                raw = _invoke_validator_callback(callback, current_context)
                reentrant_error = getattr(_validation_depth, "reentrant_error", None)
                if reentrant_error is not None:
                    _validation_depth.reentrant_error = None
                    raise reentrant_error
                outcome = _normalize_validator_result(
                    raw,
                    current_metadata=metadata,
                    current_reviewer=reviewer,
                )
            except KanbanReviewHandoffValidationError:
                raise
            except Exception as exc:
                logger.warning(
                    "Kanban review handoff validator %s raised: %s",
                    callback_name,
                    exc,
                )
                raise KanbanReviewHandoffValidationError(
                    f"review handoff validator {callback_name} raised: {exc}",
                    code="validator_exception",
                ) from exc
            outcomes.append(outcome)
            if outcome.metadata is not None:
                metadata = dict(outcome.metadata)
            reviewer = outcome.reviewer
            if not outcome.accepted:
                reason = outcome.reason or f"review handoff validator {callback_name} rejected handoff"
                raise KanbanReviewHandoffValidationError(
                    reason,
                    code=outcome.code or "validator_rejected",
                )
    finally:
        _validation_depth.value = depth
        if depth == 0:
            _validation_depth.reentrant_error = None

    return metadata, reviewer, outcomes


def _invoke_validator_callback(callback: Any, context: KanbanReviewHandoffContext) -> Any:
    payload = {
        "context": context,
        "task_id": context.task_id,
        "title": context.title,
        "body": context.body,
        "created_by": context.created_by,
        "board": context.board,
        "prior_status": context.prior_status,
        "assignee": context.assignee,
        "run_id": context.run_id,
        "summary": context.summary,
        "metadata": context.metadata,
        "reviewer": context.reviewer,
        "source": context.source,
        "surface": context.surface,
    }
    try:
        parameters = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        return callback(**payload)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return callback(**payload)
    accepted = {
        name: value
        for name, value in payload.items()
        if name in parameters
        and parameters[name].kind
        in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    return callback(**accepted)


def _normalize_reviewer(raw: Any, *, current_reviewer: str | None) -> str | None:
    if raw is None:
        return current_reviewer
    if isinstance(raw, str):
        return raw
    raise KanbanReviewHandoffValidationError(
        "review handoff validator returned non-string reviewer",
        code="validator_bad_result",
    )


def _normalize_validator_result(
    raw: Any,
    *,
    current_metadata: Mapping[str, Any],
    current_reviewer: str | None,
) -> KanbanReviewHandoffResult:
    if raw is None or isinstance(raw, bool):
        raise KanbanReviewHandoffValidationError(
            "review handoff validator must return an explicit Result or mapping decision",
            code="validator_bad_result",
        )
    if isinstance(raw, KanbanReviewHandoffResult):
        if not isinstance(raw.accepted, bool):
            raise KanbanReviewHandoffValidationError(
                "review handoff validator Result.accepted must be bool",
                code="validator_bad_result",
            )
        if raw.metadata is not None and not isinstance(raw.metadata, Mapping):
            raise KanbanReviewHandoffValidationError(
                "review handoff validator Result.metadata must be a mapping",
                code="validator_bad_result",
            )
        if raw.reason is not None and not isinstance(raw.reason, str):
            raise KanbanReviewHandoffValidationError(
                "review handoff validator Result.reason must be a string",
                code="validator_bad_result",
            )
        if raw.code is not None and not isinstance(raw.code, str):
            raise KanbanReviewHandoffValidationError(
                "review handoff validator Result.code must be a string",
                code="validator_bad_result",
            )
        reviewer = _normalize_reviewer(raw.reviewer, current_reviewer=current_reviewer)
        return KanbanReviewHandoffResult(
            accepted=raw.accepted,
            reason=raw.reason,
            code=raw.code,
            metadata=dict(raw.metadata) if raw.metadata is not None else None,
            reviewer=reviewer,
        )
    if isinstance(raw, Mapping):
        action = raw.get("action")
        decision_keys = [key for key in ("accepted", "ok", "allow") if key in raw]
        explicit_decisions = {key: raw[key] for key in decision_keys}
        if isinstance(action, str):
            normalized_action = action.strip().lower()
            expected = None
            if normalized_action in {"block", "veto", "reject", "deny"}:
                expected = False
            elif normalized_action == "allow":
                expected = True
            if expected is not None:
                for value in explicit_decisions.values():
                    if not isinstance(value, bool) or value is not expected:
                        raise KanbanReviewHandoffValidationError(
                            "review handoff validator returned contradictory decisions",
                            code="validator_bad_result",
                        )
            if normalized_action in {"block", "veto", "reject", "deny"}:
                reason = raw.get("reason") or raw.get("message") or "review handoff validator rejected handoff"
                code = raw.get("code") or "validator_rejected"
                reviewer = _normalize_reviewer(raw.get("reviewer"), current_reviewer=current_reviewer)
                return KanbanReviewHandoffResult.veto(
                    str(reason), code=str(code), reviewer=reviewer
                )
            if normalized_action == "allow":
                projected = raw.get("metadata", current_metadata)
                if projected is not None and not isinstance(projected, Mapping):
                    raise KanbanReviewHandoffValidationError(
                        "review handoff validator returned non-object metadata",
                        code="validator_bad_result",
                    )
                reviewer = _normalize_reviewer(raw.get("reviewer"), current_reviewer=current_reviewer)
                return KanbanReviewHandoffResult.accept(
                    dict(projected or {}), reviewer=reviewer
                )
            raise KanbanReviewHandoffValidationError(
                f"review handoff validator returned unsupported action: {action}",
                code="validator_bad_result",
            )
        if len(decision_keys) != 1 or not isinstance(raw[decision_keys[0]], bool):
            raise KanbanReviewHandoffValidationError(
                "review handoff validator mapping must include exactly one explicit boolean decision",
                code="validator_bad_result",
            )
        accepted_raw = raw[decision_keys[0]]
        accepted = bool(accepted_raw)
        reason = raw.get("reason") or raw.get("message")
        code = raw.get("code")
        projected = raw.get("metadata", current_metadata)
        if projected is not None and not isinstance(projected, Mapping):
            raise KanbanReviewHandoffValidationError(
                "review handoff validator returned non-object metadata",
                code="validator_bad_result",
            )
        reviewer = _normalize_reviewer(raw.get("reviewer"), current_reviewer=current_reviewer)
        return KanbanReviewHandoffResult(
            accepted=accepted,
            reason=str(reason) if reason is not None else None,
            code=str(code) if code is not None else None,
            metadata=dict(projected or {}),
            reviewer=reviewer,
        )
    raise KanbanReviewHandoffValidationError(
        f"review handoff validator returned unsupported result type: {type(raw).__name__}",
        code="validator_bad_result",
    )
