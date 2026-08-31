"""Plugin-facing Kanban completion validation seam.

This module owns the pre-completion extension point used by Kanban core,
the CLI/manual surface, and worker tools.  Validators run before any durable
completion mutation and may veto the handoff or project normalized metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect
import logging
import threading
from types import MappingProxyType
from typing import Any, Mapping

logger = logging.getLogger(__name__)

HOOK_NAME = "validate_kanban_completion"
_validation_depth = threading.local()


@dataclass(frozen=True)
class KanbanCompletionValidationContext:
    """Typed, minimal context exposed to completion-validator plugins."""

    task_id: str
    title: str | None
    body: str | None
    created_by: str | None
    board: str | None
    prior_status: str | None
    assignee: str | None
    run_id: int | None
    summary: str | None
    result: str | None
    metadata: Mapping[str, Any]
    created_cards: tuple[str, ...]
    source: str
    surface: str
    dry_run: bool = False


@dataclass(frozen=True)
class KanbanCompletionValidationResult:
    """Normalized validator outcome."""

    accepted: bool
    reason: str | None = None
    code: str | None = None
    metadata: Mapping[str, Any] | None = None

    @classmethod
    def accept(cls, metadata: Mapping[str, Any] | None = None) -> "KanbanCompletionValidationResult":
        return cls(accepted=True, metadata=metadata)

    @classmethod
    def veto(
        cls, reason: str, *, code: str | None = None, metadata: Mapping[str, Any] | None = None
    ) -> "KanbanCompletionValidationResult":
        return cls(accepted=False, reason=reason, code=code, metadata=metadata)


class KanbanCompletionValidationError(ValueError):
    """Raised when a validator blocks or the seam must fail closed."""

    def __init__(self, reason: str, *, code: str = "validator_rejected"):
        self.reason = reason
        self.code = code
        super().__init__(reason)


def project_kanban_completion_validation(
    context: KanbanCompletionValidationContext,
) -> tuple[dict[str, Any], list[KanbanCompletionValidationResult]]:
    """Run completion validators and return projected metadata.

    No registered validators is an accept: the seam is optional. Once a
    validator is present, malformed returns, explicit vetoes, exceptions, and
    recursive validation attempts fail closed before any Kanban mutation.
    """
    depth = int(getattr(_validation_depth, "value", 0) or 0)
    if depth > 0:
        error = KanbanCompletionValidationError(
            "recursive Kanban completion validation is not allowed",
            code="validator_reentrant",
        )
        _validation_depth.reentrant_error = error
        raise error
    _validation_depth.reentrant_error = None

    try:
        from hermes_cli.plugins import get_plugin_manager, has_hook
    except Exception as exc:  # pragma: no cover - import failure is fail-closed by contract
        raise KanbanCompletionValidationError(
            f"completion validator discovery failed: {exc}",
            code="validator_discovery_error",
        ) from exc

    try:
        if not has_hook(HOOK_NAME):
            return dict(context.metadata), []
    except Exception as exc:
        raise KanbanCompletionValidationError(
            f"completion validator discovery failed: {exc}",
            code="validator_discovery_error",
        ) from exc

    try:
        manager = get_plugin_manager()
        callbacks = tuple(getattr(manager, "iter_hook_callbacks")(HOOK_NAME))
    except Exception as exc:
        raise KanbanCompletionValidationError(
            f"completion validator discovery failed: {exc}",
            code="validator_discovery_error",
        ) from exc
    metadata = dict(context.metadata)
    outcomes: list[KanbanCompletionValidationResult] = []
    if not callbacks:
        return metadata, outcomes

    _validation_depth.value = depth + 1
    try:
        for callback in callbacks:
            callback_name = getattr(callback, "__name__", repr(callback))
            current_context = KanbanCompletionValidationContext(
                task_id=context.task_id,
                title=context.title,
                body=context.body,
                created_by=context.created_by,
                board=context.board,
                prior_status=context.prior_status,
                assignee=context.assignee,
                run_id=context.run_id,
                summary=context.summary,
                result=context.result,
                metadata=MappingProxyType(dict(metadata)),
                created_cards=context.created_cards,
                source=context.source,
                surface=context.surface,
                dry_run=context.dry_run,
            )
            try:
                raw = _invoke_validator_callback(callback, current_context)
                reentrant_error = getattr(_validation_depth, "reentrant_error", None)
                if reentrant_error is not None:
                    _validation_depth.reentrant_error = None
                    raise reentrant_error
                outcome = _normalize_validator_result(raw, current_metadata=metadata)
            except KanbanCompletionValidationError:
                raise
            except Exception as exc:
                logger.warning(
                    "Kanban completion validator %s raised: %s",
                    callback_name,
                    exc,
                )
                raise KanbanCompletionValidationError(
                    f"completion validator {callback_name} raised: {exc}",
                    code="validator_exception",
                ) from exc
            outcomes.append(outcome)
            if outcome.metadata is not None:
                metadata = dict(outcome.metadata)
            if not outcome.accepted:
                reason = outcome.reason or f"completion validator {callback_name} rejected completion"
                raise KanbanCompletionValidationError(
                    reason,
                    code=outcome.code or "validator_rejected",
                )
    finally:
        _validation_depth.value = depth
        if depth == 0:
            _validation_depth.reentrant_error = None

    return metadata, outcomes


def _invoke_validator_callback(callback: Any, context: KanbanCompletionValidationContext) -> Any:
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
        "result": context.result,
        "metadata": context.metadata,
        "created_cards": context.created_cards,
        "source": context.source,
        "surface": context.surface,
        "dry_run": context.dry_run,
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


def _normalize_validator_result(
    raw: Any, *, current_metadata: Mapping[str, Any]
) -> KanbanCompletionValidationResult:
    if raw is None or isinstance(raw, bool):
        raise KanbanCompletionValidationError(
            "completion validator must return an explicit Result or mapping decision",
            code="validator_bad_result",
        )
    if isinstance(raw, KanbanCompletionValidationResult):
        if not isinstance(raw.accepted, bool):
            raise KanbanCompletionValidationError(
                "completion validator Result.accepted must be bool",
                code="validator_bad_result",
            )
        if raw.metadata is not None and not isinstance(raw.metadata, Mapping):
            raise KanbanCompletionValidationError(
                "completion validator Result.metadata must be a mapping",
                code="validator_bad_result",
            )
        if raw.reason is not None and not isinstance(raw.reason, str):
            raise KanbanCompletionValidationError(
                "completion validator Result.reason must be a string",
                code="validator_bad_result",
            )
        if raw.code is not None and not isinstance(raw.code, str):
            raise KanbanCompletionValidationError(
                "completion validator Result.code must be a string",
                code="validator_bad_result",
            )
        return KanbanCompletionValidationResult(
            accepted=raw.accepted,
            reason=raw.reason,
            code=raw.code,
            metadata=dict(raw.metadata) if raw.metadata is not None else None,
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
                        raise KanbanCompletionValidationError(
                            "completion validator returned contradictory decisions",
                            code="validator_bad_result",
                        )
            if normalized_action in {"block", "veto", "reject", "deny"}:
                reason = raw.get("reason") or raw.get("message") or "completion validator rejected completion"
                code = raw.get("code") or "validator_rejected"
                return KanbanCompletionValidationResult.veto(str(reason), code=str(code))
            if normalized_action == "allow":
                projected = raw.get("metadata", current_metadata)
                if projected is not None and not isinstance(projected, Mapping):
                    raise KanbanCompletionValidationError(
                        "completion validator returned non-object metadata",
                        code="validator_bad_result",
                    )
                return KanbanCompletionValidationResult.accept(dict(projected or {}))
            raise KanbanCompletionValidationError(
                f"completion validator returned unsupported action: {action}",
                code="validator_bad_result",
            )
        if len(decision_keys) != 1 or not isinstance(raw[decision_keys[0]], bool):
            raise KanbanCompletionValidationError(
                "completion validator mapping must include exactly one explicit boolean decision",
                code="validator_bad_result",
            )
        accepted_raw = raw[decision_keys[0]]
        accepted = bool(accepted_raw)
        reason = raw.get("reason") or raw.get("message")
        code = raw.get("code")
        projected = raw.get("metadata", current_metadata)
        if projected is not None and not isinstance(projected, Mapping):
            raise KanbanCompletionValidationError(
                "completion validator returned non-object metadata",
                code="validator_bad_result",
            )
        return KanbanCompletionValidationResult(
            accepted=accepted,
            reason=str(reason) if reason is not None else None,
            code=str(code) if code is not None else None,
            metadata=dict(projected or {}),
        )
    raise KanbanCompletionValidationError(
        f"completion validator returned unsupported result type: {type(raw).__name__}",
        code="validator_bad_result",
    )
