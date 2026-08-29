"""Durable, user-actionable provider-auth reporting for Kanban workers.

This module is deliberately environment-driven: dispatcher-owned workers
already receive the exact board, task, run and profile identifiers.  Regular
interactive Hermes sessions have no ``HERMES_KANBAN_TASK`` and therefore no-op.
"""
from __future__ import annotations

import contextlib
import os
from typing import Optional

from hermes_cli.auth import AuthError, is_rate_limited_auth_error


_HUMAN_AUTH_CODES = {
    "account_missing",
    "codex_auth_invalid_shape",
    "codex_auth_missing",
    "codex_auth_missing_access_token",
    "codex_auth_missing_refresh_token",
    "insufficient_credits",
    "invalid_provider",
    "invalid_token",
    "member_spend_cap_exceeded",
    "missing_api_key",
    "missing_copilot_cli",
    "no_access_token",
    "no_aws_credentials",
    "no_provider_configured",
    "no_refresh_token",
    "no_usable_credits",
    "not_logged_in",
    "qwen_access_token_missing",
    "qwen_auth_invalid",
    "qwen_auth_missing",
    "qwen_refresh_token_missing",
    "spotify_access_token_missing",
    "spotify_auth_missing",
    "spotify_client_id_missing",
    "spotify_refresh_token_missing",
    "subscription_expired",
    "subscription_required",
    "xai_auth_invalid_shape",
    "xai_auth_missing",
    "xai_auth_missing_access_token",
    "xai_auth_missing_refresh_token",
    "xai_oauth_tier_denied",
}


def auth_failure_requires_human(error: BaseException) -> bool:
    """Return True only for credential/account states a human can repair."""
    if not isinstance(error, AuthError) or is_rate_limited_auth_error(error):
        return False
    if error.relogin_required:
        return True
    code = str(error.code or "").strip().casefold()
    if code in _HUMAN_AUTH_CODES:
        return True
    return any(
        marker in code
        for marker in (
            "auth_missing",
            "invalid_token",
            "missing_access_token",
            "missing_refresh_token",
            "not_logged_in",
        )
    )


def _profile() -> str:
    return (os.environ.get("HERMES_PROFILE") or "unknown").strip() or "unknown"


def _provider(provider: Optional[str], error: Optional[BaseException] = None) -> str:
    if provider and str(provider).strip():
        return str(provider).strip()
    if isinstance(error, AuthError) and error.provider:
        return error.provider.strip()
    return "unknown"


def _recovery_action(profile: str, provider: str, error: BaseException) -> str:
    code = str(getattr(error, "code", "") or "").casefold()
    if code in {
        "subscription_required",
        "subscription_expired",
        "insufficient_credits",
        "no_usable_credits",
        "member_spend_cap_exceeded",
        "account_missing",
        "xai_oauth_tier_denied",
    }:
        return (
            f"Vérifier le compte, l’abonnement et les crédits de {provider}, "
            f"puis relancer le profil {profile}."
        )
    return (
        f"Reconnecter {provider} dans le profil {profile} avec "
        f"`hermes -p {profile} model`, puis relancer la carte."
    )


def report_kanban_auth_required(
    error: BaseException,
    *,
    provider: Optional[str] = None,
    fallback: Optional[str] = None,
    confirmed: bool = False,
) -> bool:
    """Notify once and optionally block when a Kanban provider auth is broken.

    ``confirmed`` is for a persistent HTTP 401/403 that already exhausted the
    runtime's refresh/rotation path.  Structured startup ``AuthError`` values
    are filtered more conservatively by :func:`auth_failure_requires_human`.
    """
    task_id = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    if not task_id:
        return False
    if not confirmed and not auth_failure_requires_human(error):
        return False

    from hermes_cli import kanban_db as kb

    profile = _profile()
    provider_name = _provider(provider, error)
    action = _recovery_action(profile, provider_name, error)
    fallback_label = (fallback or "").strip() or None
    fallback_active = fallback_label is not None
    expected_run_id = None
    try:
        raw_run_id = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
        expected_run_id = int(raw_run_id) if raw_run_id else None
    except ValueError:
        expected_run_id = None

    try:
        with contextlib.closing(kb.connect()) as conn:
            kb.record_provider_auth_incident(
                conn,
                task_id,
                profile=profile,
                provider=provider_name,
                error=str(error),
                action=action,
                fallback_active=fallback_active,
                fallback=fallback_label,
                # When no fallback exists, block_task emits the single human
                # notification; avoid a duplicate provider-auth event.
                emit_event=fallback_active,
            )
            if not fallback_active:
                kb.block_task(
                    conn,
                    task_id,
                    reason=(
                        f"gate:provider-auth — Authentification du profil "
                        f"{profile} ({provider_name}) à corriger. {action}"
                    ),
                    kind="capability",
                    expected_run_id=expected_run_id,
                )
    except Exception:
        # Reporting must never replace the original provider failure or stop a
        # healthy fallback from continuing.  The worker log keeps the source
        # exception if the board is temporarily unavailable.
        return False
    return True


def mark_kanban_auth_healthy(*, provider: Optional[str]) -> bool:
    """Resolve a prior profile incident after primary credentials work again."""
    if not (os.environ.get("HERMES_KANBAN_TASK") or "").strip():
        return False
    from hermes_cli import kanban_db as kb

    try:
        with contextlib.closing(kb.connect()) as conn:
            return kb.resolve_provider_auth_incident(
                conn,
                profile=_profile(),
                provider=_provider(provider),
            )
    except Exception:
        return False
