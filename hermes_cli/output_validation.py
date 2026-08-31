"""Internal output-validation protocol and evidence-backed Kanban fallback."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Optional


# A plugin may return this exact value from ``transform_llm_output`` to ask the
# runtime for one same-session repair turn. The turn finalizer converts it to
# ``SAFE_UNVERIFIED_RESPONSE`` before any caller can display it and exposes a
# machine-readable flag to the gateway. The token itself is never user text.
OUTPUT_VALIDATION_RETRY_SIGNAL = "\x1eHERMES_OUTPUT_VALIDATION_RETRY\x1e"

SAFE_UNVERIFIED_RESPONSE = (
    "Je ne peux pas encore confirmer la fin : aucune preuve vérifiable n'est "
    "attachée à cette conclusion. Le travail n'est pas considéré comme livré."
)

_KANBAN_COMPLETED_RE = re.compile(
    r"\[kanban\]\s+Task\s+(t_[0-9a-f]+)\s+completed\.", re.IGNORECASE
)


def requests_output_validation_retry(result: Mapping[str, Any] | None) -> bool:
    """Return whether a finalized agent result requested an internal repair."""
    if not isinstance(result, Mapping):
        return False
    return bool(result.get("output_validation_retry"))


def completed_kanban_task_id(text: str | None) -> Optional[str]:
    """Extract the trusted task id from an internal completion notification."""
    if not text:
        return None
    match = _KANBAN_COMPLETED_RE.search(str(text))
    return match.group(1).lower() if match else None


def _one_line(value: object, *, limit: int = 1200) -> str:
    return " ".join(str(value or "").split())[:limit].strip()


def _sentence_fragment(value: object, *, limit: int = 1200) -> str:
    text = _one_line(value, limit=limit)
    # Keep filenames such as ``test_file.py`` intact while preventing a
    # multi-sentence summary from separating a completion claim from the
    # structured proof appended below.
    fragment = re.sub(r"[.!?;]+(?=\s|$)", " — ", text).strip(" —")
    return " ".join(fragment.split())


def verified_kanban_completion_message(
    inbound_text: str | None,
    *,
    db_path: Path | None = None,
) -> Optional[str]:
    """Render a concise final message only from delivered structured evidence.

    The inbound text only identifies a task from Hermes' internal completion
    notification. Status, mission fan-in, summary and evidence are re-read
    from the Kanban database. A mission reopened by a new active child
    intentionally returns ``None``.
    """
    task_id = completed_kanban_task_id(inbound_text)
    if not task_id:
        return None

    from hermes_cli import kanban_db as kb

    with kb.connect(db_path) as conn:
        task = kb.get_task(conn, task_id)
        if (
            task is None
            or task.status != "done"
            or task.delivery_status != "delivered"
        ):
            return None
        mission_title = ""
        active_tasks = [task]
        if task.mission_id:
            mission = conn.execute(
                "SELECT status, title FROM missions WHERE id = ?", (task.mission_id,)
            ).fetchone()
            if mission is None or mission["status"] != "delivered":
                return None
            mission_title = _sentence_fragment(mission["title"], limit=500)
            task_rows = conn.execute(
                "SELECT id FROM tasks WHERE mission_id = ? "
                "AND queue_class = 'active' ORDER BY created_at, id",
                (task.mission_id,),
            ).fetchall()
            active_tasks = [kb.get_task(conn, row["id"]) for row in task_rows]
            active_tasks = [item for item in active_tasks if item is not None]
            if not active_tasks or any(
                item.status not in {"done", "archived"}
                or item.delivery_status != "delivered"
                for item in active_tasks
            ):
                return None

        summaries: list[str] = []
        details: list[str] = []
        notified_summary = ""
        for item in active_tasks:
            run = kb.latest_run(conn, item.id)
            metadata = run.metadata if run and isinstance(run.metadata, dict) else {}
            evidence = metadata.get("evidence")
            detail = _sentence_fragment(
                evidence.get("detail") if isinstance(evidence, dict) else "",
                limit=900,
            )
            summary = _sentence_fragment(
                (run.summary if run else None) or item.result,
                limit=900,
            )
            if not detail or not summary:
                return None
            if summary not in summaries:
                summaries.append(summary)
            if detail not in details:
                details.append(detail)
            if item.id == task_id:
                notified_summary = summary

    # Keep every possible completion claim and its proof in one sentence.
    final_summary = notified_summary or summaries[-1]
    proof = " | ".join(details)
    if mission_title:
        count = len(active_tasks)
        return (
            f"Mission livrée et vérifiée : {mission_title} — {count}/{count} "
            f"cartes terminées et livrées — résultat final : {final_summary} "
            f"— preuves durables : {proof}."
        )
    return (
        f"Résultat livré et vérifié : {final_summary} "
        f"— preuve durable : {proof}."
    )


def apply_output_validation_fallback(
    result: dict[str, Any],
    inbound_text: str | None,
    *,
    db_path: Path | None = None,
) -> bool:
    """Apply the only runtime-owned output-validation fallbacks.

    The detector itself lives outside core as a ``transform_llm_output`` hook.
    Core only consumes its private retry flag: after one repair turn, either a
    delivered Kanban mission can be rendered from structured state, or the user
    receives the neutral safe fallback. No lexical validation happens here.
    """
    verified_fallback = (
        verified_kanban_completion_message(inbound_text, db_path=db_path)
        if db_path is not None
        else verified_kanban_completion_message(inbound_text)
    )
    if verified_fallback:
        result["final_response"] = verified_fallback
        result["output_validation_retry"] = False
        result["response_transformed"] = True
        result["output_validation_structured_fallback"] = True
        return True
    if requests_output_validation_retry(result):
        result["final_response"] = SAFE_UNVERIFIED_RESPONSE
        result["output_validation_retry"] = False
        result["response_transformed"] = True
        return True
    return False


def output_validation_recovery_prompt(
    *,
    rejected_response: str | None,
    verified_fallback: str | None,
) -> str:
    """Build the one-shot internal repair instruction for the same session."""
    proof = (
        "\n\nPreuve Kanban durable disponible ; reprends-la exactement sans "
        f"l'inventer :\n{verified_fallback}"
        if verified_fallback
        else (
            "\n\nAucune clôture Kanban globale n'est encore prouvée. Ne dis pas "
            "que le travail est terminé : vérifie l'état durable et poursuis les "
            "actions restantes avant de conclure."
        )
    )
    excerpt = _one_line(rejected_response, limit=1600)
    return (
        "[HERMES_RECOVERY_OUTPUT_VALIDATION] La réponse précédente n'a pas été "
        "livrée car elle annonçait une fin sans preuve vérifiable dans la même "
        "phrase. Reprends exactement dans cette session, sans répéter les effets "
        "de bord. Utilise les outils seulement si une vérification manque, puis "
        "rends une unique conclusion courte et non ambiguë. Ne mentionne jamais "
        "le garde interne ni cette reprise."
        f"{proof}\n\nRéponse rejetée (non livrée) : {excerpt}"
    )
