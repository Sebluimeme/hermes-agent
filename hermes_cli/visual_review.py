"""Deterministic policy helpers for web visual review gates.

The model decides whether a screenshot looks correct; this module decides
whether that review really happened on the exact candidate being completed.
It deliberately contains no LLM call.  The Kanban domain uses it to:

* identify web/UI work that needs visual evidence;
* normalise and hash desktop/mobile screenshots before review dispatch;
* validate the final Gemini evidence, or its explicit Coder/GPT fallback,
  against those same immutable hashes.

The final evidence file is produced by
``/home/seb/.hermes/scripts/gemini_review_image.py``. Worker profiles have an
isolated ``HOME``, so a tilde path would resolve to a profile-private directory
where the shared operational script does not exist.
Keeping hashes and routing policy here prevents a prose-only "looks good"
claim from bypassing the review lane.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


SCHEMA = "hermes.visual-review.v1"
FINAL_EVIDENCE_ROOT = Path(
    os.environ.get(
        "HERMES_VISUAL_REVIEW_EVIDENCE_DIR",
        str(Path.home() / ".hermes" / "state" / "visual-review-evidence"),
    )
)

PRODUCTION_PROOF_SCHEMA = "hermes.production-proof.v1"
PRODUCTION_PROOF_EVIDENCE_ROOT = Path(
    os.environ.get(
        "HERMES_PRODUCTION_PROOF_EVIDENCE_DIR",
        str(Path.home() / ".hermes" / "state" / "production-proof-evidence"),
    )
)
# Default freshness window for a production check: a proof older than this
# cannot be reused to close a later, different candidate. Overridable for
# slower deploy pipelines.
_PRODUCTION_PROOF_MAX_AGE_SECONDS = 24 * 60 * 60

_EXPLICIT_MARKERS = ("[visual]", "[visual-web]", "[web-visual]")
_OPT_OUT_MARKERS = ("[no-visual]", "[sans-visuel]")
# Distinct from _OPT_OUT_MARKERS on purpose: Sébastien may want a live
# production check without demanding visual review (e.g. a config-only
# deploy) or vice versa. Presence of this marker in the card is itself the
# durable trace of his explicit authorization (same convention as
# [NO-VISUAL]) — see incident t_9fbb7396.
_PRODUCTION_PROOF_EXEMPT_MARKERS = ("[no-prod-proof]", "[sans-preuve-prod]")
# Task title/body are immutable after creation (hermes_cli.kanban_db has no
# update-title/body path), so the marker's presence necessarily reflects what
# the card's *creator* wrote. That is only a real trace of Sébastien's
# authorization if the creator is a channel that positively traces back to
# him — an *allowlist*, not merely "not one of the four named execution
# profiles". tools/kanban_tools.py stamps created_by=$HERMES_PROFILE, falling
# back to the literal "worker" when that env var is unset (e.g. the kernel's
# own auto-escalation/consolidation paths, or any ad-hoc script missing the
# profile env). A denylist limited to {claude1, claude2, coder, spark} let
# any of those unaccounted-for values — "worker" (the dispatcher's own
# fallback), "task-consolidation", "codex-worker", "codex-safe-update", or
# any future profile name — pass through as if trusted, which is exactly the
# laundering path a worker could exploit by creating a card through a
# codepath that leaves HERMES_PROFILE unset. Only creators that this system's
# own architecture treats as tracing directly to Sébastien's decision are
# honoured: the Kanban dashboard UI, the default/Telegram operational
# interlocutor (which only creates cards from Sébastien's chat messages or
# narrowly-scoped autonomous corrections, never its own execution follow-up),
# and an unset/empty creator (legacy rows predating created_by tracking).
_TRUSTED_EXEMPTION_CREATORS = {"dashboard", "default", "telegram", "user", "seb"}
_LOCAL_HOST_RE = re.compile(
    r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])(:\d+)?(/|$)",
    re.IGNORECASE,
)
_VISUAL_TEXT_RE = re.compile(
    r"(?:\bsite\s+web\b|\bwebsite\b|\blanding\s+page\b|\bpage\s+web\b|"
    r"\binterface\b|\bfront[ -]?end\b|\bresponsive\b|\bdesktop\b|"
    r"\bmobile\b|\bui\b|\bux\b|\bcss\b|\btailwind\b|\blayout\b|"
    r"\bmise\s+en\s+page\b|\brendu\s+visuel\b|\bcomposant\b|"
    r"\bcomponent\b|\bpopup\b|\bmodal\b|\bhero\b|\bheader\b|"
    r"\bfooter\b|\bformulaire\b|\banimation\b)",
    re.IGNORECASE,
)
_CAPTURE_ONLY_RE = re.compile(
    r"(?:\bcaptur(?:e|er|es)\b|\bscreenshot(?:s)?\b|\bphotos?\s+de\s+(?:la\s+)?page\b)",
    re.IGNORECASE,
)
_NO_RENDER_CHANGE_RE = re.compile(
    r"(?:\bne\s+modifi(?:er|e|é)\s+aucun\s+fichier\b|"
    r"\bsans\s+(?:aucune\s+)?modification\b|\baucune\s+modification\b|"
    r"\bcapture\s+seule\b|\bread[ -]?only\b|\blecture\s+seule\b)",
    re.IGNORECASE,
)
_VISUAL_EXTENSIONS = {
    ".css", ".scss", ".sass", ".less", ".styl", ".html", ".htm",
    ".jsx", ".tsx", ".vue", ".svelte", ".astro",
}
_ADS_CONFIGURATION_TITLE_RE = re.compile(
    r"\b(?:google\s+)?ads\b",
    re.IGNORECASE,
)
_ADS_CONFIGURATION_OBJECT_RE = re.compile(
    r"\b(?:conversion(?:s)?|calendrier|budget|ench[eè]re(?:s)?|"
    r"mot(?:s)?[ -]?cl[eé](?:s)?|campagne(?:s)?|annonce(?:s)?|"
    r"ciblage|audience(?:s)?)\b",
    re.IGNORECASE,
)


class VisualReviewError(ValueError):
    """Raised when visual-review evidence is missing, stale, or inconsistent."""


class ProductionProofError(VisualReviewError):
    """Raised when a render-changing card lacks a verified production check.

    Subclasses ``VisualReviewError`` so every existing ``except
    VisualReviewError`` boundary (notably
    ``kanban_db.visual_completion_projection``) already surfaces it without
    change, and it maps to the same ``VisualReviewGateError`` at
    ``kanban_complete``.
    """


def _metadata_changed_files(metadata: Optional[dict[str, Any]]) -> list[str]:
    if not isinstance(metadata, dict):
        return []
    values = metadata.get("changed_files") or metadata.get("files") or []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def is_capture_only_delivery(
    title: str,
    body: str = "",
    metadata: Optional[dict[str, Any]] = None,
) -> bool:
    """Return whether the card only delivers views of unchanged rendering."""
    # DB adapters and partially mocked legacy callers can supply a non-string
    # sentinel. Classification is fail-closed to the declared changed files;
    # never pass such an object into a regex or let completion crash.
    title_text = title if isinstance(title, str) else ""
    body_text = body if isinstance(body, str) else ""
    text = f"{title_text}\n{body_text}".lower()
    return bool(
        _CAPTURE_ONLY_RE.search(text)
        and _NO_RENDER_CHANGE_RE.search(text)
        and not _metadata_changed_files(metadata)
    )


def is_visual_web_task(
    title: str,
    body: str = "",
    metadata: Optional[dict[str, Any]] = None,
) -> bool:
    """Return whether a task must traverse the web visual-review gate.

    Explicit markers are authoritative.  The conservative text/file heuristic
    catches ordinary UI cards whose creator forgot the marker, while the opt-out
    marker gives the orchestrator a deterministic escape hatch for false
    positives such as a CSS parser with no rendered output.
    """
    title_text = title if isinstance(title, str) else ""
    body_text = body if isinstance(body, str) else ""
    text = f"{title_text}\n{body_text}".lower()
    if any(marker in text for marker in _OPT_OUT_MARKERS):
        return False
    if any(marker in text for marker in _EXPLICIT_MARKERS):
        return True
    visual = metadata.get("visual_review") if isinstance(metadata, dict) else None
    if isinstance(visual, dict) and visual.get("required") is not None:
        return bool(visual.get("required"))
    # The final Coder + Gemini gate protects a candidate that CHANGES the
    # rendered interface.  A read-only delivery task whose sole output is to
    # capture an already-existing page must not enter that implementation
    # review lane: doing so spends two reviewers merely to send the requested
    # files and can strand delivery behind Gemini quota.  Require both a
    # capture intent and an explicit no-change statement, and refuse the
    # shortcut when changed files were supplied.
    if is_capture_only_delivery(title_text, body_text, metadata):
        return False
    # A Google Ads configuration card can legitimately mention UI words in
    # its acceptance text (for example an action named "Formulaire Devis",
    # or "si l'interface l'impose").  Those nouns do not mean that the card
    # changes a rendered web surface.  Use the title as the operation intent:
    # when it clearly targets an Ads configuration object and no visual source
    # file is declared, keep the card out of the screenshot review lane.
    # Explicit [VISUAL] markers and actual visual changed files remain
    # authoritative, so real landing-page work is still gated.
    changed_files = _metadata_changed_files(metadata)
    has_visual_file = any(
        Path(path).suffix.lower() in _VISUAL_EXTENSIONS for path in changed_files
    )
    if (
        _ADS_CONFIGURATION_TITLE_RE.search(title_text)
        and _ADS_CONFIGURATION_OBJECT_RE.search(title_text)
        and not has_visual_file
    ):
        return False
    if _VISUAL_TEXT_RE.search(text):
        return True
    return has_visual_file


def requires_production_proof(
    title: str,
    body: str = "",
    metadata: Optional[dict[str, Any]] = None,
    *,
    created_by: Optional[str] = None,
) -> bool:
    """Return whether a card must prove a live production check before closing.

    Root-cause guard for the incident where a landing-page card was validated
    on screenshots that were never actually deployed. Scope matches
    :func:`is_visual_web_task` (any Web/UI/code card that changes a rendered
    surface) so the two gates classify the same candidates the same way; the
    exemption marker is deliberately separate from ``[NO-VISUAL]`` so
    Sébastien can authorize skipping one without the other.

    ``created_by`` gates the marker itself: a card's title/body are fixed at
    creation (there is no update path), so the marker's presence traces back
    to whoever created the card. It only counts as "authorized by Sébastien"
    when the creator is a channel that positively traces to him —
    ``created_by in _TRUSTED_EXEMPTION_CREATORS`` — not merely "not an
    execution-worker profile"; an unaccounted-for value (``worker``, the
    dispatcher's own fallback when ``$HERMES_PROFILE`` is unset,
    ``task-consolidation``, a ``codex-*`` maintenance job, or any future
    profile name) is denied by default rather than trusted by omission. An
    unset/empty ``created_by`` (``None`` or ``""``, e.g. legacy rows
    predating this field) is still honoured, since there the marker cannot
    be attributed to an execution worker either.
    """
    text = f"{title or ''}\n{body or ''}".lower()
    if any(marker in text for marker in _PRODUCTION_PROOF_EXEMPT_MARKERS):
        creator = str(created_by or "").strip().lower()
        if not creator or creator in _TRUSTED_EXEMPTION_CREATORS:
            return False
    return is_visual_web_task(title, body, metadata)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_dimensions(path: Path) -> tuple[int, int]:
    """Read common screenshot dimensions without making Pillow mandatory."""
    with path.open("rb") as handle:
        header = handle.read(32)
    if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
        return struct.unpack(">II", header[16:24])
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception as exc:  # pragma: no cover - Pillow covers non-PNG installs
        raise VisualReviewError(
            f"capture illisible ou format non pris en charge: {path} ({exc})"
        ) from exc


def _normalise_viewport(value: Any, width: int) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"desktop", "bureau", "large"}:
        return "desktop"
    if raw in {"mobile", "phone", "telephone", "téléphone", "small"}:
        return "mobile"
    if width <= 600:
        return "mobile"
    if width >= 900:
        return "desktop"
    return raw or "other"


def normalise_screenshots(values: Any) -> list[dict[str, Any]]:
    """Validate screenshot paths and return immutable, hash-backed facts."""
    if not isinstance(values, (list, tuple)):
        raise VisualReviewError(
            "visual_review.screenshots doit contenir les captures desktop et mobile"
        )
    normalised: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for value in values:
        item = value if isinstance(value, dict) else {"path": value}
        raw_path = str(item.get("path") or "").strip()
        if not raw_path:
            raise VisualReviewError("chaque capture visuelle doit fournir path")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            raise VisualReviewError(f"le chemin de capture doit être absolu: {raw_path}")
        if not path.is_file() or path.stat().st_size <= 0:
            raise VisualReviewError(f"capture absente ou vide: {path}")
        width, height = _image_dimensions(path)
        digest = sha256_file(path)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        normalised.append(
            {
                "path": str(path.resolve()),
                "viewport": _normalise_viewport(item.get("viewport"), width),
                "width": width,
                "height": height,
                "size": path.stat().st_size,
                "sha256": digest,
            }
        )
    viewports = {item["viewport"] for item in normalised}
    missing = [name for name in ("desktop", "mobile") if name not in viewports]
    if missing:
        raise VisualReviewError(
            "captures visuelles incomplètes: il manque " + " et ".join(missing)
        )
    return normalised


def prepare_review_handoff(
    *,
    task_id: str,
    title: str,
    body: str,
    metadata: Optional[dict[str, Any]],
    reviewer: Optional[str],
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Normalise a visual handoff and select Coder as independent reviewer."""
    if not is_visual_web_task(title, body, metadata):
        return metadata, reviewer
    result = dict(metadata or {})
    visual = dict(result.get("visual_review") or {})
    screenshots = normalise_screenshots(visual.get("screenshots"))
    visual.update(
        {
            "schema": SCHEMA,
            "required": True,
            "stage": "implementation_handoff",
            "task_id": task_id,
            "screenshots": screenshots,
        }
    )
    result["visual_review"] = visual
    # A fresh Coder review run is the default independent visual analyst.  The
    # orchestrator remains the control plane and does not consume its long-lived
    # conversation by inspecting every screenshot.
    return result, "coder"


def screenshot_hashes(metadata: Optional[dict[str, Any]]) -> set[str]:
    if not isinstance(metadata, dict):
        return set()
    visual = metadata.get("visual_review")
    if not isinstance(visual, dict):
        return set()
    screenshots = visual.get("screenshots")
    if not isinstance(screenshots, list):
        return set()
    return {
        str(item.get("sha256") or "").lower()
        for item in screenshots
        if isinstance(item, dict) and str(item.get("sha256") or "").strip()
    }


def load_final_evidence(path_value: Any) -> tuple[Path, dict[str, Any]]:
    raw = str(path_value or "").strip()
    if not raw:
        raise VisualReviewError(
            "preuve visuelle finale absente; lancer gemini_review_image.py avec --task-id"
        )
    path = Path(raw).expanduser().resolve()
    root = FINAL_EVIDENCE_ROOT.expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise VisualReviewError(
            f"la preuve visuelle doit se trouver dans le dossier d'évidence Hermes: {root}"
        ) from exc
    if not path.is_file() or path.stat().st_size <= 0:
        raise VisualReviewError(f"preuve visuelle introuvable: {path}")
    if path.stat().st_size > 1024 * 1024:
        raise VisualReviewError("preuve visuelle anormalement volumineuse")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisualReviewError(f"preuve visuelle illisible: {exc}") from exc
    if not isinstance(data, dict):
        raise VisualReviewError("preuve visuelle invalide")
    return path, data


def validate_final_review(
    *,
    task_id: str,
    handoff_metadata: Optional[dict[str, Any]],
    completion_metadata: Optional[dict[str, Any]],
    reviewer_profile: Optional[str],
    native_checked_hashes: Iterable[str],
) -> dict[str, Any]:
    """Validate Coder-native plus Gemini/GPT-final evidence for one candidate."""
    expected = screenshot_hashes(handoff_metadata)
    if len(expected) < 2:
        raise VisualReviewError("handoff visuel sans captures desktop/mobile vérifiables")
    if str(reviewer_profile or "").strip().lower() != "coder":
        raise VisualReviewError("la revue visuelle indépendante doit être exécutée par Coder")
    checked = {str(value).lower() for value in native_checked_hashes if str(value).strip()}
    missing_native = sorted(expected - checked)
    if missing_native:
        raise VisualReviewError(
            "Coder n'a pas chargé en vision native toutes les captures du candidat"
        )
    metadata = completion_metadata if isinstance(completion_metadata, dict) else {}
    visual = metadata.get("visual_review")
    if not isinstance(visual, dict) or str(visual.get("coder_verdict") or "").upper() != "PASS":
        raise VisualReviewError("verdict Coder PASS absent de metadata.visual_review")
    evidence_path, evidence = load_final_evidence(visual.get("gemini_evidence"))
    if evidence.get("schema") != SCHEMA or evidence.get("stage") != "final":
        raise VisualReviewError("preuve visuelle d'un type ou d'une étape inattendue")
    if str(evidence.get("task_id") or "") != task_id:
        raise VisualReviewError("preuve visuelle liée à une autre tâche")
    if str(evidence.get("verdict") or "").upper() != "OK":
        raise VisualReviewError(
            "le contrôleur visuel final n'a pas validé le candidat; "
            "demander les corrections avant clôture"
        )
    model = str(evidence.get("model") or "").lower()
    fallback_from = str(evidence.get("fallback_from") or "").lower()
    fallback_basis = str(evidence.get("fallback_basis") or "")
    gemini_final = "gemini" in model and evidence.get("fallback") is not True
    gpt_fallback = (
        evidence.get("fallback") is True
        and model == "coder-native-gpt-fallback"
        and "gemini" in fallback_from
        and fallback_basis == "coder_native_pass_required_by_gate"
    )
    if not gemini_final and not gpt_fallback:
        raise VisualReviewError(
            "la preuve finale ne provient ni de Gemini ni du fallback GPT Coder autorisé"
        )
    evidence_hashes = {
        str(item.get("sha256") or "").lower()
        for item in evidence.get("screenshots", [])
        if isinstance(item, dict)
    }
    if evidence_hashes != expected:
        raise VisualReviewError("la preuve finale ne correspond pas aux captures relues par Coder")
    # Recompute every hash now: neither evidence file can bless an image that
    # changed after the two reviews.
    for item in (handoff_metadata or {}).get("visual_review", {}).get("screenshots", []):
        path = Path(str(item.get("path") or ""))
        if not path.is_file() or sha256_file(path) != str(item.get("sha256") or ""):
            raise VisualReviewError(f"capture modifiée ou supprimée après revue: {path}")
    result = {
        "schema": SCHEMA,
        "required": True,
        "coder_verdict": "PASS",
        "reviewer": "coder",
        "final_route": "gpt_fallback" if gpt_fallback else "gemini",
        "final_verdict": "OK",
        "final_model": evidence.get("model"),
        "final_evidence": str(evidence_path),
        "gemini_verdict": "UNAVAILABLE" if gpt_fallback else "OK",
        "gemini_model": evidence.get("fallback_from") if gpt_fallback else evidence.get("model"),
        "gemini_evidence": str(evidence_path),
        "screenshots": (handoff_metadata or {}).get("visual_review", {}).get("screenshots", []),
    }
    if gpt_fallback:
        result.update(
            {
                "gpt_fallback_verdict": "OK",
                "gpt_fallback_model": evidence.get("model"),
            }
        )
    return result


def _parse_timestamp(value: Any) -> Optional[float]:
    """Accept an epoch number or an ISO-8601 string (``Z`` or offset)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            pass
        try:
            normalised = text[:-1] + "+00:00" if text.endswith("Z") else text
            return datetime.fromisoformat(normalised).timestamp()
        except ValueError:
            return None
    return None


def load_production_proof(path_value: Any) -> tuple[Path, dict[str, Any]]:
    raw = str(path_value or "").strip()
    if not raw:
        raise ProductionProofError(
            "preuve de production absente; lancer "
            "scripts/verify_production_proof.py --url <url-prod> --attendu <marqueur> --task-id <carte>"
        )
    path = Path(raw).expanduser().resolve()
    root = PRODUCTION_PROOF_EVIDENCE_ROOT.expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ProductionProofError(
            f"la preuve de production doit se trouver dans le dossier d'évidence Hermes: {root}"
        ) from exc
    if not path.is_file() or path.stat().st_size <= 0:
        raise ProductionProofError(f"preuve de production introuvable: {path}")
    if path.stat().st_size > 1024 * 1024:
        raise ProductionProofError("preuve de production anormalement volumineuse")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionProofError(f"preuve de production illisible: {exc}") from exc
    if not isinstance(data, dict):
        raise ProductionProofError("preuve de production invalide")
    return path, data


def validate_production_proof(
    *,
    task_id: str,
    metadata: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Validate a live production check before a render-changing card closes.

    Reads ``metadata.production_proof.evidence_path`` — the JSON file written
    by ``scripts/verify_production_proof.py`` — and refuses a prose-only
    claim, a stale check from an earlier candidate, or one pointed at a
    non-production/local URL.
    """
    meta = metadata if isinstance(metadata, dict) else {}
    proof = meta.get("production_proof")
    if not isinstance(proof, dict):
        raise ProductionProofError(
            "preuve de production absente; fournir metadata.production_proof.evidence_path "
            "produit par scripts/verify_production_proof.py, ou marquer la carte [NO-PROD-PROOF] "
            "avec l'autorisation explicite de Sébastien"
        )
    evidence_path, evidence = load_production_proof(proof.get("evidence_path"))
    if evidence.get("schema") != PRODUCTION_PROOF_SCHEMA:
        raise ProductionProofError("preuve de production d'un type inattendu")
    if str(evidence.get("task_id") or "") != task_id:
        raise ProductionProofError("preuve de production liée à une autre tâche")
    if str(evidence.get("verdict") or "").upper() != "OK":
        raise ProductionProofError(
            "le contrôle de production n'a pas confirmé le déploiement; corriger avant clôture"
        )
    url = str(evidence.get("url") or "").strip()
    if not url:
        raise ProductionProofError("preuve de production sans URL vérifiée")
    if not url.lower().startswith("https://"):
        raise ProductionProofError("la preuve de production doit cibler une URL https réelle")
    if _LOCAL_HOST_RE.match(url):
        raise ProductionProofError("la preuve de production ne peut pas cibler un hôte local")
    checked_ts = _parse_timestamp(evidence.get("fetched_at"))
    if checked_ts is None:
        raise ProductionProofError("preuve de production sans horodatage exploitable")
    max_age = float(
        os.environ.get(
            "HERMES_PRODUCTION_PROOF_MAX_AGE_SECONDS",
            str(_PRODUCTION_PROOF_MAX_AGE_SECONDS),
        )
    )
    age = time.time() - checked_ts
    if age > max_age:
        raise ProductionProofError(
            f"preuve de production trop ancienne ({int(age)}s); relancer verify_production_proof.py"
        )
    if age < -60:
        raise ProductionProofError("preuve de production horodatée dans le futur")
    return {
        "schema": PRODUCTION_PROOF_SCHEMA,
        "required": True,
        "url": url,
        "verdict": "OK",
        "fetched_at": evidence.get("fetched_at"),
        "evidence_path": str(evidence_path),
        "matched": evidence.get("matched"),
    }
