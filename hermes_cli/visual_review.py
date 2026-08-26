"""Deterministic policy helpers for web visual review gates.

The model decides whether a screenshot looks correct; this module decides
whether that review really happened on the exact candidate being completed.
It deliberately contains no LLM call.  The Kanban domain uses it to:

* identify web/UI work that needs visual evidence;
* normalise and hash desktop/mobile screenshots before review dispatch;
* validate the final Gemini evidence against those same immutable hashes.

The final evidence file is produced by ``~/.hermes/scripts/gemini_review_image.py``.
Keeping hashes and routing policy here prevents a prose-only "looks good"
claim from bypassing the review lane.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from pathlib import Path
from typing import Any, Iterable, Optional


SCHEMA = "hermes.visual-review.v1"
FINAL_EVIDENCE_ROOT = Path(
    os.environ.get(
        "HERMES_VISUAL_REVIEW_EVIDENCE_DIR",
        str(Path.home() / ".hermes" / "state" / "visual-review-evidence"),
    )
)

_EXPLICIT_MARKERS = ("[visual]", "[visual-web]", "[web-visual]")
_OPT_OUT_MARKERS = ("[no-visual]", "[sans-visuel]")
_VISUAL_TEXT_RE = re.compile(
    r"(?:\bsite\s+web\b|\bwebsite\b|\blanding\s+page\b|\bpage\s+web\b|"
    r"\binterface\b|\bfront[ -]?end\b|\bresponsive\b|\bdesktop\b|"
    r"\bmobile\b|\bui\b|\bux\b|\bcss\b|\btailwind\b|\blayout\b|"
    r"\bmise\s+en\s+page\b|\brendu\s+visuel\b|\bcomposant\b|"
    r"\bcomponent\b|\bpopup\b|\bmodal\b|\bhero\b|\bheader\b|"
    r"\bfooter\b|\bformulaire\b|\banimation\b)",
    re.IGNORECASE,
)
_VISUAL_EXTENSIONS = {
    ".css", ".scss", ".sass", ".less", ".styl", ".html", ".htm",
    ".jsx", ".tsx", ".vue", ".svelte", ".astro",
}


class VisualReviewError(ValueError):
    """Raised when visual-review evidence is missing, stale, or inconsistent."""


def _metadata_changed_files(metadata: Optional[dict[str, Any]]) -> list[str]:
    if not isinstance(metadata, dict):
        return []
    values = metadata.get("changed_files") or metadata.get("files") or []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


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
    text = f"{title or ''}\n{body or ''}".lower()
    if any(marker in text for marker in _OPT_OUT_MARKERS):
        return False
    if any(marker in text for marker in _EXPLICIT_MARKERS):
        return True
    visual = metadata.get("visual_review") if isinstance(metadata, dict) else None
    if isinstance(visual, dict) and visual.get("required") is not None:
        return bool(visual.get("required"))
    if _VISUAL_TEXT_RE.search(text):
        return True
    return any(Path(path).suffix.lower() in _VISUAL_EXTENSIONS for path in _metadata_changed_files(metadata))


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
            "preuve Gemini finale absente; lancer gemini_review_image.py avec --task-id"
        )
    path = Path(raw).expanduser().resolve()
    root = FINAL_EVIDENCE_ROOT.expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise VisualReviewError(
            f"la preuve Gemini doit se trouver dans le dossier d'évidence Hermes: {root}"
        ) from exc
    if not path.is_file() or path.stat().st_size <= 0:
        raise VisualReviewError(f"preuve Gemini introuvable: {path}")
    if path.stat().st_size > 1024 * 1024:
        raise VisualReviewError("preuve Gemini anormalement volumineuse")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisualReviewError(f"preuve Gemini illisible: {exc}") from exc
    if not isinstance(data, dict):
        raise VisualReviewError("preuve Gemini invalide")
    return path, data


def validate_final_review(
    *,
    task_id: str,
    handoff_metadata: Optional[dict[str, Any]],
    completion_metadata: Optional[dict[str, Any]],
    reviewer_profile: Optional[str],
    native_checked_hashes: Iterable[str],
) -> dict[str, Any]:
    """Validate Coder-native and Gemini-final evidence for one candidate."""
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
        raise VisualReviewError("preuve Gemini d'un type ou d'une étape inattendue")
    if str(evidence.get("task_id") or "") != task_id:
        raise VisualReviewError("preuve Gemini liée à une autre tâche")
    if str(evidence.get("verdict") or "").upper() != "OK":
        raise VisualReviewError(
            "Gemini n'a pas validé le candidat final; demander les corrections avant clôture"
        )
    model = str(evidence.get("model") or "").lower()
    if "gemini" not in model:
        raise VisualReviewError("la preuve finale ne provient pas de Gemini")
    evidence_hashes = {
        str(item.get("sha256") or "").lower()
        for item in evidence.get("screenshots", [])
        if isinstance(item, dict)
    }
    if evidence_hashes != expected:
        raise VisualReviewError("la preuve Gemini ne correspond pas aux captures relues par Coder")
    # Recompute every hash now: neither evidence file can bless an image that
    # changed after the two reviews.
    for item in (handoff_metadata or {}).get("visual_review", {}).get("screenshots", []):
        path = Path(str(item.get("path") or ""))
        if not path.is_file() or sha256_file(path) != str(item.get("sha256") or ""):
            raise VisualReviewError(f"capture modifiée ou supprimée après revue: {path}")
    return {
        "schema": SCHEMA,
        "required": True,
        "coder_verdict": "PASS",
        "reviewer": "coder",
        "gemini_verdict": "OK",
        "gemini_model": evidence.get("model"),
        "gemini_evidence": str(evidence_path),
        "screenshots": (handoff_metadata or {}).get("visual_review", {}).get("screenshots", []),
    }
