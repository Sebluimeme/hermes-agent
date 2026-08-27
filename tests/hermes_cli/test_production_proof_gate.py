"""Root-cause guard for incident t_9fbb7396: a landing-page card was validated
even though the deployment to production was never checked. These tests
simulate exactly that shape of card — a Web/UI render change completing on
visual review alone — and assert `kanban_complete` now refuses it without a
verified `scripts/verify_production_proof.py` evidence file, unless the card
carries Sébastien's explicit `[NO-PROD-PROOF]` exemption.
"""
from __future__ import annotations

import json
import struct
import time
import zlib
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import visual_review as vr


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated Hermes state so these tests never touch the live board."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def png(path: Path, width: int, height: int, rgb: tuple[int, int, int]) -> Path:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    row = b"\x00" + bytes(rgb) * width
    raw = row * height
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return path


def visual_metadata(desktop: Path, mobile: Path) -> dict:
    return {
        "visual_review": {
            "screenshots": [
                {"path": str(desktop), "viewport": "desktop"},
                {"path": str(mobile), "viewport": "mobile"},
            ]
        }
    }


def _write_production_evidence(
    root: Path,
    *,
    task_id: str,
    url: str = "https://flamme-traiteur.fr/",
    verdict: str = "OK",
    fetched_at: str | None = None,
    schema: str = vr.PRODUCTION_PROOF_SCHEMA,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{task_id}.json"
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "task_id": task_id,
                "url": url,
                "expected": "build-9f2a",
                "status_code": 200,
                "matched": verdict == "OK",
                "verdict": verdict,
                "fetched_at": fetched_at or _iso_now(),
            }
        ),
        encoding="utf-8",
    )
    return path


def _iso_now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat()


def _complete_visual_task(conn, tmp_path: Path, evidence_root: Path, monkeypatch, title: str, body: str = ""):
    """Drive one card through request_review + Coder native check + Gemini
    final evidence, exactly as the two-stage visual gate expects, and return
    its task_id with the completion metadata still pending production proof.
    """
    evidence_root.mkdir(parents=True, exist_ok=True)
    desktop = png(tmp_path / f"desktop-{title[:8]}.png", 1200, 800, (10, 20, 30))
    mobile = png(tmp_path / f"mobile-{title[:8]}.png", 390, 844, (40, 50, 60))
    monkeypatch.setattr(vr, "FINAL_EVIDENCE_ROOT", evidence_root)
    task_id = kb.create_task(conn, title=title, body=body, assignee="claude2")
    assert kb.request_review(
        conn, task_id, summary="candidate ready", metadata=visual_metadata(desktop, mobile),
    )
    assert kb.claim_review_task(conn, task_id) is not None
    handoff = kb._latest_review_handoff_metadata(conn, task_id)
    screenshots = handoff["visual_review"]["screenshots"]
    for screenshot in screenshots:
        assert kb.record_visual_check(
            conn, task_id, engine="native", sha256=screenshot["sha256"], size=screenshot["size"],
        )
    gemini_evidence = evidence_root / "gemini-final.json"
    gemini_evidence.write_text(
        json.dumps(
            {
                "schema": vr.SCHEMA,
                "stage": "final",
                "task_id": task_id,
                "model": "gemini-3.5-flash",
                "verdict": "OK",
                "screenshots": [
                    {"path": item["path"], "sha256": item["sha256"]} for item in screenshots
                ],
            }
        ),
        encoding="utf-8",
    )
    completion_metadata = {
        "visual_review": {"coder_verdict": "PASS", "gemini_evidence": str(gemini_evidence)},
    }
    return task_id, completion_metadata


def test_requires_production_proof_matches_visual_scope_with_its_own_exemption() -> None:
    assert vr.requires_production_proof("Corriger la landing page")
    assert not vr.requires_production_proof("Corriger la landing page [NO-PROD-PROOF]")
    # The visual opt-out also implies nothing rendered was changed, so no
    # production check is owed either.
    assert not vr.requires_production_proof("Corriger le parseur CSS [NO-VISUAL]")
    # But the two markers are independent: [NO-VISUAL] alone does not grant
    # a [NO-PROD-PROOF] exemption once a render change is otherwise detected.
    assert vr.requires_production_proof("Refactor", metadata={"changed_files": ["src/Landing.tsx"]})


def test_render_change_cannot_complete_on_visual_review_alone(
    kanban_home, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulation of the exact incident: Coder PASS + Gemini OK exist, but no
    production check was ever run. Completion must still be refused."""
    evidence_root = tmp_path / "visual-evidence"
    with kb.connect() as conn:
        task_id, completion_metadata = _complete_visual_task(
            conn, tmp_path, evidence_root, monkeypatch,
            title="Publier la nouvelle landing page",
        )
        with pytest.raises(kb.VisualReviewGateError, match="preuve de production"):
            kb.complete_task(
                conn, task_id, summary="visuel validé", metadata=completion_metadata,
            )
        # The Coder reviewer claimed the review run (status moves to
        # "running" for that run); the point under test is that the render
        # change stays open rather than reaching "done" on screenshots alone.
        assert kb.get_task(conn, task_id).status != "done"


def test_valid_production_proof_unlocks_completion(
    kanban_home, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = tmp_path / "visual-evidence"
    production_root = tmp_path / "production-evidence"
    monkeypatch.setattr(vr, "PRODUCTION_PROOF_EVIDENCE_ROOT", production_root)
    with kb.connect() as conn:
        task_id, completion_metadata = _complete_visual_task(
            conn, tmp_path, evidence_root, monkeypatch,
            title="Publier la nouvelle landing page",
        )
        proof_path = _write_production_evidence(production_root, task_id=task_id)
        completion_metadata["production_proof"] = {"evidence_path": str(proof_path)}
        assert kb.complete_task(
            conn, task_id, summary="visuel + production vérifiés", metadata=completion_metadata,
        )
        task = kb.get_task(conn, task_id)
        assert task.status == "done"
        stored = kb.list_runs(conn, task_id)[-1].metadata
        assert stored["production_proof"]["verdict"] == "OK"


def test_explicit_no_prod_proof_marker_exempts_the_gate(
    kanban_home, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = tmp_path / "visual-evidence"
    with kb.connect() as conn:
        task_id, completion_metadata = _complete_visual_task(
            conn, tmp_path, evidence_root, monkeypatch,
            title="Publier la landing page interne [NO-PROD-PROOF]",
            body="Autorisation explicite de Sébastien: outil interne sans domaine public.",
        )
        assert kb.complete_task(
            conn, task_id, summary="visuel validé, prod exemptée", metadata=completion_metadata,
        )
        assert kb.get_task(conn, task_id).status == "done"


def test_production_proof_for_a_different_task_is_rejected(
    kanban_home, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = tmp_path / "visual-evidence"
    production_root = tmp_path / "production-evidence"
    monkeypatch.setattr(vr, "PRODUCTION_PROOF_EVIDENCE_ROOT", production_root)
    with kb.connect() as conn:
        task_id, completion_metadata = _complete_visual_task(
            conn, tmp_path, evidence_root, monkeypatch,
            title="Publier la nouvelle landing page",
        )
        proof_path = _write_production_evidence(production_root, task_id="t_une_autre_carte")
        completion_metadata["production_proof"] = {"evidence_path": str(proof_path)}
        with pytest.raises(kb.VisualReviewGateError, match="autre tâche"):
            kb.complete_task(
                conn, task_id, summary="visuel validé", metadata=completion_metadata,
            )


def test_production_proof_with_probleme_verdict_is_rejected(
    kanban_home, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = tmp_path / "visual-evidence"
    production_root = tmp_path / "production-evidence"
    monkeypatch.setattr(vr, "PRODUCTION_PROOF_EVIDENCE_ROOT", production_root)
    with kb.connect() as conn:
        task_id, completion_metadata = _complete_visual_task(
            conn, tmp_path, evidence_root, monkeypatch,
            title="Publier la nouvelle landing page",
        )
        proof_path = _write_production_evidence(production_root, task_id=task_id, verdict="PROBLEME")
        completion_metadata["production_proof"] = {"evidence_path": str(proof_path)}
        with pytest.raises(kb.VisualReviewGateError, match="n'a pas confirmé"):
            kb.complete_task(
                conn, task_id, summary="visuel validé", metadata=completion_metadata,
            )


def test_production_proof_targeting_localhost_is_rejected(
    kanban_home, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = tmp_path / "visual-evidence"
    production_root = tmp_path / "production-evidence"
    monkeypatch.setattr(vr, "PRODUCTION_PROOF_EVIDENCE_ROOT", production_root)
    with kb.connect() as conn:
        task_id, completion_metadata = _complete_visual_task(
            conn, tmp_path, evidence_root, monkeypatch,
            title="Publier la nouvelle landing page",
        )
        proof_path = _write_production_evidence(
            production_root, task_id=task_id, url="https://localhost:3000/",
        )
        completion_metadata["production_proof"] = {"evidence_path": str(proof_path)}
        with pytest.raises(kb.VisualReviewGateError, match="hôte local"):
            kb.complete_task(
                conn, task_id, summary="visuel validé", metadata=completion_metadata,
            )


def test_stale_production_proof_is_rejected(
    kanban_home, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = tmp_path / "visual-evidence"
    production_root = tmp_path / "production-evidence"
    monkeypatch.setattr(vr, "PRODUCTION_PROOF_EVIDENCE_ROOT", production_root)
    monkeypatch.setattr(vr, "_PRODUCTION_PROOF_MAX_AGE_SECONDS", 60)
    with kb.connect() as conn:
        task_id, completion_metadata = _complete_visual_task(
            conn, tmp_path, evidence_root, monkeypatch,
            title="Publier la nouvelle landing page",
        )
        old_fetched_at = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - 3600))
        proof_path = _write_production_evidence(
            production_root, task_id=task_id, fetched_at=old_fetched_at,
        )
        completion_metadata["production_proof"] = {"evidence_path": str(proof_path)}
        with pytest.raises(kb.VisualReviewGateError, match="trop ancienne"):
            kb.complete_task(
                conn, task_id, summary="visuel validé", metadata=completion_metadata,
            )
