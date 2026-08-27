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
    """Isolated Hermes state so visual-review tests never touch the live board."""
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


def test_visual_classifier_supports_heuristic_marker_and_opt_out() -> None:
    assert vr.is_visual_web_task("Corriger l'interface responsive")
    assert vr.is_visual_web_task("Refactor", "[VISUAL] rendu à vérifier")
    assert not vr.is_visual_web_task("Corriger le parseur CSS [NO-VISUAL]")
    assert vr.is_visual_web_task("Refactor", metadata={"changed_files": ["src/Card.tsx"]})


def test_capture_only_delivery_does_not_enter_implementation_review() -> None:
    body = (
        "Prendre une capture desktop et mobile de la vraie page Entreprise. "
        "Ne modifier aucun fichier, ne rien committer ni pousser."
    )
    assert not vr.is_visual_web_task("Capturer la page Entreprise", body)
    # A real changed-file receipt remains authoritative even if the prose says
    # capture-only: the no-change shortcut cannot hide an implementation.
    assert vr.is_visual_web_task(
        "Capturer la page Entreprise",
        body,
        metadata={"changed_files": ["src/Page.tsx"]},
    )
    # Explicit review markers also remain authoritative.
    assert vr.is_visual_web_task("[VISUAL] Capturer la page", body)


def test_capture_only_card_can_escape_a_historical_false_positive_handoff(
    kanban_home, tmp_path: Path,
) -> None:
    desktop = png(tmp_path / "desktop.png", 1200, 800, (1, 2, 3))
    mobile = png(tmp_path / "mobile.png", 390, 844, (4, 5, 6))
    body = (
        "Prendre une capture desktop et mobile de la page existante. "
        "Ne modifier aucun fichier."
    )
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Capturer la page Entreprise",
            body=body,
            assignee="spark",
        )
        # Simulate the handoff produced by the former over-broad classifier.
        forced_visual = visual_metadata(desktop, mobile)
        forced_visual["visual_review"]["required"] = True
        assert kb.request_review(
            conn,
            task_id,
            summary="captures ready",
            metadata=forced_visual,
        )
        assert kb.complete_task(
            conn,
            task_id,
            summary="captures delivered, no files changed",
            metadata={"changed_files": []},
        )
        assert kb.get_task(conn, task_id).status == "done"


def test_visual_request_review_requires_two_viewports_and_routes_to_coder(kanban_home, tmp_path: Path) -> None:
    desktop = png(tmp_path / "desktop.png", 1200, 800, (1, 2, 3))
    mobile = png(tmp_path / "mobile.png", 390, 844, (4, 5, 6))
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Corriger l'interface web", assignee="claude2")
        ok, reason = kb.request_review(
            conn,
            task_id,
            summary="candidate ready",
            metadata=visual_metadata(desktop, mobile),
            with_reason=True,
        )
        assert ok, reason
        task = kb.get_task(conn, task_id)
        assert task.status == "review"
        assert task.assignee == "coder"
        handoff = kb.list_runs(conn, task_id)[-1]
        visual = handoff.metadata["visual_review"]
        assert {item["viewport"] for item in visual["screenshots"]} == {"desktop", "mobile"}
        assert all(len(item["sha256"]) == 64 for item in visual["screenshots"])


def test_visual_task_cannot_complete_without_review_handoff(kanban_home) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Modifier la mise en page du site", assignee="claude2")
        with pytest.raises(kb.VisualReviewGateError, match="kanban_request_review"):
            kb.complete_task(
                conn,
                task_id,
                summary="done",
                metadata={"evidence": {"kind": "test", "detail": "tests pass"}},
            )


def test_coder_native_and_matching_gemini_evidence_unlock_completion(
    kanban_home, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    desktop = png(tmp_path / "desktop.png", 1200, 800, (10, 20, 30))
    mobile = png(tmp_path / "mobile.png", 390, 844, (40, 50, 60))
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    monkeypatch.setattr(vr, "FINAL_EVIDENCE_ROOT", evidence_root)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Finaliser le rendu visuel du site", assignee="claude2")
        assert kb.request_review(
            conn,
            task_id,
            summary="candidate ready",
            metadata=visual_metadata(desktop, mobile),
        )
        assert kb.claim_review_task(conn, task_id) is not None
        handoff = kb._latest_review_handoff_metadata(conn, task_id)
        screenshots = handoff["visual_review"]["screenshots"]
        for screenshot in screenshots:
            assert kb.record_visual_check(
                conn,
                task_id,
                engine="native",
                sha256=screenshot["sha256"],
                size=screenshot["size"],
            )
        evidence = evidence_root / "final.json"
        evidence.write_text(
            json.dumps(
                {
                    "schema": vr.SCHEMA,
                    "stage": "final",
                    "task_id": task_id,
                    "model": "gemini-3.5-flash",
                    "verdict": "OK",
                    "screenshots": [
                        {"path": item["path"], "sha256": item["sha256"]}
                        for item in screenshots
                    ],
                }
            ),
            encoding="utf-8",
        )
        assert kb.complete_task(
            conn,
            task_id,
            summary="Coder et Gemini valident le même candidat",
            metadata={
                "visual_review": {
                    "coder_verdict": "PASS",
                    "gemini_evidence": str(evidence),
                }
            },
        )
        assert kb.get_task(conn, task_id).status == "done"


def test_transient_final_review_deferral_auto_resumes_review_session(
    kanban_home, tmp_path: Path,
) -> None:
    desktop = png(tmp_path / "desktop.png", 1200, 800, (1, 1, 1))
    mobile = png(tmp_path / "mobile.png", 390, 844, (2, 2, 2))
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Vérifier la page web mobile", assignee="claude2")
        assert kb.request_review(
            conn,
            task_id,
            summary="candidate ready",
            metadata=visual_metadata(desktop, mobile),
        )
        claimed = kb.claim_review_task(conn, task_id)
        assert claimed is not None
        retry_at = int(time.time()) + 900
        ok, reason = kb.defer_review_task(
            conn,
            task_id,
            reason="quota Gemini temporairement atteint",
            retry_at=retry_at,
            metadata={"worker_session_id": "review-session-1"},
        )
        assert ok, reason
        task = kb.get_task(conn, task_id)
        assert task.status == "review"
        assert task.next_retry_at == retry_at
        assert kb.list_runs(conn, task_id)[-1].outcome == "review_deferred"
    assert kb._transient_resume_session_id(task_id, board=None) == "review-session-1"
