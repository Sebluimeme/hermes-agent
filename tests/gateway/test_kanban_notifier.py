import asyncio
import sqlite3
import time
from pathlib import Path


from gateway.config import Platform
from gateway.kanban_watchers import (
    _acquire_singleton_lock,
    _release_singleton_lock,
)
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb


class RecordingAdapter:
    def __init__(self):
        self.sent = []
        self.handled = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})

    async def handle_message(self, event):
        self.handled.append(event)


class DisconnectedAdapters(dict):
    """Expose a platform during collection, then simulate disconnect on get()."""

    def get(self, key, default=None):
        return None


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _make_runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    # Most tests model the default gateway after its dispatcher acquired the
    # singleton lock. Tests for startup or non-owner gateways clear this.
    runner._kanban_dispatcher_lock_handle = object()
    return runner


def _create_completed_subscription(summary="done once"):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="notify once", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(conn, tid, summary=summary)
        return tid
    finally:
        conn.close()


def _unseen_terminal_events(tid):
    conn = kb.connect()
    try:
        _, events = kb.unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


def test_kanban_notifier_replays_telegram_dm_topic_delivery_metadata(tmp_path, monkeypatch):
    """DM-topic metadata replays onto the raw ping for a `blocked` event.

    Uses `blocked` (not `completed`) deliberately: since t_62e8c688, a
    `completed` event on a push adapter with an owning session defers to the
    wake instead of sending a raw ping at all (see
    test_completed_event_defers_raw_ping_to_wake_synthesis below), so it no
    longer exercises this metadata-replay path. `blocked` is an "alerte
    pertinente" that still sends its own clean ping regardless of wake, so it
    keeps this regression (DM-topic reply routing) covered.
    """
    db_path = tmp_path / "dm-topic-metadata.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="dm topic task",
            assignee="worker",
            session_id="agent:main:telegram:dm:chat-1",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            thread_id="20197",
            delivery_mode="notify+wake",
            delivery_metadata={
                "chat_type": "dm",
                "direct_messages_topic_id": "20197",
                "telegram_dm_topic_reply_fallback": True,
                "telegram_reply_to_message_id": "462",
                "thread_id": "20197",
            },
        )
        assert kb.block_task(conn, tid, reason="needs a decision", kind="capability")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert adapter.sent[0]["metadata"] == {
        "chat_type": "dm",
        "direct_messages_topic_id": "20197",
        "telegram_dm_topic_reply_fallback": True,
        "telegram_reply_to_message_id": "462",
        "thread_id": "20197",
    }
    assert len(adapter.handled) == 1
    assert adapter.handled[0].source.chat_type == "dm"
    assert adapter.handled[0].source.thread_id == "20197"


def test_active_named_profile_subscription_is_delivered(tmp_path, monkeypatch):
    """A sub stamped with the gateway's own named profile uses self.adapters.

    Regression for #71340: on a standalone (non-multiplex) gateway running a
    named profile, _authorization_adapter() used to treat the active name as a
    multiplex secondary, find no _profile_adapters entry, fail closed, and
    rewind the claim forever — silent zero-delivery.

    Uses a `completed` event (not `blocked`, which is now board-only and
    intentionally silent — see test_notifier_sends_no_technical_ping_for_blocked_task)
    to exercise the same named-profile adapter routing this regression pins.
    """
    db_path = tmp_path / "actionable-completion.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    summary = "AGE-39 — https://linear.example/AGE-39 — publishing verified."
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="approval", assignee="publisher")
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            notifier_profile="main",
        )
        kb.complete_task(conn, tid, summary=summary)
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "main"

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    message = adapter.sent[0]["text"]
    assert summary in message


def test_completed_notification_carries_structured_closure_evidence(tmp_path, monkeypatch):
    db_path = tmp_path / "completion-proof-notifier.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="proof", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(
            conn,
            tid,
            summary="Correction terminee",
            metadata={
                "evidence": {
                    "kind": "test",
                    "detail": "python -m pytest tests/hermes_cli/test_kanban_closure_gate.py OK",
                }
            },
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    message = adapter.sent[0]["text"]
    assert "Preuve Kanban : test" in message
    assert "python -m pytest tests/hermes_cli/test_kanban_closure_gate.py OK" in message
    assert "Correction terminee — Preuve Kanban" in message


def test_completed_event_defers_raw_ping_to_wake_synthesis(tmp_path, monkeypatch):
    """t_62e8c688: a `completed` event must reach Sébastien exactly once.

    Before this fix, a push-adapter subscription with an owning session
    (the normal shape for any interactive Telegram/Discord card, since
    ``_maybe_auto_subscribe`` always stamps gateway sessions
    ``delivery_mode="notify+wake"``) delivered the raw technical ping
    below ("✔ [...] Kanban t_xxx done — title — Preuve Kanban : ...")
    immediately, THEN woke the creator session, which produced its own
    normal "clôture automatique" synthesis a moment later — two messages
    for one completion (AGENTS.md "Silence Kanban intermédiaire": only the
    human synthesis should ever reach him).

    This pins the fix: the raw technical ping is suppressed entirely
    (`adapter.sent` stays empty — no task id, no checkmark, no board tag,
    no raw "Preuve Kanban" line ever reaches the chat directly) while the
    wake still fires exactly once, carrying the worker's summary and
    evidence into the synthetic turn so the woken agent can still compose
    an informed human synthesis from it.
    """
    db_path = tmp_path / "completed-defers-to-wake.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="Corriger le double message Kanban",
            assignee="worker",
            session_id="agent:main:telegram:group:chat-1",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            delivery_mode="notify+wake",
        )
        kb.complete_task(
            conn,
            tid,
            summary="Correction terminee",
            metadata={
                "evidence": {
                    "kind": "test",
                    "detail": "pytest tests/gateway/test_kanban_notifier.py OK",
                }
            },
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # No raw technical ping reaches the chat at all — not the checkmark
    # line, not the task id, not a duplicate.
    assert adapter.sent == [], (
        f"a completed event with an owning session must defer entirely to "
        f"the wake synthesis, got a raw ping too: {adapter.sent}"
    )
    # Exactly one wake, carrying the worker's summary and evidence so the
    # woken agent's own synthesis stays informed by the real result.
    assert len(adapter.handled) == 1
    wake_text = adapter.handled[0].text
    assert tid in wake_text  # internal wake context may reference the id
    assert "Correction terminee" in wake_text
    assert "pytest tests/gateway/test_kanban_notifier.py OK" in wake_text

    # Cursor advanced — the completed event is not left claimable forever.
    conn = kb.connect()
    try:
        _, remaining = kb.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-1",
            kinds=["completed"],
        )
    finally:
        conn.close()
    assert remaining == []


def test_crashed_event_defers_raw_ping_to_wake_synthesis(tmp_path, monkeypatch):
    """t_07db0331: `crashed` must reach Sébastien exactly once, like `completed`.

    Live evidence (2026-09-02/03 session on the main Telegram topic) showed
    the same double-message shape t_62e8c688 fixed for `completed`, still
    present for `crashed`/`timed_out`: the raw "Impact/Solution/Preuve" ping
    was sent directly AND the owning session was woken, producing its own
    follow-up synthesis a moment later — two messages for one auto-retried
    crash the dispatcher already handles without any decision from him.
    """
    db_path = tmp_path / "crashed-defers-to-wake.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="Worker mort en cours de route",
            assignee="worker",
            session_id="agent:main:telegram:group:chat-1",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            delivery_mode="notify+wake",
        )
        kb._append_event(conn, tid, "crashed", {})
        conn.commit()
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.sent == [], (
        f"a crashed event with an owning session must defer entirely to "
        f"the wake synthesis, got a raw ping too: {adapter.sent}"
    )
    assert len(adapter.handled) == 1

    conn = kb.connect()
    try:
        _, remaining = kb.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-1",
            kinds=["crashed"],
        )
    finally:
        conn.close()
    assert remaining == []


def test_timed_out_event_defers_raw_ping_to_wake_synthesis(tmp_path, monkeypatch):
    """t_07db0331: same fix as `crashed`, applied to `timed_out`."""
    db_path = tmp_path / "timedout-defers-to-wake.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="Tâche au-delà du délai",
            assignee="worker",
            session_id="agent:main:telegram:group:chat-1",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            delivery_mode="notify+wake",
        )
        kb._append_event(conn, tid, "timed_out", {"limit_seconds": 600})
        conn.commit()
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.sent == [], (
        f"a timed_out event with an owning session must defer entirely to "
        f"the wake synthesis, got a raw ping too: {adapter.sent}"
    )
    assert len(adapter.handled) == 1

    conn = kb.connect()
    try:
        _, remaining = kb.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-1",
            kinds=["timed_out"],
        )
    finally:
        conn.close()
    assert remaining == []


def test_gave_up_ping_has_no_raw_task_id_or_english_jargon(tmp_path, monkeypatch):
    """t_07db0331: `gave_up` keeps its guaranteed direct ping (retries are

    truly exhausted here, unlike crashed/timed_out which auto-retry — a real
    decision may be needed), but the wording must stop leaking the raw task
    id and English internals ("Kanban t_xxx gave up after repeated spawn
    failures") that Sébastien flagged as incomprehensible technical spam.
    """
    db_path = tmp_path / "gave-up-clean-ping.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="Ne demarre plus",
            assignee="worker",
            session_id="agent:main:telegram:group:chat-1",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            delivery_mode="notify+wake",
        )
        kb._append_event(conn, tid, "gave_up", {"error": "spawn failed: quota"})
        conn.commit()
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1, (
        f"gave_up must still send exactly one guaranteed human message, got: {adapter.sent}"
    )
    message = adapter.sent[0]["text"]
    assert tid not in message, "no raw task id in a direct chat message"
    assert "Kanban" not in message, "no internal 'Kanban' jargon in a direct chat message"
    assert "spawn failures" not in message, "no raw English worker-internal wording"
    assert "Impact :" in message
    assert "Solution :" in message
    # The wake self-post still fires so the creator agent stays informed.
    assert len(adapter.handled) == 1


def test_non_dispatch_gateway_claims_only_its_profile_subscriptions(
    tmp_path, monkeypatch,
):
    """A profile gateway delivers its events while another gateway dispatches."""
    db_path = tmp_path / "cross-profile-notifier.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        foreign_tid = kb.create_task(
            conn, title="default-owned", assignee="worker",
        )
        kb.add_notify_sub(
            conn,
            task_id=foreign_tid,
            platform="telegram",
            chat_id="default-chat",
            notifier_profile="default",
        )
        kb.complete_task(conn, foreign_tid, summary="default done")

        owned_tid = kb.create_task(
            conn, title="writer-owned", assignee="worker",
        )
        kb.add_notify_sub(
            conn,
            task_id=owned_tid,
            platform="telegram",
            chat_id="writer-chat",
            notifier_profile="writer",
        )
        kb.complete_task(conn, owned_tid, summary="writer done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "writer"
    runner._kanban_dispatcher_lock_handle = None

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert [delivery["chat_id"] for delivery in adapter.sent] == ["writer-chat"]
    assert owned_tid in adapter.sent[0]["text"]
    assert len(_unseen_terminal_events_for(foreign_tid, "default-chat")) == 1


def test_legacy_subscription_requires_confirmed_dispatcher_lock_owner(
    tmp_path, monkeypatch,
):
    """Startup and lock-losing gateways cannot claim legacy notifications."""
    db_path = tmp_path / "legacy-lock-owner.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="legacy", assignee="worker")
        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="legacy-chat",
        )
        kb.complete_task(conn, task_id, summary="legacy done")
    finally:
        conn.close()

    startup_adapter = RecordingAdapter()
    startup_runner = _make_runner(startup_adapter)
    startup_runner._kanban_dispatcher_lock_handle = None
    asyncio.run(_run_one_notifier_tick(monkeypatch, startup_runner))
    assert startup_adapter.sent == []
    assert len(_unseen_terminal_events_for(task_id, "legacy-chat")) == 1

    lock_path = tmp_path / ".dispatcher.lock"
    winner_handle, winner_state = _acquire_singleton_lock(lock_path)
    loser_handle, loser_state = _acquire_singleton_lock(lock_path)
    try:
        assert winner_state == "held"
        assert loser_state == "contended"

        loser_adapter = RecordingAdapter()
        loser_runner = _make_runner(loser_adapter)
        loser_runner._kanban_dispatcher_lock_handle = loser_handle
        asyncio.run(_run_one_notifier_tick(monkeypatch, loser_runner))
        assert loser_adapter.sent == []
        assert len(_unseen_terminal_events_for(task_id, "legacy-chat")) == 1

        winner_adapter = RecordingAdapter()
        winner_runner = _make_runner(winner_adapter)
        winner_runner._kanban_dispatcher_lock_handle = winner_handle
        asyncio.run(_run_one_notifier_tick(monkeypatch, winner_runner))
        assert [item["chat_id"] for item in winner_adapter.sent] == ["legacy-chat"]
        assert task_id in winner_adapter.sent[0]["text"]
    finally:
        _release_singleton_lock(loser_handle)
        _release_singleton_lock(winner_handle)


class FailingAdapter:
    """Adapter whose send() always raises, simulating a transient send error."""

    def __init__(self):
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        self.attempts += 1
        raise RuntimeError("simulated send failure")


class ReportedFailureAdapter:
    """Adapter that REPORTS failure via SendResult(success=False) instead of
    raising — the exact contract the Telegram adapter uses for 'Not connected'
    and degraded-send paths."""

    def __init__(self):
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        self.attempts += 1
        from gateway.platforms.base import SendResult
        return SendResult(success=False, error="Not connected")


def test_notifier_redelivers_same_kind_on_dispatch_cycle(tmp_path, monkeypatch):
    """A retry cycle (crashed → reclaimed → crashed) notifies the user twice.

    Before #21398 the notifier auto-unsubscribed on any terminal event kind
    (gave_up / crashed / timed_out), so the second crash in a respawn cycle
    silently dropped — the subscription was already gone. This test pins the
    new contract: subscription survives non-final terminal events; the
    cursor handles dedup.

    Two crashes ten seconds apart on the same task — both should land on
    the adapter.
    """
    db_path = tmp_path / "redeliver-cycle.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="cycle test", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        # First crash — fired by the dispatcher when the worker PID dies.
        kb._append_event(conn, tid, kind="crashed")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # First crash delivered.
    assert len(adapter.sent) == 1
    assert "Impact : le worker s’est arrêté avant la fin." in adapter.sent[0]["text"]
    assert "Solution : relance automatique engagée." in adapter.sent[0]["text"]

    # Subscription survives — the cursor advanced past event #1, but the
    # row is still there.
    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1, (
            "Subscription must survive a crashed event so a respawn-cycle "
            "second crash also notifies the user (issue #21398)."
        )

        # Second crash — same task, same dispatcher (or a respawn). Append
        # another event to simulate the dispatcher firing crashed a second
        # time during retry.
        kb._append_event(conn, tid, kind="crashed")
    finally:
        conn.close()

    # New tick: the second event has a fresh id past the cursor advance,
    # so it gets claimed and delivered.
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 2, (
        f"Second crashed event should also notify; got {len(adapter.sent)} "
        f"deliveries (texts: {[d['text'] for d in adapter.sent]})"
    )
    assert "Preuve : processus de la carte absent." in adapter.sent[1]["text"]


def test_notifier_subscription_survives_done_reopen_until_archive(
    tmp_path, monkeypatch,
):
    """Done is reversible; archive alone ends notification ownership.

    All events here are `completed` on a push adapter with an owning
    session, so since t_62e8c688 the raw ping is deferred to the wake
    (`adapter.sent` stays empty throughout) — the wake carries the same
    chat/thread/profile routing the raw ping used to.
    """
    db_path = tmp_path / "done-reopen-archive.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="review continuation",
            assignee="worker",
            session_id="origin-session",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="origin-chat",
            thread_id="origin-thread",
            user_id="origin-user",
            chat_type="group",
            notifier_profile="reviewer",
            delivery_mode="notify+wake",
        )
        assert kb.complete_task(conn, tid, summary="first completion")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "reviewer"
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.sent == [], "completed defers to the wake; no raw ping"
    assert len(adapter.handled) == 1
    assert adapter.handled[0].source.thread_id == "origin-thread"
    assert adapter.handled[0].source.profile == "reviewer"

    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1, "completion must retain the origin subscription"
        first_cursor = subs[0]["last_event_id"]
    finally:
        conn.close()

    # A quiet tick proves the completed event cannot replay after its cursor
    # was advanced, even though the subscription now remains present.
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "reviewer"
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert adapter.sent == []
    assert len(adapter.handled) == 1

    conn = kb.connect()
    try:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
            kb._append_event(conn, tid, "status", {"status": "ready"})
        assert kb.complete_task(conn, tid, summary="corrected completion")
    finally:
        conn.close()

    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "reviewer"
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # Internal reopen status is silent; only the second completion delivers
    # and wakes the exact original session/thread.
    assert adapter.sent == []
    assert len(adapter.handled) == 2
    assert adapter.handled[-1].source.thread_id == "origin-thread"
    assert adapter.handled[-1].source.profile == "reviewer"

    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1
        assert subs[0]["last_event_id"] > first_cursor
        assert kb.archive_task(conn, tid)
    finally:
        conn.close()

    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "reviewer"
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # Archive itself is intentionally silent, but consumes its event and
    # removes the subscription so no later historical event can replay.
    assert adapter.sent == []
    assert len(adapter.handled) == 2
    conn = kb.connect()
    try:
        assert kb.list_notify_subs(conn, tid) == []
    finally:
        conn.close()


def test_notifier_wakeup_uses_subscription_chat_type(tmp_path, monkeypatch):
    db_path = tmp_path / "chat-type-wakeup.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="dm requester",
            assignee="worker",
            session_id="origin-session",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-dm",
            chat_type="dm",
            delivery_mode="notify+wake",
        )
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    # completed defers to the wake (t_62e8c688); no raw ping is sent.
    assert adapter.sent == []
    assert len(adapter.handled) == 1
    assert adapter.handled[0].source.chat_type == "dm"

    # The wake must resume the creator's real DM session key — the whole bug
    # was that a hardcoded chat_type="group" made build_session_key() produce
    # a group-scoped key (a NEW session) instead of the ":dm:<chat_id>" shape
    # the original conversation runs under (#56580 / #68874).
    from gateway.session import build_session_key

    wake_key = build_session_key(adapter.handled[0].source)
    assert wake_key == "agent:main:telegram:dm:chat-dm"
    assert ":group:" not in wake_key


def _unseen_terminal_events_for(tid, chat_id):
    conn = kb.connect()
    try:
        _, events = kb.unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id=chat_id,
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


def test_kanban_notifier_isolates_per_subscription_failure(tmp_path, monkeypatch):
    """One bad subscription must not block delivery for all others.

    Regression for #59269: when claim_unseen_events_for_sub raises for one
    subscription, the entire notifier tick used to abort — silently blocking
    delivery for every other subscription.
    """
    db_path = tmp_path / "isolation.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    # Create two tasks with subscriptions and complete both. The BAD task is
    # created first: list_notify_subs() has no ORDER BY, so SQLite's natural
    # scan returns insertion order — the failing subscription must be
    # processed BEFORE the good one or this test passes even without the
    # per-subscription isolation (the good delivery happens before the tick
    # aborts). A deterministic-order shim below removes the reliance on the
    # scan order entirely.
    conn = kb.connect()
    try:
        tid_bad = kb.create_task(conn, title="bad task", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid_bad, platform="telegram", chat_id="chat-bad")
        kb.complete_task(conn, tid_bad, summary="done")

        tid_good = kb.create_task(conn, title="good task", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid_good, platform="telegram", chat_id="chat-good")
        kb.complete_task(conn, tid_good, summary="done")
    finally:
        conn.close()

    original_claim = kb.claim_unseen_events_for_sub

    def selective_claim(conn, task_id, **kwargs):
        if task_id == tid_bad:
            raise RuntimeError("simulated DB corruption for bad task")
        return original_claim(conn, task_id=task_id, **kwargs)

    monkeypatch.setattr(kb, "claim_unseen_events_for_sub", selective_claim)

    # Force the failing subscription to be iterated FIRST regardless of the
    # unordered SELECT's scan order.
    original_list = kb.list_notify_subs

    def bad_first(conn, task_id=None, **kwargs):
        subs = original_list(conn, task_id, **kwargs)
        return sorted(subs, key=lambda s: 0 if s["task_id"] == tid_bad else 1)

    monkeypatch.setattr(kb, "list_notify_subs", bad_first)

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # The good task must still be delivered despite the bad task failing.
    assert len(adapter.sent) == 1
    assert tid_good in adapter.sent[0]["text"]


def test_notifier_delivers_block_loop_detected_triage_ping(tmp_path, monkeypatch):
    """A `block_loop_detected` event must reach the subscriber as ONE clean
    human message — never the raw internal `triage` wording, and never
    silence either.

    Regression for the silent-triage gap (PR #62712): kanban_db routes a task
    to `triage` after BLOCK_RECURRENCE_LIMIT re-blocks for the same cause and
    emits ONLY a `block_loop_detected` event — no `blocked`/`status` event.
    Before `block_loop_detected` joined TERMINAL_KINDS with its own message
    branch, that one transition (the whole point of which is to force human
    attention) produced zero notification and the task stalled in triage
    silently.

    A later incident showed the fix for that gap had overshot: the message
    was the *raw* worker/internal wording — "Kanban <task_id> routed to
    TRIAGE — needs a human decision (blocked Nx for the same cause): <raw
    reason>" — partly in English, leaking the raw task id and internal
    `gate:` marker. Sébastien flagged this as exactly the kind of opaque
    ping the `blocked` case was already fixed to avoid (see
    ``test_notifier_sends_clean_human_ping_for_blocked_task``). The correct
    behavior mirrors that fix: one clean French human message, no task id,
    no `gate:` marker, no English, just the plain-language reason.
    """
    db_path = tmp_path / "block-loop.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="Identifier compte e-mail gérant API Ecobloc",
            assignee="worker",
            session_id="origin-session",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            delivery_mode="notify+wake",
        )
        kb._append_event(
            conn, tid, "block_loop_detected",
            {
                "reason": (
                    "gate:credentials — Consentement OAuth Google Business Profile "
                    "obtenu et jeton valide, mais l’appel API échoue en 403 car "
                    "mybusinessaccountmanagement.googleapis.com est désactivée pour "
                    "le projet 362154063865 ; décision attendue de Sébastien - "
                    "activer l’API dans ce projet ou autoriser Hermes à le faire. "
                    "Une fois activée, je relance la vérification en lecture seule "
                    "pour clore la carte."
                ),
                "kind": "capability",
                "recurrences": 2,
                "limit": kb.BLOCK_RECURRENCE_LIMIT,
            },
        )
        # Mirror kanban_db.block_task's real recurrence-limit transition: the
        # task actually lands in `triage` and stays there (no auto-decompose
        # resolution in this scenario) — the case that genuinely needs a
        # human decision.
        conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (tid,))
        conn.commit()
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1, "block_loop_detected must produce a notification"
    text = adapter.sent[0]["text"]
    assert "TRIAGE" not in text, "no raw internal TRIAGE wording in a human message"
    assert tid not in text, "no raw task id in a direct chat message"
    assert "gate:" not in text, "internal gate: marker must not leak"
    assert "Blocage :" in text, "the diagnosis must be labelled separately"
    assert "l’appel API échoue en 403" in text, "the plain-language cause must survive"
    assert "Identifier compte e-mail gérant API Ecobloc" in text
    assert "Action attendue de toi :" in text
    assert (
        "activer l’API dans ce projet ou autoriser Hermes à le faire." in text
    ), "the exact Ecobloc action must survive after the long technical diagnosis"
    assert "clore la carte." in text, "the single-task action must not be truncated"
    assert "réutili\n" not in text, "copy must never be cut in the middle of a word"
    assert "Ensuite : Hermes reprend automatiquement" in text
    assert "@worker" not in text, "no assignee/profile tag on a human-decision message"
    assert "[default]" not in text, "no bracketed board-tag prefix on a human-decision message"
    assert len(adapter.handled) == 1, (
        "block_loop_detected must wake the owning Hermes conversation after notifying"
    )
    # Cursor advanced: the event is claimed and not re-delivered.
    conn = kb.connect()
    try:
        _, remaining = kb.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-1",
            kinds=["block_loop_detected"],
        )
    finally:
        conn.close()
    assert remaining == []


def test_notifier_delivers_one_clean_final_coder_relay(tmp_path, monkeypatch):
    db_path = tmp_path / "coder-relay.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="internal task", assignee="coder")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb._append_event(conn, tid, "relayed_to_coder", {
            "message": "Relais automatique vers Coder.\nRetour estimé Claude 2 : 09:30",
        })
        conn.commit()
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert adapter.sent[0]["text"] == "Relais automatique vers Coder.\nRetour estimé Claude 2 : 09:30"
    assert tid not in adapter.sent[0]["text"]
    assert "@default" not in adapter.sent[0]["text"]


def test_notifier_explains_provider_auth_action_while_fallback_continues(
    tmp_path, monkeypatch,
):
    db_path = tmp_path / "provider-auth-required.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="Vérifier Ecobloc", assignee="claude2")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb._append_event(
            conn,
            tid,
            "provider_auth_required",
            {
                "profile": "claude2",
                "provider": "anthropic",
                "error": "OAuth token expired",
                "action": "Reconnecter anthropic dans le profil claude2.",
                "fallback_active": True,
                "fallback": "openai-codex/gpt-5.5",
            },
        )
        conn.commit()
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    message = adapter.sent[0]["text"]
    assert "Authentification à corriger" in message
    assert "claude2 (anthropic)" in message
    assert "OAuth token expired" in message
    assert "Reconnecter anthropic" in message
    assert "continue automatiquement via openai-codex/gpt-5.5" in message
    assert tid not in message


def test_notifier_explains_visual_retry_requires_no_human_action(tmp_path, monkeypatch):
    db_path = tmp_path / "visual-review-deferred.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    retry_at = int(time.time()) + 900
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="Finaliser la page mobile", assignee="coder")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb._append_event(
            conn,
            tid,
            "visual_review_deferred",
            {"reason": "quota Gemini", "retry_at": retry_at},
        )
        conn.commit()
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    message = adapter.sent[0]["text"]
    assert "validation visuelle finale Gemini" in message
    assert "Reprise automatique" in message
    assert "Aucune action requise" in message
    assert tid not in message


def test_notifier_suppresses_block_loop_ping_when_already_auto_resolved(
    tmp_path, monkeypatch,
):
    """A raw `block_loop_detected` event must send ZERO message once the
    underlying task has already left `triage` by the time the notifier
    processes it.

    The dispatcher's auto-decomposer polls every `triage`-status task
    (including ones routed there by this exact recurrence-limit path) on its
    own independent tick and can turn it back into ready/running work with
    no human input needed — a real race between the two loops. Pinging
    Sébastien anyway for a transition that resolved itself is exactly the
    raw-internal-event noise this task exists to silence: a genuine human
    decision is only needed while the task is still actually sitting in
    `triage`.
    """
    db_path = tmp_path / "block-loop-auto-resolved.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="loops forever", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb._append_event(
            conn, tid, "block_loop_detected",
            {"reason": "needs credentials", "kind": "needs_input",
             "recurrences": 2, "limit": kb.BLOCK_RECURRENCE_LIMIT},
        )
        # Simulate the auto-decomposer having already resolved the task back
        # to ready/fanned-out work in the gap between the DB write and this
        # notifier tick.
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
        conn.commit()
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.sent == [], (
        "a self-resolved triage transition must not ping the user"
    )
    # Cursor still advances — the event is claimed and not re-delivered
    # forever just because it produced no message.
    conn = kb.connect()
    try:
        _, remaining = kb.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-1",
            kinds=["block_loop_detected"],
        )
    finally:
        conn.close()
    assert remaining == []


def test_notifier_sends_clean_human_ping_for_blocked_task(tmp_path, monkeypatch):
    """A `blocked` event must reach the user as ONE clean human message.

    Two failure modes bracket this contract, both observed in production:
    (1) the original raw ping — "⛔ [default] @codex-worker … gate:credentials
    …" — leaked task ids, profile-tag brackets, and the internal `gate:`
    marker as noise Sébastien explicitly flagged; (2) a since-reverted "fix"
    made `blocked` fully silent, relying only on the passive "Travail en
    cours" board card — which reproduced the actual incident this task fixes:
    a resolved/actionable state with no message telling him so. The correct
    behavior is neither raw noise nor silence: send exactly one message,
    strip the `gate:<type> —` marker, and keep the plain-language reason a
    worker is required to write.
    """
    db_path = tmp_path / "blocked-clean-ping.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="needs a token", assignee="codex-worker",
            session_id="origin-session",
        )
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-1",
            delivery_mode="notify+wake",
        )
        ok = kb.block_task(
            conn, tid,
            reason="gate:credentials — needs the Ecobloc GSC token to continue",
            kind="capability",
        )
        assert ok, "block_task should succeed from the default ready state"
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # Exactly one terminal message — not silence, not a duplicate.
    assert len(adapter.sent) == 1, (
        f"blocked must send exactly one human message, got: {adapter.sent}"
    )
    message = adapter.sent[0]["text"]
    assert tid not in message, "no raw task id in a direct chat message"
    assert "gate:" not in message, "internal gate: marker must not leak"
    assert "needs the Ecobloc GSC token to continue" in message, (
        "the plain-language reason must survive"
    )
    assert "Blocage :" in message
    assert "Action attendue de toi :" in message
    assert "Ensuite : Hermes reprend automatiquement" in message
    assert "codex-worker" not in message, (
        "no assignee/profile tag on a human-decision message"
    )
    assert "[default]" not in message, (
        "no bracketed board-tag prefix on a human-decision message"
    )

    # The wake self-post still fires — the creator agent is still informed
    # internally and can act (e.g. route a real button-based permission ask).
    assert len(adapter.handled) == 1

    # Cursor still advances past the blocked event — it must not be
    # redelivered forever.
    conn = kb.connect()
    try:
        _, remaining = kb.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-1",
            kinds=["blocked"],
        )
    finally:
        conn.close()
    assert remaining == []


def test_notifier_classifies_prompt_timeout_as_internal_authorization_failure(
    tmp_path, monkeypatch,
):
    """A worker-approval timeout is an internal relay/sync failure, not a
    request for Sébastien to repeat an authorization manually."""
    db_path = tmp_path / "blocked-internal-auth-timeout.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="protected write relay", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        assert kb.block_task(
            conn,
            tid,
            reason=(
                "approval prompt timed out without a user response. "
                "Silence is not consent. Run authorize-instruction-edit if needed."
            ),
            kind="capability",
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    message = adapter.sent[0]["text"]
    assert "Problème interne d’autorisation" in message
    assert "aucune action de votre part" in message
    assert "Reprise automatique en cours" in message
    assert "Action requise" not in message


def test_notifier_recovers_blocked_task_when_durable_grant_already_exists(
    tmp_path, monkeypatch,
):
    """A present durable grant proves the failure is local guard sync, so the
    notifier must not ask the user again and must make dispatcher recovery
    possible by unblocking the task."""
    db_path = tmp_path / "blocked-grant-sync-failure.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    target = tmp_path / "AGENTS.md"
    target.write_text("rules")

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="grant already exists", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        assert kb.authorize_instruction_edit(
            conn,
            tid,
            str(target),
            granted_by="operator",
            reason="Sébastien already authorized this exact file",
        )
        assert kb.block_task(
            conn,
            tid,
            reason="approval prompt timed out despite a durable grant",
            kind="capability",
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    message = adapter.sent[0]["text"]
    assert "Problème interne d’autorisation" in message
    assert "Action requise" not in message
    conn = kb.connect()
    try:
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"
    finally:
        conn.close()


def test_notifier_delivers_final_message_after_block_genuinely_resolved(
    tmp_path, monkeypatch,
):
    """Block → unblock → completed must end in exactly one clear final message.

    This pins the "blocage réellement résolu" case from the incident: a task
    that was blocked, got unblocked, and then genuinely finished must not
    leave Sébastien needing to ask again for the result. The intermediate
    `unblocked` transition stays silent (internal), but the block itself and
    the eventual completion both reach the user — never zero, never a
    stuck/ambiguous state.
    """
    db_path = tmp_path / "block-resolved-then-complete.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="resumable task", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        assert kb.block_task(
            conn, tid, reason="besoin d'une décision sur le format", kind="needs_input",
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1, "the block itself must reach the user"
    assert "besoin d'une décision sur le format" in adapter.sent[0]["text"]

    conn = kb.connect()
    try:
        assert kb.unblock_task(conn, tid)
        assert kb.complete_task(conn, tid, summary="fait selon le format choisi")
    finally:
        conn.close()

    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # Exactly one more message: the completion. `unblocked` stays silent —
    # it must not add a second, redundant ping.
    assert len(adapter.sent) == 2, (
        f"expected block + completion only, got: {[d['text'] for d in adapter.sent]}"
    )
    assert "fait selon le format choisi" in adapter.sent[1]["text"]

    # A third tick with no new events must not resend anything (dedup holds).
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert len(adapter.sent) == 2, "no event left to redeliver; count must not grow"


def test_notifier_resumes_delivery_after_crash_then_completion(tmp_path, monkeypatch):
    """A crash → retry → completion cycle must not strand the final result.

    Mirrors the incident report: an announced task that crashes mid-run, gets
    reclaimed by the dispatcher, and eventually finishes must still surface
    ONE clear terminal message for the completion — not just the earlier
    crash ping, and not a hang waiting on a manual follow-up.
    """
    db_path = tmp_path / "crash-then-resume-complete.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="flaky task", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb._append_event(conn, tid, kind="crashed")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert len(adapter.sent) == 1
    assert "Impact : le worker s’est arrêté avant la fin." in adapter.sent[0]["text"]

    # Dispatcher reclaims and respawns; this run succeeds.
    conn = kb.connect()
    try:
        assert kb.complete_task(conn, tid, summary="terminé après relance")
    finally:
        conn.close()

    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 2, (
        f"the post-crash completion must still be delivered, got: "
        f"{[d['text'] for d in adapter.sent]}"
    )
    assert "terminé après relance" in adapter.sent[1]["text"]


def test_notifier_does_not_double_send_same_event(tmp_path, monkeypatch):
    """Re-polling with no new events must never resend the last message.

    The cursor is the sole dedup mechanism; this pins that a completed
    event, once delivered and the cursor advanced, is never replayed by a
    later tick that finds nothing new to claim.
    """
    db_path = tmp_path / "no-double-send.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription(summary="fini une seule fois")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert len(adapter.sent) == 1

    # Three more ticks, no new DB writes in between.
    for _ in range(3):
        runner = _make_runner(adapter)
        asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1, (
        f"same event must not be redelivered across idle ticks, got: "
        f"{[d['text'] for d in adapter.sent]}"
    )

# ---------------------------------------------------------------------------
# Handoffs that hand a decision back to the origin must wake it, not only ping
# it: `review_requested` (implementation done, waiting for a reviewer) and
# `block_loop_detected` (routed to triage) are terminal kinds just like
# `blocked`.
# ---------------------------------------------------------------------------


def _wake_text(adapter):
    """Text of the single synthetic wake turn injected into the adapter."""
    assert len(adapter.handled) == 1, (
        f"expected exactly one wake turn, got {len(adapter.handled)}"
    )
    return getattr(adapter.handled[0], "text", "") or ""


def _review_handoff_task(
    *,
    delivery_mode="notify+wake",
    summary="PR ready: https://example.invalid/pr/7\nfull details below",
):
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="implement the thing",
            assignee="worker",
            session_id="agent:main:telegram:dm:chat-1",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            chat_type="dm",
            delivery_mode=delivery_mode,
        )
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        assert kb.request_review(
            conn, tid, summary=summary, expected_run_id=run_id,
        ) is True
        return tid
    finally:
        conn.close()


def test_review_requested_wakes_the_origin_session(tmp_path, monkeypatch):
    """A review handoff wakes the origin and carries the worker's summary."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "review-wake.db"))
    kb.init_db()
    tid = _review_handoff_task()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1, "the passive review ping is unchanged"
    # t_e5cb4411: the raw ping must read as a plain French human message —
    # no raw task id, no "Kanban" literal, no English jargon.
    text = adapter.sent[0]["text"]
    assert tid not in text
    assert "Kanban" not in text
    assert "en attente de vérification" in text

    wake = _wake_text(adapter)
    assert tid in wake
    assert "PR ready: https://example.invalid/pr/7" in wake, (
        "the worker's handoff must ride the wake turn like it does for "
        "`completed`, otherwise the woken reviewer has to re-read the board"
    )


def test_block_loop_detected_wakes_the_origin_session(tmp_path, monkeypatch):
    """A triage escalation wakes the origin so a decision gets made."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "triage-wake.db"))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="loops forever",
            assignee="worker",
            session_id="agent:main:telegram:dm:chat-1",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            chat_type="dm",
            delivery_mode="notify+wake",
        )
        # A live block-loop escalation is actionable only while the card is
        # still in triage. If it has already returned to ready/running, the
        # local notifier intentionally suppresses the stale human ping.
        conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (tid,))
        kb._append_event(
            conn, tid, "block_loop_detected",
            {"reason": "needs credentials", "kind": "needs_input",
             "recurrences": 2, "limit": kb.BLOCK_RECURRENCE_LIMIT},
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert tid in _wake_text(adapter)


def test_review_requested_does_not_wake_a_notify_only_subscription(
    tmp_path, monkeypatch,
):
    """delivery_mode still decides whether a wake-worthy kind wakes at all."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "review-notify.db"))
    kb.init_db()
    _review_handoff_task(delivery_mode="notify")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert adapter.handled == [], (
        "notify-only subscriptions must not be woken by a review handoff"
    )
