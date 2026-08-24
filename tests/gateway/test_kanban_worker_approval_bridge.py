"""Gateway side of the Kanban worker approval bridge (t_f98bc92d).

``GatewayKanbanWatchersMixin._dispatch_kanban_worker_approval`` is the piece
that turns one claimed worker-approval request into a real chat prompt via
the adapter's existing ``send_exec_approval`` and relays the human's
decision (through the unmodified ``resolve_gateway_approval``) back to the
worker's file. These tests exercise it directly against a minimal fake
``self`` — no real gateway/adapters needed — covering the two contractual
edges the task calls out:

* a transport failure (``send_exec_approval`` raising, or no reachable
  subscriber) must resolve to refusal, never leave a phantom approval; and
* a real button decision is relayed byte-for-byte into the request file so
  the worker's poll loop (tested separately) sees it.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="worker-approval-bridge-test", assignee="test-worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    return tid


class _FakeAdapter:
    def __init__(self, *, raise_on_send=False, success=True):
        self.raise_on_send = raise_on_send
        self.success = success
        self.calls = []

    async def send_exec_approval(self, **kwargs):
        from gateway.platforms.base import SendResult

        self.calls.append(kwargs)
        if self.raise_on_send:
            raise RuntimeError("transport is down")
        return SendResult(success=self.success, message_id="1", error=None if self.success else "boom")


class _FakeSelf:
    """Minimal stand-in exposing exactly what the mixin method touches."""

    def __init__(self, adapter):
        self._adapter = adapter

    def _authorization_adapter(self, platform, profile):
        return self._adapter


def _bind(fake_self):
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    return GatewayKanbanWatchersMixin._dispatch_kanban_worker_approval.__get__(fake_self)


def test_no_subscriber_leaves_the_request_for_the_workers_own_timeout(worker_env):
    """No chat is watching this task: the bridge must NOT write any decision
    — the worker's local timeout is the only thing allowed to close it out,
    and it always closes to refusal."""
    from tools import worker_approval as wa

    task_id = worker_env
    paths = ["/repo/AGENTS.md"]
    request_id = wa._request_id(task_id, paths)
    wa._atomic_write(wa._request_path(request_id), {
        "request_id": request_id, "task_id": task_id, "board": "default",
        "title": "t", "description": "d", "paths": paths,
        "created_at": 0, "expires_at": 9e18, "worker_pid": 1,
        "decision": None, "reason": None, "decided_by": None, "decided_at": None,
    })

    fake_self = _FakeSelf(adapter=None)
    method = _bind(fake_self)
    asyncio.run(method({
        "request_id": request_id, "task_id": task_id, "board": "default",
        "title": "t", "description": "d", "paths": paths,
        "expires_at": 9e18,
    }))

    assert wa._read(wa._request_path(request_id))["decision"] is None


def test_send_failure_resolves_to_refusal_never_a_phantom_approval(worker_env):
    from tools import worker_approval as wa
    from hermes_cli import kanban_db as kb

    task_id = worker_env
    paths = ["/repo/AGENTS.md"]
    request_id = wa._request_id(task_id, paths)
    wa._atomic_write(wa._request_path(request_id), {
        "request_id": request_id, "task_id": task_id, "board": "default",
        "title": "t", "description": "d", "paths": paths,
        "created_at": 0, "expires_at": 9e18, "worker_pid": 1,
        "decision": None, "reason": None, "decided_by": None, "decided_at": None,
    })
    conn = kb.connect()
    try:
        kb.add_notify_sub(
            conn, task_id=task_id, platform="telegram", chat_id="123",
        )
        conn.commit()
    finally:
        conn.close()

    adapter = _FakeAdapter(raise_on_send=True)
    fake_self = _FakeSelf(adapter=adapter)
    method = _bind(fake_self)
    asyncio.run(method({
        "request_id": request_id, "task_id": task_id, "board": "default",
        "title": "t", "description": "d", "paths": paths,
        "expires_at": 9e18,
    }))

    assert len(adapter.calls) == 1
    # The bridge must not have written any decision itself on a transport
    # failure — refusal comes only from the worker's own timeout closing
    # the file, never fabricated here.
    assert wa._read(wa._request_path(request_id))["decision"] is None

    from tools import approval

    assert approval.list_gateway_approvals(f"kanban-worker-approval:{request_id}") == []


def test_button_click_is_relayed_to_the_request_file(worker_env):
    from tools import worker_approval as wa
    from tools import approval
    from hermes_cli import kanban_db as kb

    task_id = worker_env
    paths = ["/repo/AGENTS.md"]
    request_id = wa._request_id(task_id, paths)
    wa._atomic_write(wa._request_path(request_id), {
        "request_id": request_id, "task_id": task_id, "board": "default",
        "title": "t", "description": "d", "paths": paths,
        "created_at": 0, "expires_at": 9e18, "worker_pid": 1,
        "decision": None, "reason": None, "decided_by": None, "decided_at": None,
    })
    conn = kb.connect()
    try:
        kb.add_notify_sub(
            conn, task_id=task_id, platform="telegram", chat_id="456",
        )
        conn.commit()
    finally:
        conn.close()

    adapter = _FakeAdapter()
    fake_self = _FakeSelf(adapter=adapter)
    method = _bind(fake_self)

    async def _run_and_click():
        task = asyncio.ensure_future(method({
            "request_id": request_id, "task_id": task_id, "board": "default",
            "title": "t", "description": "d", "paths": paths,
            "expires_at": 9e18,
        }))
        # Wait until the entry is actually enqueued (send_exec_approval
        # called), then click it exactly the way the Telegram ``ea:``
        # callback handler does — same public function, untouched.
        for _ in range(200):
            if adapter.calls:
                break
            await asyncio.sleep(0.01)
        session_key = f"kanban-worker-approval:{request_id}"
        count = approval.resolve_gateway_approval(session_key, "once")
        assert count == 1
        await task

    asyncio.run(_run_and_click())

    assert adapter.calls[0]["chat_id"] == "456"
    final = wa._read(wa._request_path(request_id))
    assert final["decision"] == "once"
    assert final["decided_by"] == "456"
