"""Concurrent protected-instruction edits require one exact, atomic consent."""

import threading
import time


def test_concurrent_protected_instruction_edits_share_one_exact_prompt(monkeypatch):
    from tools import approval
    from tools import file_tools as ft

    session_key = "protected-group-test"
    approval.clear_session(session_key)
    approval.unregister_gateway_notify(session_key)
    monkeypatch.setattr(approval, "_get_approval_timeout", lambda: 5)
    monkeypatch.setattr(ft, "_PROTECTED_APPROVAL_BATCH_WINDOW_SECONDS", 0.05)
    monkeypatch.setattr(approval, "get_current_session_key", lambda: session_key)

    notified = []
    approval.register_gateway_notify(session_key, notified.append)
    paths = ["/repo/AGENTS.md", "/repo/CLAUDE.md"]
    results: list[str | None] = [None, None]
    threads = [
        threading.Thread(
            target=lambda index=index: results.__setitem__(
                index,  # type: ignore[index]
                ft._request_protected_instruction_approval(
                    [paths[index].rsplit("/", 1)[-1]], [paths[index]]
                ),
            ),
        )
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for _ in range(200):
        if notified:
            break
        time.sleep(0.01)

    assert len(notified) == 1
    assert notified[0]["protected_instruction_paths"] == paths
    assert "/repo/AGENTS.md" in notified[0]["description"]
    assert "/repo/CLAUDE.md" in notified[0]["description"]

    assert approval.resolve_gateway_approval(session_key, "once") == 1
    for thread in threads:
        thread.join(timeout=5)
    assert results == [None, None]
    assert approval.list_gateway_approvals(session_key) == []


def test_same_session_does_not_merge_relative_paths_from_distinct_tasks(monkeypatch):
    from tools import approval
    from tools import file_tools as ft

    session_key = "protected-group-distinct-tasks"
    approval.clear_session(session_key)
    approval.unregister_gateway_notify(session_key)
    monkeypatch.setattr(approval, "_get_approval_timeout", lambda: 5)
    monkeypatch.setattr(ft, "_PROTECTED_APPROVAL_BATCH_WINDOW_SECONDS", 0.05)
    monkeypatch.setattr(approval, "get_current_session_key", lambda: session_key)
    monkeypatch.setattr(
        ft,
        "_resolve_path_for_task",
        lambda path, task_id: f"/workspaces/{task_id}/{path}",
    )
    notified = []
    approval.register_gateway_notify(session_key, notified.append)
    results: list[str | None] = [None, None]
    threads = [
        threading.Thread(
            target=lambda index=index: results.__setitem__(
                index,
                ft._request_protected_instruction_approval(
                    ["AGENTS.md"], ["AGENTS.md"], f"task-{index}"
                ),
            ),
        )
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for _ in range(200):
        if len(notified) == 2:
            break
        time.sleep(0.01)

    assert len(notified) == 2
    assert {tuple(item["protected_instruction_paths"]) for item in notified} == {
        ("/workspaces/task-0/AGENTS.md",),
        ("/workspaces/task-1/AGENTS.md",),
    }
    for _ in range(2):
        assert approval.resolve_gateway_approval(session_key, "once") == 1
    for thread in threads:
        thread.join(timeout=5)
    assert results == [None, None]
    approval.unregister_gateway_notify(session_key)


def test_grouped_prompt_never_authorizes_a_later_path(monkeypatch):
    from tools import approval
    from tools import file_tools as ft

    session_key = "protected-group-later-path"
    approval.clear_session(session_key)
    approval.unregister_gateway_notify(session_key)
    monkeypatch.setattr(approval, "_get_approval_timeout", lambda: 5)
    monkeypatch.setattr(ft, "_PROTECTED_APPROVAL_BATCH_WINDOW_SECONDS", 0)
    monkeypatch.setattr(approval, "get_current_session_key", lambda: session_key)

    notified = []
    approval.register_gateway_notify(session_key, notified.append)
    result = {}
    thread = threading.Thread(
        target=lambda: result.setdefault(
            "value",
            ft._request_protected_instruction_approval(["AGENTS.md"], ["/repo/AGENTS.md"]),
        )
    )
    thread.start()
    for _ in range(200):
        if notified:
            break
        time.sleep(0.01)
    approval.resolve_gateway_approval(session_key, "once")
    thread.join(timeout=5)

    assert result["value"] is None
    assert ft._request_protected_instruction_approval(
        ["CLAUDE.md"], ["/repo/CLAUDE.md"]
    ) is not None
    assert len(notified) == 2
    approval.unregister_gateway_notify(session_key)


def test_late_path_waits_for_the_first_prompt_instead_of_becoming_a_sibling(monkeypatch):
    from tools import approval
    from tools import file_tools as ft

    session_key = "protected-group-no-sibling"
    approval.clear_session(session_key)
    approval.unregister_gateway_notify(session_key)
    monkeypatch.setattr(approval, "_get_approval_timeout", lambda: 5)
    monkeypatch.setattr(ft, "_PROTECTED_APPROVAL_BATCH_WINDOW_SECONDS", 0)
    monkeypatch.setattr(approval, "get_current_session_key", lambda: session_key)
    notified = []
    approval.register_gateway_notify(session_key, notified.append)

    first = threading.Thread(
        target=lambda: ft._request_protected_instruction_approval(
            ["AGENTS.md"], ["/repo/AGENTS.md"]
        )
    )
    first.start()
    for _ in range(200):
        if notified:
            break
        time.sleep(0.01)
    late = threading.Thread(
        target=lambda: ft._request_protected_instruction_approval(
            ["CLAUDE.md"], ["/repo/CLAUDE.md"]
        )
    )
    late.start()
    time.sleep(0.05)

    assert len(notified) == 1
    assert len(approval.list_gateway_approvals(session_key)) == 1
    approval.resolve_gateway_approval(session_key, "deny")
    first.join(timeout=5)
    late.join(timeout=5)
    assert len(notified) == 1
    assert approval.list_gateway_approvals(session_key) == []
    approval.unregister_gateway_notify(session_key)


def test_expired_group_cancels_every_waiting_protected_edit(monkeypatch):
    from tools import approval
    from tools import file_tools as ft

    session_key = "protected-group-expiry"
    approval.clear_session(session_key)
    approval.unregister_gateway_notify(session_key)
    monkeypatch.setattr(approval, "_get_approval_timeout", lambda: 0.05)
    monkeypatch.setattr(ft, "_PROTECTED_APPROVAL_BATCH_WINDOW_SECONDS", 0)
    monkeypatch.setattr(approval, "get_current_session_key", lambda: session_key)
    notified = []
    approval.register_gateway_notify(session_key, notified.append)
    results: list[str | None] = [None, None]
    paths = ["/repo/AGENTS.md", "/repo/CLAUDE.md"]
    threads = [
        threading.Thread(
            target=lambda index=index: results.__setitem__(
                index,
                ft._request_protected_instruction_approval(
                    [paths[index].rsplit("/", 1)[-1]], [paths[index]]
                ),
            ),
        )
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(notified) == 1
    assert all(result and "timed out" in result for result in results)
    assert approval.list_gateway_approvals(session_key) == []
    approval.unregister_gateway_notify(session_key)
