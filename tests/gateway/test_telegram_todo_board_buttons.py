"""Telegram pagination for the single persistent TODO task message."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.platforms.telegram import todo_board_pagination as pagination


def make_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.name = "Telegram"
    adapter._is_callback_user_authorized = MagicMock(return_value=True)
    return adapter


def make_query(data: str, *, message_id: int = 222):
    query = AsyncMock()
    query.data = data
    query.message = MagicMock()
    query.message.chat_id = -100
    query.message.message_id = message_id
    query.message.message_thread_id = 10214
    query.message.chat.type = "supergroup"
    query.from_user = SimpleNamespace(id=777, first_name="Tester")
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    return query


def write_state(tmp_path, *, message_id: int = 222, pages: list[str] | None = None) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "todo-hub.json").write_text(json.dumps({
        "chat_id": "-100",
        "thread_id": 10214,
        "message_id": message_id,
        "rendered_pages": pages or ["<b>Page une</b>", "<b>Page deux</b>"],
    }), encoding="utf-8")


@pytest.mark.asyncio
async def test_todo_button_edits_same_official_message_to_requested_page(tmp_path) -> None:
    write_state(tmp_path)
    query = make_query("tb:1")
    update = SimpleNamespace(callback_query=query)
    adapter = make_adapter()
    markup = object()

    with patch("hermes_constants.get_hermes_home", return_value=tmp_path), \
         patch.object(pagination, "_page_keyboard", return_value=markup) as page_keyboard:
        await pagination.handle_todo_board_callback(update, MagicMock(), adapter)

    query.edit_message_text.assert_awaited_once()
    kwargs = query.edit_message_text.await_args.kwargs
    assert kwargs["text"] == "<b>Page deux</b>"
    assert kwargs["reply_markup"] is markup
    page_keyboard.assert_called_once_with(1, 2)
    query.answer.assert_awaited_once_with(text="Page 2/2")


@pytest.mark.asyncio
async def test_todo_button_rejects_an_old_generated_message(tmp_path) -> None:
    write_state(tmp_path)
    query = make_query("tb:1", message_id=111)
    update = SimpleNamespace(callback_query=query)
    adapter = make_adapter()

    with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
        await pagination.handle_todo_board_callback(update, MagicMock(), adapter)

    query.edit_message_text.assert_not_awaited()
    assert "no longer active" in query.answer.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_todo_button_rejects_unauthorized_user_before_state_read(tmp_path) -> None:
    query = make_query("tb:1")
    update = SimpleNamespace(callback_query=query)
    adapter = make_adapter()
    adapter._is_callback_user_authorized.return_value = False

    with patch("hermes_constants.get_hermes_home") as get_home:
        await pagination.handle_todo_board_callback(update, MagicMock(), adapter)

    get_home.assert_not_called()
    query.edit_message_text.assert_not_awaited()
    query.answer.assert_awaited_once_with(
        text="⛔ You are not authorized to browse this board."
    )


def test_register_wires_only_scoped_telegram_callback_handler() -> None:
    application = MagicMock()
    handler = object()

    with patch("telegram.ext.CallbackQueryHandler", return_value=handler) as factory:
        pagination.wire_todo_board_pagination(application, make_adapter())

    factory.assert_called_once()
    assert factory.call_args.kwargs["pattern"] == r"^tb:"
    application.add_handler.assert_called_once_with(handler)
