"""Telegram pagination for the single persistent TODO task message."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
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


@pytest.mark.asyncio
async def test_todo_button_edits_same_official_message_to_requested_page(tmp_path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "todo-hub.json").write_text(json.dumps({
        "chat_id": "-100",
        "thread_id": 10214,
        "message_id": 222,
        "rendered_pages": ["<b>Page une</b>", "<b>Page deux</b>"],
    }), encoding="utf-8")
    query = make_query("tb:1")
    update = SimpleNamespace(callback_query=query)
    adapter = make_adapter()

    def button(text, callback_data):
        return SimpleNamespace(text=text, callback_data=callback_data)

    with patch("hermes_constants.get_hermes_home", return_value=tmp_path), \
         patch("plugins.platforms.telegram.adapter.InlineKeyboardButton", side_effect=button), \
         patch("plugins.platforms.telegram.adapter.InlineKeyboardMarkup", side_effect=lambda rows: rows):
        await adapter._handle_callback_query(update, MagicMock())

    query.edit_message_text.assert_awaited_once()
    kwargs = query.edit_message_text.await_args.kwargs
    assert kwargs["text"] == "<b>Page deux</b>"
    assert kwargs["reply_markup"][0][0].callback_data == "tb:0"
    query.answer.assert_awaited_once_with(text="Page 2/2")


@pytest.mark.asyncio
async def test_todo_button_rejects_an_old_generated_message(tmp_path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "todo-hub.json").write_text(json.dumps({
        "chat_id": "-100",
        "thread_id": 10214,
        "message_id": 222,
        "rendered_pages": ["Page une", "Page deux"],
    }), encoding="utf-8")
    query = make_query("tb:1", message_id=111)
    update = SimpleNamespace(callback_query=query)
    adapter = make_adapter()

    with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
        await adapter._handle_callback_query(update, MagicMock())

    query.edit_message_text.assert_not_awaited()
    assert "no longer active" in query.answer.await_args.kwargs["text"]
