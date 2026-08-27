"""One-tap Telegram recovery after bounded automatic retries are exhausted."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    adapter._is_callback_user_authorized = MagicMock(return_value=True)
    return adapter


@pytest.mark.asyncio
async def test_resume_prompt_renders_one_button_and_pins_session() -> None:
    adapter = _adapter()
    sent = MagicMock(message_id=301)
    adapter._bot.send_message = AsyncMock(return_value=sent)

    with (
        patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardButton",
            side_effect=lambda text, callback_data: SimpleNamespace(
                text=text, callback_data=callback_data,
            ),
        ),
        patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup",
            side_effect=lambda rows: SimpleNamespace(inline_keyboard=rows),
        ),
    ):
        result = await adapter.send_resume_prompt(
            chat_id="12345",
            session_key="agent:main:telegram:dm:12345",
            metadata={"thread_id": "77"},
        )

    assert result.success is True
    kwargs = adapter._bot.send_message.await_args.kwargs
    button = kwargs["reply_markup"].inline_keyboard[0][0]
    assert button.text == "▶️ Relancer maintenant"
    assert button.callback_data.startswith("hr:")
    resume_id = int(button.callback_data.split(":", 1)[1])
    assert adapter._resume_prompt_state[resume_id] == "agent:main:telegram:dm:12345"


@pytest.mark.asyncio
async def test_resume_button_dispatches_exact_session_once() -> None:
    adapter = _adapter()
    adapter._resume_prompt_state[9] = "agent:main:telegram:dm:12345"
    adapter.handle_message = AsyncMock()

    query = AsyncMock()
    query.data = "hr:9"
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.message.message_id = 302
    query.message.message_thread_id = None
    query.message.chat.type = "private"
    query.message.chat.title = None
    query.message.chat.full_name = "Sébastien"
    query.from_user = SimpleNamespace(
        id=777,
        first_name="Sébastien",
        full_name="Sébastien",
    )
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    await adapter._handle_callback_query(
        SimpleNamespace(callback_query=query),
        MagicMock(),
    )

    query.answer.assert_awaited_once_with(text="Relance engagée")
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.internal is True
    assert event.metadata["gateway_session_key"] == "agent:main:telegram:dm:12345"
    assert "Reprends exactement" in event.text
    assert 9 not in adapter._resume_prompt_state

    # The same tap is idempotent/stale-safe: it cannot queue a second retry.
    await adapter._handle_callback_query(
        SimpleNamespace(callback_query=query),
        MagicMock(),
    )
    adapter.handle_message.assert_awaited_once()
    assert "déjà été utilisée" in query.answer.await_args.kwargs["text"]
