"""Telegram callback-query bridge for Sébastien's external TODO board.

The TODO producer remains outside Hermes core (``~/.hermes/scripts/todo_hub.py``).
It pre-renders full HTML pages into ``state/todo-hub.json``; this plugin only
validates that a button belongs to the sole official Telegram message and asks
Telegram to edit that same message to the requested page.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_CALLBACK_PREFIX = "tb:"


def wire_todo_board_pagination(application: Any, adapter: Any) -> None:
    """Register the scoped TODO-board callback handler via the public hook."""
    from telegram.ext import CallbackQueryHandler

    application.add_handler(
        CallbackQueryHandler(
            lambda update, context: handle_todo_board_callback(update, context, adapter),
            pattern=r"^tb:",
        )
    )


async def handle_todo_board_callback(update: Any, _context: Any, adapter: Any) -> None:
    """Switch the sole task-table message between externally rendered pages."""
    query = getattr(update, "callback_query", None)
    data = getattr(query, "data", None)
    if query is None or not isinstance(data, str) or not data.startswith(_CALLBACK_PREFIX):
        return

    query_message = getattr(query, "message", None)
    query_chat_id = getattr(query_message, "chat_id", None)
    query_chat = getattr(query_message, "chat", None)
    query_chat_type = getattr(query_chat, "type", None)
    query_thread_id = getattr(query_message, "message_thread_id", None)
    query_user = getattr(query, "from_user", None)
    caller_id = str(getattr(query_user, "id", ""))
    query_user_name = getattr(query_user, "first_name", None)

    if not adapter._is_callback_user_authorized(  # noqa: SLF001 - platform auth API is not public yet.
        caller_id,
        chat_id=query_chat_id,
        chat_type=str(query_chat_type) if query_chat_type is not None else None,
        thread_id=str(query_thread_id) if query_thread_id is not None else None,
        user_name=query_user_name,
    ):
        await query.answer(text="⛔ You are not authorized to browse this board.")
        return

    try:
        from hermes_constants import get_hermes_home

        state_path = get_hermes_home() / "state" / "todo-hub.json"
        state = await asyncio.to_thread(
            lambda: json.loads(state_path.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        logger.warning("[%s] TODO board state unavailable: %s", adapter.name, exc)
        await query.answer(text="⚠️ Task board temporarily unavailable.")
        return

    query_message_id = getattr(query_message, "message_id", None)
    official = (
        str(query_chat_id) == str(state.get("chat_id"))
        and str(query_thread_id) == str(state.get("thread_id"))
        and str(query_message_id) == str(state.get("message_id"))
    )
    pages = state.get("rendered_pages")
    if not official or not isinstance(pages, list) or not pages:
        await query.answer(text="⌛ This task board is no longer active.")
        return

    token = data.split(":", 1)[1]
    if token == "noop":
        await query.answer()
        return
    try:
        page_index = int(token)
    except (TypeError, ValueError):
        await query.answer(text="Invalid page.")
        return
    if not 0 <= page_index < len(pages):
        await query.answer(text="Page no longer available.")
        return

    try:
        from telegram.constants import ParseMode

        await query.edit_message_text(
            text=str(pages[page_index]),
            parse_mode=ParseMode.HTML,
            reply_markup=_page_keyboard(page_index, len(pages)),
        )
        await query.answer(text=f"Page {page_index + 1}/{len(pages)}")
    except Exception as exc:
        logger.warning("[%s] TODO board page edit failed: %s", adapter.name, exc)
        await query.answer(text="⚠️ Unable to change page right now.")


def _page_keyboard(page_index: int, total_pages: int) -> Any:
    """Build Telegram navigation controls for the already-rendered pages."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    buttons = []
    if page_index > 0:
        buttons.append(InlineKeyboardButton(
            "◀ Précédente", callback_data=f"tb:{page_index - 1}"
        ))
    buttons.append(InlineKeyboardButton(
        f"{page_index + 1}/{total_pages}", callback_data="tb:noop"
    ))
    if page_index + 1 < total_pages:
        buttons.append(InlineKeyboardButton(
            "Suivante ▶", callback_data=f"tb:{page_index + 1}"
        ))
    return InlineKeyboardMarkup([buttons])
