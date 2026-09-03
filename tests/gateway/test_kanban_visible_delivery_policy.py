from gateway.platforms.base import _compact_kanban_notification_text


def test_short_plain_notification_is_preserved():
    text = "La note 4,9/5 est publiée et vérifiée."
    assert _compact_kanban_notification_text(text) == text


def test_technical_completion_is_replaced_by_plain_title():
    text = (
        "Résultat livré : PID 1234, commit abcdef123456789, "
        "pytest tests/gateway/test_x.py, /home/seb/private/file.py"
    )
    assert _compact_kanban_notification_text(
        text,
        title="Corriger les notifications Telegram",
        event_kind="completed",
    ) == "Validé : Corriger les notifications Telegram."


def test_technical_block_keeps_action_label_and_title():
    text = (
        "Blocage : PID 1234, commit abcdef123456789, "
        "voir /home/seb/private/file.py"
    )
    assert _compact_kanban_notification_text(
        text,
        title="Reconnecter le compte",
        event_kind="blocked",
    ) == "Action requise : Reconnecter le compte."


def test_long_notification_is_bounded():
    result = _compact_kanban_notification_text("mot " * 200)
    assert len(result) <= 420
    assert result.endswith("…")
