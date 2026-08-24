"""Tests for agent/display.py — build_tool_preview() and inline diff previews."""

import json
import pytest
from unittest.mock import MagicMock

import agent.display as display_module
from agent.display import (
    build_tool_preview,
    capture_local_edit_snapshot,
    extract_edit_diff,
    get_cute_tool_message,
    prepare_tool_preview,
    redact_tool_args_for_display,
    set_tool_preview_max_len,
    _render_inline_unified_diff,
    _summarize_rendered_diff_sections,
    render_edit_diff_with_delta,
)


@pytest.fixture(autouse=True)
def reset_tool_preview_max_len():
    set_tool_preview_max_len(0)
    yield
    set_tool_preview_max_len(0)


def test_cute_tool_message_falls_back_when_renderer_raises(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise RuntimeError("cosmetic failure")

    monkeypatch.setattr(display_module, "_get_cute_tool_message", _boom)

    assert get_cute_tool_message("web_extract", {"urls": []}, 0.25) == (
        "┊ ⚡ web_extra completed  0.2s"
    )


class TestBuildToolPreview:
    """Tests for build_tool_preview defensive handling and normal operation."""

    def test_none_args_returns_none(self):
        """PR #453: None args should not crash, should return None."""
        assert build_tool_preview("terminal", None) is None

    def test_empty_dict_returns_none(self):
        """Empty dict has no keys to preview."""
        assert build_tool_preview("terminal", {}) is None








    def test_browser_type_preview_redacts_api_key(self):
        secret = "sk-proj-ABCD1234567890EFGH"
        result = build_tool_preview("browser_type", {"ref": "@e3", "text": secret})
        assert result is not None
        assert secret not in result
        assert "sk-pro" in result and "..." in result

    def test_browser_type_preview_keeps_normal_text(self):
        text = "hello world search query"
        result = build_tool_preview("browser_type", {"ref": "@e3", "text": text})
        assert result is not None
        assert text in result

    def test_browser_type_display_args_redact_api_key(self):
        secret = "ghp_ABCDEFGHIJ1234567890"
        safe_args = redact_tool_args_for_display(
            "browser_type", {"ref": "@e3", "text": secret}
        )
        assert secret not in str(safe_args)
        assert safe_args["ref"] == "@e3"
        assert safe_args["text"].startswith("ghp_AB")















    def test_delegate_task_batch_preview_respects_max_len(self):
        result = build_tool_preview(
            "delegate_task",
            {"tasks": [{"goal": "A" * 80}, {"goal": "B" * 80}]},
            max_len=30,
        )
        assert result == "2 tasks: AAAAAAAAAAAAAAAAAA..."
        assert len(result) == 30

    def test_false_like_args_zero(self):
        """Non-dict falsy values should return None, not crash."""
        assert build_tool_preview("terminal", 0) is None
        assert build_tool_preview("terminal", "") is None
        assert build_tool_preview("terminal", []) is None


class TestPrepareToolPreview:
    def test_recovers_and_describes_truncated_url(self):
        url = "https://example.com/a/very/long/path/to/a/page"
        set_tool_preview_max_len(20)

        preview = prepare_tool_preview(
            "web_extract",
            {"urls": [url]},
            fallback=url[:17] + "...",
            max_len=20,
        )

        assert preview.text == url[:17] + "..."
        assert preview.truncated is True
        assert preview.url == url

    def test_untruncated_url_has_no_link_target(self):
        url = "https://example.com/page"
        preview = prepare_tool_preview(
            "browser_navigate", None, fallback=url, max_len=40
        )

        assert preview.text == url
        assert preview.truncated is False
        assert preview.url is None

    def test_truncated_non_url_has_no_link_target(self):
        preview = prepare_tool_preview(
            "web_search",
            {"query": "how to parse a URL"},
            fallback="how to parse a URL",
            max_len=12,
        )

        assert preview.truncated is True
        assert preview.url is None


class TestCuteToolMessagePreviewLength:


    def test_search_files_preview_uses_positive_configured_limit_not_default(self):
        set_tool_preview_max_len(80)
        pattern = "function.formatToolCall.context.preview.compactPreview.maxLength.truncate"

        line = get_cute_tool_message("search_files", {"pattern": pattern}, 0.1)

        assert pattern in line
        assert "..." not in line





    def test_browser_type_cute_message_redacts_api_key(self):
        secret = "sk-proj-ABCD1234567890EFGH"
        line = get_cute_tool_message(
            "browser_type",
            {"ref": "@password", "text": secret},
            0.1,
            result='{"success": true, "typed": "sk-pro...EFGH"}',
        )

        assert secret not in line
        assert "sk-pro" in line

    def test_browser_type_cute_message_keeps_normal_text(self):
        text = "hello world"
        line = get_cute_tool_message(
            "browser_type",
            {"ref": "@search", "text": text},
            0.1,
            result='{"success": true, "typed": "hello world"}',
        )

        assert text in line


class TestEditDiffPreview:



    def test_extract_edit_diff_uses_local_snapshot_for_write_file(self, tmp_path):
        target = tmp_path / "note.txt"
        target.write_text("old\n", encoding="utf-8")

        snapshot = capture_local_edit_snapshot("write_file", {"path": str(target)})

        target.write_text("new\n", encoding="utf-8")

        diff = extract_edit_diff(
            "write_file",
            '{"bytes_written": 4}',
            function_args={"path": str(target)},
            snapshot=snapshot,
        )

        assert diff is not None
        assert "--- a/" in diff
        assert "+++ b/" in diff
        assert "-old" in diff
        assert "+new" in diff



    def test_render_edit_diff_with_delta_handles_renderer_errors(self, monkeypatch):
        printer = MagicMock()

        monkeypatch.setattr("agent.display._summarize_rendered_diff_sections", MagicMock(side_effect=RuntimeError("boom")))

        rendered = render_edit_diff_with_delta(
            "patch",
            '{"diff": "--- a/x\\n+++ b/x\\n"}',
            print_fn=printer,
        )

        assert rendered is False
        assert printer.call_count == 0


    def test_summarize_rendered_diff_sections_limits_file_count(self):
        diff = "".join(
            f"--- a/file{i}.py\n+++ b/file{i}.py\n+line{i}\n"
            for i in range(8)
        )

        rendered = _summarize_rendered_diff_sections(diff, max_files=3, max_lines=50)

        assert any("a/file0.py" in line for line in rendered)
        assert any("a/file1.py" in line for line in rendered)
        assert any("a/file2.py" in line for line in rendered)
        assert not any("a/file7.py" in line for line in rendered)
        assert "additional file" in rendered[-1]


class TestBuildToolLabel:
    """Friendly human-phrased tool labels for built-in tools."""

    @pytest.fixture(autouse=True)
    def _enable_friendly(self):
        from agent.display import set_friendly_tool_labels
        set_friendly_tool_labels(True)
        yield
        set_friendly_tool_labels(True)

    def test_web_search_uses_for_connector(self):
        from agent.display import build_tool_label
        label = build_tool_label("web_search", {"query": "weather in NYC"})
        assert label == 'Searching the web for weather in NYC'

    def test_web_extract_reads_url(self):
        from agent.display import build_tool_label
        label = build_tool_label("web_extract", {"urls": ["https://example.com/page"]})
        assert label is not None
        assert label.startswith("Reading ")
        assert "example.com/page" in label







    def test_disabled_falls_back_to_preview(self):
        from agent.display import (
            build_tool_label,
            build_tool_preview,
            set_friendly_tool_labels,
        )
        set_friendly_tool_labels(False)
        args = {"query": "weather in NYC"}
        label = build_tool_label("web_search", args)
        # With the feature off, must match the raw preview exactly
        assert label == build_tool_preview("web_search", args)
        assert "Searching the web" not in (label or "")



class TestBuildStatusPhrase:
    """build_status_phrase — live working-state text for Slack's status line."""



    def test_verb_only_when_args_none(self):
        # live_status: "verb" mode passes args=None to suppress previews.
        from agent.display import build_status_phrase
        assert build_status_phrase("terminal", None) == "is running…"
        assert build_status_phrase("read_file", None) == "is reading…"



    def test_caps_length_for_slack_status_line(self):
        from agent.display import build_status_phrase
        phrase = build_status_phrase(
            "terminal", {"command": "x" * 300}, max_len=49
        )
        assert phrase is not None and len(phrase) <= 49
        assert phrase.endswith("…")


    def test_respects_friendly_labels_toggle(self):
        from agent.display import build_status_phrase, set_friendly_tool_labels
        set_friendly_tool_labels(False)
        try:
            assert build_status_phrase("terminal", {"command": "ls"}) is None
        finally:
            set_friendly_tool_labels(True)


# ---------------------------------------------------------------------------
# get_cute_tool_message() -> live agent-activity board event emission
# (see /home/seb/.hermes/workspace/agent-live-activity-contract.md)
# ---------------------------------------------------------------------------

class TestToolActivityEventEmission:
    def test_emits_activity_event_when_inside_a_kanban_task(self, monkeypatch):
        # Patch the real hermes_cli.kanban_db symbol -- display.py does a
        # lazy `from hermes_cli import kanban_db as _kdb` inside the guarded
        # call, so once that module is already imported in-process (it is,
        # by the wider test suite) faking sys.modules alone does not
        # intercept it; the attribute must be patched on the real module.
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
        calls = []
        monkeypatch.setattr(
            "hermes_cli.kanban_db.append_activity_event",
            lambda **kw: calls.append(kw),
        )
        get_cute_tool_message("read_file", {"path": "foo.py"}, 0.1)
        assert len(calls) == 1
        assert calls[0]["action"] == "read_file"
        assert "foo.py" in calls[0]["target"]

    def test_no_event_outside_kanban_context(self, monkeypatch):
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        calls = []
        monkeypatch.setattr(
            "hermes_cli.kanban_db.append_activity_event",
            lambda **kw: calls.append(kw),
        )
        get_cute_tool_message("read_file", {"path": "foo.py"}, 0.1)
        assert calls == []

    def test_activity_action_maps_git_and_test_run_from_terminal_command(self):
        from agent.display import _activity_action_for_tool
        assert _activity_action_for_tool("terminal", {"command": "git status"}) == "git"
        assert _activity_action_for_tool("terminal", {"command": "pytest -q"}) == "test_run"
        assert _activity_action_for_tool("terminal", {"command": "ls -la"}) == "bash"
        assert _activity_action_for_tool("unknown_tool_xyz", {}) == "other"

    def test_a_broken_event_emitter_never_breaks_the_rendered_line(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
        monkeypatch.setattr(
            "hermes_cli.kanban_db.append_activity_event",
            MagicMock(side_effect=RuntimeError("db glitch")),
        )
        line = get_cute_tool_message("read_file", {"path": "foo.py"}, 0.1)
        assert "read" in line

    def test_terminal_activity_target_never_carries_the_raw_command(self, monkeypatch):
        # Review finding t_6b360247 (run #142): a real worker activity wrote
        # a raw command straight into task_events.payload.target, e.g.
        # "HERMES_KANBAN_TASK=... uv run python ...". sanitize_activity_text
        # only redacts recognizable secret PATTERNS and a small banned-
        # vocabulary word list -- it never rejects a raw command line on
        # principle, so an admin env-var assignment or an arbitrary argument
        # sails straight through it. The fix must never hand the raw/full
        # command to append_activity_event's target in the first place
        # (agent-live-activity-contract.md §5: "cible structurée/fermée
        # ... pour terminal").
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
        calls = []
        monkeypatch.setattr(
            "hermes_cli.kanban_db.append_activity_event",
            lambda **kw: calls.append(kw),
        )
        raw_command = (
            "HERMES_KANBAN_TASK=t_9075 uv run python worker.py "
            "--profile claude2 --secret sk-abcdef1234567890"
        )
        get_cute_tool_message("terminal", {"command": raw_command}, 0.1)
        assert len(calls) == 1
        target = calls[0]["target"] or ""
        assert "HERMES_KANBAN_TASK" not in target
        assert "t_9075" not in target
        assert "worker.py" not in target
        assert "--secret" not in target
        assert "sk-abcdef1234567890" not in target
        assert target != raw_command

    def test_execute_code_activity_target_never_carries_the_raw_code(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
        calls = []
        monkeypatch.setattr(
            "hermes_cli.kanban_db.append_activity_event",
            lambda **kw: calls.append(kw),
        )
        raw_code = "requests.post(url, headers={'Authorization': 'Bearer sk-abcdef1234567890'})"
        get_cute_tool_message("execute_code", {"code": raw_code}, 0.1)
        assert len(calls) == 1
        target = calls[0]["target"] or ""
        assert "sk-abcdef1234567890" not in target
        assert "Authorization" not in target
        assert target != raw_code


# ---------------------------------------------------------------------------
# _activity_target_for_tool() -- closed target vocabulary (review finding
# t_6b360247 run #147, BLOCK): every tool other than terminal/execute_code
# fell through to build_tool_preview(), a free-text renderer, so a sensitive
# web query / browser text / delegated goal / generation prompt could land
# in task_events.payload.target and the Telegram board. Fix: a closed table
# — file basename, program name, controlled host, or no target at all for
# any text/prompt-carrying tool.
# ---------------------------------------------------------------------------

class TestActivityTargetClosedVocabulary:
    SENSITIVE = "MARKER-do-not-leak-sk-abcdef1234567890"

    def _target(self, tool_name, args):
        from agent.display import _activity_target_for_tool
        return _activity_target_for_tool(tool_name, args)

    # -- texte/prompt tools : never a target, only the closed action -------

    def test_web_search_query_never_becomes_the_target(self):
        target = self._target("web_search", {"query": self.SENSITIVE})
        assert self.SENSITIVE not in target

    def test_browser_type_text_never_becomes_the_target(self):
        target = self._target("browser_type", {"ref": "e3", "text": self.SENSITIVE})
        assert target == ""

    def test_browser_exec_comment_never_becomes_the_target(self):
        target = self._target(
            "browser_exec", {"code": f"# {self.SENSITIVE}\nclick('e1')"}
        )
        assert target == ""

    def test_delegate_task_goal_never_becomes_the_target(self):
        target = self._target("delegate_task", {"goal": self.SENSITIVE})
        assert target == ""

    def test_image_generate_prompt_never_becomes_the_target(self):
        target = self._target("image_generate", {"prompt": self.SENSITIVE})
        assert target == ""

    def test_text_to_speech_text_never_becomes_the_target(self):
        target = self._target("text_to_speech", {"text": self.SENSITIVE})
        assert target == ""

    def test_vision_analyze_question_never_becomes_the_target(self):
        target = self._target("vision_analyze", {"question": self.SENSITIVE})
        assert target == ""

    # -- fichier : basename only, never full free text ----------------------

    def test_read_file_target_is_basename_only(self):
        target = self._target("read_file", {"path": "/home/seb/secret-project/foo.py"})
        assert target == "foo.py"

    def test_write_file_target_is_basename_only(self):
        target = self._target("write_file", {"path": "/home/seb/secret-project/bar.py", "content": self.SENSITIVE})
        assert target == "bar.py"
        assert self.SENSITIVE not in target

    def test_patch_target_is_basename_only(self):
        target = self._target("patch", {"path": "/home/seb/secret-project/baz.py", "old_string": self.SENSITIVE})
        assert target == "baz.py"
        assert self.SENSITIVE not in target

    def test_search_files_target_never_carries_the_pattern(self):
        target = self._target("search_files", {"pattern": self.SENSITIVE, "path": "."})
        assert self.SENSITIVE not in target

    # -- web/navigation : controlled host only, never query/full URL -------

    def test_web_extract_target_is_host_only(self):
        target = self._target("web_extract", {"urls": ["https://example.com/secret/path?token=abc"]})
        assert target == "example.com"
        assert "token" not in target
        assert "secret" not in target

    def test_browser_navigate_target_strips_query_string(self):
        target = self._target(
            "browser_navigate", {"url": f"https://example.com/account?session={self.SENSITIVE}"}
        )
        assert self.SENSITIVE not in target
        assert "example.com" in target

    def test_browser_navigate_target_never_carries_the_path(self):
        # Reviewer reproduction (t_da242e47 run #152, codex-worker): a
        # navigation URL whose *path* segment carries a sensitive marker --
        # not just the query string -- must not leak either. The path is
        # unqualifiable free text (an attacker/user can put anything there),
        # so the only safe target is the bare hostname.
        target = self._target(
            "browser_navigate",
            {"url": f"https://example.com/{self.SENSITIVE}?q=ignored"},
        )
        assert self.SENSITIVE not in target
        assert target == "example.com"

    def test_browser_navigate_falls_back_when_url_unparseable(self):
        target = self._target("browser_navigate", {"url": "not a url"})
        assert target == "navigation web"

    # -- unknown tool : no target -------------------------------------------

    def test_unknown_tool_never_gets_a_target(self):
        target = self._target("some_future_tool_xyz", {"prompt": self.SENSITIVE})
        assert target == ""
