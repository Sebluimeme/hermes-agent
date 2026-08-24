"""Regression tests for the narrow non-interactive Python diagnostic allowlist."""

from tools.approval import detect_dangerous_command


def test_allows_read_only_file_diagnostic():
    command = "python3 -c \"from pathlib import Path; print(Path('/tmp/report.txt').read_text(encoding='utf-8'))\""
    assert detect_dangerous_command(command) == (False, None, None)


def test_allows_read_only_sqlite_uri_diagnostic():
    command = "python3 -c \"import sqlite3; c=sqlite3.connect('file:/tmp/a.db?mode=ro', uri=True); print(c.execute('SELECT 1').fetchall())\""
    assert detect_dangerous_command(command) == (False, None, None)


def test_allows_read_only_sqlite_pragma_diagnostic():
    command = "python3 -c \"import sqlite3; c=sqlite3.connect('file:/tmp/a.db?mode=ro', uri=True); print(c.execute('PRAGMA table_info(users)').fetchall())\""
    assert detect_dangerous_command(command) == (False, None, None)


def test_rejects_sqlite_uri_with_mode_ro_only_in_an_unrelated_parameter():
    command = "python3 -c \"import sqlite3; c=sqlite3.connect('file:/tmp/a.db?x=mode=ro', uri=True); print(c.execute('SELECT 1').fetchall())\""
    assert detect_dangerous_command(command)[0] is True


def test_rejects_sqlite_uri_with_conflicting_mode_parameters():
    command = "python3 -c \"import sqlite3; c=sqlite3.connect('file:/tmp/a.db?mode=rw&mode=ro', uri=True); print(c.execute('SELECT 1').fetchall())\""
    assert detect_dangerous_command(command)[0] is True


def test_rejects_mutating_sqlite_pragma():
    command = "python3 -c \"import sqlite3; c=sqlite3.connect('file:/tmp/a.db?mode=ro', uri=True); print(c.execute('PRAGMA journal_mode=WAL').fetchall())\""
    assert detect_dangerous_command(command)[0] is True


def test_rejects_file_write_even_when_using_pathlib():
    command = "python3 -c \"from pathlib import Path; Path('/tmp/report.txt').write_text('changed')\""
    assert detect_dangerous_command(command)[0] is True


def test_rejects_sqlite_write_even_with_read_only_uri():
    command = "python3 -c \"import sqlite3; c=sqlite3.connect('file:/tmp/a.db?mode=ro', uri=True); c.execute('DELETE FROM users')\""
    assert detect_dangerous_command(command)[0] is True


def test_rejects_process_network_and_dynamic_execution():
    for program in (
        "import subprocess; subprocess.run(['id'])",
        "import urllib.request; urllib.request.urlopen('https://example.test')",
        "eval('1 + 1')",
    ):
        assert detect_dangerous_command(f"python3 -c {program!r}")[0] is True
