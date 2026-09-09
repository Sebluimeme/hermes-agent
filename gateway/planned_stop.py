"""Lightweight systemd ``ExecStop`` helper for the gateway.

For ``systemctl stop/restart``, systemd runs ``ExecStop`` before it sends
``KillSignal`` to the remaining service process and exposes that exact PID to
the control process through the ``MAINPID`` environment variable.  For a
self-requested exit 75 the main process is already gone and ``MAINPID`` is
absent, so the later ``ExecStop`` becomes a clean no-op.  This lets the
shutdown handler distinguish a managed stop from an unrelated external
SIGTERM without ever discovering or marking a replacement gateway.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) > 1:
        return 2

    try:
        from gateway.status import write_planned_stop_marker

        if args:
            raw_pid = args[0]
        else:
            # systemd supplies MAINPID only while this unit still owns a main
            # process. Never fall back to the shared pidfile here: a successor
            # may already have claimed it after a self-requested exit 75.
            raw_pid = os.environ.get("MAINPID", "").strip()
            if not raw_pid:
                return 0

        try:
            target_pid = int(raw_pid)
        except (TypeError, ValueError):
            return 2
        if target_pid <= 0:
            return 2

        return 0 if write_planned_stop_marker(target_pid) else 1
    except Exception:
        # The systemd unit prefixes this helper with ``-`` so shutdown still
        # proceeds if the filesystem is unavailable.  A failure deliberately
        # leaves SIGTERM unmarked and therefore on the conservative non-zero
        # gateway exit path.
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
