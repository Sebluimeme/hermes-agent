"""Lightweight systemd ``ExecStop`` helper for the gateway.

systemd runs ``ExecStop`` before it sends ``KillSignal`` to the remaining
service processes.  Recording the gateway PID here lets the normal shutdown
handler distinguish ``systemctl stop/restart`` from an unrelated external
SIGTERM without weakening the unexpected-signal recovery path.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        return 2
    try:
        target_pid = int(args[0])
    except (TypeError, ValueError):
        return 2
    if target_pid <= 0:
        return 2

    try:
        from gateway.status import write_planned_stop_marker

        return 0 if write_planned_stop_marker(target_pid) else 1
    except Exception:
        # The systemd unit prefixes this helper with ``-`` so shutdown still
        # proceeds if the filesystem is unavailable.  A failure deliberately
        # leaves SIGTERM unmarked and therefore on the conservative non-zero
        # gateway exit path.
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
