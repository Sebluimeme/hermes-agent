"""Durable ownership contracts for dispatcher-spawned Kanban workers.

Only PIDs recorded here by the dispatcher are candidates for termination.
This deliberately excludes interactive Claude/Desktop/manual processes.
"""
from __future__ import annotations

import json
import os
import signal
import time
from typing import Any, Callable

# A Kanban worker may legitimately spend several minutes in one bounded tool
# operation before it can publish its next descriptive checkpoint.  The
# dispatcher already owns explicit runtime limits via ``max_runtime_seconds``;
# this contract guard is only an orphan detector.  Keep its grace aligned with
# the documented hourly worker heartbeat contract so it cannot pre-empt a
# healthy uncapped worker at the former ten-minute threshold.
CHECKPOINT_STALE_SECONDS = 3600
ACTIVE_TASK_STATUSES = {"running", "review", "changes_requested"}
TERMINAL_TASK_STATUSES = {"done", "blocked", "archived", "cancelled", "triage", "todo", "ready"}


def ensure_schema(conn: Any) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS worker_contracts (
            task_id TEXT PRIMARY KEY,
            run_id INTEGER NOT NULL,
            profile TEXT NOT NULL,
            model TEXT,
            pid INTEGER NOT NULL,
            start_identity TEXT NOT NULL,
            process_group INTEGER,
            workspace_path TEXT NOT NULL,
            max_runtime_seconds INTEGER,
            max_retries INTEGER,
            created_at INTEGER NOT NULL,
            state TEXT NOT NULL DEFAULT 'active',
            stopped_at INTEGER,
            anomaly TEXT
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_worker_contracts_state ON worker_contracts(state)"
    )


def proc_start_identity(pid: int) -> str | None:
    """Linux /proc starttime ticks; PID alone is never a kill authority."""
    if pid <= 0 or not sys_platform_linux():
        return None
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            fields = handle.read().split()
        # /proc/<pid>/stat field 22 (starttime), zero-indexed after splitting.
        return fields[21] if len(fields) > 21 and fields[21].isdigit() else None
    except OSError:
        return None


def sys_platform_linux() -> bool:
    return os.path.isdir("/proc")


def process_group(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def register(
    conn: Any,
    *,
    task_id: str,
    run_id: int,
    profile: str,
    model: str | None,
    pid: int,
    workspace_path: str,
    max_runtime_seconds: int | None,
    max_retries: int | None,
    now: int | None = None,
) -> bool:
    """Persist ownership before any watchdog can manage the PID.

    A missing start identity means the PID is deliberately unmanaged: killing a
    recycled PID is worse than leaving an anomaly for an operator to inspect.
    """
    identity = proc_start_identity(pid)
    if identity is None:
        return False
    conn.execute(
        """INSERT INTO worker_contracts(
            task_id,run_id,profile,model,pid,start_identity,process_group,
            workspace_path,max_runtime_seconds,max_retries,created_at,state)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,'active')
           ON CONFLICT(task_id) DO UPDATE SET
             run_id=excluded.run_id, profile=excluded.profile, model=excluded.model,
             pid=excluded.pid, start_identity=excluded.start_identity,
             process_group=excluded.process_group, workspace_path=excluded.workspace_path,
             max_runtime_seconds=excluded.max_runtime_seconds,
             max_retries=excluded.max_retries, created_at=excluded.created_at,
             state='active', stopped_at=NULL, anomaly=NULL""",
        (task_id, run_id, profile, model, pid, identity, process_group(pid),
         workspace_path, max_runtime_seconds, max_retries, int(now or time.time())),
    )
    return True


def latest_descriptive_checkpoint(conn: Any, task_id: str) -> int | None:
    row = conn.execute(
        """SELECT created_at, payload FROM task_events
           WHERE task_id=? AND kind='heartbeat' ORDER BY id DESC LIMIT 1""",
        (task_id,),
    ).fetchone()
    if row is None:
        return None
    try:
        payload = json.loads(row["payload"] or "{}")
    except (TypeError, ValueError):
        return None
    note = payload.get("note") if isinstance(payload, dict) else None
    return int(row["created_at"]) if isinstance(note, str) and note.strip() else None


def safe_stop(contract: Any, *, kill: Callable[[int, int], None] | None = None) -> bool:
    """Stop only the exact recorded process group after identity re-validation."""
    pid = int(contract["pid"])
    if proc_start_identity(pid) != contract["start_identity"]:
        return False
    if kill is None:
        kill = os.kill
    pgid = contract["process_group"]
    try:
        if pgid and process_group(pid) == int(pgid):
            kill(-int(pgid), signal.SIGTERM)
        else:
            kill(pid, signal.SIGTERM)
    except OSError:
        return not process_alive(pid)
    return True


def reconcile(conn: Any, *, now: int | None = None, stale_seconds: int = CHECKPOINT_STALE_SECONDS) -> list[dict[str, Any]]:
    """Return actionable contract anomalies and stop only proven owned workers.

    Caller owns task-state transitions. This function durably marks every
    observed action, making each tick idempotent and audit-friendly.
    """
    current = int(now or time.time())
    actions: list[dict[str, Any]] = []
    rows = conn.execute("SELECT * FROM worker_contracts WHERE state='active'").fetchall()
    for c in rows:
        task = conn.execute("SELECT status,current_run_id FROM tasks WHERE id=?", (c["task_id"],)).fetchone()
        reason: str | None = None
        should_stop = False
        if task is None:
            reason, should_stop = "task_missing", True
        elif task["status"] not in ACTIVE_TASK_STATUSES:
            reason, should_stop = f"task_{task['status']}", True
        elif task["current_run_id"] != c["run_id"]:
            reason = "run_mismatch"
        elif proc_start_identity(int(c["pid"])) != c["start_identity"]:
            reason = "pid_identity_mismatch"
        else:
            checkpoint = latest_descriptive_checkpoint(conn, c["task_id"])
            # A new worker needs time to produce its first descriptive
            # checkpoint. Its durable contract creation time is a bounded
            # grace baseline, never a substitute for progress afterwards.
            checkpoint_or_start = checkpoint if checkpoint is not None else int(c["created_at"])
            if current - checkpoint_or_start > stale_seconds:
                reason, should_stop = "checkpoint_stale", True
        if reason is None:
            continue
        stopped = safe_stop(c) if should_stop else False
        state = "stopped" if stopped else "anomaly"
        conn.execute(
            "UPDATE worker_contracts SET state=?, stopped_at=?, anomaly=? WHERE task_id=? AND state='active'",
            (state, current, reason, c["task_id"]),
        )
        actions.append({"task_id": c["task_id"], "pid": int(c["pid"]), "reason": reason, "stopped": stopped})
    return actions
