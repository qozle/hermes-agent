"""Cooperative pause for dispatcher-spawned kanban workers (LOCAL COMMIT, frankentrait).

The quota guard (``~/.hermes/scripts/quota_guard.py``) used to freeze a worker's cgroup and
SIGSTOP its agent loop at 95 % of the Anthropic usage window. The process woke at the window
reset to a dead model stream, an expired session turn lease and a liveness watchdog that had
seen no progress for the whole freeze: the turn died (exit 130), the board called it a crash
and the card restarted from whatever was on disk (Max, 2026-11-16: "it should finish its
current turn and then just wait until the reset time").

Now the guard writes the ``WORKER_PAUSE`` sentinel next to the board (``kanban_home()``) and a
worker reads it before every model call (``begin_iteration``). While the sentinel holds the
worker stays awake and idle: it touches the activity clock so the liveness watchdog stays
quiet, its lease refresher keeps running on the usual timer, and it heartbeats the card with a
``paused:`` note that the guard reads as the acknowledgement. When the guard removes the
sentinel (or ``RESUME_GRACE_S`` after the recorded ``until``, should the guard be late or
dead) the same turn resumes with its context intact.

Only kanban workers pause (``HERMES_KANBAN_TASK`` set); the gateway's own chats never do.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

PAUSE_SENTINEL = "WORKER_PAUSE"
# A heartbeat note starting with this is the guard's acknowledgement match (quota_guard.acked_tasks).
NOTE_PREFIX = "paused:"
# Sentinel re-read + activity touch cadence; the liveness watchdog defaults to 600 s idle.
POLL_S = 20.0
# Noted board heartbeat cadence (the auto heartbeat from ``_touch_activity`` covers liveness).
NOTE_INTERVAL_S = 180.0
# Sentinel still present this long past its ``until``: the guard is late or dead, resume anyway.
RESUME_GRACE_S = 300.0


def pause_path() -> Path:
    from hermes_cli.kanban_db import kanban_home

    return kanban_home() / PAUSE_SENTINEL


def read_pause(path: Optional[Path] = None) -> Optional[dict]:
    """The sentinel's JSON (``reason``, ``until`` ISO-8601 or null); a reason-only dict when
    the file is not JSON or unreadable (a bare touch holds too); None when absent."""
    target = path or pause_path()
    try:
        raw = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return {"reason": "worker pause sentinel (unreadable)"}
    try:
        data = json.loads(raw)
    except ValueError:
        return {"reason": raw.strip()[:120] or "worker pause sentinel"}
    return data if isinstance(data, dict) else {"reason": str(data)[:120]}


def until_timestamp(info: dict) -> Optional[float]:
    raw = info.get("until")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _local_clock(ts: float) -> str:
    local = datetime.fromtimestamp(ts).astimezone()
    return local.strftime("%I:%M %p %Z").lstrip("0")


def pause_note(info: dict) -> str:
    reason = str(info.get("reason") or "worker pause")
    until = until_timestamp(info)
    when = f"until {_local_clock(until)}" if until is not None else "until the guard releases it"
    return f"{NOTE_PREFIX} {reason} — waiting {when} between steps"


def _heartbeat_note(note: str) -> bool:
    """Board heartbeat carrying ``note`` for this worker's own run (the auto bridge in
    ``_touch_activity`` never carries one). Best-effort; never raises."""
    tid = os.environ.get("HERMES_KANBAN_TASK")
    if not tid:
        return False
    try:
        from hermes_cli import kanban_db_dispatch as kbd
        from tools.kanban_tools import _board, _worker_run_id

        with _board(None, quiet_close=True) as (kb, conn):
            kb.heartbeat_claim(conn, tid, claimer=os.environ.get("HERMES_KANBAN_CLAIM_LOCK"))
            return bool(kbd.heartbeat_worker(conn, tid, note=note, expected_run_id=_worker_run_id(tid)))
    except Exception:
        logger.debug("worker pause: noted heartbeat failed", exc_info=True)
        return False


def wait_out_worker_pause(
    agent: Any,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], float] = time.time,
    poll_s: float = POLL_S,
    heartbeat_fn: Callable[[str], bool] = _heartbeat_note,
) -> bool:
    """Block while the pause sentinel holds; True iff the turn was interrupted meanwhile.

    No-op (False) outside a dispatcher-spawned worker or when no sentinel is present, so the
    cost on every other model call is one ``stat``."""
    if not os.environ.get("HERMES_KANBAN_TASK"):
        return False
    info = read_pause()
    if info is None:
        return False
    started = now_fn()
    last_note: Optional[float] = None
    announced: Optional[str] = None
    while True:
        if getattr(agent, "_interrupt_requested", False):
            logger.info("worker pause: interrupted after %d s", now_fn() - started)
            return True
        info = read_pause()
        if info is None:
            break
        now = now_fn()
        until = until_timestamp(info)
        if until is not None and now >= until + RESUME_GRACE_S:
            logger.warning(
                "worker pause: sentinel still present %d s past its until; resuming on the recorded time",
                now - until,
            )
            break
        note = pause_note(info)
        if note != announced:
            logger.info("worker pause: %s", note)
            announced = note
            last_note = None
        touch = getattr(agent, "_touch_activity", None)
        if callable(touch):
            touch(note)
        if last_note is None or now - last_note >= NOTE_INTERVAL_S:
            heartbeat_fn(note)
            last_note = now
        sleep_fn(poll_s)
    logger.info("worker pause: over after %d s; resuming the turn", now_fn() - started)
    return False
