"""Reap worker scopes whose card already left ``running``.

A kanban worker runs inside a transient ``hermes-worker-kanban-<task>-run-<n>.scope``
cgroup.  When it calls ``kanban_complete`` / ``kanban_request_review`` /
``kanban_block`` the run is closed in the DB, but the OS process keeps going —
it can post more comments and hold an API session indefinitely.  The dispatcher
stops tracking a run the moment the card leaves ``running``, so nothing reaped
those scopes: one orphan per finished card, quota burnt after the deliverable,
and a false positive for the fleet's orphan-scope signal.

The tick therefore asks the user manager which worker scopes exist, maps each
back to its ``(task_id, run_id)``, and stops the ones whose run is closed.
Deliberate properties:

* **SIGTERM only** (``systemctl --user stop``), never SIGKILL — the worker's own
  handler flushes and exits, and the scope reaps its whole cgroup including
  double-forked descendants.
* **A frozen scope is thawed first.** A frozen cgroup cannot process SIGTERM, so
  a stop against it would hang until ``TimeoutStopSec``.
* **Unknown task ids are left alone.** Scope names carry no board, so a unit
  this board cannot resolve belongs to another board's dispatcher.

Systemd access goes through an injectable runner: a unit test must never reach
the real user manager.
"""

from __future__ import annotations

import logging
import re
import shutil
import sqlite3
import subprocess
from typing import Optional, Protocol

logger = logging.getLogger(__name__)

# hermes-worker-kanban-<task_id>-run-<run_id>.scope — minted by
# ``_restart_safe_worker_argv`` via ``restart_safe_gateway_child_argv``.
_WORKER_SCOPE_RE = re.compile(
    r"^hermes-worker-kanban-(?P<task>.+)-run-(?P<run>\d+)\.scope$"
)

_SYSTEMCTL_TIMEOUT_SECONDS = 15


def parse_worker_scope_unit(unit: str) -> Optional[tuple[str, int]]:
    """``(task_id, run_id)`` for a kanban worker scope, else ``None``."""
    match = _WORKER_SCOPE_RE.match((unit or "").strip())
    if match is None:
        return None
    return match.group("task"), int(match.group("run"))


class WorkerUnits(Protocol):
    """The systemd surface this module needs (injected so tests never touch it)."""

    def list_worker_scopes(self) -> list[str]: ...
    def freezer_state(self, unit: str) -> str: ...
    def thaw(self, unit: str) -> bool: ...
    def stop(self, unit: str) -> bool: ...


class SystemdUserUnits:
    """Real ``systemctl --user`` runner. Every method is a no-op without systemctl."""

    def _run(self, *args: str) -> Optional[subprocess.CompletedProcess]:
        binary = shutil.which("systemctl")
        if binary is None:
            return None
        from tools.process_registry import systemd_user_bus_env

        try:
            return subprocess.run(
                [binary, "--user", *args],
                capture_output=True,
                check=False,
                timeout=_SYSTEMCTL_TIMEOUT_SECONDS,
                stdin=subprocess.DEVNULL,
                env=systemd_user_bus_env(),
            )
        except Exception as exc:
            logger.debug("systemctl --user %s failed: %s", " ".join(args), exc)
            return None

    def list_worker_scopes(self) -> list[str]:
        result = self._run(
            "list-units", "--type=scope", "--all", "--plain",
            "--no-legend", "--no-pager", "hermes-worker-kanban-*.scope",
        )
        if result is None or result.returncode != 0:
            return []
        units = []
        for line in (result.stdout or b"").decode("utf-8", "replace").splitlines():
            unit = line.split(None, 1)[0] if line.split() else ""
            if parse_worker_scope_unit(unit):
                units.append(unit)
        return units

    def freezer_state(self, unit: str) -> str:
        result = self._run("show", "-p", "FreezerState", "--value", unit)
        if result is None or result.returncode != 0:
            return ""
        return (result.stdout or b"").decode("utf-8", "replace").strip()

    def thaw(self, unit: str) -> bool:
        result = self._run("thaw", unit)
        return result is not None and result.returncode == 0

    def stop(self, unit: str) -> bool:
        """SIGTERM the whole cgroup. A unit that is already gone counts as stopped."""
        result = self._run("stop", unit)
        if result is None:
            return False
        if result.returncode == 0:
            return True
        stderr = (result.stderr or b"").decode("utf-8", "replace").lower()
        if any(m in stderr for m in ("not loaded", "not found", "does not exist")):
            return True
        logger.debug("systemctl --user stop %s exited %d: %s", unit, result.returncode, stderr.strip())
        return False


def reap_enabled() -> bool:
    """``kanban.reap_finished_worker_scopes`` (default on)."""
    try:
        from hermes_cli.config import load_config

        return bool(
            (load_config() or {}).get("kanban", {}).get("reap_finished_worker_scopes", True)
        )
    except Exception:
        return True


def _run_is_closed(conn: sqlite3.Connection, task_id: str, run_id: int) -> Optional[bool]:
    """True when this run has ended, False while it is live, ``None`` if unknown here.

    ``None`` means the scope names a task or run this board has never seen —
    another board's worker — and must not be touched.
    """
    row = conn.execute(
        "SELECT ended_at FROM task_runs WHERE id = ? AND task_id = ?",
        (int(run_id), task_id),
    ).fetchone()
    if row is None:
        return None
    return row["ended_at"] is not None


def reap_finished_worker_scopes(
    conn: sqlite3.Connection, *, systemd: Optional[WorkerUnits] = None,
) -> list[str]:
    """Stop worker scopes whose run is closed; returns the units actually stopped."""
    if not reap_enabled():
        return []
    runner: WorkerUnits = systemd if systemd is not None else SystemdUserUnits()

    reaped: list[str] = []
    for unit in runner.list_worker_scopes():
        parsed = parse_worker_scope_unit(unit)
        if parsed is None:
            continue
        task_id, run_id = parsed
        closed = _run_is_closed(conn, task_id, run_id)
        if closed is not True:
            # Live run, or a task this board does not own.
            continue

        # A frozen cgroup cannot process SIGTERM: thaw first or the stop hangs
        # until TimeoutStopSec.
        try:
            if runner.freezer_state(unit) == "frozen":
                runner.thaw(unit)
        except Exception as exc:
            logger.debug("freezer probe failed for %s: %s", unit, exc)

        try:
            stopped = runner.stop(unit)
        except Exception as exc:
            logger.warning("kanban reap: stopping %s failed: %s", unit, exc)
            continue
        if not stopped:
            continue

        reaped.append(unit)
        logger.info(
            "kanban reap: stopped leftover worker scope %s (task %s run %s "
            "already left running)", unit, task_id, run_id,
        )
        _record_reap_event(conn, task_id, run_id, unit)
    return reaped


def _record_reap_event(conn: sqlite3.Connection, task_id: str, run_id: int, unit: str) -> None:
    """Audit the reap on the card; never let bookkeeping break the tick."""
    from hermes_cli import kanban_db as kb

    try:
        with kb.write_txn(conn):
            kb._append_event(
                conn, task_id, "worker_scope_reaped",
                {"unit": unit, "run_id": int(run_id)}, run_id=int(run_id),
            )
    except Exception as exc:
        logger.debug("kanban reap: could not record event for %s: %s", task_id, exc)


__all__ = [
    "SystemdUserUnits",
    "WorkerUnits",
    "parse_worker_scope_unit",
    "reap_enabled",
    "reap_finished_worker_scopes",
]
