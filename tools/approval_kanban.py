"""LOCAL COMMIT (frankentrait): kanban worker approvals routed to the coordinator.

A dispatched kanban worker is a single-query context: nobody sits at its prompt,
so upstream resolves every flagged command from ``approvals.single_query_mode``
(deny here) and the card parks on its last step. With
``approvals.kanban_mode: coordinator`` the worker instead posts an
``approval_requested`` event on its own card (the gateway's kanban notifier
wakes the coordinator session with it), heartbeats its claim, and waits up to
``approvals.timeout`` for an ``approval_decided`` event carrying the same token
(``hermes kanban approve|deny <task> <token> [note]``). No answer = denied,
exactly as before.
"""
from __future__ import annotations

import logging
import os
import secrets
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

APPROVAL_REQUESTED = "approval_requested"
APPROVAL_DECIDED = "approval_decided"
_POLL_SECONDS = 5.0


def worker_task_id() -> Optional[str]:
    return (os.environ.get("HERMES_KANBAN_TASK") or "").strip() or None


def kanban_mode() -> str:
    """``approvals.kanban_mode``: ``coordinator`` | ``deny`` (default: deny = upstream)."""
    try:
        from hermes_cli.config import load_config_readonly
        value = ((load_config_readonly() or {}).get("approvals", {}) or {}).get("kanban_mode")
    except Exception:
        return "deny"
    return str(value or "deny").strip().lower()


def approval_timeout_seconds() -> int:
    try:
        from hermes_cli.config import load_config_readonly
        value = ((load_config_readonly() or {}).get("approvals", {}) or {}).get("timeout")
        return max(30, int(value or 1200))
    except Exception:
        return 1200


def coordinator_relay_active() -> bool:
    return worker_task_id() is not None and kanban_mode() == "coordinator"


def post_request(conn: Any, task_id: str, *, command: str, description: str,
                 pattern_key: str, profile: Optional[str]) -> str:
    """Append the ``approval_requested`` event; returns its token."""
    from hermes_cli import kanban_db as kb
    token = secrets.token_hex(4)
    with kb.write_txn(conn):
        kb._append_event(conn, task_id, APPROVAL_REQUESTED, {
            "token": token,
            "command": command[:2000],
            "aim": description[:600],
            "pattern_key": pattern_key,
            "profile": profile,
            "run_id": _current_run_id(conn, task_id),
        })
    return token


def _current_run_id(conn: Any, task_id: str) -> Optional[int]:
    row = conn.execute("SELECT current_run_id FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def read_decision(conn: Any, task_id: str, token: str) -> Optional[dict]:
    """The ``approval_decided`` payload for ``token``, or None while unanswered."""
    from hermes_cli import kanban_db as kb
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id DESC LIMIT 20",
        (task_id, APPROVAL_DECIDED),
    ).fetchall()
    for row in rows:
        payload = kb._json_dict(row[0])
        if payload.get("token") == token:
            return payload
    return None


def post_decision(conn: Any, task_id: str, token: str, *, decision: str,
                  note: Optional[str], author: str) -> bool:
    """Coordinator side: record approve/deny for ``token``. False when the token
    is unknown on this card or already answered."""
    from hermes_cli import kanban_db as kb
    if decision not in ("approve", "deny"):
        raise ValueError("decision must be approve or deny")
    asked = any(
        kb._json_dict(r[0]).get("token") == token
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id DESC LIMIT 50",
            (task_id, APPROVAL_REQUESTED),
        ).fetchall()
    )
    if not asked:
        return False
    if read_decision(conn, task_id, token) is not None:
        return False
    with kb.write_txn(conn):
        kb._append_event(conn, task_id, APPROVAL_DECIDED, {
            "token": token, "decision": decision, "note": (note or "")[:600], "by": author,
        })
    return True


def wait_for_decision(conn: Any, task_id: str, token: str, *, timeout_s: int,
                      heartbeat: Optional[Any] = None, sleep=time.sleep,
                      clock=time.monotonic) -> tuple[str, str]:
    """Poll until answered or ``timeout_s``; returns ``(decision, note)`` with
    decision ``approve`` / ``deny`` / ``timeout``. ``heartbeat`` runs each poll so
    the card's claim stays alive while the worker waits."""
    deadline = clock() + timeout_s
    while True:
        payload = read_decision(conn, task_id, token)
        if payload is not None:
            return str(payload.get("decision") or "deny"), str(payload.get("note") or "")
        if clock() >= deadline:
            return "timeout", ""
        if heartbeat is not None:
            try:
                heartbeat()
            except Exception as exc:  # pragma: no cover
                logger.debug("kanban approval relay: heartbeat failed: %s", exc)
        sleep(_POLL_SECONDS)


def coordinator_decision(*, command: str, description: str, pattern_key: str) -> tuple[str, str]:
    """Worker side, end to end: post the request on the card, wait, return
    ``(decision, note)``. Any board failure reads as ``deny`` with the error as note."""
    task_id = worker_task_id()
    assert task_id is not None
    try:
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from agent.redact import redact_sensitive_text
        conn = kbc.connect()
        token = post_request(
            conn, task_id, command=redact_sensitive_text(command),
            description=redact_sensitive_text(description), pattern_key=pattern_key,
            profile=os.environ.get("HERMES_PROFILE") or None,
        )
        claimer = os.environ.get("HERMES_KANBAN_CLAIM_LOCK")

        def _beat():
            kb.heartbeat_claim(conn, task_id, claimer=claimer)

        logger.info("kanban approval relay: %s asked the coordinator (token %s): %s", task_id, token, description)
        return wait_for_decision(conn, task_id, token, timeout_s=approval_timeout_seconds(), heartbeat=_beat)
    except Exception as exc:
        logger.warning("kanban approval relay failed for %s: %s", task_id, exc)
        return "deny", f"approval relay failed: {exc}"
