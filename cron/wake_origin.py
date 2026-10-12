"""``wake_origin``: hand a cron job's output to the agent in the job's ORIGIN conversation as a
turn, instead of (or beside) posting it as a chat message a human reads.

``deliver`` and ``wake_origin`` are orthogonal. ``deliver`` sends a MESSAGE — a human reads it and
decides. ``wake_origin`` injects the same text into the origin session as a turn, exactly the way
a kanban watcher notice does (``gateway/kanban_watchers_notifier.py::wake``), so the agent living
in that conversation sees it and decides. ``deliver: local`` + ``wake_origin: true`` is the "tell
the agent, not me" combination: a watchdog whose raw output is noise to a human but is precisely
what the coordinating agent needs in order to act.

The wake is driven from ``cron.scheduler_delivery._deliver_result``, the single seam BOTH the
in-process ticker and the external restart-safe worker's ``delivery_queue`` → gateway drain pass
through. Silence is inherited for free: ``_save_compose_deliver`` already returns before
``_deliver_result`` when a run had nothing to say (empty stdout, ``wakeAgent:false``, an unchanged
monitor, ``[SILENT]``), so a silent run never wakes anybody.
"""

from __future__ import annotations

import logging
import weakref
from typing import Optional

# Log-record parity with the rest of cron delivery.
logger = logging.getLogger("cron.scheduler")

# Same ceiling as the live text-send lane (``_live_send_text``): admission is cheap, but the
# gateway loop may be mid-turn, and a wedged loop must not pin the cron worker forever.
_WAKE_SCHEDULE_TIMEOUT_SECONDS = 60

_WAKE_FOOTER = (
    "This is an automatic scheduled-job notification; decide whether anything needs doing.")


def _wake_text(job: dict, content: str) -> str:
    """Outward-facing wake payload: job name + the run's output, both redacted.

    ``content`` is the RAW job output — ``_deliver_result`` redacts a *copy* for the chat lanes, so
    this lane scrubs it itself (the same reason ``mirror_text`` is re-redacted there). A transcript
    outlives the message it came from.
    """
    from cron.scheduler_delivery import _cron_display_name, _redact_cron_payload
    from gateway.platforms.base import BasePlatformAdapter

    _, body = BasePlatformAdapter.extract_media(content or "")
    body = _redact_cron_payload((body or "").strip(), "wake payload")
    return f"[cron] {_cron_display_name(job)}\n\n{body}\n\n{_WAKE_FOOTER}"


def _resolve_wake_chat_type(origin: dict) -> Optional[str]:
    """``chat_type`` for the origin session, or None when it cannot be established.

    This is the one correctness-critical input: ``build_session_key()`` keys a DM differently from
    a group/thread, so a wrong value does not degrade gracefully — it lands the wake in a
    BRAND-NEW session nobody is reading and burns a full agent turn every time the job fires. Jobs
    created after this feature capture it on the origin; older jobs resolve against the live
    session row (precedent: ``gateway/mirror.py::_find_session_id``). Neither → the caller skips.
    """
    declared = str(origin.get("chat_type") or "").strip()
    if declared:
        return declared
    try:
        from hermes_state_registry import acquire, release_or_close
        db = acquire()
        try:
            session_id = db.find_session_by_origin(
                platform=str(origin["platform"]), chat_id=str(origin["chat_id"]),
                thread_id=origin.get("thread_id") or None, user_id=origin.get("user_id") or None)
            row = db.get_session(str(session_id)) if session_id else None
        finally:
            release_or_close(db)
    except Exception as exc:
        logger.debug("wake_origin: session lookup for chat_type failed: %s", exc)
        return None
    return str((row or {}).get("chat_type") or "").strip() or None


def wake_origin_session(job: dict, content: str, *, adapters=None, loop=None) -> Optional[str]:
    """Inject *content* into the job's origin session as an agent turn.

    Returns None on success OR on a deliberate skip (no origin, no live gateway, unresolvable
    chat_type), and an error string when a wake that should have happened failed. The caller runs
    this AFTER the chat sends, so a failure here is reported but never blocks a delivery.
    """
    from cron.scheduler_delivery import _resolve_origin

    job_id = job.get("id", "?")
    origin = _resolve_origin(job)
    if not origin:
        # Same spirit as ``_unresolved_delivery_outcome``'s origin case: a CLI-created job never
        # captured an origin, and making that a per-run error would cry wolf on every fire.
        logger.info(
            "Job '%s': wake_origin is set but the job captured no origin — nothing to wake", job_id)
        return None
    if adapters is None or loop is None or not loop.is_running():
        # Standalone ``hermes cron run`` has no gateway loop to inject a turn on, and an external
        # restart-safe worker reaches the gateway through the delivery queue, which re-enters this
        # function with both live.
        logger.info(
            "Job '%s': wake_origin requested but no live gateway adapters — skipping the wake", job_id)
        return None

    chat_type = _resolve_wake_chat_type(origin)
    if not chat_type:
        logger.warning(
            "Job '%s': wake_origin cannot resolve the origin session's chat_type for %s:%s "
            "(no live session row); skipping rather than waking a fresh session nobody reads. "
            "Set chat_type on the job's origin to fix this.",
            job_id, origin.get("platform"), origin.get("chat_id"))
        return None

    from agent.async_utils import safe_schedule_threadsafe
    from gateway.config import Platform, load_gateway_config
    from gateway.delivery import resolve_delivery_transport
    from gateway.session import SessionSource
    from gateway.wake import deliver_wake
    from hermes_constants import get_hermes_home, profile_name_for_home

    platform_name = str(origin["platform"]).lower()
    try:
        platform = Platform(platform_name)
        transport = resolve_delivery_transport(platform, load_gateway_config(), adapters)
    except Exception as exc:
        return f"origin session wake could not resolve a transport: {exc}"
    adapter = getattr(transport, "adapter", None)
    if adapter is None:
        return f"origin session wake has no live {platform_name} adapter"

    source = SessionSource(
        platform=platform, chat_id=str(origin["chat_id"]), chat_type=chat_type,
        thread_id=origin.get("thread_id") or None, user_id=origin.get("user_id"),
        # The job's store is already bound to its owning profile (the ticker's cron scope, or the
        # queued drain's ``_profile_runtime_scope``), so the home IS the job's profile. The default
        # home resolves to "default", which keys as ``agent:main`` — byte-identical to a live
        # default-profile session key.
        profile=profile_name_for_home(get_hermes_home()),
        scope_id=origin.get("scope_id"))
    # Relay-fronted platforms answer through the relay adapter: pin the transport we resolved
    # rather than letting key derivation re-guess the owning bot (same as the kanban notifier).
    source._transport_adapter_ref = weakref.ref(adapter)

    future = safe_schedule_threadsafe(
        deliver_wake(adapter, text=_wake_text(job, content), source=source,
                     notification_category="result"),
        loop)
    if future is None:
        return "origin session wake could not be scheduled on the gateway loop"
    try:
        future.result(timeout=_WAKE_SCHEDULE_TIMEOUT_SECONDS)
    except Exception as exc:
        logger.warning("Job '%s': origin session wake failed: %s", job_id, exc)
        return f"origin session wake failed: {exc}"
    job["_wake_origin_delivered"] = True
    logger.info("Job '%s': woke origin session %s:%s (profile %s)",
                job_id, platform_name, origin.get("chat_id"), source.profile)
    return None
