"""``wake_origin``: a job's output is injected into its ORIGIN session as an agent turn.

``deliver`` and ``wake_origin`` are orthogonal lanes. ``deliver`` sends a MESSAGE a human reads;
``wake_origin`` hands the same text to the agent living in the origin conversation as a turn, the
way a kanban watcher notice does. ``deliver: local`` + ``wake_origin: true`` is the "tell the
agent, not me" combination behind the live sys-health watchdog.

The contracts pinned here:

* a requested wake fires exactly ONCE per run (not once per delivery target) and carries the
  origin's own ``chat_type``/``thread_id``/``user_id`` into the synthetic event's ``SessionSource``
  — a wrong key lands the turn in a fresh session nobody reads;
* silence is inherited from ``_save_compose_deliver``'s early return, with no second gate: a run
  with nothing to say never reaches ``_deliver_result`` and so never wakes anybody;
* a job WITHOUT the field behaves byte-identically, including the external-worker enqueue gate;
* a failed wake is REPORTED but never blocks a chat send that already left the process;
* ``deliver: local`` + ``wake_origin`` under a restart-safe worker still reaches the durable
  delivery queue — the worker holds no live adapters, so the gateway's drain is the only process
  that can perform the wake. This is the re-order the feature depends on: before it, the
  ``if not targets`` early return swallowed exactly the sys-health shape.

Production is patched where it READS (``delivery._wake_origin.wake_origin_session``,
``gateway.wake.deliver_wake``), never at the defining module.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

ORIGIN = {
    "platform": "telegram",
    "chat_id": "555",
    "chat_name": "Max",
    "chat_type": "dm",
    "thread_id": "topic-7",
    "user_id": "u-42",
    "scope_id": None,
}
OUTPUT = "disk 71%, mem 44%, 3 cron jobs wedged"


# --------------------------------------------------------------------------------------------
# Fakes: a push-capable adapter and a genuinely running gateway loop.
# --------------------------------------------------------------------------------------------

class _FakeAdapter:
    """Push-capable adapter: records the synthetic wake event and acks admission."""

    supports_async_delivery = True

    def __init__(self):
        self.events = []
        self.sends = []

    async def handle_message(self, event):
        event._gateway_accepted = True
        self.events.append(event)

    async def send(self, chat_id, content, metadata=None):
        self.sends.append((chat_id, content, metadata))
        return {"success": True, "message_id": "m1"}


@contextlib.contextmanager
def _live_loop():
    """A real running event loop on another thread — ``wake_origin`` requires ``is_running()``."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True, name="wake-origin-test-loop")
    thread.start()
    deadline = time.monotonic() + 5
    while not loop.is_running() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert loop.is_running(), "test loop never started"
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


def _gateway_cfg(*platforms):
    """Gateway config whose named platforms are enabled; absent ones are honored as 'no block'."""
    from gateway.config import Platform

    cfg = MagicMock()
    enabled = MagicMock()
    enabled.enabled = True
    cfg.platforms = {Platform(name): enabled for name in platforms}
    return cfg


@pytest.fixture
def temp_cron_home(tmp_path, monkeypatch):
    """Real cron store in a temp home (``save_job_output``/``update_job`` write through it)."""
    from cron import jobs as cron_jobs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with cron_jobs.use_cron_store(tmp_path):
        cron_jobs.ensure_dirs()
        yield tmp_path


@pytest.fixture(autouse=True)
def _not_an_external_worker(monkeypatch):
    """Default every test to the in-process ticker; the queue tests opt back in explicitly."""
    monkeypatch.delenv("_HERMES_CRON_EXTERNAL_WORKER", raising=False)


def _job(**overrides) -> dict:
    job = {
        "id": "sys-health",
        "name": "sys-health (15-min watchdog)",
        "execution_id": "exec-1",
        "deliver": "local",
        "origin": dict(ORIGIN),
        "no_agent": True,
    }
    job.update(overrides)
    return job


def _run_through_save_compose_deliver(job, final_response, *, adapters=None, loop=None):
    """Drive the REAL ``_save_compose_deliver`` seam (silence gate included) for one finished run.

    Calling ``_deliver_result`` directly would skip the gate that makes a silent run silent, so
    the silence contract would prove nothing. A job with no ``fire_claim`` has no owner, so the
    side-effect fences are no-ops.
    """
    from cron import scheduler as sched

    d = sched._RunDelivery(job=job, success=True, error=None)
    fence = sched._FireOwnership(job)
    sched._save_compose_deliver(
        d, fence, final_response, final_response,
        adapters=adapters, loop=loop, verbose=False, execution_token=object())
    return d


# --------------------------------------------------------------------------------------------
# 1. The happy path, end to end through the real wake module.
# --------------------------------------------------------------------------------------------

def test_wake_fires_once_and_keys_the_turn_to_the_origin_session(temp_cron_home):
    """A ``wake_origin`` run injects ONE turn carrying the output, keyed to the origin's session.

    ``chat_type``/``thread_id``/``user_id`` are the session-key inputs: a wrong (or defaulted)
    value opens a brand-new session nobody is reading and burns an agent turn every fire, so they
    are asserted individually rather than through the adapter's say-so.
    """
    from gateway.config import Platform

    adapter = _FakeAdapter()
    with _live_loop() as loop, \
            patch("gateway.config.load_gateway_config", return_value=_gateway_cfg()):
        d = _run_through_save_compose_deliver(
            _job(wake_origin=True), OUTPUT,
            adapters={Platform.TELEGRAM: adapter}, loop=loop)

    assert d.delivery_attempted and d.delivery_error is None, d.delivery_error
    assert len(adapter.events) == 1, "the wake must fire exactly once per run"
    event = adapter.events[0]
    assert event.internal is True
    assert OUTPUT in event.text, "the job's own output never reached the woken turn"
    assert event.source.platform is Platform.TELEGRAM
    assert event.source.chat_id == ORIGIN["chat_id"]
    assert event.source.chat_type == ORIGIN["chat_type"]
    assert event.source.thread_id == ORIGIN["thread_id"]
    assert event.source.user_id == ORIGIN["user_id"]
    # The ledger must count a wake as a real delivery, or a `deliver: local` run that DID reach an
    # agent is recorded "suppressed" and a failure incident never gets marked alerted.
    from cron import scheduler as sched
    assert sched._classify_delivery_outcome(
        delivery_error=None, should_deliver=True, unresolved_origin=False,
        normalized_deliver="local", incident_acked=False, success=True,
        woke_origin=bool(d.job.get("_wake_origin_delivered"))) == "delivered"


# --------------------------------------------------------------------------------------------
# 2. Silence is inherited, not re-implemented.
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "final_response, why",
    [("", "empty stdout / wakeAgent:false"), ("   ", "whitespace-only"), ("[SILENT]", "sentinel")])
def test_a_silent_run_never_wakes_anybody(temp_cron_home, monkeypatch, final_response, why):
    """Nothing to say -> ``_deliver_result`` is never reached, so no wake and no second gate.

    Driven through ``_save_compose_deliver`` deliberately: calling ``_deliver_result`` directly
    would bypass the only gate under test.
    """
    from cron import scheduler_delivery as delivery

    wake = Mock(return_value=None)
    monkeypatch.setattr(delivery._wake_origin, "wake_origin_session", wake)
    adapter = _FakeAdapter()

    from gateway.config import Platform
    with _live_loop() as loop, \
            patch("gateway.config.load_gateway_config", return_value=_gateway_cfg()):
        d = _run_through_save_compose_deliver(
            _job(wake_origin=True), final_response,
            adapters={Platform.TELEGRAM: adapter}, loop=loop)

    assert d.should_deliver is False, f"{why}: the run was not treated as silent"
    assert d.delivery_attempted is False, f"{why}: _deliver_result ran for a silent run"
    wake.assert_not_called()
    assert adapter.events == []


# --------------------------------------------------------------------------------------------
# 3. A job without the field is unchanged.
# --------------------------------------------------------------------------------------------

def test_job_without_the_field_neither_wakes_nor_changes_delivery(temp_cron_home, monkeypatch):
    """No ``wake_origin`` -> no wake, same return value, and the enqueue gate is untouched.

    The enqueue arm is the one that matters: the branch was MOVED above the ``if not targets``
    early return, so a field-less ``deliver: local`` job must still fall straight through to the
    unresolved-target outcome instead of newly queuing a row.
    """
    from cron import delivery_queue
    from cron import scheduler_delivery as delivery

    wake = Mock(return_value=None)
    monkeypatch.setattr(delivery._wake_origin, "wake_origin_session", wake)
    enqueue = Mock(side_effect=AssertionError("a field-less job must not enqueue"))
    monkeypatch.setattr(delivery_queue, "enqueue_and_wait", enqueue)
    # Pose as the restart-safe worker that owns this very execution: the ONLY shape that can take
    # the enqueue branch. Without the field it must not.
    monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", "exec-1")

    job = _job()
    assert "wake_origin" not in job
    assert delivery._deliver_result(job, OUTPUT, adapters=None, loop=None) is None
    wake.assert_not_called()
    enqueue.assert_not_called()
    assert "_wake_origin_delivered" not in job


# --------------------------------------------------------------------------------------------
# 4. One wake per run, not one per target.
# --------------------------------------------------------------------------------------------

def test_multi_target_delivery_wakes_the_origin_exactly_once(temp_cron_home, monkeypatch):
    """Two chat targets both get the message; the origin gets ONE turn, not one per target.

    The wake sits after the per-target loop for exactly this reason — inside it, a job fanning out
    to N chats would burn N agent turns on one run's output.
    """
    from cron import scheduler_delivery as delivery

    wake = Mock(return_value=None)
    monkeypatch.setattr(delivery._wake_origin, "wake_origin_session", wake)
    send = AsyncMock(return_value={"success": True, "message_id": "m1"})

    job = _job(deliver="telegram:111,telegram:222", no_agent=False, wake_origin=True)
    with patch("gateway.config.load_gateway_config", return_value=_gateway_cfg("telegram")), \
            patch("tools.send_message_tool._send_to_platform", new=send), \
            patch("sys.is_finalizing", return_value=False):
        error = delivery._deliver_result(job, OUTPUT, adapters=None, loop=None)

    assert error is None, error
    assert send.await_count == 2, "both chat targets must still be delivered"
    assert wake.call_count == 1, "the wake is per RUN, not per target"
    # ...and it got the run's own output, not a per-target slice of it.
    assert OUTPUT in wake.call_args.args[1]


# --------------------------------------------------------------------------------------------
# 5. A failed wake is reported, never allowed to swallow a send that already left.
# --------------------------------------------------------------------------------------------

def test_a_failing_wake_is_reported_without_losing_the_chat_send(temp_cron_home):
    """The wake coroutine raising surfaces in ``_deliver_result``'s error string, after the send.

    Failing at the real boundary (``deliver_wake``) rather than mocking the wake entry point: the
    error has to survive ``wake_origin_session``'s own handling to reach the caller. The chat
    delivery already left the process, so it must not be rolled back or re-reported as lost.
    """
    from cron import scheduler_delivery as delivery
    from gateway.config import Platform

    adapter = _FakeAdapter()
    live_sends = []

    def _record_live_send(t, cleaned_text, media_files, **kwargs):
        live_sends.append((t.platform_name, t.chat_id, cleaned_text))
        return True  # delivered: no standalone fallback

    job = _job(deliver="telegram:111", no_agent=False, wake_origin=True)
    with _live_loop() as loop, \
            patch("gateway.config.load_gateway_config", return_value=_gateway_cfg("telegram")), \
            patch.object(delivery, "_deliver_via_live_adapter", side_effect=_record_live_send), \
            patch("gateway.wake.deliver_wake",
                  new=AsyncMock(side_effect=RuntimeError("session store is wedged"))):
        error = delivery._deliver_result(
            job, OUTPUT, adapters={Platform.TELEGRAM: adapter}, loop=loop)

    assert len(live_sends) == 1, "the chat send was skipped or rolled back by the wake failure"
    assert OUTPUT in live_sends[0][2]
    assert error and "session store is wedged" in error, error
    assert not job.get("_wake_origin_delivered")


# --------------------------------------------------------------------------------------------
# 6. The regression the enqueue re-order exists for.
# --------------------------------------------------------------------------------------------

def test_local_plus_wake_origin_still_reaches_the_delivery_queue(temp_cron_home, monkeypatch):
    """A restart-safe worker with ``deliver: local`` + ``wake_origin`` MUST enqueue the row.

    The worker holds no live adapters, so the gateway's drain of this queue is the only process
    that can perform the wake. Before the enqueue branch was moved above ``if not targets``, this
    shape — the live sys-health job's exact shape — returned early and silently never woke
    anything. The field-less control proves the move did not start queuing unrelated jobs.
    """
    from cron import delivery_queue
    from cron import scheduler_delivery as delivery

    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", temp_cron_home / "deliveries.db")
    # The worker's wait budget: no gateway will drain inside this test, and a row left `pending`
    # is a deferral the next gateway still performs, not a failure.
    monkeypatch.setattr(delivery_queue, "DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", "exec-1")

    assert delivery._deliver_result(
        _job(wake_origin=True), OUTPUT, adapters=None, loop=None) is None
    row = delivery_queue.get_status("exec-1")
    assert row is not None and row["status"] == "pending", row
    assert OUTPUT in row["content"]
    assert row["job_json"] and '"wake_origin": true' in row["job_json"]

    # Control: the same job without the field resolves no targets and queues nothing.
    monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", "exec-2")
    assert delivery._deliver_result(
        _job(execution_id="exec-2"), OUTPUT, adapters=None, loop=None) is None
    assert delivery_queue.get_status("exec-2") is None


# --------------------------------------------------------------------------------------------
# E2E: real imports, real store, real queue, fake adapter — the whole queued lane.
# --------------------------------------------------------------------------------------------

def test_e2e_queued_worker_result_wakes_the_origin_on_the_gateway_drain(temp_cron_home, monkeypatch):
    """Worker enqueues -> gateway drains -> the origin session is woken with the job's output.

    Nothing is mocked between the two halves: the real ``cron.delivery_queue`` sqlite file under a
    temp ``HERMES_HOME`` carries the job across, and ``cron.scheduler.drain_delivery_queue`` runs
    the gateway side with live fakes. This is the only test that proves the lane the live
    ``deliver: local`` + ``wake_origin`` job actually travels.
    """
    from cron import delivery_queue, scheduler
    from cron import scheduler_delivery as delivery
    from gateway.config import Platform

    monkeypatch.setattr(delivery_queue, "DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS", 0.0)

    # --- worker half: no adapters, owns this execution ---
    monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", "exec-e2e")
    assert delivery._deliver_result(
        _job(execution_id="exec-e2e", wake_origin=True), OUTPUT,
        adapters=None, loop=None) is None
    assert delivery_queue.get_status("exec-e2e")["status"] == "pending"

    # --- gateway half: live adapters, not a worker ---
    monkeypatch.delenv("_HERMES_CRON_EXTERNAL_WORKER", raising=False)
    adapter = _FakeAdapter()
    with _live_loop() as loop, \
            patch("gateway.config.load_gateway_config", return_value=_gateway_cfg()):
        assert scheduler.drain_delivery_queue({Platform.TELEGRAM: adapter}, loop) == 1

    assert delivery_queue.get_status("exec-e2e")["status"] == "delivered"
    assert len(adapter.events) == 1, "the drained row did not wake the origin session"
    event = adapter.events[0]
    assert OUTPUT in event.text
    assert event.source.chat_id == ORIGIN["chat_id"]
    assert event.source.chat_type == ORIGIN["chat_type"]
    assert event.source.thread_id == ORIGIN["thread_id"]
