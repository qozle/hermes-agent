"""The cooperative worker pause (agent/worker_pause.py, LOCAL COMMIT frankentrait).

A kanban worker that finds the WORKER_PAUSE sentinel before a model call waits awake —
touching its activity clock, heartbeating a ``paused:`` note — and resumes the same turn
when the sentinel goes, when its ``until`` is well past, or breaks out on an interrupt.
Everything else (no worker env, no sentinel) must cost nothing.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from agent import worker_pause as wp


@pytest.fixture
def board_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test1234")
    return tmp_path


def _write_sentinel(home, *, until=None, reason="quota guard: session usage 95 %"):
    (home / wp.PAUSE_SENTINEL).write_text(json.dumps({"reason": reason, "until": until}))


def _agent():
    agent = SimpleNamespace(_interrupt_requested=False, touches=[])
    agent._touch_activity = lambda desc, **_kw: agent.touches.append(desc)
    return agent


class _Clock:
    def __init__(self, start=1_000_000.0):
        self.now = start
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def time(self):
        return self.now


def test_no_worker_env_is_a_noop_without_touching_the_board(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    _write_sentinel(tmp_path)
    clock = _Clock()
    assert wp.wait_out_worker_pause(_agent(), sleep_fn=clock.sleep, now_fn=clock.time) is False
    assert clock.sleeps == []


def test_no_sentinel_returns_at_once(board_home):
    clock = _Clock()
    agent = _agent()
    assert wp.wait_out_worker_pause(agent, sleep_fn=clock.sleep, now_fn=clock.time) is False
    assert clock.sleeps == [] and agent.touches == []


def test_waits_until_the_sentinel_is_removed_then_resumes(board_home):
    _write_sentinel(board_home, until=None)
    clock = _Clock()
    agent = _agent()
    notes = []

    def sleep(seconds):
        clock.sleep(seconds)
        if len(clock.sleeps) == 3:
            (board_home / wp.PAUSE_SENTINEL).unlink()

    out = wp.wait_out_worker_pause(
        agent, sleep_fn=sleep, now_fn=clock.time, poll_s=20.0, heartbeat_fn=lambda n: notes.append(n) or True,
    )
    assert out is False
    assert len(clock.sleeps) == 3
    # The activity clock was touched on every poll so the liveness watchdog never sees idle.
    assert len(agent.touches) == 3
    assert all(t.startswith(wp.NOTE_PREFIX) for t in agent.touches)
    # The first poll carried the acknowledgement note to the board; later polls inside
    # NOTE_INTERVAL_S did not repeat it.
    assert len(notes) == 1
    assert notes[0].startswith("paused: quota guard: session usage 95 %")
    assert "until the guard releases it" in notes[0]


def test_resumes_on_its_own_once_until_is_well_past(board_home):
    until = datetime.now(timezone.utc) + timedelta(minutes=10)
    _write_sentinel(board_home, until=until.isoformat())
    clock = _Clock(start=until.timestamp() - 60)
    agent = _agent()
    out = wp.wait_out_worker_pause(agent, sleep_fn=clock.sleep, now_fn=clock.time, poll_s=60.0, heartbeat_fn=lambda n: True)
    assert out is False
    assert (board_home / wp.PAUSE_SENTINEL).exists()  # the guard never removed it
    # 60 s to ``until`` plus the grace, at 60 s polls.
    assert len(clock.sleeps) == 1 + int(wp.RESUME_GRACE_S // 60)
    assert "until" in agent.touches[0]


def test_an_interrupt_during_the_wait_breaks_out(board_home):
    _write_sentinel(board_home)
    clock = _Clock()
    agent = _agent()

    def sleep(seconds):
        clock.sleep(seconds)
        if len(clock.sleeps) == 2:
            agent._interrupt_requested = True

    out = wp.wait_out_worker_pause(agent, sleep_fn=sleep, now_fn=clock.time, heartbeat_fn=lambda n: True)
    assert out is True
    assert len(clock.sleeps) == 2


def test_noted_heartbeat_repeats_every_interval(board_home):
    _write_sentinel(board_home)
    clock = _Clock()
    notes = []

    def sleep(seconds):
        clock.sleep(seconds)
        if clock.now - 1_000_000.0 >= wp.NOTE_INTERVAL_S * 2 + 1:
            (board_home / wp.PAUSE_SENTINEL).unlink()

    wp.wait_out_worker_pause(
        _agent(), sleep_fn=sleep, now_fn=clock.time, poll_s=wp.NOTE_INTERVAL_S / 3,
        heartbeat_fn=lambda n: notes.append(n) or True,
    )
    assert len(notes) == 3


def test_a_bare_touch_or_garbage_file_still_holds(board_home):
    (board_home / wp.PAUSE_SENTINEL).write_text("")
    info = wp.read_pause()
    assert info is not None and wp.until_timestamp(info) is None
    (board_home / wp.PAUSE_SENTINEL).write_text("not json {")
    info = wp.read_pause()
    assert info is not None and info["reason"].startswith("not json")


def test_noted_heartbeat_lands_on_the_running_card_as_the_guards_acknowledgement(monkeypatch, tmp_path):
    """The real board write path: a ``paused:`` heartbeat event on the worker's own run, which
    ``quota_guard.acked_tasks`` reads (``payload LIKE '%paused:%'``, note prefix)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "parent-worker")
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="paused card", assignee="parent-worker", workspace_kind="scratch",
                             workspace_path=str(tmp_path))
        claim = kb.claim_task(conn, tid)
        assert claim is not None
        run_id = conn.execute("SELECT current_run_id FROM tasks WHERE id = ?", (tid,)).fetchone()[0]
        assert run_id is not None
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    note = wp.pause_note({"reason": "quota guard: session usage 95 %", "until": None})
    assert wp._heartbeat_note(note) is True

    conn = kbc.connect()
    try:
        rows = conn.execute(
            "SELECT run_id, payload FROM task_events WHERE task_id = ? AND kind = 'heartbeat' "
            "AND payload LIKE '%paused:%'", (tid,),
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0][0] == run_id
    assert json.loads(rows[0][1])["note"] == note


def test_begin_iteration_breaks_when_the_pause_is_interrupted(monkeypatch):
    import sys

    from agent import turn_iteration_prep as prep

    # ``begin_iteration`` imports the conversation loop lazily; its bootstrap reads the real
    # home on this install (pre-existing red in test_turn_iteration_prep.py), so stub it.
    monkeypatch.setitem(sys.modules, "agent.conversation_loop", SimpleNamespace(
        _apply_active_turn_redirect=lambda *a, **k: None, _review_input_budget_exhausted=lambda a: False,
    ))
    monkeypatch.setattr(wp, "wait_out_worker_pause", lambda agent, **kw: True)
    monkeypatch.setattr(prep, "interrupt_issuer", lambda agent: "session turn lease lost")
    agent = SimpleNamespace(
        _interrupt_requested=False, quiet_mode=True, _drain_pending_redirect=lambda: None,
        _checkpoint_mgr=SimpleNamespace(new_turn=lambda: None),
    )
    verdict = prep.begin_iteration(
        agent, messages=[], conversation_history=[], original_user_message="x",
        api_call_count=0, interrupted=False, _turn_exit_reason=None,
    )
    assert verdict.action == "break"
    assert verdict.interrupted is True
    assert verdict._turn_exit_reason == "interrupted_by_system(session turn lease lost)"
    assert verdict.api_call_count == 0
