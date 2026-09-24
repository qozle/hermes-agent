"""Dispatcher-side reap of worker scopes whose card already left ``running``.

A worker that calls ``kanban_complete`` closes its run in the DB, but the OS
process (and its ``hermes-worker-kanban-<id>-run-<n>.scope`` cgroup) keeps
holding an API session.  The dispatcher stops tracking the run the moment the
card leaves ``running``, so nothing reaped it and the scope leaked per finished
card.

Every test here drives an INJECTED systemd runner: a unit test must never touch
the real user manager (one that ran ``systemctl freeze`` for real froze every
live worker scope including its own).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_reap as reap


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeSystemd:
    """In-memory stand-in for the user manager. Records every call, touches nothing."""

    def __init__(self, units: dict[str, str] | None = None):
        # unit -> freezer state ("running" | "frozen")
        self.units = dict(units or {})
        self.calls: list[tuple[str, str]] = []

    def list_worker_scopes(self) -> list[str]:
        self.calls.append(("list", ""))
        return sorted(self.units)

    def freezer_state(self, unit: str) -> str:
        return self.units.get(unit, "")

    def thaw(self, unit: str) -> bool:
        self.calls.append(("thaw", unit))
        if unit in self.units:
            self.units[unit] = "running"
        return True

    def stop(self, unit: str) -> bool:
        self.calls.append(("stop", unit))
        self.units.pop(unit, None)
        return True

    def stopped(self) -> list[str]:
        return [unit for verb, unit in self.calls if verb == "stop"]


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _scope(task_id: str, run_id: int) -> str:
    return f"hermes-worker-kanban-{task_id}-run-{run_id}.scope"


def _claimed(conn, *, title: str = "card") -> tuple[str, int]:
    task_id = kb.create_task(conn, title=title, assignee="worker")
    task = kb.claim_task(conn, task_id, claimer="worker:1")
    assert task is not None and task.current_run_id is not None
    return task_id, int(task.current_run_id)


# ---------------------------------------------------------------------------
# Unit-name parsing
# ---------------------------------------------------------------------------


def test_scope_name_round_trips_task_and_run():
    parsed = reap.parse_worker_scope_unit(_scope("t_4e41c6c7", 64))
    assert parsed == ("t_4e41c6c7", 64)


def test_unrelated_units_are_not_parsed():
    for unit in (
        "hermes-worker-cron-job-7.scope",
        "hermes-gateway.service",
        "hermes-worker-kanban-t_abc-run-notanumber.scope",
        "",
    ):
        assert reap.parse_worker_scope_unit(unit) is None


# ---------------------------------------------------------------------------
# Reap decisions
# ---------------------------------------------------------------------------


def test_scope_of_completed_card_is_stopped(board):
    """The incident: worker called kanban_complete, run closed, scope still live."""
    with kbc.connect() as conn:
        task_id, run_id = _claimed(conn)
        assert kb.complete_task(conn, task_id, summary="done", expected_run_id=run_id)

        systemd = FakeSystemd({_scope(task_id, run_id): "running"})
        reaped = reap.reap_finished_worker_scopes(conn, systemd=systemd)

    assert reaped == [_scope(task_id, run_id)]
    assert systemd.stopped() == [_scope(task_id, run_id)]


def test_running_card_scope_is_left_alone(board):
    with kbc.connect() as conn:
        task_id, run_id = _claimed(conn)
        systemd = FakeSystemd({_scope(task_id, run_id): "running"})
        reaped = reap.reap_finished_worker_scopes(conn, systemd=systemd)

    assert reaped == []
    assert systemd.stopped() == []


def test_blocked_and_review_cards_are_reaped_too(board):
    """kanban_block / kanban_request_review end the run exactly like completion."""
    with kbc.connect() as conn:
        blocked_id, blocked_run = _claimed(conn, title="blocked card")
        assert kb.block_task(
            conn, blocked_id, reason="need input", kind="needs_input",
            expected_run_id=blocked_run,
        )
        review_id, review_run = _claimed(conn, title="review card")
        ok, _ = kb.request_review(
            conn, review_id, summary="implemented", expected_run_id=review_run,
            with_reason=True,
        )
        assert ok

        systemd = FakeSystemd({
            _scope(blocked_id, blocked_run): "running",
            _scope(review_id, review_run): "running",
        })
        reaped = reap.reap_finished_worker_scopes(conn, systemd=systemd)

    assert sorted(reaped) == sorted([
        _scope(blocked_id, blocked_run), _scope(review_id, review_run),
    ])


def test_stale_scope_of_earlier_run_is_reaped_while_current_run_survives(board):
    """A respawn must not let the previous run's scope hide behind the new one."""
    with kbc.connect() as conn:
        task_id, first_run = _claimed(conn)
        # First run ends (blocked), the card is unblocked and claimed again ->
        # a second, LIVE run beside the first run's leaked scope.
        assert kb.block_task(
            conn, task_id, reason="need input", kind="needs_input",
            expected_run_id=first_run,
        )
        assert kb.unblock_task(conn, task_id)
        task = kb.claim_task(conn, task_id, claimer="worker:2")
        assert task is not None and task.current_run_id is not None
        second_run = int(task.current_run_id)
        assert second_run != first_run

        systemd = FakeSystemd({
            _scope(task_id, first_run): "running",
            _scope(task_id, second_run): "running",
        })
        reaped = reap.reap_finished_worker_scopes(conn, systemd=systemd)

    assert reaped == [_scope(task_id, first_run)]
    assert systemd.stopped() == [_scope(task_id, first_run)]


def test_scope_for_a_task_on_another_board_is_never_touched(board):
    """Scope names carry no board; an unknown task id belongs to someone else."""
    with kbc.connect() as conn:
        systemd = FakeSystemd({_scope("t_notonthisboard", 3): "running"})
        reaped = reap.reap_finished_worker_scopes(conn, systemd=systemd)

    assert reaped == []
    assert systemd.stopped() == []


def test_frozen_scope_is_thawed_before_it_is_stopped(board):
    """A frozen cgroup cannot process SIGTERM, so `stop` would hang on it."""
    with kbc.connect() as conn:
        task_id, run_id = _claimed(conn)
        assert kb.complete_task(conn, task_id, summary="done", expected_run_id=run_id)

        unit = _scope(task_id, run_id)
        systemd = FakeSystemd({unit: "frozen"})
        reaped = reap.reap_finished_worker_scopes(conn, systemd=systemd)

    assert reaped == [unit]
    assert systemd.calls == [("list", ""), ("thaw", unit), ("stop", unit)], (
        "a frozen scope must be thawed BEFORE the stop, never after"
    )


def test_reap_records_an_auditable_event_on_the_card(board):
    with kbc.connect() as conn:
        task_id, run_id = _claimed(conn)
        assert kb.complete_task(conn, task_id, summary="done", expected_run_id=run_id)
        systemd = FakeSystemd({_scope(task_id, run_id): "running"})
        reap.reap_finished_worker_scopes(conn, systemd=systemd)

        kinds = [row["kind"] for row in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ?", (task_id,)
        )]
    assert "worker_scope_reaped" in kinds


def test_a_failed_stop_is_not_reported_as_reaped(board):
    class StubbornSystemd(FakeSystemd):
        def stop(self, unit: str) -> bool:
            self.calls.append(("stop", unit))
            return False

    with kbc.connect() as conn:
        task_id, run_id = _claimed(conn)
        assert kb.complete_task(conn, task_id, summary="done", expected_run_id=run_id)
        systemd = StubbornSystemd({_scope(task_id, run_id): "running"})
        reaped = reap.reap_finished_worker_scopes(conn, systemd=systemd)

    assert reaped == []


def test_disabled_by_config(board, monkeypatch):
    monkeypatch.setattr(reap, "reap_enabled", lambda: False)
    with kbc.connect() as conn:
        task_id, run_id = _claimed(conn)
        assert kb.complete_task(conn, task_id, summary="done", expected_run_id=run_id)
        systemd = FakeSystemd({_scope(task_id, run_id): "running"})
        reaped = reap.reap_finished_worker_scopes(conn, systemd=systemd)

    assert reaped == []
    assert systemd.stopped() == []


# ---------------------------------------------------------------------------
# Wiring: the dispatcher tick must actually call it
# ---------------------------------------------------------------------------


def test_dispatch_tick_reaps_finished_scopes(board, monkeypatch):
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect() as conn:
        task_id, run_id = _claimed(conn)
        assert kb.complete_task(conn, task_id, summary="done", expected_run_id=run_id)

        systemd = FakeSystemd({_scope(task_id, run_id): "running"})
        monkeypatch.setattr(
            kbd, "reap_finished_worker_scopes",
            lambda c: reap.reap_finished_worker_scopes(c, systemd=systemd),
        )
        res = kbd.dispatch_once(conn, spawn_fn=lambda *a, **kw: 4242)

    assert systemd.stopped() == [_scope(task_id, run_id)]
    assert res.reaped_scopes == [_scope(task_id, run_id)]


def test_real_systemd_runner_is_a_noop_without_systemctl(monkeypatch):
    """No systemd (macOS, containers, CI) must be a silent no-op, never an error."""
    monkeypatch.setattr(reap.shutil, "which", lambda _name: None)
    runner = reap.SystemdUserUnits()
    assert runner.list_worker_scopes() == []
    assert runner.freezer_state("hermes-worker-kanban-t_x-run-1.scope") == ""
    assert runner.stop("hermes-worker-kanban-t_x-run-1.scope") is False
