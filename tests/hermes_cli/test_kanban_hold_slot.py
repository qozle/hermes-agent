"""LOCAL COMMIT (frankentrait): ``kanban.hold_slot_through_review``.

Max, 2026-11-27: two cards at once, and "each card should hold its slot until
it's finished completely (reviewed)" — a card that finishes its build and waits
for review, or is in fix rounds, keeps its slot; no new card takes it.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import pytest


@pytest.fixture()
def home(monkeypatch):
    test_home = tempfile.mkdtemp(prefix="kanban_hold_slot_test_")
    for prof in ("alpha", "beta", "gamma", "rev", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
        with open(os.path.join(test_home, "profiles", prof, "config.yaml"), "w") as fh:
            fh.write("{}\n")
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    from hermes_cli import kanban_db_dispatch
    monkeypatch.setattr(kanban_db_dispatch, "_system_memory_sample", dict)
    yield kanban_db


def _spawn(*args, **kwargs):
    return 12345


def _started(conn, tid, status):
    """Give a card a finished run and put it in ``status``."""
    conn.execute(
        "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome) "
        "VALUES (?, 'done', ?, ?, 'review_requested')",
        (tid, int(time.time()) - 60, int(time.time())),
    )
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, tid))
    conn.commit()


def _tick(conn, hold=True, cap=2):
    from hermes_cli import kanban_db_dispatch as kbd
    return kbd.dispatch_once(
        conn, spawn_fn=_spawn, max_in_progress=cap, hold_slots=hold,
        max_in_progress_per_lane={"ready": 2},
    )


def _setup_running_plus_review(kb, kbc):
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        a = kb.create_task(conn, title="A building", assignee="alpha")
    with kbc.connect_closing() as conn:
        assert [s[0] for s in _tick(conn).spawned] == [a]
        b = kb.create_task(conn, title="B waiting for its reviewer", assignee="rev")
        c = kb.create_task(conn, title="C new", assignee="gamma")
        _started(conn, b, "review")
        # the review lane is off here so B stays waiting: the slot must still be held
    return a, b, c


def test_card_waiting_for_review_keeps_its_slot(home, monkeypatch):
    kb = home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    monkeypatch.setattr(kbd, "review_dispatch_enabled", lambda: False)
    _a, _b, c = _setup_running_plus_review(kb, kbc)
    with kbc.connect_closing() as conn:
        res = _tick(conn)
    assert res.spawned == []
    assert res.skipped_slot_held == [c]


def test_without_the_setting_the_new_card_takes_the_slot(home, monkeypatch):
    kb = home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    monkeypatch.setattr(kbd, "review_dispatch_enabled", lambda: False)
    _a, _b, c = _setup_running_plus_review(kb, kbc)
    with kbc.connect_closing() as conn:
        res = _tick(conn, hold=False)
    assert [s[0] for s in res.spawned] == [c]


def test_fix_round_card_spawns_into_the_held_slot(home):
    kb = home
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        a = kb.create_task(conn, title="A building", assignee="alpha")
    with kbc.connect_closing() as conn:
        assert [s[0] for s in _tick(conn).spawned] == [a]
        b = kb.create_task(conn, title="B changes requested", assignee="beta")
        fix = kb.create_task(conn, title="Round 1 fixes: B", assignee="beta")
        kb.create_task(conn, title="C new", assignee="gamma")
        conn.execute("INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)", (fix, b))
        conn.commit()
        _started(conn, b, "todo")
    with kbc.connect_closing() as conn:
        res = _tick(conn)
    assert [s[0] for s in res.spawned] == [fix]  # C never starts
    with kbc.connect_closing() as conn:
        # fix round running: B is covered by it, still no room for C
        res = _tick(conn)
    assert res.spawned == []


def test_two_new_cards_start_when_nothing_is_held(home):
    kb = home
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        kb.create_task(conn, title="a", assignee="alpha")
        kb.create_task(conn, title="b", assignee="beta")
        kb.create_task(conn, title="c", assignee="gamma")
    with kbc.connect_closing() as conn:
        res = _tick(conn)
    assert len(res.spawned) == 2


def test_setting_parses(home, monkeypatch):
    from hermes_cli import kanban_db_dispatch as kbd
    import hermes_cli.config as cfg
    monkeypatch.setattr(cfg, "load_config_readonly", lambda: {"kanban": {"hold_slot_through_review": True}})
    assert kbd.configured_hold_slot_through_review() is True
    monkeypatch.setattr(cfg, "load_config_readonly", lambda: {"kanban": {}})
    assert kbd.configured_hold_slot_through_review() is False
