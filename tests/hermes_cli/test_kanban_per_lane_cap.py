"""LOCAL COMMIT (frankentrait): ``kanban.max_in_progress_per_lane``.

``{ready: 1, review: 1}`` means at most one implementation worker and one
reviewer live at once — never two runners, never two reviewers — even when
``max_in_progress`` is 2 and the assignees differ.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest


@pytest.fixture()
def home(monkeypatch):
    test_home = tempfile.mkdtemp(prefix="kanban_per_lane_cap_test_")
    for prof in ("alpha", "beta", "rev", "default"):
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


def _fake_spawn(*args, **kwargs):
    return 12345


LANES = {"ready": 1, "review": 1}


def test_two_ready_rows_spawn_only_one_runner(home):
    kb = home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        kb.create_task(conn, title="a", assignee="alpha")
        kb.create_task(conn, title="b", assignee="beta")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, max_in_progress=2, max_in_progress_per_lane=LANES,
        )
    assert len(res.spawned) == 1
    with kbc.connect_closing() as conn:
        # Second tick: the runner slot is taken, nothing else spawns.
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, max_in_progress=2, max_in_progress_per_lane=LANES,
        )
    assert res.spawned == []
    assert [c[1] for c in res.skipped_per_lane_capped] == ["ready"]


def test_runner_plus_reviewer_both_spawn_but_never_two_reviewers(home):
    kb = home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        kb.create_task(conn, title="a", assignee="alpha")
        r1 = kb.create_task(conn, title="r1", assignee="rev")
        r2 = kb.create_task(conn, title="r2", assignee="rev")
        for tid in (r1, r2):
            conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (tid,))
        conn.commit()
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, max_in_progress=2, max_in_progress_per_lane=LANES,
        )
    assert len(res.spawned) == 2
    with kbc.connect_closing() as conn:
        rows = {r["id"]: r["status"] for r in conn.execute("SELECT id, status FROM tasks")}
    assert sorted(rows.values()) == ["review", "running", "running"]
    assert kbd.count_running_tasks_by_lane(kbc.connect()) == {"ready": 1, "review": 1}
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, max_in_progress=3, max_in_progress_per_lane=LANES,
        )
    assert res.spawned == []
    assert [c[1] for c in res.skipped_per_lane_capped] == ["review"]


def test_unset_lane_caps_change_nothing(home):
    kb = home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        kb.create_task(conn, title="a", assignee="alpha")
        kb.create_task(conn, title="b", assignee="beta")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, max_in_progress=2, max_in_progress_per_lane=None,
        )
    assert len(res.spawned) == 2
    assert kbd.parse_max_in_progress_per_lane({"ready": "x", "review": 0}) == {}
    assert kbd.parse_max_in_progress_per_lane({"ready": 1}) == {"ready": 1}
