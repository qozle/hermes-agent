"""LOCAL COMMIT (frankentrait): ``kanban.max_in_progress_per_origin``.

Max's rule (2026-11-08): every agent conversation may have N cards running at
once — the WhatsApp chat N, each terminal session N — independent of each
other. A card's origin is the conversation subscribed to it at creation
(``kanban_notify_subs`` platform:chat_id); cards with no subscription share one
"no origin" bucket.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest


@pytest.fixture()
def home(monkeypatch):
    test_home = tempfile.mkdtemp(prefix="kanban_per_origin_cap_test_")
    for prof in ("alpha", "beta", "gamma", "default"):
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


def _subscribe(conn, task_id, platform, chat_id):
    from hermes_cli import kanban_db_notify as kbn
    kbn.add_notify_sub(conn, task_id=task_id, platform=platform, chat_id=chat_id,
                       notifier_profile="default")


def _statuses(kbc):
    with kbc.connect_closing() as conn:
        return {r["id"]: r["status"] for r in conn.execute("SELECT id, status FROM tasks")}


def test_each_conversation_gets_its_own_two_slots(home):
    """Three WhatsApp cards + two terminal cards, cap 2 per origin: two of each
    spawn, the third WhatsApp card waits and is reported as origin-capped."""
    kb = home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        wa = [kb.create_task(conn, title=f"wa{i}", assignee="alpha") for i in range(3)]
        cli = [kb.create_task(conn, title=f"cli{i}", assignee="beta") for i in range(2)]
        for tid in wa:
            _subscribe(conn, tid, "whatsapp", "1234@lid")
        for tid in cli:
            _subscribe(conn, tid, "local", "local-abc")
        conn.commit()
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, max_in_progress=10, max_in_progress_per_origin=2,
        )
    spawned = {t for (t, _a, _w) in res.spawned}
    assert len(spawned) == 4
    assert len(spawned & set(wa)) == 2
    assert set(cli) <= spawned
    assert [(t, key) for (t, key, _n) in res.skipped_per_origin_capped] == [
        (wa[2], "whatsapp:1234@lid"),
    ]
    with kbc.connect_closing() as conn:
        assert kbd.count_running_tasks_by_origin(conn) == {
            "whatsapp:1234@lid": 2, "local:local-abc": 2,
        }
    # Next tick: nothing new — both conversations are at their cap.
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, max_in_progress=10, max_in_progress_per_origin=2,
        )
    assert res.spawned == []
    assert [t for (t, _k, n) in res.skipped_per_origin_capped] == [wa[2]]
    assert res.skipped_per_origin_capped[0][2] == 2


def test_origin_cap_frees_a_slot_when_a_card_finishes(home):
    kb = home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        ids = [kb.create_task(conn, title=f"c{i}", assignee="alpha") for i in range(3)]
        for tid in ids:
            _subscribe(conn, tid, "whatsapp", "1234@lid")
        conn.commit()
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, max_in_progress=10, max_in_progress_per_origin=2,
        )
    assert len(res.spawned) == 2
    first = res.spawned[0][0]
    with kbc.connect_closing() as conn:
        kb.complete_task(conn, first, result="done")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, max_in_progress=10, max_in_progress_per_origin=2,
        )
    assert [t for (t, _a, _w) in res.spawned] == [ids[2]]
    statuses = _statuses(kbc)
    assert sorted(statuses.values()) == ["done", "running", "running"]


def test_unsubscribed_cards_share_one_bucket_and_unset_cap_changes_nothing(home):
    kb = home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(3):
            kb.create_task(conn, title=f"n{i}", assignee="gamma")
        conn.commit()
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, max_in_progress=10, max_in_progress_per_origin=2,
        )
    assert len(res.spawned) == 2
    assert [key for (_t, key, _n) in res.skipped_per_origin_capped] == [kbd.NO_ORIGIN_KEY]
    # Unset cap: the third card spawns too.
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, max_in_progress=10, max_in_progress_per_origin=None,
        )
    assert len(res.spawned) == 1
    assert res.skipped_per_origin_capped == []
    assert kbd.parse_max_in_progress_per_origin("x") is None
    assert kbd.parse_max_in_progress_per_origin(0) is None
    assert kbd.parse_max_in_progress_per_origin(True) is None
    assert kbd.parse_max_in_progress_per_origin("2") == 2


def test_review_rows_respect_the_origin_cap_and_do_not_reserve_a_slot(home):
    """A review row from a conversation at its cap must neither spawn nor hold
    the ready-lane reservation back from another conversation's ready card."""
    kb = home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        wa_run = [kb.create_task(conn, title=f"wa{i}", assignee="alpha") for i in range(2)]
        for tid in wa_run:
            _subscribe(conn, tid, "whatsapp", "1234@lid")
        conn.commit()
    # Tick 1: the two WhatsApp cards take that conversation's two slots.
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, max_in_progress=3, max_in_progress_per_origin=2,
        )
    assert sorted(t for (t, _a, _w) in res.spawned) == sorted(wa_run)
    with kbc.connect_closing() as conn:
        wa_review = kb.create_task(conn, title="wa-review", assignee="alpha")
        cli_ready = kb.create_task(conn, title="cli", assignee="beta")
        _subscribe(conn, wa_review, "whatsapp", "1234@lid")
        _subscribe(conn, cli_ready, "local", "local-abc")
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (wa_review,))
        conn.commit()
    # Tick 2: one slot of shared budget left. The WhatsApp review row is
    # origin-capped, so it must not reserve that slot — the terminal card gets it.
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, max_in_progress=3, max_in_progress_per_origin=2,
        )
    assert [t for (t, _a, _w) in res.spawned] == [cli_ready]
    statuses = _statuses(kbc)
    assert statuses[wa_review] == "review"
    assert statuses[cli_ready] == "running"
