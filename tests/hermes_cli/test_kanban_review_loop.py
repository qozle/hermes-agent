"""LOCAL COMMIT (frankentrait): the review loop.

With ``kanban.review_fix_cards`` on, a reviewer's ``request_changes`` opens a
"Round N fixes" card carrying the findings, the review card drops to ``todo``
gated on it, and when the fix card completes the review card returns to the
review lane for a re-review. A review-only card (no implementer run) gets the
same treatment, the fix card going to ``kanban.review_fix_assignee`` /
``default_assignee``. After ``review_max_rounds`` rounds the review card blocks
``needs_input`` naming every round.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _cfg(monkeypatch, **kanban):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", lambda: {"kanban": kanban},
    )


def _review_card_with_implementer(conn):
    tid = kb.create_task(conn, title="Fix the quote line", assignee="builder")
    claimed = kb.claim_task(conn, tid)
    assert kb.request_review(
        conn, tid, summary="v1", reviewer="reviewer", expected_run_id=claimed.current_run_id,
    )
    review = kb.claim_review_task(conn, tid)
    assert review is not None
    return tid, review


def _events(conn, tid, kind):
    return [
        kb._json_dict(r["payload"]) for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
            (tid, kind),
        ).fetchall()
    ]


def test_loop_off_keeps_the_handback(kanban_home, monkeypatch):
    _cfg(monkeypatch, review_fix_cards=False)
    with kbc.connect() as conn:
        tid, review = _review_card_with_implementer(conn)
        assert kb.request_changes(
            conn, tid, reason="fix it", expected_run_id=review.current_run_id,
        ) == (True, "builder")
        assert kb.get_task(conn, tid).status == "ready"
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE created_by = 'review-loop'"
        ).fetchone()[0] == 0


def test_changes_open_a_fix_card_and_the_review_card_waits(kanban_home, monkeypatch):
    _cfg(monkeypatch, review_fix_cards=True, review_max_rounds=4)
    with kbc.connect() as conn:
        tid, review = _review_card_with_implementer(conn)
        ok, child = kb.request_changes(
            conn, tid, reason="price blanks every 20 s", expected_run_id=review.current_run_id,
        )
        assert ok and child and child.startswith("t_") and child != tid
        rc = kb.get_task(conn, tid)
        fx = kb.get_task(conn, child)
        # review card parked in todo, still owned by the reviewer, gated on the fix card
        assert rc.status == "todo"
        assert rc.assignee == "reviewer"
        assert kb.unsatisfied_parents(conn, tid) == [(child, "ready")]
        # fix card goes to the implementer, carries the findings and the round
        assert fx.status == "ready"
        assert fx.assignee == "builder"
        assert fx.title.startswith("Round 1 fixes: Fix the quote line")
        assert "price blanks every 20 s" in (fx.body or "")
        assert tid in (fx.body or "")
        ev = _events(conn, tid, "changes_requested")
        assert ev[-1]["round"] == 1 and ev[-1]["fix_task"] == child
        assert ev[-1]["resume_status"] == "review"
        # the reviewer cannot be dispatched again while the fix card is open
        assert kb.claim_review_task(conn, tid) is None


def test_fix_card_done_returns_the_review_card_to_review(kanban_home, monkeypatch):
    _cfg(monkeypatch, review_fix_cards=True)
    with kbc.connect() as conn:
        tid, review = _review_card_with_implementer(conn)
        ok, child = kb.request_changes(
            conn, tid, reason="round one", expected_run_id=review.current_run_id,
        )
        assert ok
        fixer = kb.claim_task(conn, child)
        assert fixer is not None
        assert kb.complete_task(conn, child, summary="pushed", expected_run_id=fixer.current_run_id)
        kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).status == "review"
        # round two: the same reviewer picks it up and a second fix card opens
        review2 = kb.claim_review_task(conn, tid)
        assert review2 is not None
        ok, child2 = kb.request_changes(
            conn, tid, reason="round two", expected_run_id=review2.current_run_id,
        )
        assert ok and child2 != child
        assert kb.get_task(conn, child2).title.startswith("Round 2 fixes:")
        assert kb.get_task(conn, tid).status == "todo"


def test_review_only_card_uses_the_fallback_assignee(kanban_home, monkeypatch):
    """A review card created straight for the reviewer (a foreground PR) has
    no implementer run; the loop still opens the fix card instead of refusing."""
    _cfg(monkeypatch, review_fix_cards=True, default_assignee="sonnet-med")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Review PR #1031", assignee="opus-x")
        assert kb.request_review(conn, tid, summary="foreground PR", reviewer="opus-x")
        review = kb.claim_review_task(conn, tid)
        assert review is not None
        ok, child = kb.request_changes(
            conn, tid, reason="two blockers", expected_run_id=review.current_run_id,
        )
        assert ok and child.startswith("t_")
        assert kb.get_task(conn, child).assignee == "sonnet-med"
        assert kb.get_task(conn, tid).status == "todo"


def test_round_limit_blocks_the_review_card_for_max(kanban_home, monkeypatch):
    _cfg(monkeypatch, review_fix_cards=True, review_max_rounds=2)
    with kbc.connect() as conn:
        tid, review = _review_card_with_implementer(conn)
        for n in (1, 2):
            ok, child = kb.request_changes(
                conn, tid, reason=f"round {n} findings", expected_run_id=review.current_run_id,
            )
            assert ok and child.startswith("t_")
            fixer = kb.claim_task(conn, child)
            assert kb.complete_task(conn, child, summary="done", expected_run_id=fixer.current_run_id)
            kb.recompute_ready(conn)
            assert kb.get_task(conn, tid).status == "review"
            review = kb.claim_review_task(conn, tid)
            assert review is not None
        # third request: over the limit of two
        ok, detail = kb.request_changes(
            conn, tid, reason="round 3 findings", expected_run_id=review.current_run_id,
        )
        assert ok and detail is None
        row = kb.get_task(conn, tid)
        assert row.status == "blocked"
        blocked = _events(conn, tid, "blocked")[-1]
        assert blocked["kind"] == "needs_input"
        assert "after 2 rounds" in blocked["reason"]
        assert "Round 1: round 1 findings" in blocked["reason"]
        assert "Round 3: round 3 findings" in blocked["reason"]
        # no fourth fix card
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE created_by = 'review-loop'"
        ).fetchone()[0] == 2
        # a sticky block: the dispatcher never promotes it on its own
        kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).status == "blocked"
