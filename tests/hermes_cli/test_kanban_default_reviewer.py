"""LOCAL COMMIT (frankentrait): ``kanban.default_reviewer``.

A hand-off that names no reviewer, on a card with no earlier review round, goes
to the board's default reviewer instead of staying with the builder's tier. On
2026-11-19 two Sonnet-built cards called kanban_request_review() without a
reviewer although their bodies named one, and Sonnet reviewed its own work.
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


def _built_card(conn, assignee="sonnet-high"):
    tid = kb.create_task(conn, title="Portfolio numbers (6 of 6)", assignee=assignee)
    return tid, kb.claim_task(conn, tid)


def _review_requested_reviewer(conn, tid):
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'review_requested' "
        "ORDER BY id DESC LIMIT 1", (tid,),
    ).fetchone()
    return kb._json_dict(row["payload"]).get("reviewer")


def test_an_unnamed_reviewer_goes_to_the_board_default(kanban_home, monkeypatch):
    _cfg(monkeypatch, default_reviewer="opus-x")
    with kbc.connect() as conn:
        tid, claimed = _built_card(conn)
        assert kb.request_review(conn, tid, summary="PR #1137", expected_run_id=claimed.current_run_id)
        task = kb.get_task(conn, tid)
        assert task.status == "review"
        assert task.assignee == "opus-x"
        assert _review_requested_reviewer(conn, tid) == "opus-x"


def test_a_named_reviewer_still_wins(kanban_home, monkeypatch):
    _cfg(monkeypatch, default_reviewer="opus-x")
    with kbc.connect() as conn:
        tid, claimed = _built_card(conn)
        assert kb.request_review(
            conn, tid, summary="PR #1137", reviewer="opus-high", expected_run_id=claimed.current_run_id,
        )
        assert kb.get_task(conn, tid).assignee == "opus-high"
        assert _review_requested_reviewer(conn, tid) == "opus-high"


def test_no_default_keeps_the_card_with_its_builder(kanban_home, monkeypatch):
    _cfg(monkeypatch)
    with kbc.connect() as conn:
        tid, claimed = _built_card(conn)
        assert kb.request_review(conn, tid, summary="PR #1137", expected_run_id=claimed.current_run_id)
        assert kb.get_task(conn, tid).assignee == "sonnet-high"
        assert _review_requested_reviewer(conn, tid) is None


def test_a_re_review_keeps_the_earlier_rounds_reviewer(kanban_home, monkeypatch):
    """The default fills only a FIRST hand-off; a re-review goes back to whoever
    asked for the changes."""
    _cfg(monkeypatch, default_reviewer="opus-x", review_fix_cards=False)
    with kbc.connect() as conn:
        tid, claimed = _built_card(conn)
        assert kb.request_review(
            conn, tid, summary="v1", reviewer="opus-high", expected_run_id=claimed.current_run_id,
        )
        review = kb.claim_review_task(conn, tid)
        assert review is not None
        ok, _ = kb.request_changes(conn, tid, reason="fix it", expected_run_id=review.current_run_id)
        assert ok
        again = kb.claim_task(conn, tid)
        assert again is not None
        assert kb.request_review(conn, tid, summary="v2", expected_run_id=again.current_run_id)
        assert kb.get_task(conn, tid).assignee == "opus-high"


def test_the_default_is_read_from_the_board_owner_not_the_worker_profile(kanban_home, monkeypatch):
    """The hand-off runs inside the builder's worker, whose HERMES_HOME is its tier
    profile; the default lives in the config next to kanban.db."""
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  default_reviewer: opus-x\n", encoding="utf-8",
    )
    profile = kanban_home / "profiles" / "sonnet-high"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("model:\n  default: claude-sonnet-5\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(profile))
    assert kb.kanban_home() == kanban_home
    assert kb._default_reviewer() == "opus-x"
    with kbc.connect() as conn:
        tid, claimed = _built_card(conn)
        assert kb.request_review(conn, tid, summary="PR #1137", expected_run_id=claimed.current_run_id)
        assert kb.get_task(conn, tid).assignee == "opus-x"
