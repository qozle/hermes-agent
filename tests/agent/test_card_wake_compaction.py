"""Card-wake turn compaction: a kanban ping over 150K tokens compacts before any model
call, focused on this chat's own open cards (id + title, kept word for word).

On Nov 19-20, pings into 250K-470K token chats were 18% of all usage; compacting costs
~60 cents, one ping into a 440K-token chat costs ~$2 (see
``agent/turn_context_compaction.py::CARD_WAKE_COMPACT_THRESHOLD_TOKENS``). This builds on
the card-wake gate from ``agent/prompt_caching.py::is_card_wake_message`` (the 1h cache-tier
PR): a typed (non-wake) turn never sets ``agent._card_wake_turn`` and so never hits this rule,
regardless of size.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent import turn_context as _tc
from agent.prompt_caching import is_card_wake_message
from agent.turn_context_compaction import (
    CARD_WAKE_COMPACT_THRESHOLD_TOKENS,
    _card_wake_focus_topic,
    run_turn_start_compaction,
)

PING = ("[kanban] Task t_1 handed off for review; the implementation is done.\n"
        "Title: Buy steps (4 of 12)\nAssignee: @sonnet-high\nBoard: default\n\n"
        "Check the result or decide the next step.")


def _agent(**kw):
    compressor = SimpleNamespace(
        protect_first_n=3, protect_last_n=3, threshold_tokens=999_000_000,
        context_length=2_000_000_000, summary_target_ratio=0.5,
    )
    base = dict(
        compression_enabled=True, context_compressor=compressor, session_id="s1",
        model="m", platform="telegram", _chat_id="42", _thread_id=None,
        _clear_context_overflow_warn=MagicMock(),
        _compress_context=MagicMock(return_value=([{"role": "user", "content": "summary"}], "sys")),
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _run(agent, tokens, user_message, monkeypatch):
    """Drive ``run_turn_start_compaction`` the way ``build_turn_context`` does: stamp the
    wake flag from the message, then pin the token estimate so the test controls the gate.

    ``_reanchor`` is stubbed: on this host, ``reanchor_current_turn_user_idx`` pulls in
    ``agent.conversation_loop`` -> ``hermes_bootstrap``, which reads the REAL home's
    manifest.json and trips ``tests/home_io_guard.py`` regardless of ``HERMES_HOME``
    (``tests/agent/test_prompt_cache_ttl_auto.py::test_real_agent_resolves_auto_from_its_platform``
    fails the same way with no changes in this tree at all -- a pre-existing environment gap,
    not something this change introduces).
    """
    monkeypatch.setattr(_tc, "_preflight_request_tokens", lambda *a, **k: tokens)
    monkeypatch.setattr("agent.turn_context_compaction._reanchor", lambda *a, **k: 0)
    agent._card_wake_turn = is_card_wake_message(user_message)
    msgs = [{"role": "user", "content": "hi"}]
    out = run_turn_start_compaction(
        agent, messages=msgs, system_message="sys", active_system_prompt="sys",
        conversation_history=None, current_turn_user_idx=0, user_message=user_message,
        effective_task_id="t",
    )
    return out, msgs


class TestCardWakeCompaction:
    def test_ping_over_threshold_compacts_first_with_open_cards_named(self, tmp_path, monkeypatch):
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_notify as kbn

        monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
        conn = kbc.connect()
        try:
            t1 = kb.create_task(conn, title="Buy steps", assignee="sonnet-high")
            t2 = kb.create_task(conn, title="Sell steps", assignee="sonnet-high")
            for tid in (t1, t2):
                kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="42",
                                   chat_type="dm", delivery_mode="notify+wake")
        finally:
            conn.close()

        agent = _agent()
        out, msgs = _run(agent, 200_000, PING, monkeypatch)
        assert out.compressed is True
        assert out.messages == [{"role": "user", "content": "summary"}]
        agent._compress_context.assert_called_once()
        call = agent._compress_context.call_args
        assert call.args[0] is msgs, "card-wake compaction ran first, against the ORIGINAL list"
        assert "force" not in call.kwargs, "not forced: honors the same automatic gate as idle compaction"
        assert call.kwargs["trigger"] == "card_wake"
        focus = call.kwargs["focus_topic"]
        assert f"{t1} (Buy steps)" in focus
        assert f"{t2} (Sell steps)" in focus
        assert "word for word" in focus

    def test_ping_at_or_under_threshold_does_not_compact(self, monkeypatch):
        agent = _agent()
        out, _ = _run(agent, 100_000, PING, monkeypatch)
        assert out.compressed is False
        agent._compress_context.assert_not_called()

        agent2 = _agent()
        out2, _ = _run(agent2, CARD_WAKE_COMPACT_THRESHOLD_TOKENS, PING, monkeypatch)
        assert out2.compressed is False, "exactly at the threshold is not OVER it"
        agent2._compress_context.assert_not_called()

    def test_typed_message_into_a_big_chat_is_not_compacted_by_this_rule(self, monkeypatch):
        agent = _agent()
        out, _ = _run(agent, 200_000, "please check the dashboard", monkeypatch)
        assert agent._card_wake_turn is False
        assert out.compressed is False
        agent._compress_context.assert_not_called()

    def test_disabled_compression_never_forces_a_card_wake_compaction(self, monkeypatch):
        agent = _agent(compression_enabled=False)
        out, _ = _run(agent, 200_000, PING, monkeypatch)
        assert out.compressed is False
        agent._compress_context.assert_not_called()


class TestCardWakeFocusTopic:
    def test_names_every_open_card_by_id_and_title(self, monkeypatch):
        monkeypatch.setattr(
            "agent.turn_context_compaction._card_wake_open_cards",
            lambda a: [("t_1", "Buy steps"), ("t_2", "")],
        )
        focus = _card_wake_focus_topic(SimpleNamespace())
        assert "t_1 (Buy steps)" in focus
        assert "t_2" in focus and "t_2 (" not in focus, "no title falls back to the bare id"
        assert "word for word" in focus

    def test_generic_ask_when_no_open_card_resolves(self, monkeypatch):
        monkeypatch.setattr("agent.turn_context_compaction._card_wake_open_cards", lambda a: [])
        focus = _card_wake_focus_topic(SimpleNamespace())
        assert "word for word" in focus
        assert "kanban card pings" in focus


class TestCardWakeOpenCardsFromBoard:
    """``_card_wake_open_cards`` against a real (temp) board: the exact query this chat's
    focus topic is built from."""

    def _board(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_notify as kbn
        return kb, kbc, kbn

    def test_open_cards_only_this_chat_id_and_title(self, tmp_path, monkeypatch):
        from agent.turn_context_compaction import _card_wake_open_cards
        kb, kbc, kbn = self._board(tmp_path, monkeypatch)

        conn = kbc.connect()
        try:
            open_id = kb.create_task(conn, title="Buy steps (4 of 12)", assignee="sonnet-high")
            blocked_id = kb.create_task(conn, title="Sell steps", assignee="sonnet-high")
            done_id = kb.create_task(conn, title="Finished card", assignee="sonnet-high")
            other_chat_id = kb.create_task(conn, title="Someone else's card", assignee="sonnet-high")
            kb.block_task(conn, blocked_id, reason="waiting on approval")
            kb.complete_task(conn, done_id, summary="shipped")
            for tid in (open_id, blocked_id, done_id):
                kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="42",
                                   chat_type="dm", delivery_mode="notify+wake")
            kbn.add_notify_sub(conn, task_id=other_chat_id, platform="telegram", chat_id="99",
                               chat_type="dm", delivery_mode="notify+wake")
        finally:
            conn.close()

        agent = SimpleNamespace(platform="telegram", _chat_id="42", _thread_id=None)
        cards = _card_wake_open_cards(agent)
        assert {cid for cid, _ in cards} == {open_id, blocked_id}
        assert dict(cards)[open_id] == "Buy steps (4 of 12)"
        # A different platform/chat_id (unsubscribed here) sees nothing.
        assert _card_wake_open_cards(SimpleNamespace(platform="telegram", _chat_id="99", _thread_id=None)) == \
            [(other_chat_id, "Someone else's card")]

    def test_no_gateway_chat_identity_resolves_nothing(self, tmp_path, monkeypatch):
        from agent.turn_context_compaction import _card_wake_open_cards
        self._board(tmp_path, monkeypatch)
        assert _card_wake_open_cards(SimpleNamespace(platform=None, _chat_id=None, _thread_id=None)) == []
        assert _card_wake_open_cards(
            SimpleNamespace(platform="cli", _chat_id=None, _thread_id=None, session_id=None)
        ) == []

    def test_missing_board_file_resolves_nothing(self, tmp_path, monkeypatch):
        from agent.turn_context_compaction import _card_wake_open_cards
        monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "does-not-exist.db"))
        agent = SimpleNamespace(platform="telegram", _chat_id="42", _thread_id=None)
        assert _card_wake_open_cards(agent) == []

    def test_cli_agent_matches_its_local_subscription_by_session_id(self, tmp_path, monkeypatch):
        """Terminal chats never set ``_chat_id``; their subscription is stored as platform
        ``local`` with ``chat_id`` equal to the chat's ``session_id``. A cli agent must be
        matched on ``session_id`` against that shape, with or without a (stale/unset) ``_chat_id``."""
        from agent.turn_context_compaction import _card_wake_open_cards
        kb, kbc, kbn = self._board(tmp_path, monkeypatch)
        sid = "local-ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12ab12"

        conn = kbc.connect()
        try:
            tid = kb.create_task(conn, title="Notifications (8 of 8)", assignee="sonnet-high")
            kbn.add_notify_sub(conn, task_id=tid, platform="local", chat_id=sid,
                                chat_type="dm", delivery_mode="notify+wake")
        finally:
            conn.close()

        agent_with_stale_chat_id = SimpleNamespace(
            platform="cli", _chat_id=sid, _thread_id=None, session_id=sid
        )
        agent_without_chat_id = SimpleNamespace(
            platform="cli", _chat_id=None, _thread_id=None, session_id=sid
        )
        assert _card_wake_open_cards(agent_with_stale_chat_id) == [(tid, "Notifications (8 of 8)")]
        assert _card_wake_open_cards(agent_without_chat_id) == [(tid, "Notifications (8 of 8)")]
