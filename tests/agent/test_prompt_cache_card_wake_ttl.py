"""LOCAL COMMIT (frankentrait): a kanban card-wake turn caches at 1h; every other turn keeps the
configured tier.

A chat subscribed to a chain of cards is woken every 5-30 minutes with nobody typing in between,
so on the 5m tier each ping re-wrote the whole chat into the cache (2026-11-20: 14 such rewrites
of 380K-430K tokens in one morning). ``build_turn_context`` stamps ``agent._card_wake_turn`` from
the turn's user message; both request sites read the tier through ``turn_cache_ttl``.

The wiring checks read the source files instead of importing them: on this host importing the
conversation loop reads the real home's manifest.json and trips ``tests/home_io_guard.py`` (the
upstream real-agent test in ``test_prompt_cache_ttl_auto.py`` fails the same way without this
change).
"""
from pathlib import Path
from types import SimpleNamespace

from agent.prompt_caching import CARD_WAKE_PREFIX, is_card_wake_message, turn_cache_ttl

REPO = Path(__file__).resolve().parents[2]
WAKE = ("[kanban] Task t_7e9411f4 handed off for review; the implementation is done.\n"
        "Title: Buy steps (4 of 12)\nAssignee: @sonnet-high\nBoard: default\n\nCheck the result or decide the next step.")


def test_wake_message_is_recognised():
    assert is_card_wake_message(WAKE)
    assert is_card_wake_message("  " + WAKE), "leading whitespace from a transport is tolerated"
    assert WAKE.startswith(CARD_WAKE_PREFIX)


def test_human_and_other_automation_turns_are_not_wakes():
    assert not is_card_wake_message("why did we use 40% of the 5 hour usage?")
    assert not is_card_wake_message("what does [kanban] Task t_1 mean in the log?")
    assert not is_card_wake_message("[cron] sys-health: disk at 91%")
    assert not is_card_wake_message([{"type": "text", "text": WAKE}]), "multimodal content is a human turn"
    assert not is_card_wake_message(None)


def test_wake_turn_uses_1h_and_human_turn_keeps_configured_tier():
    agent = SimpleNamespace(_cache_ttl="5m", _card_wake_turn=True)
    assert turn_cache_ttl(agent) == "1h"
    agent._card_wake_turn = False
    assert turn_cache_ttl(agent) == "5m"
    assert turn_cache_ttl(SimpleNamespace(_cache_ttl="5m")) == "5m", "an agent never stamped is a human turn"
    assert turn_cache_ttl(SimpleNamespace(_cache_ttl="1h", _card_wake_turn=False)) == "1h"


def test_disabled_caching_is_never_switched_on():
    assert turn_cache_ttl(SimpleNamespace(_cache_ttl=None, _card_wake_turn=True)) is None


def _flat(rel: str) -> str:
    return "".join((REPO / rel).read_text(encoding="utf-8").split())


def test_both_request_sites_read_the_turn_tier():
    """Contract for the daily re-apply onto upstream: a merge that restores ``agent._cache_ttl`` at a
    request site silently returns pings to 5m."""
    for rel in ("agent/conversation_loop.py", "agent/turn_request_assembly.py"):
        flat = _flat(rel)
        assert "effective_cache_ttl(turn_cache_ttl(agent),provider=agent.provider" in flat, rel
        assert "effective_cache_ttl(agent._cache_ttl" not in flat, rel


def test_turn_start_stamps_the_wake_flag():
    src = (REPO / "agent/turn_context.py").read_text(encoding="utf-8")
    start = src.index("def build_turn_context(")
    body = src[start:src.index("\ndef ", start + 1)]
    assert "agent._card_wake_turn = is_card_wake_message(user_message)" in body
