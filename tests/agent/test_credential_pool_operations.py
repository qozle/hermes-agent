"""Durable pool administration and selection invariants."""
import time
from dataclasses import replace

import pytest

from agent.credential_pool import CredentialPool, PooledCredential
from hermes_cli.auth import read_credential_pool, write_credential_pool


def _pool(provider="openrouter", *, exhausted=False):
    rows = [PooledCredential(
        provider=provider, id=f"row{i}", label=f"account{i}", source="manual",
        auth_type="api_key", access_token=f"fixture-{i}", priority=i,
        last_status="exhausted" if exhausted else None,
        last_status_at=time.time() if exhausted else None,
        last_error_code=429 if exhausted else None,
        last_error_reset_at=time.time() + 3600 if exhausted else None,
    ) for i in range(2)]
    write_credential_pool(provider, [e.to_dict() for e in rows])
    return CredentialPool(provider, rows)


def test_target_reset_preserves_sibling_cooldown():
    pool = _pool(exhausted=True)
    before = read_credential_pool(pool.provider)
    assert pool.reset_status("missing") is None
    assert read_credential_pool(pool.provider) == before
    assert pool.reset_status("row1").last_status is None
    after = {e["id"]: e for e in read_credential_pool(pool.provider)}
    assert after["row0"] == before[0]
    assert after["row1"].get("last_error_reset_at") is None
    assert pool.reset_statuses() == 1
    assert all(e.get("last_status") is None for e in read_credential_pool(pool.provider))


@pytest.mark.parametrize("strategy", ["fill_first", "round_robin", "random", "least_used"])
def test_selection_counts_only_returned_selections(strategy):
    pool = _pool()
    pool._strategy = strategy
    selected = [pool.select().id for _ in range(2)]
    assert {e.id: e.request_count for e in pool.entries()} == {
        e.id: selected.count(e.id) for e in pool.entries()}
    pool.reset_status("row0")  # existing persistence boundary, not a selection
    assert sum(e.get("request_count", 0) for e in read_credential_pool(pool.provider)) == 2
    pool._current_id = None
    pool.try_refresh_matching()  # API key is not refreshable; lookup must not count
    assert sum(e.request_count for e in pool.entries()) == 2
    assert pool.peek() is not None
    assert sum(e.request_count for e in pool.entries()) == 2


def test_priority_persists_contiguous_order_without_clearing_cooldown():
    pool = _pool(exhausted=True)
    before = {e.id: e.last_error_reset_at for e in pool.entries()}
    assert pool.move_entry("row1", -5).priority == 0
    assert [e["id"] for e in read_credential_pool(pool.provider)] == ["row1", "row0"]
    assert pool.move_entry("row1", 99).priority == 1
    assert [(e.id, e.priority) for e in pool.entries()] == [("row0", 0), ("row1", 1)]
    assert {e.id: e.last_error_reset_at for e in pool.entries()} == before
    snapshot = read_credential_pool(pool.provider)
    assert pool.move_entry("missing", 0) is None
    assert read_credential_pool(pool.provider) == snapshot


def test_priority_honors_anthropic_manual_first():
    # Without an explicit order the load-time rule still puts manual logins first.
    from agent.credential_pool import _normalize_pool_priorities

    pool = _pool("anthropic")
    entries = [replace(pool._entries[0], priority=1),
               replace(pool._entries[1], source="env:ANTHROPIC_API_KEY", priority=0)]
    assert _normalize_pool_priorities("anthropic", entries)
    assert sorted((e.priority, e.id) for e in entries) == [(0, "row0"), (1, "row1")]


def test_explicit_priority_puts_a_seeded_anthropic_login_first_and_survives_reload():
    # `hermes auth priority` is the user's order: it may put a seeded login (claude_code)
    # ahead of a manual one, and the load-time normalizer must not undo it.
    from agent.credential_pool import _normalize_pool_priorities

    pool = _pool("anthropic")
    pool._entries[1] = replace(pool._entries[1], source="claude_code")
    assert pool.move_entry("row1", 0).priority == 0
    assert [e.id for e in pool.entries()] == ["row1", "row0"]
    stored = read_credential_pool("anthropic")
    assert [e["id"] for e in stored] == ["row1", "row0"]
    assert [e.get("priority_pinned") for e in stored] == [1, 2]
    reloaded = [PooledCredential.from_dict("anthropic", e) for e in stored]
    assert not _normalize_pool_priorities("anthropic", reloaded)
    assert [e.id for e in sorted(reloaded, key=lambda e: e.priority)] == ["row1", "row0"]


def test_a_stale_process_rewriting_priorities_manual_first_is_undone_on_the_next_load():
    # A gateway, dashboard or terminal still running pre-#38 code re-sorts ``priority``
    # manual-first and persists it, carrying the extra keys through untouched.
    from agent.credential_pool import _normalize_pool_priorities

    pool = _pool("anthropic")
    pool._entries[1] = replace(pool._entries[1], source="claude_code")
    pool.move_entry("row1", 0)
    by_id = {e.id: e for e in pool.entries()}
    stale = [replace(by_id["row0"], priority=0), replace(by_id["row1"], priority=1)]
    assert _normalize_pool_priorities("anthropic", stale)
    assert [e.id for e in sorted(stale, key=lambda e: e.priority)] == ["row1", "row0"]


def test_a_legacy_true_pin_keeps_the_stored_priority_order():
    from agent.credential_pool import _normalize_pool_priorities

    pool = _pool("anthropic")
    rows = [replace(pool._entries[0], priority=1, extra={"priority_pinned": True}),
            replace(pool._entries[1], source="claude_code", priority=0,
                    extra={"priority_pinned": True})]
    assert not _normalize_pool_priorities("anthropic", rows)
    assert [e.id for e in sorted(rows, key=lambda e: e.priority)] == ["row1", "row0"]


def test_a_login_added_after_an_explicit_order_goes_after_it():
    from agent.credential_pool import _normalize_pool_priorities

    pool = _pool("anthropic")
    pool._entries[1] = replace(pool._entries[1], source="claude_code")
    pool.move_entry("row1", 0)
    newcomer = PooledCredential(
        provider="anthropic", id="row2", label="account2", source="manual",
        auth_type="api_key", access_token="fixture-2", priority=0,
    )
    entries = [*pool.entries(), newcomer]
    _normalize_pool_priorities("anthropic", entries)
    assert [e.id for e in sorted(entries, key=lambda e: e.priority)] == ["row1", "row0", "row2"]
