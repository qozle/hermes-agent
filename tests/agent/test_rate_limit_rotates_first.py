"""Local commit #39: a 429 on one seat rotates to a free seat before any wait.

Live case (2026-11-26): the claude_code seat 429'd ("This request would exceed your account's
rate limit") with a ~22 h Retry-After; the ladder retried the same seat after the 600 s cap
while seat2 sat idle. The retry-same-seat rung now runs only when no other entry can serve
the model.
"""

import time

import run_agent  # imported at collection: its dependency activation reads the real manifest.json, which the home-IO guard refuses inside a test

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.credential_pool import CredentialPool, PooledCredential

_BASE = "https://api.anthropic.com"
_MODEL = "claude-opus-5-5"
_CTX = {"message": "This request would exceed your account's rate limit. Please try again later."}


def _seat(entry_id, label, priority):
    return PooledCredential.from_dict("anthropic", {
        "id": entry_id, "label": label, "auth_type": "oauth", "priority": priority,
        "access_token": f"tok-{entry_id}", "refresh_token": f"rt-{entry_id}",
        "expires_at_ms": int((time.time() + 86400) * 1000), "base_url": _BASE, "source": "manual",
    })


class _Agent:
    _fallback_activated = False
    provider = "anthropic"
    model = _MODEL
    base_url = _BASE

    def __init__(self, pool):
        self._credential_pool = pool
        first = pool.select()
        self._credential_pool_entry_id = first.id
        self.api_key = first.runtime_api_key
        self.swaps = []

    def _swap_credential(self, entry):
        self.swaps.append(entry.id)
        self.api_key = entry.runtime_api_key
        self._credential_pool_entry_id = entry.id
        return True

    def _is_entitlement_failure(self, error_context, status_code):
        return False


def test_first_429_rotates_to_a_free_seat_without_waiting():
    pool = CredentialPool(provider="anthropic", entries=[_seat("cc000000", "claude_code", 0), _seat("s2000000", "seat2", 1)])
    agent = _Agent(pool)
    assert agent._credential_pool_entry_id == "cc000000"

    recovered, has_retried = recover_with_credential_pool(
        agent, status_code=429, has_retried_429=False, error_context=_CTX,
    )

    assert (recovered, has_retried) == (True, False), "first 429 waited on the throttled seat instead of rotating"
    assert agent.swaps == ["s2000000"]
    # The throttled seat is benched for this model only; a fresh select lands on seat2 too.
    assert pool.select(model=_MODEL).id == "s2000000"


def test_sole_seat_keeps_the_retry_once_rung():
    pool = CredentialPool(provider="anthropic", entries=[_seat("cc000000", "claude_code", 0)])
    agent = _Agent(pool)

    recovered, has_retried = recover_with_credential_pool(
        agent, status_code=429, has_retried_429=False, error_context=_CTX,
    )

    assert (recovered, has_retried) == (False, True)
    assert agent.swaps == []
    assert pool.select(model=_MODEL).id == "cc000000", "a sole seat was benched on its first 429"


def test_other_seat_benched_for_the_model_keeps_the_retry_once_rung():
    pool = CredentialPool(provider="anthropic", entries=[_seat("cc000000", "claude_code", 0), _seat("s2000000", "seat2", 1)])
    pool.mark_exhausted_and_rotate(
        status_code=429, credential_id="s2000000", failure_reason="rate_limit", error_context=_CTX, model=_MODEL,
    )
    agent = _Agent(pool)
    assert agent._credential_pool_entry_id == "cc000000"

    recovered, has_retried = recover_with_credential_pool(
        agent, status_code=429, has_retried_429=False, error_context=_CTX,
    )

    assert (recovered, has_retried) == (False, True)
    assert agent.swaps == []


def test_has_available_besides():
    pool = CredentialPool(provider="anthropic", entries=[_seat("cc000000", "claude_code", 0), _seat("s2000000", "seat2", 1)])
    assert pool.has_available_besides("cc000000", model=_MODEL) is True
    pool.mark_exhausted_and_rotate(
        status_code=429, credential_id="s2000000", failure_reason="rate_limit", error_context=_CTX, model=_MODEL,
    )
    assert pool.has_available_besides("cc000000", model=_MODEL) is False
    # Benched for one model only: another model still has seat2.
    assert pool.has_available_besides("cc000000", model="claude-sonnet-5") is True
