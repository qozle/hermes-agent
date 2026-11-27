"""Local commit #40: a kanban worker / delegated subagent only ever rides the seat pool.

Live case (2026-11-26, card t_98f53e96 run 1402, tier opus-high): the worker's claude_code
seat answered ``429 This request would exceed your account's rate limit`` and the turn waited
600 s twice while seat2 sat idle — no ``rotating`` line, no ``rotated to pool entry`` line.
A one-shot through the same gateway rotated 12 minutes later.

Two independent holes produced that silence, both covered here:

1. ``resolve_runtime_provider``'s Anthropic env rung (``ANTHROPIC_TOKEN`` /
   ``CLAUDE_CODE_OAUTH_TOKEN`` / ``ANTHROPIC_API_KEY`` / the borrowed ``~/.claude`` login)
   handed back a runtime with **no credential_pool**, so ``recover_with_credential_pool``
   returned at its first line (``pool is None``) without a word. The ladder reaches that rung
   whenever ``pool.select(model=…)`` misses for a moment (every seat benched for the model, a
   deferred token refresh), and the session then holds ONE static credential for its whole life.
2. ``_other_entry_available`` is strict about the failed identity: with no identifiable entry
   it answers False and ``_recover_rate_limit`` returns ``(False, True)`` — no rotation and no
   log — so a 429 on an unattributable key waits out the retry ladder even with a free seat.
"""

import time

import run_agent  # imported at collection: its dependency activation reads the real manifest.json, which the home-IO guard refuses inside a test

from agent.agent_runtime_helpers import _other_entry_available, recover_with_credential_pool
from agent.credential_pool import CredentialPool, PooledCredential

_BASE = "https://api.anthropic.com"
_MODEL = "claude-opus-5"
_CTX = {"message": "This request would exceed your account's rate limit. Please try again later."}


def _seat(entry_id, label, priority):
    return PooledCredential.from_dict("anthropic", {
        "id": entry_id, "label": label, "auth_type": "oauth", "priority": priority,
        "access_token": f"tok-{entry_id}", "refresh_token": f"rt-{entry_id}",
        "expires_at_ms": int((time.time() + 86400) * 1000), "base_url": _BASE, "source": "manual",
    })


def _two_seats():
    return CredentialPool(
        provider="anthropic",
        entries=[_seat("cc000000", "claude_code", 0), _seat("s2000000", "seat2", 1)],
    )


class _WorkerAgent:
    """A worker whose credential came from outside the pool: no entry id, unmatched key."""

    _fallback_activated = False
    provider = "anthropic"
    model = _MODEL
    base_url = _BASE

    def __init__(self, pool, api_key):
        self._credential_pool = pool
        self._credential_pool_entry_id = None
        self.api_key = api_key
        self.swaps = []

    def _swap_credential(self, entry):
        self.swaps.append(entry.id)
        self.api_key = entry.runtime_api_key
        self._credential_pool_entry_id = entry.id
        return True

    def _is_entitlement_failure(self, error_context, status_code):
        return False


def test_unidentified_seat_still_rotates_on_the_first_429():
    """The failed key matches no pool row (it came from the env rung) — rotate anyway."""
    pool = _two_seats()
    agent = _WorkerAgent(pool, "tok-from-the-environment")

    recovered, has_retried = recover_with_credential_pool(
        agent, status_code=429, has_retried_429=False, error_context=_CTX,
    )

    assert (recovered, has_retried) == (True, False), (
        "a 429 on an unattributable credential waited instead of rotating to the free seat"
    )
    assert agent.swaps, "no seat was swapped in"


def test_other_entry_available_falls_back_to_the_model_probe():
    """With no identifiable failed entry the probe asks 'can ANY seat serve this model?'."""
    pool = _two_seats()
    assert _other_entry_available(pool, None, _MODEL) is True

    for entry_id in ("cc000000", "s2000000"):
        pool.mark_exhausted_and_rotate(
            status_code=429, credential_id=entry_id, failure_reason="rate_limit",
            error_context=_CTX, model=_MODEL,
        )
    assert _other_entry_available(pool, None, _MODEL) is False, (
        "every seat is benched for this model; nothing to rotate to"
    )


def test_anthropic_env_rung_binds_the_seat_pool(monkeypatch):
    """The env/borrowed-login rung must still carry the pool, so the session can rotate."""
    from hermes_cli import runtime_provider as rp

    pool = _two_seats()
    monkeypatch.setattr(rp, "load_pool", lambda provider: pool if provider == "anthropic" else None)
    monkeypatch.setattr(rp, "_anthropic_token_or_raise", lambda *, model=None: "tok-from-the-environment")

    runtime = rp._anthropic_env_runtime("anthropic", {}, _MODEL)

    assert runtime.get("credential_pool") is pool, (
        "an Anthropic runtime resolved outside the pool cannot rotate: "
        "recover_with_credential_pool returns at 'pool is None'"
    )
    assert runtime["api_key"] == "tok-from-the-environment", "the rung's own credential must be kept"


def test_anthropic_env_rung_binds_no_pool_for_a_foreign_endpoint(monkeypatch):
    """An Azure Foundry Anthropic endpoint is not what the seats serve — bind nothing."""
    from hermes_cli import runtime_provider as rp

    pool = _two_seats()
    monkeypatch.setattr(rp, "load_pool", lambda provider: pool if provider == "anthropic" else None)
    monkeypatch.setattr(rp, "_azure_anthropic_env_key", lambda model_cfg: "azure-key")

    runtime = rp._anthropic_env_runtime(
        "anthropic", {"provider": "anthropic", "base_url": "https://x.services.ai.azure.com"}, _MODEL,
    )

    assert runtime.get("credential_pool") is None
