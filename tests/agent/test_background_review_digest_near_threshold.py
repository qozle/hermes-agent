"""A same-model review fork near the PARENT chat's own compression threshold must replay the
bounded digest (``_digest_history``), not the full snapshot — exactly like the already-existing
routed (different-model) path. Without this, a review spawned while the live chat sits near its
compression trigger replays the full near-threshold snapshot into the fork, which immediately
trips the fork's OWN in-place compaction (``_detach_fork_compression``) on a copy that is thrown
away by design: tokens and wall-clock are spent compressing something nobody keeps, while the
live chat that actually needed shrinking stays exactly as big.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import run_agent
import agent.background_review as bg_review
from agent.background_review import (
    _chat_near_compression_threshold,
    _digest_history,
)
from agent.model_metadata import estimate_messages_tokens_rough
from agent.prompt_cache_scope import resolve_prompt_cache_scope
from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# _chat_near_compression_threshold: pure predicate
# ---------------------------------------------------------------------------


def _compressor(threshold_tokens, last_prompt_tokens):
    return SimpleNamespace(threshold_tokens=threshold_tokens, last_prompt_tokens=last_prompt_tokens)


def test_near_threshold_true_at_and_above_the_fraction():
    agent = SimpleNamespace(context_compressor=_compressor(100_000, 80_000))
    assert _chat_near_compression_threshold(agent) is True  # exactly 80%

    agent = SimpleNamespace(context_compressor=_compressor(100_000, 85_000))
    assert _chat_near_compression_threshold(agent) is True

    agent = SimpleNamespace(context_compressor=_compressor(100_000, 100_000))
    assert _chat_near_compression_threshold(agent) is True


def test_near_threshold_false_below_the_fraction():
    agent = SimpleNamespace(context_compressor=_compressor(100_000, 79_999))
    assert _chat_near_compression_threshold(agent) is False

    agent = SimpleNamespace(context_compressor=_compressor(100_000, 50_000))
    assert _chat_near_compression_threshold(agent) is False


def test_near_threshold_false_on_missing_or_invalid_compressor_state():
    # No context_compressor attribute at all (bare test stub / fresh agent).
    assert _chat_near_compression_threshold(SimpleNamespace()) is False
    # Compressor present but counters not yet populated.
    assert _chat_near_compression_threshold(SimpleNamespace(context_compressor=_compressor(0, 0))) is False
    assert _chat_near_compression_threshold(SimpleNamespace(context_compressor=_compressor(100_000, 0))) is False
    assert _chat_near_compression_threshold(SimpleNamespace(context_compressor=_compressor(100_000, -1))) is False
    assert _chat_near_compression_threshold(SimpleNamespace(context_compressor=_compressor(None, 90_000))) is False
    # bool must not pass isinstance(int) checks as a real count/threshold.
    assert _chat_near_compression_threshold(SimpleNamespace(context_compressor=_compressor(True, 90_000))) is False
    assert _chat_near_compression_threshold(SimpleNamespace(context_compressor=_compressor(100_000, True))) is False


def test_near_threshold_custom_fraction_override():
    agent = SimpleNamespace(context_compressor=_compressor(100_000, 60_000))
    assert _chat_near_compression_threshold(agent, fraction=0.5) is True
    assert _chat_near_compression_threshold(agent, fraction=0.7) is False


# ---------------------------------------------------------------------------
# End-to-end wiring: _spawn_background_review -> the fork's run_conversation call
# ---------------------------------------------------------------------------


def _make_agent_stub(agent_cls, *, threshold_tokens, last_prompt_tokens):
    """Minimal AIAgent-like stub carrying everything build_cache_parity_fork/_run_review_fork
    reads, plus a parent context_compressor reporting the given threshold/usage pair."""
    agent = object.__new__(agent_cls)
    agent.model = "test-model"
    agent.platform = "test"
    agent.provider = "openai"
    agent.base_url = ""
    agent.api_key = ""
    agent.api_mode = ""
    agent.session_id = "sess-123"
    agent.quiet_mode = True
    agent._parent_session_id = ""
    agent._credential_pool = None
    agent._memory_store = None
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._memory_nudge_interval = 5
    agent._skill_nudge_interval = 5
    agent.background_review_callback = None
    agent.status_callback = None
    agent._safe_print = lambda *_a, **_k: None
    agent._cached_system_prompt = "PARENT-SYSTEM-PROMPT"
    agent.ephemeral_system_prompt = ""
    import datetime as _dt

    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    agent.enabled_toolsets = None
    agent.disabled_toolsets = None
    agent.reasoning_config = None
    agent.prefill_messages = None
    agent.providers_allowed = None
    agent.providers_ignored = None
    agent.providers_order = None
    agent.provider_sort = None
    agent.provider_require_parameters = None
    agent.provider_data_collection = None
    agent.request_overrides = {}
    agent.max_tokens = None
    agent.acp_command = None
    agent.acp_args = []
    agent._warned_bg_review_routing = False
    agent._warned_bg_review_reasoning_effort = False
    agent._emit_warning = lambda *_a, **_k: None
    agent.client = object()
    agent.context_compressor = _compressor(threshold_tokens, last_prompt_tokens)
    return agent


class _SyncThread:
    def __init__(self, *, target=None, daemon=None, name=None):
        self._target = target

    def start(self):
        if self._target:
            self._target()


def _make_history_capturing_recorder(captured):
    class _Recorder:
        def __init__(self, *args, **kwargs):
            self._cached_system_prompt = None
            self._memory_write_origin = None
            self._memory_write_context = None
            self._memory_store = None
            self._memory_enabled = None
            self._user_profile_enabled = None
            self._memory_nudge_interval = None
            self._skill_nudge_interval = None
            self.suppress_status_output = None
            self.session_start = None
            self.session_id = None
            self.tools = None
            self.valid_tool_names = set()
            self._tool_snapshot_generation = 0
            self._session_messages = []

        def run_conversation(self, **kwargs):
            captured["conversation_history"] = kwargs.get("conversation_history")
            captured["user_message"] = kwargs.get("user_message")

        def release_clients(self):
            pass

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    return _Recorder


def _big_messages_snapshot(n: int = 30) -> list[dict]:
    """A long alternating transcript — stands in for the 468k-token buy-steps chat. Few turns,
    each with a large tool-output-shaped body: real near-threshold chats are long because of a
    handful of huge messages (tool dumps), not thousands of tiny ones — _digest_history's
    per-message preview cap (300/200 chars) only pays off against that realistic shape."""
    msgs = []
    for i in range(n):
        msgs.append({"role": "user", "content": f"user turn {i}: " + ("lorem ipsum " * 400)})
        msgs.append({"role": "assistant", "content": f"assistant turn {i}: " + ("dolor sit amet " * 400)})
    return msgs


def test_same_model_review_fork_gets_full_snapshot_at_50_percent_of_parent_threshold():
    """Below the 80% line, nothing changes: the fork still replays the full chat (today's
    warm-cache-read behavior)."""
    agent = _make_agent_stub(run_agent.AIAgent, threshold_tokens=100_000, last_prompt_tokens=50_000)
    snapshot = _big_messages_snapshot()
    captured = {}
    _Recorder = _make_history_capturing_recorder(captured)

    with patch.object(run_agent, "AIAgent", _Recorder), patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(messages_snapshot=snapshot, review_memory=True, review_skills=False)

    assert "conversation_history" in captured, "run_conversation was never invoked"
    assert captured["conversation_history"] == snapshot


# ---------------------------------------------------------------------------
# Real fork, real conversation loop, real compression gates (only ``compress`` stubbed): at
# 85% of the parent's threshold the fork's FIRST request must carry the digest and the fork
# must never compress it — replacing a recorder-based proxy (asserted only history selection,
# never ran a fork) and a non-fork token-estimate proxy (never ran ``_run_review_in_thread`` at
# all) with one test that drives the real code path end to end.
# ---------------------------------------------------------------------------


def _tool_call_response(prompt_tokens: int) -> SimpleNamespace:
    message = SimpleNamespace(
        content=None, reasoning_content=None, reasoning=None,
        tool_calls=[SimpleNamespace(
            id="call_1", type="function",
            function=SimpleNamespace(name="skill_view", arguments='{"name": "big-skill"}'),
        )],
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")], model="test/model",
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=1, total_tokens=prompt_tokens + 1),
    )


def _final_response(prompt_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=SimpleNamespace(
            role="assistant", content="Nothing to save.", tool_calls=None, reasoning_content=None,
        ), finish_reason="stop")], model="test/model",
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=5, total_tokens=prompt_tokens + 5),
    )


def _run_real_review_fork(tmp_path: Path, last_prompt_tokens: int) -> dict:
    """Drive the REAL ``_run_review_in_thread`` -> ``run_conversation`` with real compression
    gates; only ``compress`` (the aux-LLM call) is stubbed. Adapted from
    ``test_compression_concurrent_fork.py::test_review_fork_compacts_oversized_snapshot_in_memory``."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent-sid", source="cli")
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        parent = run_agent.AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
            quiet_mode=True, session_db=db, session_id="parent-sid",
            skip_context_files=True, skip_memory=True,
        )
    parent._cached_system_prompt = "stable parent prompt"
    parent.context_compressor.threshold_tokens = 100_000
    parent.context_compressor.last_prompt_tokens = last_prompt_tokens
    # ~84K rough tokens of transcript: the parent really is at ~85% of its own 100K trigger.
    snapshot = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " + "y" * 5_600}
        for i in range(60)
    ]
    captured = {"sent": []}
    real_run_conversation = run_agent.AIAgent.run_conversation

    def _wrapped(self, *args, **kwargs):
        compressor = self.context_compressor
        compressor.threshold_tokens = 100_000  # same model -> same trigger as the parent
        compressor.compress = MagicMock(return_value=[
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "ok"},
        ])
        compressor.select_context = MagicMock(return_value=None)
        self._compression_feasibility_checked = True
        self._disable_streaming, self._use_prompt_caching = True, False
        self.client = MagicMock()
        calls = {"n": 0}

        def _create(**create_kwargs):
            messages = create_kwargs.get("messages") or []
            captured["sent"].append(messages)
            calls["n"] += 1
            prompt_tokens = estimate_messages_tokens_rough(messages)
            return _tool_call_response(prompt_tokens) if calls["n"] == 1 else _final_response(prompt_tokens)

        self.client.chat.completions.create.side_effect = _create

        def _execute_tool_calls(assistant_message, messages, *_args):
            tool_call = assistant_message.tool_calls[0]
            # A 20K-token skill body: what pushes a near-threshold replay over the line.
            messages.append({
                "role": "tool", "name": tool_call.function.name, "tool_call_id": tool_call.id,
                "content": "z" * 80_000,
            })

        self._execute_tool_calls = _execute_tool_calls
        try:
            return real_run_conversation(self, *args, **kwargs)
        finally:
            captured["compress_calls"] = compressor.compress.call_count

    try:
        with patch.object(run_agent.AIAgent, "run_conversation", _wrapped):
            bg_review._run_review_in_thread(parent, snapshot, "review this conversation")
    finally:
        db.close()
    first_request = captured["sent"][0] if captured["sent"] else []
    captured["first_has_digest"] = any(
        "[Earlier conversation digest" in str(message.get("content", "")) for message in first_request
    )
    captured["first_tokens"] = estimate_messages_tokens_rough(first_request)
    captured["requests"] = len(captured["sent"])
    return captured


def test_r2_real_fork_at_85_percent_gets_digest_and_never_compresses(tmp_path):
    """Card step 4, the untested half: at 85% of the parent's threshold the review fork's
    first request must carry the digest AND the fork must never compress it. On the base
    (full-snapshot replay) this same scenario compresses once after the tool round."""
    captured = _run_real_review_fork(tmp_path, 85_000)
    assert captured["requests"] == 2
    assert captured["first_has_digest"]
    assert captured["compress_calls"] == 0


def test_r2_control_real_fork_at_50_percent_replays_full_chat_and_compresses(tmp_path):
    """Below the 80% line, the same real-fork harness still replays the full chat — proving the
    85% assertions above are not just an artifact of the harness."""
    captured = _run_real_review_fork(tmp_path, 50_000)
    assert not captured["first_has_digest"]
    assert captured["compress_calls"] == 1


# ---------------------------------------------------------------------------
# R1 — cache scope of the fork's FIRST request on a slot-keyed (xAI) route: a same-model fork
# whose request #1 is the digest diverges from the parent's prompt right after the system
# prompt, and must not ride the parent's xAI slot key until its own (possibly-never-reached,
# for a disposable digest fork) compaction commits.
# ---------------------------------------------------------------------------


def _make_scope_capturing_recorder(captured):
    class _Recorder:
        def __init__(self, *args, **kwargs):
            self.provider = kwargs.get("provider")
            self.model = kwargs.get("model")
            self.base_url = kwargs.get("base_url") or ""
            self._session_db = None
            self.context_compressor = SimpleNamespace(compression_count=0)
            self._cached_system_prompt = None
            self._memory_write_origin = None
            self._memory_write_context = None
            self._memory_store = None
            self._memory_enabled = None
            self._user_profile_enabled = None
            self._memory_nudge_interval = None
            self._skill_nudge_interval = None
            self.suppress_status_output = None
            self.session_start = None
            self.session_id = None
            self.tools = None
            self.valid_tool_names = set()
            self._tool_snapshot_generation = 0
            self._session_messages = []

        def run_conversation(self, **kwargs):
            captured["conversation_history"] = kwargs.get("conversation_history")
            captured["scope"] = resolve_prompt_cache_scope(self)

        def release_clients(self):
            pass

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    return _Recorder


def test_r1_xai_digest_fork_must_not_ride_the_parents_slot():
    """At >=80%, a same-model xAI/Grok fork's request #1 is the digest — a stream that already
    diverges from the parent's prompt right after the system prompt. It must resolve its own
    ``<scope>::review`` slot-keyed cache scope from request #1, not the parent's slot key (which
    it would otherwise evict, forcing the parent's next turn to re-prefill the whole chat cold)."""
    agent = _make_agent_stub(run_agent.AIAgent, threshold_tokens=100_000, last_prompt_tokens=85_000)
    agent.provider, agent.model = "xai-oauth", "grok-4.3"
    snapshot = _big_messages_snapshot()
    captured = {}
    _Recorder = _make_scope_capturing_recorder(captured)

    with patch.object(run_agent, "AIAgent", _Recorder), patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(messages_snapshot=snapshot, review_memory=True, review_skills=False)

    parent_scope = resolve_prompt_cache_scope(agent)
    assert captured["conversation_history"] != snapshot, "fixture bug: the fork must actually get the digest"
    assert captured["scope"] != parent_scope, (
        "a same-model fork whose request #1 replays the digest (diverged from the parent's prompt) "
        "must not send the parent's xAI slot key, or it evicts the parent's warm slot"
    )


def test_r1c_anthropic_digest_fork_shares_scope_harmlessly():
    """Control: Anthropic's content-addressed cache makes a shared scope harmless even when the
    fork's first request is the digest — this fix must not touch non-slot-keyed routes."""
    agent = _make_agent_stub(run_agent.AIAgent, threshold_tokens=100_000, last_prompt_tokens=85_000)
    agent.provider, agent.model = "anthropic", "claude-opus-4-8"
    snapshot = _big_messages_snapshot()
    captured = {}
    _Recorder = _make_scope_capturing_recorder(captured)

    with patch.object(run_agent, "AIAgent", _Recorder), patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(messages_snapshot=snapshot, review_memory=True, review_skills=False)

    assert captured["conversation_history"] != snapshot
    assert captured["scope"] == resolve_prompt_cache_scope(agent)


def test_routed_path_unchanged_uses_digest_regardless_of_parent_threshold():
    """Routing to a different model already forces the digest independent of the parent's
    threshold state — this fix must not touch that existing behavior."""
    # Parent chat nowhere near its threshold (10%): routed path must still get the digest.
    agent = _make_agent_stub(run_agent.AIAgent, threshold_tokens=100_000, last_prompt_tokens=10_000)
    snapshot = _big_messages_snapshot()
    captured = {}
    _Recorder = _make_history_capturing_recorder(captured)

    routed_runtime = {
        "provider": "openrouter", "model": "aux-cheap-model", "api_key": "test-key",
        "base_url": None, "api_mode": None, "credential_pool": None, "request_overrides": {},
        "max_tokens": None, "command": None, "args": [], "routed": True,
    }

    with patch.object(run_agent, "AIAgent", _Recorder), \
         patch.object(bg_review, "_resolve_review_runtime", return_value=routed_runtime), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(messages_snapshot=snapshot, review_memory=True, review_skills=False)

    assert captured["conversation_history"] == _digest_history(snapshot)
