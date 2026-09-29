"""Kanban workers get a wrap-up instruction at 80% of the iteration budget, not the
generic "checkpoint and continue" notice — the heavy tail on a kanban card (PR -> merge
-> mirror -> report -> kanban_complete) is long enough that "continue" burns the
remaining budget on new work and the hard cap lands before kanban_complete ever runs.
Non-kanban sessions must be untouched (same 0.9 default, same generic template).
"""
from copy import deepcopy

import pytest
from tests.agent.test_iteration_budget_warning import _agent


def test_kanban_worker_gets_wrapup_notice_naming_kanban_complete_at_80_percent(
    tmp_path, monkeypatch
):
    from agent.turn_iteration_prep import prepare_iteration

    # No delegated_child_context / non_dispatcher_owned_context here: plain
    # HERMES_KANBAN_TASK plus a real kanban_complete tool is exactly the
    # dispatcher-owned-worker scope (test_kanban_budget_checkpoint's "owner" case).
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_wrapup")
    agent = _agent(tmp_path, monkeypatch, "null")  # no explicit ratio -> kanban default
    try:
        # max_iterations=10; consume 8 so budget.used/max_total == 0.8, which is >= the
        # NEW 0.8 kanban default (min(0.8*10, 9) == 8, so 8 < 8 is False -> fires) but
        # still BELOW the OLD 0.9 default (min(0.9*10, 9) == 9, so 8 < 9 is True -> would
        # NOT have fired) -- this budget size is the one that actually distinguishes the
        # two ratios, unlike max_iterations=4 where they coincide.
        agent.iteration_budget.max_total = 10
        for _ in range(8):
            agent.iteration_budget.consume()
        messages = [
            {"role": "user", "content": "work"},
            {"role": "assistant", "tool_calls": [
                {"id": "t", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t", "content": "verified artifact"},
        ]
        original = deepcopy(messages)
        prepare_iteration(agent, messages=messages, api_call_count=8)
        assert messages[:-1] == original[:-1]
        notice = messages[-1]["content"]
        assert notice != "verified artifact"
        assert "kanban_complete" in notice
        assert "Wrap up now" in notice
        assert "kanban_create" in notice
        # The old "do not stop solely because of this warning" / "continue the
        # task" phrasing must be gone from the kanban path.
        assert "do not stop solely because of this warning" not in notice
        assert "Do not start new work" in notice
    finally:
        agent._session_db.close()


def test_non_kanban_session_at_85_percent_gets_no_notice(tmp_path, monkeypatch):
    """No HERMES_KANBAN_TASK, no explicit ratio -> ratio stays None (the kanban-only
    default of 0.8 must not leak to ordinary sessions), so nothing is injected even
    well past the old 0.9 threshold."""
    from agent.turn_iteration_prep import prepare_iteration

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    agent = _agent(tmp_path, monkeypatch, "null")
    try:
        assert getattr(agent, "budget_warning_ratio", None) is None
        for _ in range(3):  # 3 of 4 == 0.75, well above the removed inline check point
            agent.iteration_budget.consume()
        messages = [
            {"role": "user", "content": "work"},
            {"role": "assistant", "tool_calls": [
                {"id": "t", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t", "content": "verified artifact"},
        ]
        prepare_iteration(agent, messages=messages, api_call_count=3)
        assert messages[-1]["content"] == "verified artifact"
    finally:
        agent._session_db.close()
