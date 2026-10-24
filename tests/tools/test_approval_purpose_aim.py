"""The approval prompt says what a command/script is FOR (``Aim: …``), not only why it was flagged.

Both the terminal tool and execute_code carry a ``purpose`` argument; the guard renders it first
in the prompt's reason. Max's standing order (2026-10-24): no approval card without the aim.
"""
from tools import approval
from tools.code_execution_tool import EXECUTE_CODE_SCHEMA, build_execute_code_schema
from tools.terminal_tool import TERMINAL_SCHEMA


def test_with_purpose_prefixes_aim_when_set():
    token = approval.set_command_purpose("Reorder the open cards so small fixes run first.")
    try:
        text = approval.with_purpose("flagged: database write")
    finally:
        approval.reset_command_purpose(token)
    assert text == "Aim: Reorder the open cards so small fixes run first. Flagged: flagged: database write"


def test_with_purpose_is_plain_description_without_purpose():
    assert approval.with_purpose("flagged: x") == "flagged: x"


def test_execute_code_guard_description_carries_aim(monkeypatch):
    captured = {}

    def fake_human_decision(gate, **kw):
        captured.update(kw)
        return {"approved": True}

    monkeypatch.setattr(approval, "_human_decision", fake_human_decision)
    monkeypatch.setattr(approval, "_should_skip_container_guards", lambda *a, **k: False)
    monkeypatch.setattr(approval, "_yolo_active", lambda: False)
    monkeypatch.setattr(approval.approval_context, "_get_approval_mode", lambda: "ask")
    monkeypatch.setattr(approval, "_presence", lambda: (None, False, True, False))
    monkeypatch.setattr(approval, "_unattended_contexts", lambda: [])
    monkeypatch.setattr(approval, "_is_local_viewer_turn", lambda: False)
    monkeypatch.setattr(approval, "get_current_session_key", lambda: "s1")
    monkeypatch.setattr(approval, "is_approved", lambda *a, **k: False)

    token = approval.set_command_purpose("Reprioritize the kanban board")
    try:
        approval.check_execute_code_guard("print(1)", "local")
    finally:
        approval.reset_command_purpose(token)
    assert captured["description"].startswith("Aim: Reprioritize the kanban board. Flagged: ")


def test_both_tool_schemas_expose_purpose():
    assert "purpose" in TERMINAL_SCHEMA["parameters"]["properties"]
    assert "purpose" in EXECUTE_CODE_SCHEMA["parameters"]["properties"]
    assert "purpose" in build_execute_code_schema()["parameters"]["properties"]
