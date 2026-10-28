"""LOCAL COMMIT (frankentrait): kanban worker approvals routed to the coordinator.

``approvals.kanban_mode: coordinator`` turns a dispatched worker's flagged
command into an ``approval_requested`` event on its card (the gateway notifier
wakes the coordinator with it); the worker waits for ``approval_decided`` with
the same token, written by ``hermes kanban approve|deny``. No answer = denied.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from tools import approval_kanban as ak


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _cfg(monkeypatch, **approvals):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"approvals": approvals})


def _running_card(conn):
    tid = kb.create_task(conn, title="restart the dev dashboard", assignee="sonnet-high")
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    return tid


def test_request_and_approve_round_trip(kanban_home, monkeypatch):
    _cfg(monkeypatch, kanban_mode="coordinator", timeout=60)
    with kbc.connect() as conn:
        tid = _running_card(conn)
        token = ak.post_request(conn, tid, command="sudo systemctl restart etc-dashboard-dev",
                                description="Aim: serve the new template", pattern_key="service restart",
                                profile="sonnet-high")
        ev = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'approval_requested'", (tid,),
        ).fetchall()
        assert len(ev) == 1
        payload = kb._json_dict(ev[0][0])
        assert payload["token"] == token and "restart etc-dashboard-dev" in payload["command"]
        assert payload["aim"].startswith("Aim:")
        assert ak.read_decision(conn, tid, token) is None
        # coordinator answers
        assert ak.post_decision(conn, tid, token, decision="approve", note="go", author="default")
        # answering twice or with an unknown token is refused
        assert not ak.post_decision(conn, tid, token, decision="deny", note=None, author="default")
        assert not ak.post_decision(conn, tid, "nope", decision="approve", note=None, author="default")
        assert ak.read_decision(conn, tid, token)["decision"] == "approve"


def test_wait_returns_the_decision_and_heartbeats(kanban_home, monkeypatch):
    _cfg(monkeypatch, kanban_mode="coordinator")
    with kbc.connect() as conn:
        tid = _running_card(conn)
        token = ak.post_request(conn, tid, command="rm -rf build", description="clean",
                                pattern_key="rm -rf", profile=None)
        beats = []
        polls = {"n": 0}

        def sleep(_s):
            polls["n"] += 1
            if polls["n"] == 2:
                ak.post_decision(conn, tid, token, decision="deny", note="not that dir", author="default")

        clock = [0.0]

        def mono():
            clock[0] += 1.0
            return clock[0]

        decision, note = ak.wait_for_decision(
            conn, tid, token, timeout_s=100, heartbeat=lambda: beats.append(1), sleep=sleep, clock=mono)
        assert (decision, note) == ("deny", "not that dir")
        assert len(beats) == 2


def test_wait_times_out_as_timeout(kanban_home, monkeypatch):
    _cfg(monkeypatch, kanban_mode="coordinator")
    with kbc.connect() as conn:
        tid = _running_card(conn)
        token = ak.post_request(conn, tid, command="x", description="y", pattern_key="z", profile=None)
        clock = [0.0]

        def mono():
            clock[0] += 50.0
            return clock[0]

        assert ak.wait_for_decision(conn, tid, token, timeout_s=100, sleep=lambda s: None, clock=mono) \
            == ("timeout", "")


def test_gate_relays_only_for_a_worker_in_coordinator_mode(kanban_home, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    _cfg(monkeypatch, kanban_mode="coordinator")
    assert not ak.coordinator_relay_active()
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc")
    assert ak.coordinator_relay_active()
    _cfg(monkeypatch, kanban_mode="deny")
    assert not ak.coordinator_relay_active()


def test_terminal_guard_uses_the_relay_for_a_dangerous_command(kanban_home, monkeypatch):
    """The worker's shell gate: a dangerous command goes to the coordinator and
    the gate returns approved / blocked from the answer; a harmless command
    never asks."""
    from tools import approval
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc")
    _cfg(monkeypatch, kanban_mode="coordinator")
    monkeypatch.setattr(approval, "_presence", lambda cb=None: (None, False, False, False))
    asked = []

    def fake_decision(*, command, description, pattern_key):
        asked.append((command, pattern_key))
        return ("approve", "ok") if "restart" in command else ("deny", "no")

    monkeypatch.setattr(ak, "coordinator_decision", fake_decision)
    ok = approval.check_all_command_guards("sudo systemctl restart etc-dashboard-dev", "host")
    assert ok["approved"] is True
    bad = approval.check_all_command_guards("rm -rf ~/build", "host")
    assert bad["approved"] is False and "denied" in bad["message"]
    assert approval.check_all_command_guards("ls -la", "host")["approved"] is True
    assert len(asked) == 2


def test_notifier_formats_the_approval_ping(kanban_home):
    from types import SimpleNamespace
    from gateway import kanban_watchers_notifier as kwn
    assert "approval_requested" in kwn.TERMINAL_KINDS and "approval_requested" in kwn._WAKE_KINDS
    ev = SimpleNamespace(kind="approval_requested", payload={
        "token": "ab12", "aim": "serve the new template", "command": "sudo systemctl restart x"})
    n = SimpleNamespace(head="[kanban] t_abc", task_id="t_abc", title="x", board_tag="", task=None)
    msg, handoff, detail = kwn._EVENT_FORMATTERS["approval_requested"](ev, n)
    assert "ab12" in msg and "t_abc" in msg
    assert handoff and "ab12" in handoff
    assert detail is None
    assert kwn.diagnostic_event(ev) is False
