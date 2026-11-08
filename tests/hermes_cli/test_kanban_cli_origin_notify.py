"""LOCAL COMMIT (frankentrait): ``hermes kanban create`` subscribes the conversation it was run
from (the terminal tool's subprocess inherits ``HERMES_SESSION_*``), and falls back to
``kanban.default_notify`` only from a bare shell — so a card opened from a terminal chat pings
that chat, one opened from WhatsApp pings WhatsApp, and nothing created inside a worker
subscribes itself."""
import argparse
import json

import pytest

HOME_CHAT = {
    "platform": "whatsapp", "chat_id": "home@lid", "chat_type": "dm",
    "user_id": "owner@s.whatsapp.net", "notifier_profile": "default", "delivery_mode": "notify+wake",
}

SESSION_VARS = (
    "HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID", "HERMES_SESSION_CHAT_TYPE",
    "HERMES_SESSION_USER_ID", "HERMES_SESSION_USER_ID_ALT", "HERMES_SESSION_THREAD_ID",
    "HERMES_SESSION_PROFILE", "HERMES_SESSION_KEY", "HERMES_SESSION_ID", "HERMES_KANBAN_TASK",
)


@pytest.fixture
def board(tmp_path, monkeypatch):
    """A fresh board under a temp HERMES_HOME whose config carries the home-chat fallback,
    with every session variable cleared (a bare shell)."""
    import yaml
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for var in SESSION_VARS:
        monkeypatch.delenv(var, raising=False)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"kanban": {"default_notify": dict(HOME_CHAT)}}))
    kb.init_db()
    return tmp_path


def _create(monkeypatch, capsys, title="card"):
    from hermes_cli.kanban import kanban_command
    from hermes_cli.kanban_parser import build_parser

    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers())
    assert kanban_command(parser.parse_args(["kanban", "create", title, "--json"])) == 0
    return json.loads(capsys.readouterr().out)["id"]


def _subs(tid):
    from hermes_cli import kanban_db_connect as kbc, kanban_db_notify as kn

    with kbc.connect_closing() as conn:
        return kn.list_notify_subs(conn, tid)


def _in_conversation(monkeypatch, platform, chat_id, user_id):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", platform)
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", chat_id)
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "dm")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", user_id)
    monkeypatch.setenv("HERMES_SESSION_ID", chat_id)


def test_terminal_conversation_card_pings_that_conversation_not_the_home_chat(board, monkeypatch, capsys):
    _in_conversation(monkeypatch, "local", "local-abc123", "uid:1000")
    subs = _subs(_create(monkeypatch, capsys))
    assert [(s["platform"], s["chat_id"]) for s in subs] == [("local", "local-abc123")]
    sub = subs[0]
    assert sub["chat_type"] == "dm" and sub["user_id"] == "uid:1000"
    assert sub["delivery_mode"] == "notify+wake" and sub["notifier_profile"] == "default"


def test_whatsapp_conversation_card_pings_whatsapp_as_before(board, monkeypatch, capsys):
    _in_conversation(monkeypatch, "whatsapp", "home@lid", "owner@s.whatsapp.net")
    subs = _subs(_create(monkeypatch, capsys))
    assert len(subs) == 1
    assert (subs[0]["platform"], subs[0]["chat_id"], subs[0]["delivery_mode"]) == ("whatsapp", "home@lid", "notify+wake")


def test_bare_shell_card_falls_back_to_the_home_chat(board, monkeypatch, capsys):
    subs = _subs(_create(monkeypatch, capsys))
    assert [(s["platform"], s["chat_id"]) for s in subs] == [("whatsapp", "home@lid")]


def test_worker_created_card_subscribes_nothing_on_its_own(board, monkeypatch, capsys):
    _in_conversation(monkeypatch, "local", "local-abc123", "uid:1000")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_notarealcreator")
    assert _subs(_create(monkeypatch, capsys)) == []


def test_auto_subscribe_switched_off_keeps_only_the_home_chat_fallback(board, monkeypatch, capsys):
    import yaml

    (board / "config.yaml").write_text(yaml.safe_dump(
        {"kanban": {"default_notify": dict(HOME_CHAT), "auto_subscribe_on_create": False}}))
    _in_conversation(monkeypatch, "local", "local-abc123", "uid:1000")
    subs = _subs(_create(monkeypatch, capsys))
    assert [(s["platform"], s["chat_id"]) for s in subs] == [("whatsapp", "home@lid")]
