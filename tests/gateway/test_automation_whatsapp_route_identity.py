"""A WhatsApp completion's route identity must not depend on which JID spelling the bridge used.

The session-store origin a background completion is rebuilt from can carry the LID
(``999@lid``) in ``user_id`` while the committed human envelope carries the phone JID
(``6012@s.whatsapp.net``). Both key the same route; a raw dict compare refused every
async-delegation / watch completion on the DM with ``admission_conflict`` forever.
"""
import json

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_automation import _route_identity


def _mapped_home(tmp_path, monkeypatch, lid="999999999999999", phone="15551234567"):
    tmp_home = tmp_path / "hermes-home"
    mapping_dir = tmp_home / "platforms" / "whatsapp" / "session"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    (mapping_dir / f"lid-mapping-{lid}.json").write_text(
        json.dumps(f"{phone}@s.whatsapp.net"), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_home))


def _source(chat_id, user_id, **extra):
    return SessionSource(platform=Platform.WHATSAPP, chat_id=chat_id, user_id=user_id,
                         chat_type="dm", chat_name="Max", user_name="Max", **extra)


def test_lid_and_phone_jid_are_one_route_identity(tmp_path, monkeypatch):
    _mapped_home(tmp_path, monkeypatch)
    persisted_origin = _source("999999999999999@lid", "999999999999999@lid")
    committed_human = _source("999999999999999@lid", "15551234567@s.whatsapp.net",
                              message_id="ACC3117B9D179A69F5C6E1A409385E5C")
    assert _route_identity(persisted_origin) == _route_identity(committed_human)
    assert "message_id" not in _route_identity(committed_human)


def test_different_whatsapp_humans_stay_distinct(tmp_path, monkeypatch):
    _mapped_home(tmp_path, monkeypatch)
    a = _source("999999999999999@lid", "999999999999999@lid")
    b = _source("15559990000@s.whatsapp.net", "15559990000@s.whatsapp.net")
    assert _route_identity(a) != _route_identity(b)


def test_other_platforms_compare_raw():
    a = SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="u1", chat_type="dm")
    b = SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="u2", chat_type="dm")
    assert _route_identity(a) != _route_identity(b)
    assert _route_identity(a) == _route_identity(
        SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="u1", chat_type="dm",
                      message_id="m9"))
