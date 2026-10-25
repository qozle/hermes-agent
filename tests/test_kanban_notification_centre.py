"""Tests for the kanban -> notification centre hook (card t_325f9028).

Two kanban events move to Max's dashboard notification centre and stop being
WhatsApp pings: `done` (yellow) and a `blocked` carrying kind `needs_input`
(red). Everything else must keep going out over the subscription's platform.

The property worth defending hardest is the failure mode: a claim that could
not post must return False so the old platform ping still happens. An outage
of the new destination must not lose a notification.
"""
from __future__ import annotations

import json
import urllib.error
from types import SimpleNamespace
from unittest import mock

import pytest

from gateway import kanban_notification_centre as centre


def ev(event_kind, **payload):
    # NB: the positional is `event_kind`, not `kind` -- the payload of a
    # blocked event has its OWN `kind` ("needs_input"), and the two collide.
    return SimpleNamespace(kind=event_kind, payload=payload or {}, id=1)


def task(title="Wire every source"):
    return SimpleNamespace(title=title, status="done")


@pytest.fixture(autouse=True)
def _configured(monkeypatch):
    """A token is the on-switch; give every test one unless it says otherwise."""
    monkeypatch.setenv("NOTIFY_TOKEN", "test-token")
    monkeypatch.delenv("KANBAN_NOTIFY_CENTRE", raising=False)
    monkeypatch.setenv("NOTIFY_URL", "http://dash.example")


# ── classify: exactly two kinds, and no more ────────────────────────────────

def test_done_is_yellow():
    assert centre.classify(ev("completed")) == ("yellow", "Done")


def test_a_needs_input_block_is_red():
    assert centre.classify(ev("blocked", kind="needs_input")) == ("red", "Needs you")


@pytest.mark.parametrize("e", [
    ev("blocked", kind="dependency"),
    ev("blocked", kind="capability"),
    ev("blocked", kind="transient"),
    ev("blocked"),                      # no kind at all
    ev("crashed"),
    ev("timed_out"),
    ev("gave_up"),
    ev("review_requested"),
    ev("changes_requested"),
    ev("block_loop_detected", kind="needs_input"),
    ev("status", status="blocked"),
    ev("archived"),
    ev("unblocked"),
])
def test_every_other_event_falls_through_to_the_platform_ping(e):
    """An event swallowed by a misfiring classifier is one Max never sees."""
    assert centre.classify(e) is None


# ── build ───────────────────────────────────────────────────────────────────

def test_a_done_notification_leads_with_the_card_title():
    level, title, body = centre.build(ev("completed", summary="Shipped it"),
                                      task("Notification centre part 2"), "t_abc")
    assert level == "yellow"
    assert title == "Done: Notification centre part 2"
    assert "Shipped it" in body
    assert "t_abc" in body


def test_a_done_notification_names_its_attachments():
    _l, _t, body = centre.build(
        ev("completed", summary="s",
           artifacts=["/home/maxterd/.hermes/kanban/attachments/t_abc/report.pdf",
                      "/var/x/chart.png"]),
        task(), "t_abc")
    assert "Attached: report.pdf, chart.png" in body


def test_artifacts_nested_under_metadata_are_found_too():
    _l, _t, body = centre.build(
        ev("completed", metadata={"artifacts": ["/x/only.png"]}), task(), "t_abc")
    assert "only.png" in body


def test_attachment_paths_are_never_passed_off_as_links():
    """A host path is not a URL; the body names the file, the link stays empty."""
    _l, _t, body = centre.build(
        ev("completed", artifacts=["/home/maxterd/secret/path/report.pdf"]),
        task(), "t_abc")
    assert "/home/maxterd" not in body


def test_a_blocked_notification_carries_the_reason():
    level, title, body = centre.build(
        ev("blocked", kind="needs_input", reason="Which broker account?"),
        task("Pick an account"), "t_xyz")
    assert level == "red"
    assert title == "Needs you: Pick an account"
    assert "Which broker account?" in body


def test_a_blocked_notification_without_a_reason_still_says_something_useful():
    _l, _t, body = centre.build(ev("blocked", kind="needs_input"), task(), "t_xyz")
    assert "blocked on your input" in body


def test_a_titleless_task_falls_back_to_the_card_id():
    _l, title, _b = centre.build(ev("completed"), SimpleNamespace(title=""), "t_abc")
    assert title == "Done: t_abc"


def test_a_runaway_summary_is_clipped():
    _l, _t, body = centre.build(ev("completed", summary="x" * 5000), task(), "t_abc")
    assert len(body) < 1500
    assert "\u2026" in body


# ── enabled ─────────────────────────────────────────────────────────────────

def test_no_token_means_the_centre_was_never_configured(monkeypatch, tmp_path):
    monkeypatch.delenv("NOTIFY_TOKEN", raising=False)
    monkeypatch.setenv("NOTIFY_ENV", str(tmp_path / "absent.env"))
    assert centre.enabled() is False


def test_the_token_can_come_from_the_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("NOTIFY_TOKEN", raising=False)
    f = tmp_path / "notify.env"
    f.write_text('NOTIFY_TOKEN="from-file"\n# comment\nOTHER=1\n')
    monkeypatch.setenv("NOTIFY_ENV", str(f))
    assert centre.resolve_token() == "from-file"
    assert centre.enabled() is True


def test_an_explicit_off_switch_wins_over_a_present_token(monkeypatch):
    monkeypatch.setenv("KANBAN_NOTIFY_CENTRE", "off")
    assert centre.enabled() is False


def test_the_env_file_follows_hermes_home_and_not_the_real_one(monkeypatch, tmp_path):
    """A profile gateway (and the test harness) must not read the real ~/.hermes."""
    monkeypatch.delenv("NOTIFY_ENV", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "isolated"))
    assert centre._env_file() == tmp_path / "isolated" / "notify.env"


# ── claim: the failure contract ─────────────────────────────────────────────

def test_a_claimed_event_posts_and_suppresses_the_platform_ping():
    with mock.patch.object(centre, "post", return_value={"id": 7}) as p:
        assert centre.claim(ev("completed", summary="s"), task(), "t_abc") is True
    assert p.call_count == 1
    assert p.call_args.args[0] == "yellow"
    assert p.call_args.args[1] == "kanban"


def test_an_unclaimed_event_does_not_post_at_all():
    with mock.patch.object(centre, "post") as p:
        assert centre.claim(ev("crashed"), task(), "t_abc") is False
    p.assert_not_called()


def test_an_unconfigured_centre_claims_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv("NOTIFY_TOKEN", raising=False)
    monkeypatch.setenv("NOTIFY_ENV", str(tmp_path / "absent.env"))
    with mock.patch.object(centre, "post") as p:
        assert centre.claim(ev("completed"), task(), "t_abc") is False
    p.assert_not_called()


@pytest.mark.parametrize("boom", [
    urllib.error.HTTPError("u", 500, "err", {}, None),
    urllib.error.URLError("dashboard down"),
    TimeoutError("slow"),
    ValueError("bad json"),
])
def test_a_failed_post_is_not_a_claim_so_the_old_ping_still_happens(boom):
    """The whole point: an outage degrades to WhatsApp, it does not lose the event."""
    with mock.patch.object(centre, "post", side_effect=boom):
        assert centre.claim(ev("completed", summary="s"), task(), "t_abc") is False


def test_claim_never_raises_into_the_notifier():
    with mock.patch.object(centre, "post", side_effect=RuntimeError("anything")):
        assert centre.claim(ev("blocked", kind="needs_input"), task(), "t_x") is False


# ── post: wire format ───────────────────────────────────────────────────────

def test_post_sends_the_token_as_a_header_and_never_on_the_url():
    captured = {}

    class FakeResp:
        def read(self): return b'{"id": 3}'
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.data.decode())
        return FakeResp()

    with mock.patch.object(centre.urllib.request, "urlopen", fake_urlopen):
        out = centre.post("red", "kanban", "T", "B")

    assert out == {"id": 3}
    assert captured["url"] == "http://dash.example/dashboard/api/notifications"
    assert captured["headers"]["X-notify-token"] == "test-token"
    assert "test-token" not in captured["url"]
    assert captured["body"] == {"level": "red", "source": "kanban",
                                "title": "T", "body": "B", "link": None}
