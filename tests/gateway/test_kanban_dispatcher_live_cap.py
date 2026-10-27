"""The embedded dispatcher re-reads ``kanban.max_in_progress`` every tick.

Local commit (frankentrait): ``kanban.max_in_progress`` is the operator's live
throttle. The settings dataclass is built once at gateway start, so without a
per-tick re-resolve ``hermes config set kanban.max_in_progress N`` only applied
after a gateway cycle — a raised cap left ready cards unspawned while the
dispatcher logged "stuck" for 70+ ticks.
"""
from __future__ import annotations

from gateway import kanban_watchers_dispatcher as kwd


def _dispatcher(monkeypatch, tmp_path):
    settings = kwd._DispatcherSettings(
        interval=60.0, max_spawn=None, max_in_progress=1, failure_limit=2,
        stale_timeout_seconds=3600, reconcile_orphans=True,
        default_assignee=None, max_in_progress_per_profile=None,
    )
    d = kwd._KanbanDispatcher(kb=object(), settings=settings)
    monkeypatch.setattr(d, "board_db_fingerprint", lambda slug: (slug, 1, 1))
    monkeypatch.setattr(d, "_quarantine_lifted", lambda slug, fp: True)
    return d


def test_tick_passes_the_currently_configured_cap_not_the_startup_snapshot(monkeypatch, tmp_path):
    seen: dict = {}

    class _FakeConn:
        def close(self):
            pass

    class _FakeKbc:
        @staticmethod
        def connect(board=None):
            return _FakeConn()

    class _FakeKbd:
        @staticmethod
        def configured_max_in_progress():
            return 2  # operator raised it after the gateway started

        @staticmethod
        def resolve_max_in_progress(configured):
            return configured

        @staticmethod
        def configured_max_in_progress_per_lane():
            return {"ready": 1, "review": 1}  # per-lane caps are re-read the same way

        @staticmethod
        def dispatch_once(conn, board=None, **kwargs):
            seen.update(kwargs)
            return "ok"

    monkeypatch.setattr(kwd, "_kbc", lambda: _FakeKbc)
    monkeypatch.setattr(kwd, "_kbd", lambda: _FakeKbd)

    d = _dispatcher(monkeypatch, tmp_path)
    assert d.settings.max_in_progress == 1  # startup snapshot is stale on purpose
    assert d.tick_once_for_board("default") == "ok"
    assert seen["max_in_progress"] == 2
    assert seen["max_in_progress_per_lane"] == {"ready": 1, "review": 1}
