"""LOCAL COMMIT (frankentrait): a kanban worker whose owner link drops AFTER admission reconnects
and reads the receipt again instead of exiting 1.

The owner's managed worker watches the submitter's pid and ends the attempt when it dies, so a
poller that exits on a transport failure (a keepalive ping timeout while the gateway's event loop
stalls under load, as on 2026-11-08) throws away the whole run. Before admission the old rules
hold: a non-transient refusal fails at once, a draining owner is waited for, nothing is submitted
twice. After admission an owner REFUSAL (``GatewayRPCError``) still fails — only transport loss
is retried, and only inside ``RECONNECT_WINDOW_S``.
"""

import asyncio
from contextlib import asynccontextmanager

import pytest

from hermes_cli import gateway_client, kanban_worker_client


def _owner(plan, log):
    """``plan`` is one entry per connection attempt: a list of actions for that link's ``rpc``
    calls in order — a dict is returned, an exception instance is raised; ``"refuse"`` makes the
    connect itself fail as a dropped transport."""
    @asynccontextmanager
    async def connect(**kwargs):
        log.append(("connect", kwargs.get("ping_timeout")))
        actions = plan.pop(0)
        if actions == "refuse":
            raise gateway_client.GatewayClientError("Gateway connection failed; no local fallback")

        class _Client:
            events = asyncio.Queue()

            async def rpc(self, method, **params):
                log.append((method, params.get("admission_id")))
                action = actions.pop(0)
                if isinstance(action, Exception):
                    raise action
                return action

        yield _Client()

    return connect


ADMITTED = {"session_id": "s1", "receipt": {"status": "running", "admission_id": "a1"}}
RUNNING = {"status": "running", "admission_id": "a1"}
TERMINAL = {"status": "terminal", "admission_id": "a1"}
DROPPED = gateway_client.GatewayClientError("Gateway disconnected; turn outcome is unknown.")


def _settle(monkeypatch):
    from gateway import session_kanban
    monkeypatch.setattr(session_kanban, "worker_result", lambda *_: {"exit_code": 0, "last_output": "card closed"})
    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _no_sleep)
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)


@pytest.mark.asyncio
async def test_a_link_dropped_after_admission_is_reconnected_and_the_receipt_read_again(monkeypatch, tmp_path):
    log = []
    plan = [
        [ADMITTED, RUNNING, DROPPED],   # link 1: admitted, one poll, then the socket dies
        "refuse",                       # link 2: the owner is still unreachable for a moment
        [RUNNING, TERMINAL],            # link 3: polling resumes on the SAME admission id
    ]
    monkeypatch.setattr(gateway_client, "connect_gateway", _owner(plan, log))
    _settle(monkeypatch)

    assert await kanban_worker_client.run({"task_id": "t1"}, str(tmp_path / "kanban.db")) == 0
    assert [m for m, _ in log if m == "kanban.run"] == ["kanban.run"]          # admitted exactly once
    assert [m for m, _ in log if m == "connect"] == ["connect"] * 3
    receipts = [a for m, a in log if m == "prompt.receipt"]
    assert receipts == ["a1"] * 4                 # two polls on link 1, two on link 3, same admission
    # The worker client asks for the wider keepalive window on every link.
    assert all(t == kanban_worker_client.RECEIPT_PING_TIMEOUT_S for m, t in log if m == "connect")


@pytest.mark.asyncio
async def test_an_owner_refusal_after_admission_still_fails(monkeypatch, tmp_path):
    log = []
    refusal = gateway_client.GatewayRPCError("unknown_admission")
    plan = [[ADMITTED, refusal]]
    monkeypatch.setattr(gateway_client, "connect_gateway", _owner(plan, log))
    _settle(monkeypatch)

    with pytest.raises(gateway_client.GatewayRPCError):
        await kanban_worker_client.run({"task_id": "t1"}, str(tmp_path / "kanban.db"))
    assert [m for m, _ in log if m == "connect"] == ["connect"]


@pytest.mark.asyncio
async def test_a_drop_before_admission_still_fails_at_once(monkeypatch, tmp_path):
    log = []
    plan = [[DROPPED], [ADMITTED, TERMINAL]]
    monkeypatch.setattr(gateway_client, "connect_gateway", _owner(plan, log))
    _settle(monkeypatch)

    with pytest.raises(gateway_client.GatewayClientError):
        await kanban_worker_client.run({"task_id": "t1"}, str(tmp_path / "kanban.db"))
    assert [m for m, _ in log if m == "connect"] == ["connect"]   # no second connection, no second claim


@pytest.mark.asyncio
async def test_reconnecting_gives_up_once_the_window_has_passed(monkeypatch, tmp_path):
    log = []
    clock = {"t": 0.0}
    step = kanban_worker_client.RECONNECT_WINDOW_S / 2 + 1.0

    @asynccontextmanager
    async def connect(**kwargs):
        log.append(("connect", kwargs.get("ping_timeout")))
        if len(log) == 1:
            class _Client:
                events = asyncio.Queue()

                async def rpc(self, method, **params):
                    if method == "kanban.run":
                        return ADMITTED
                    raise DROPPED
            yield _Client()
            return
        clock["t"] += step   # every failed reconnect attempt costs half the window
        raise gateway_client.GatewayClientError("Gateway connection failed; no local fallback")

    monkeypatch.setattr(gateway_client, "connect_gateway", connect)
    monkeypatch.setattr(kanban_worker_client.time, "monotonic", lambda: clock["t"])
    _settle(monkeypatch)

    with pytest.raises(gateway_client.GatewayClientError):
        await kanban_worker_client.run({"task_id": "t1"}, str(tmp_path / "kanban.db"))
    connects = len([m for m, _ in log if m == "connect"])
    assert 3 <= connects <= 4   # the admitted link, then reconnects until the window is spent


async def _no_sleep(_seconds):
    return None
