"""LOCAL COMMIT (frankentrait): a Kanban wake for a CLI/TUI ("local") conversation reaches
``admit_route_wake`` through the LocalSessionAdapter's authority. That adapter is built by
``session_local_recovery`` rather than ``GatewayRunner._create_adapter``, so it never carries
``gateway_runner``; before the fix ``deliver_wake`` skipped the route path and the wake died in
``handle_message`` ("no gateway message handler is installed on this adapter")."""
from types import SimpleNamespace

import pytest


def _local_source():
    from gateway.config import Platform
    from gateway.session import SessionSource
    return SessionSource(platform=Platform.LOCAL, chat_id="local-abc123", user_id="uid:1000", chat_type="dm")


class _LocalAdapter:
    """What the notifier holds for a local subscription: push-capable, no gateway_runner, an authority."""
    supports_async_delivery = True

    def __init__(self, runner):
        self.authority = SimpleNamespace(runner=runner)
        self.handled = []

    async def handle_message(self, event):
        self.handled.append(event)  # the adapter has no message handler: nothing is admitted


@pytest.mark.asyncio
async def test_local_adapter_wake_is_routed_through_its_authority_runner(monkeypatch):
    import gateway.session_automation as automation
    from gateway.wake import deliver_wake

    runner = object()
    adapter = _LocalAdapter(runner)
    seen = {}

    async def fake_admit_route_wake(got_runner, got_adapter, source, text, producer, metadata=None):
        seen.update(runner=got_runner, adapter=got_adapter, source=source, text=text, producer=producer)
        return True

    monkeypatch.setattr(automation, "admit_route_wake", fake_admit_route_wake)
    await deliver_wake(adapter, text="[kanban] card done", source=_local_source())
    assert seen["runner"] is runner and seen["adapter"] is adapter
    assert seen["producer"] == "kanban-wake" and seen["text"] == "[kanban] card done"
    assert adapter.handled == []  # admitted on the route FIFO, never a classic inbound message


@pytest.mark.asyncio
async def test_adapter_without_runner_or_authority_still_takes_the_classic_path(monkeypatch):
    import gateway.session_automation as automation
    from gateway.wake import WakeNotAccepted, deliver_wake

    class _Bare:
        supports_async_delivery = True

        def __init__(self):
            self.handled = []

        async def handle_message(self, event):
            self.handled.append(event)

    called = []

    async def fake_admit_route_wake(*args, **kwargs):
        called.append(args)
        return True

    monkeypatch.setattr(automation, "admit_route_wake", fake_admit_route_wake)
    adapter = _Bare()
    with pytest.raises(WakeNotAccepted):
        await deliver_wake(adapter, text="[kanban] card done", source=_local_source())
    assert called == [] and len(adapter.handled) == 1
