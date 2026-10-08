"""Completion polls release temporary native references on every outcome."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from syntheticcx.adapters.channel import broker as module
from syntheticcx.config.models import LocatorConfig, ResponseCompletionConfig


@pytest.mark.parametrize("fresh", [False, True])
def test_completion_marker_handles_are_disposed_even_on_early_return(monkeypatch, fresh):
    async def exercise():
        nodes = [
            SimpleNamespace(
                is_visible=AsyncMock(return_value=True),
                evaluate=AsyncMock(return_value=not fresh),
                dispose=AsyncMock(),
            )
            for _ in range(2)
        ]
        marker = SimpleNamespace(element_handles=AsyncMock(return_value=nodes))
        monkeypatch.setattr(module, "locate", lambda *_args: marker)
        broker = module.NativeChatBroker("http://localhost:8765/")
        broker.owned = SimpleNamespace(locator=object())
        broker._completion = ResponseCompletionConfig(
            strategy="completion_locator", locator=LocatorConfig(strategy="css", value="#done")
        )
        assert await broker._completion_ready(0) is fresh
        for node in nodes:
            node.dispose.assert_awaited_once()

    asyncio.run(exercise())
