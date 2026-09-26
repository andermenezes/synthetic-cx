"""Cancellation of a close caller must not abandon browser cleanup."""

import asyncio
from pathlib import Path
from typing import cast

from playwright.async_api import Browser, BrowserContext

from syntheticcx.adapters.channel.playwright import PlaywrightChannel
from syntheticcx.config.loader import load_config


def test_cancelled_close_continues_and_is_idempotent() -> None:
    async def exercise() -> None:
        config = load_config(Path("examples/telecom/scenario.yaml"))
        channel = PlaywrightChannel(config.channel, config.execution)
        entered = asyncio.Event()
        release = asyncio.Event()

        class SlowContext:
            calls = 0

            async def close(self) -> None:
                self.calls += 1
                entered.set()
                await release.wait()

        class BrowserSpy:
            calls = 0

            async def close(self) -> None:
                self.calls += 1

        context, browser = SlowContext(), BrowserSpy()
        channel._context = cast(BrowserContext, context)
        channel._browser = cast(Browser, browser)
        closing = asyncio.create_task(channel.close())
        await asyncio.wait_for(entered.wait(), timeout=1)
        closing.cancel()
        try:
            await closing
        except asyncio.CancelledError:
            pass
        release.set()
        await asyncio.wait_for(channel.close(), timeout=1)
        await channel.close()
        assert context.calls == browser.calls == 1

    asyncio.run(exercise())
