"""Native channel browser mode is an explicit runtime choice."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml

from syntheticcx.adapters.channel import broker as broker_module
from syntheticcx.adapters.channel.dynamic import DynamicPlaywrightChannel
from syntheticcx.config.models import ExecutionLimits, InterfaceChannelConfig
from syntheticcx.errors import ChannelError


@pytest.mark.parametrize("headed", [False, True])
def test_native_browser_launch_uses_selected_mode(
    monkeypatch: pytest.MonkeyPatch, headed: bool
) -> None:
    launches: list[bool] = []

    class Page:
        async def goto(self, *_args: Any, **_kwargs: Any) -> Any:
            return type("Response", (), {"status": 403})()

    class Context:
        def set_default_timeout(self, _timeout: float) -> None:
            pass

        def set_default_navigation_timeout(self, _timeout: float) -> None:
            pass

        async def new_page(self) -> Page:
            return Page()

    class Browser:
        async def new_context(self) -> Context:
            return Context()

    class Chromium:
        async def launch(self, *, headless: bool) -> Browser:
            launches.append(headless)
            return Browser()

    class Playwright:
        chromium = Chromium()

    class Manager:
        async def start(self) -> Playwright:
            return Playwright()

    monkeypatch.setattr(broker_module, "async_playwright", Manager)
    source = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    channel = DynamicPlaywrightChannel(
        InterfaceChannelConfig.model_validate(source["channel"]),
        ExecutionLimits.model_validate(source["execution"]),
        headed=headed,
    )
    with pytest.raises(ChannelError, match="native target navigation failed"):
        asyncio.run(channel.start())
    assert launches == [not headed]
