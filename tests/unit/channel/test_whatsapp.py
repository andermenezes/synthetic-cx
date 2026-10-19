"""Operational privacy and cancellation-safe lifetime for the WhatsApp channel."""

import asyncio
import os
from pathlib import Path
from typing import Any, cast

import pytest
from playwright.async_api import BrowserContext, Playwright

from syntheticcx.adapters.channel.whatsapp import (
    WhatsAppWebChannel,
    resolve_recipient,
    validate_browser_profile,
)
from syntheticcx.config.models import ExecutionLimits, WhatsAppChannelConfig
from syntheticcx.errors import ConfigurationError


def config() -> WhatsAppChannelConfig:
    return WhatsAppChannelConfig.model_validate(
        {"type": "whatsapp_web_playwright", "url": "https://service.example/"}
    )


def limits() -> ExecutionLimits:
    return ExecutionLimits(
        max_turns=5,
        journey_timeout_seconds=10,
        greeting_grace_seconds=0,
        browser_action_timeout_seconds=1,
        response_timeout_seconds=1,
        model_timeout_seconds=1,
        evaluator_timeout_seconds=1,
        persistence_timeout_seconds=1,
        close_timeout_seconds=2,
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not-a-phone",
        "1555",
        "155512345678901234",
        "+1 555 123 4567",
        "15551234567?token=private",
    ],
)
def test_recipient_rejected_without_disclosing_raw_value(monkeypatch, value):
    monkeypatch.setenv("SYNTHETICCX_WHATSAPP_RECIPIENT", value)
    with pytest.raises(ConfigurationError) as error:
        resolve_recipient(config())
    assert str(error.value) == "channel.config: a valid international recipient is required"


def test_protected_recipient_is_normalized(monkeypatch):
    monkeypatch.setenv("SYNTHETICCX_WHATSAPP_RECIPIENT", "+15551234567")
    assert resolve_recipient(config()) == "15551234567"


def test_explicit_recipient_overrides_environment_without_mutating_it(monkeypatch):
    monkeypatch.setenv("SYNTHETICCX_WHATSAPP_RECIPIENT", "15557654321")
    assert resolve_recipient(config(), "+15551234567") == "15551234567"
    assert os.environ["SYNTHETICCX_WHATSAPP_RECIPIENT"] == "15557654321"


@pytest.mark.parametrize("recipient", ["", "PRIVATE-INVALID-RECIPIENT"])
def test_invalid_explicit_recipient_never_falls_back_to_valid_environment(monkeypatch, recipient):
    monkeypatch.setenv("SYNTHETICCX_WHATSAPP_RECIPIENT", "15551234567")
    with pytest.raises(ConfigurationError) as caught:
        resolve_recipient(config(), recipient)
    if recipient:
        assert recipient not in str(caught.value)


def test_recipient_fallback_uses_the_guide_configured_environment(monkeypatch):
    monkeypatch.setenv("SYNTHETICCX_WHATSAPP_RECIPIENT", "15557654321")
    monkeypatch.setenv("BUSINESS_WHATSAPP_NUMBER", "+15551234567")
    configured = config().model_copy(
        update={"recipient_environment_variable": "BUSINESS_WHATSAPP_NUMBER"}
    )
    assert resolve_recipient(configured) == "15551234567"


def test_profile_is_existing_dedicated_root_and_reusable(tmp_path):
    with pytest.raises(ConfigurationError):
        validate_browser_profile(tmp_path / "does-not-exist")
    (tmp_path / "Default").mkdir()
    (tmp_path / "Local State").write_text("{}")
    assert validate_browser_profile(tmp_path) == tmp_path.resolve()
    with pytest.raises(ConfigurationError):
        validate_browser_profile(tmp_path / "Default")
    normal = Path.home() / "Library/Application Support/Google/Chrome"
    # Existing canonical personal profiles fail closed. No user directory is created.
    if normal.exists():
        with pytest.raises(ConfigurationError):
            validate_browser_profile(normal)


def test_cancelled_close_does_not_release_profile_before_owned_context_finishes(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SYNTHETICCX_WHATSAPP_RECIPIENT", "15551234567")

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        entered, release = asyncio.Event(), asyncio.Event()
        channel._acquire_profile()

        class SlowContext:
            calls = 0

            async def close(self):
                self.calls += 1
                entered.set()
                await release.wait()

        class Driver:
            calls = 0

            async def stop(self):
                self.calls += 1

        context, driver = SlowContext(), Driver()
        channel._context = cast(BrowserContext, context)
        channel._playwright = cast(Playwright, driver)
        task = asyncio.create_task(channel.close())
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert channel._profile_lock is not None
        release.set()
        await channel.close()
        await channel.close()
        assert context.calls == driver.calls == 1
        assert channel._profile_lock is None
        assert tmp_path.is_dir()

    asyncio.run(exercise())


def test_cancelled_start_cleans_owned_driver_and_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNTHETICCX_WHATSAPP_RECIPIENT", "15551234567")

    async def exercise():
        entered = asyncio.Event()

        class Driver:
            calls = 0

            @property
            def chromium(self):
                return self

            async def launch_persistent_context(self, *args, **kwargs):
                entered.set()
                await asyncio.Event().wait()

            async def stop(self):
                self.calls += 1

        driver = Driver()

        class Factory:
            async def start(self):
                return cast(Any, driver)

        monkeypatch.setattr("syntheticcx.adapters.channel.whatsapp.async_playwright", Factory)
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        task = asyncio.create_task(channel.start())
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert driver.calls == 1
        assert channel._profile_lock is None

    asyncio.run(exercise())
