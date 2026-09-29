"""Verify candidate controls with a fresh native Playwright browser."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import urlsplit

from syntheticcx.adapters.channel.playwright import PlaywrightChannel
from syntheticcx.config.models import ChannelGuide, ExecutionLimits, GuideValidation
from syntheticcx.discovery.explore import DiscoveryCandidate
from syntheticcx.errors import ChannelError
from syntheticcx.language import Language, require_language, second_probe


def _origin(url: str) -> str:
    parsed = urlsplit(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _require_target_origin(channel: PlaywrightChannel, expected: str) -> None:
    actual = channel.current_url
    if actual is None or _origin(actual) != expected:
        raise ChannelError("discover navigation", "browser left the selected target origin")


async def verify_candidate(
    candidate: DiscoveryCandidate,
    limits: ExecutionLimits,
    probe_message: str,
    *,
    headed: bool = False,
    language: Language = "en-US",
    on_progress: Callable[[str], None] | None = None,
) -> ChannelGuide:
    """Exercise the precise candidate that later runs will consume."""
    if not probe_message.strip():
        raise ChannelError("discover probe", "synthetic probe cannot be blank")
    language = require_language(language)
    channel = (
        PlaywrightChannel(candidate.channel, limits, headed=True)
        if headed
        else PlaywrightChannel(candidate.channel, limits)
    )
    failed: BaseException | None = None
    try:
        if on_progress is not None:
            mode = "headed" if headed else "headless"
            on_progress(f"starting fresh native Playwright verification browser ({mode})")
        # Navigation, opener activation, and input resolution each have an
        # action timeout; do not cancel the whole startup before later stages.
        await asyncio.wait_for(
            channel.start(on_progress=on_progress) if on_progress is not None else channel.start(),
            3 * limits.browser_action_timeout_seconds,
        )
        expected_origin = _origin(str(candidate.channel.url))
        _require_target_origin(channel, expected_origin)
        if limits.greeting_grace_seconds > 0:
            if on_progress is not None:
                on_progress("waiting briefly for an optional assistant greeting")
            try:
                await asyncio.wait_for(
                    channel.receive(on_progress=on_progress)
                    if on_progress is not None
                    else channel.receive(),
                    limits.greeting_grace_seconds,
                )
            except TimeoutError:
                pass
            except ChannelError as error:
                if (
                    error.operation != "channel.receive"
                    or error.message != "assistant response timed out"
                ):
                    raise
        # A late greeting can look like a response to the first probe. A second
        # independent submission must also complete before the controls are trusted.
        probes = (probe_message, second_probe(language))
        for index, message in enumerate(probes, start=1):
            _require_target_origin(channel, expected_origin)
            if on_progress is not None:
                on_progress(f"sending synthetic verification probe {index}/{len(probes)}")
            receipt = await asyncio.wait_for(
                channel.send(message, on_progress=on_progress)
                if on_progress is not None
                else channel.send(message),
                limits.browser_action_timeout_seconds,
            )
            if on_progress is not None:
                on_progress(
                    f"customer echo {receipt.evidence['echo']} after probe {index}/{len(probes)}"
                )
            response = await asyncio.wait_for(
                channel.receive(on_progress=on_progress)
                if on_progress is not None
                else channel.receive(),
                limits.response_timeout_seconds,
            )
            _require_target_origin(channel, expected_origin)
            if not response.messages or any(not item.content.strip() for item in response.messages):
                raise ChannelError("discover probe", "assistant response is incomplete")
            if on_progress is not None:
                on_progress(
                    f"completed assistant response received for probe {index}/{len(probes)}"
                )
        _require_target_origin(channel, expected_origin)
        if on_progress is not None:
            on_progress("native channel verification passed")
        return ChannelGuide(
            version="1",
            channel=candidate.channel,
            validation=GuideValidation(
                status="verified",
                verified_at=datetime.now(UTC),
                target_origin=_origin(str(candidate.channel.url)),
                observed_behaviors=["open", "send", "receive", "completion"],
            ),
        )
    except TimeoutError:
        failed = ChannelError("discover probe", "native channel verification timed out")
        raise failed from None
    except BaseException as exc:
        failed = exc
        raise
    finally:
        if on_progress is not None:
            on_progress("closing native verification browser")
        try:
            await asyncio.wait_for(channel.close(), limits.close_timeout_seconds)
        except Exception:
            if failed is None:
                raise ChannelError("discover cleanup", "browser cleanup failed") from None
