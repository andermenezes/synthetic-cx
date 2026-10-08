"""Verify candidate controls with a fresh native Playwright browser."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import urlsplit

from syntheticcx.adapters.channel.playwright import PlaywrightChannel
from syntheticcx.config.models import ChannelGuide, ExecutionLimits, GuideValidation
from syntheticcx.discovery.explore import DiscoveryCandidate
from syntheticcx.domain.models import SelectionReceipt
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
    stage = "startup"
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
        # A late greeting can look like a response to the first probe. Require a
        # second independent interaction: one current guarded option, or text.
        probes = (probe_message, second_probe(language))
        next_option_id: str | None = None
        next_option_source: str | None = None
        selected_option = False
        for index, message in enumerate(probes, start=1):
            _require_target_origin(channel, expected_origin)
            if index == 2 and next_option_id is not None:
                stage = "select_option"
                if on_progress is not None:
                    on_progress("selecting one guarded informational option for verification 2/2")
                # IDs are issued by the native guard for this exact current reply.
                # Any failure is terminal, never a reason to try another ID or text.
                receipt = await asyncio.wait_for(
                    channel.select_option(next_option_id), limits.browser_action_timeout_seconds
                )
                if (
                    not isinstance(receipt, SelectionReceipt)
                    or receipt.selected_option_id != next_option_id
                    or receipt.source_message_id != next_option_source
                ):
                    raise ChannelError("discover probe", "guarded option receipt is inconsistent")
                selected_option = True
            else:
                stage = "send"
                if on_progress is not None:
                    if index == 2:
                        on_progress(
                            "no guarded options in latest reply; using second text probe"
                            if candidate.channel.reply_options is not None
                            else "reply options not configured; using second text probe"
                        )
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
            stage = "receive"
            response = await asyncio.wait_for(
                channel.receive(on_progress=on_progress)
                if on_progress is not None
                else channel.receive(),
                limits.response_timeout_seconds,
            )
            _require_target_origin(channel, expected_origin)
            if not response.messages or any(
                not item.content.strip() and not getattr(item, "options", [])
                for item in response.messages
            ):
                raise ChannelError("discover probe", "assistant response is incomplete")
            if index == 1 and candidate.channel.reply_options is not None:
                options = getattr(response.messages[-1], "options", [])
                next_option_id = options[0].id if options else None
                next_option_source = response.messages[-1].id if options else None
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
                observed_behaviors=["open", "send", "receive", "completion"]
                + (["select_option"] if selected_option else []),
            ),
        )
    except (TimeoutError, ChannelError) as error:
        if isinstance(error, ChannelError) and not (
            stage == "receive"
            and error.operation == "channel.receive"
            and error.message == "assistant response timed out"
        ):
            failed = error
            raise
        message = "native channel verification timed out"
        if stage == "select_option":
            message = "guarded option verification timed out; action not retried"
        if stage == "receive":
            message = (
                "assistant reply attached but completion was not observed before response deadline"
                if channel.response_bubble_attached
                else "assistant message locator did not observe a new bubble "
                "before response deadline"
            )
            if on_progress is not None:
                on_progress(message)
        failed = ChannelError("discover probe", message)
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
