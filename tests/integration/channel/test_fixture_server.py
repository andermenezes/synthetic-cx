"""Local fixture transport and Chromium startup/greeting checks."""

import asyncio
import hashlib
import importlib.util
import unicodedata
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest
import yaml
from playwright.async_api import Browser, BrowserContext

from syntheticcx.adapters.channel.playwright import PlaywrightChannel
from syntheticcx.adapters.storage.filesystem import FilesystemStorage
from syntheticcx.config.loader import load_config, load_config_text
from syntheticcx.config.models import ChannelConfig, LocatorConfig
from syntheticcx.domain.artifacts import ConfigArtifact, Run
from syntheticcx.domain.models import PersonaDefinition, Scenario
from syntheticcx.errors import ChannelError
from syntheticcx.privacy import Redactor
from syntheticcx.runtime.clock import RealClock, RealWaitController
from syntheticcx.runtime.journal import RunJournal
from syntheticcx.runtime.runner import run_conversation

SERVER_PATH = Path(__file__).resolve().parents[3] / "examples/telecom/webchat_fixture/server.py"
spec = importlib.util.spec_from_file_location("webchat_fixture_server", SERVER_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
FixtureServer = module.FixtureServer


def test_server_serves_every_variant_and_frame_assets() -> None:
    with FixtureServer() as fixture:
        assert fixture.base_url.startswith("http://127.0.0.1:")
        for variant in (
            "success",
            "repeated_question",
            "streamed_response",
            "missing_control",
            "silent_response",
        ):
            for view in ("main", "frame"):
                with urlopen(
                    f"{fixture.base_url}/?variant={variant}&view={view}", timeout=2
                ) as response:
                    assert response.status == 200
                    assert b'data-testid="open-chat"' in response.read()
        for path, media_type in (
            ("/frame.html", "text/html"),
            ("/style.css", "text/css"),
            ("/app.js", "javascript"),
        ):
            with urlopen(f"{fixture.base_url}{path}", timeout=2) as response:
                assert media_type in response.headers["Content-Type"]
                assert response.read()
        with urlopen(Request(f"{fixture.base_url}/app.js", method="HEAD"), timeout=2) as response:
            assert response.status == 200
            assert int(response.headers["Content-Length"]) > 0
            assert response.read() == b""


def test_server_rejects_unlisted_and_traversal_paths() -> None:
    with FixtureServer() as fixture:
        for path in ("/missing.txt", "/%2e%2e/scenario.yaml", "/%2fetc/passwd"):
            with pytest.raises(HTTPError) as caught:
                urlopen(f"{fixture.base_url}{path}", timeout=2)
            assert caught.value.code == 404


def test_server_teardown_and_independent_ephemeral_ports() -> None:
    server = FixtureServer()
    with server as first, FixtureServer() as second:
        assert first.base_url != second.base_url
        old_url = first.base_url
        with urlopen(f"{second.base_url}/health", timeout=2) as response:
            assert response.read() == b"ready"
    server.close()
    with pytest.raises(URLError):
        urlopen(f"{old_url}/health", timeout=0.3)


def test_invalid_timeout_rejected_before_binding() -> None:
    for timeout in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            FixtureServer(timeout)


@pytest.mark.parametrize("framed", [False, True])
@pytest.mark.parametrize("semantic", [False, True])
def test_chromium_greeting_excludes_history_and_closes_after_partial_start(
    framed: bool, semantic: bool
) -> None:
    async def exercise() -> None:
        with FixtureServer() as fixture:
            config = load_config(Path("examples/telecom/scenario.yaml"))
            channel_data = config.channel.model_dump(mode="json")
            channel_data["url"] = (
                f"{fixture.base_url}/?variant=success&view={'frame' if framed else 'main'}"
            )
            if framed:
                channel_data["frame_locator"] = LocatorConfig(
                    strategy="test_id", value="chat-frame"
                ).model_dump()
            if semantic:
                channel_data["open_locator"] = LocatorConfig(
                    strategy="role", value="button", name="Open chat"
                ).model_dump()
                channel_data["input_locator"] = LocatorConfig(
                    strategy="label", value="Message"
                ).model_dump()
                channel_data["assistant_message_locator"] = LocatorConfig(
                    strategy="role", value="article", name="Assistant message"
                ).model_dump()
            channel = PlaywrightChannel(
                ChannelConfig.model_validate(channel_data), config.execution
            )
            try:
                await channel.start()
                greeting = await asyncio.wait_for(channel.receive(), timeout=3)
                assert [part.content for part in greeting.messages] == [
                    "Hello! How can I help with your bill?"
                ]
                assert greeting.messages[0].id == "assistant:greeting-1"
            finally:
                await channel.close()
                await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("framed", [False, True])
def test_css_text_and_placeholder_locators_in_browser(framed: bool) -> None:
    async def exercise() -> None:
        with FixtureServer() as fixture:
            config = load_config(Path("examples/telecom/scenario.yaml"))
            channel_data = config.channel.model_dump(mode="json")
            channel_data["url"] = (
                f"{fixture.base_url}/?variant=success&view={'frame' if framed else 'main'}"
            )
            channel_data["open_locator"] = {"strategy": "text", "value": "Open chat"}
            channel_data["input_locator"] = {
                "strategy": "placeholder",
                "value": "Type your message",
            }
            channel_data["assistant_message_locator"] = {
                "strategy": "css",
                "value": '[data-testid="assistant-message"]',
            }
            channel_data["send_strategy"] = {
                "type": "click",
                "locator": {"strategy": "css", "value": '[data-testid="send-message"]'},
            }
            channel_data["response_completion"] = {
                "strategy": "completion_locator",
                "locator": {"strategy": "css", "value": '[data-testid="response-complete"]'},
            }
            if framed:
                channel_data["frame_locator"] = {
                    "strategy": "css",
                    "value": '[data-testid="chat-frame"]',
                }
            channel = PlaywrightChannel(
                ChannelConfig.model_validate(channel_data), config.execution
            )
            try:
                await channel.start()
                assert (await channel.receive()).messages[0].id == "assistant:greeting-1"
                await channel.send("Why did my invoice change?")
                assert (await channel.receive()).messages[0].id == "assistant:response-1-1"
            finally:
                await channel.close()

    asyncio.run(exercise())


def configured_channel(
    base_url: str,
    *,
    variant: str = "success",
    framed: bool = False,
    enter: bool = False,
    echo: str = "match",
    completion: str = "completion_locator",
    fallback: bool = False,
    response_timeout: float = 2,
    greeting: str = "normal",
    echo_window: float = 0.12,
    reuse_completion: bool = False,
) -> PlaywrightChannel:
    config = load_config(Path("examples/telecom/scenario.yaml"))
    channel = config.channel.model_dump(mode="json")
    channel["url"] = (
        f"{base_url}/?variant={variant}&view={'frame' if framed else 'main'}"
        f"&echo={echo}&greeting={greeting}"
        f"&identity={'none' if fallback else 'attribute'}"
        f"&completion={'reuse' if reuse_completion else 'replace'}"
    )
    if framed:
        channel["frame_locator"] = {"strategy": "test_id", "value": "chat-frame"}
    if enter:
        channel["send_strategy"] = {"type": "press_enter"}
    if fallback:
        channel["message_identity"] = {"strategy": "actor_sequence_content_hash"}
    if completion == "quiet_interval":
        channel["response_completion"] = {
            "strategy": "quiet_interval",
            "quiet_interval_seconds": 0.15,
        }
    elif completion == "streaming_indicator_disappears":
        channel["response_completion"] = {
            "strategy": completion,
            "locator": {"strategy": "test_id", "value": "streaming-indicator"},
        }
    limits = config.execution.model_copy(
        update={
            "response_timeout_seconds": response_timeout,
            "echo_observation_seconds": echo_window,
        }
    )
    return PlaywrightChannel(ChannelConfig.model_validate(channel), limits)


@pytest.mark.parametrize("variant", ["success", "repeated_question", "streamed_response"])
@pytest.mark.parametrize(
    "framed,enter", [(False, False), (False, True), (True, False), (True, True)]
)
def test_chromium_send_receive_integrity(variant: str, framed: bool, enter: bool) -> None:
    async def exercise() -> None:
        with FixtureServer() as fixture:
            channel = configured_channel(
                fixture.base_url, variant=variant, framed=framed, enter=enter
            )
            try:
                await channel.start()
                greeting = await channel.receive()
                assert [part.id for part in greeting.messages] == ["assistant:greeting-1"]
                receipt = await channel.send("My account number is SYNTHETIC-001")
                assert receipt.message_id == "customer:customer-1"
                assert receipt.evidence["echo"] == "matched"
                response = await channel.receive()
                expected_parts = 1 if variant == "repeated_question" else 2
                assert len(response.messages) == expected_parts
                assert [part.id for part in response.messages] == [
                    f"assistant:response-1-{index}" for index in range(1, expected_parts + 1)
                ]
                assert "history" not in str(response.model_dump())
                if variant == "streamed_response":
                    assert response.messages[0].content == (
                        "Your bill increased because your plan changed."
                    )
                second = await channel.send("Could you clarify?")
                assert second.message_id == "customer:customer-2"
                next_response = await channel.receive()
                assert next_response.messages[0].id == "assistant:response-2-1"
            finally:
                await channel.close()
                await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("echo", ["none", "normalized", "mismatch", "mutating", "duplicate"])
def test_chromium_optional_echo_and_failed_receipt(echo: str) -> None:
    async def exercise() -> None:
        with FixtureServer() as fixture:
            channel = configured_channel(fixture.base_url, echo=echo)
            try:
                await channel.start()
                await channel.receive()
                if echo in {"mismatch", "mutating", "duplicate"}:
                    with pytest.raises(ChannelError, match="echo differs|echo is ambiguous"):
                        await channel.send("I need help")
                    with pytest.raises(ChannelError, match="previous submission"):
                        await channel.send("Do not retry")
                else:
                    receipt = await channel.send("  My Account Number  ")
                    assert receipt.evidence["echo"] == ("absent" if echo == "none" else "matched")
                    if echo == "none":
                        assert receipt.message_id.startswith("customer:submitted-")
                    assert (await channel.receive()).messages[0].actor == "assistant"
            finally:
                await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("completion", ["quiet_interval", "streaming_indicator_disappears"])
def test_chromium_stream_completion_strategies(completion: str) -> None:
    async def exercise() -> None:
        with FixtureServer() as fixture:
            channel = configured_channel(
                fixture.base_url,
                variant="streamed_response",
                completion=completion,
                fallback=True,
            )
            try:
                await channel.start()
                await channel.receive()
                await channel.send("Explain my bill")
                response = await channel.receive()
                assert response.completion_strategy == completion
                assert len(response.messages) == 2
                assert response.messages[0].content == (
                    "Your bill increased because your plan changed."
                )
                assert all(part.id.startswith("assistant:") for part in response.messages)
                for part in response.messages:
                    normalized = unicodedata.normalize("NFKC", part.content).casefold().strip()
                    digest = hashlib.sha256(
                        f"assistant\0{part.sequence}\0{normalized}".encode()
                    ).hexdigest()
                    assert part.id == f"assistant:{digest}"
                await channel.send("Second turn")
                later = await channel.receive()
                assert later.messages[0].id != response.messages[0].id
            finally:
                await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "framed,enter", [(False, False), (False, True), (True, False), (True, True)]
)
def test_chromium_silent_response_is_classified(framed: bool, enter: bool) -> None:
    async def exercise() -> None:
        with FixtureServer() as fixture:
            channel = configured_channel(
                fixture.base_url,
                variant="silent_response",
                response_timeout=0.3,
                framed=framed,
                enter=enter,
            )
            try:
                await channel.start()
                await channel.receive()
                await channel.send("Please explain my bill")
                with pytest.raises(ChannelError, match="assistant response timed out"):
                    await channel.receive()
            finally:
                await channel.close()

    asyncio.run(exercise())


def test_completion_locator_reused_node_is_fresh_for_each_response() -> None:
    async def exercise() -> None:
        with FixtureServer() as fixture:
            channel = configured_channel(fixture.base_url, reuse_completion=True)
            try:
                await channel.start()
                await channel.receive()
                for index in range(2):
                    await channel.send(f"Turn {index}")
                    result = await channel.receive()
                    assert result.messages[0].id == f"assistant:response-{index + 1}-1"
            finally:
                await channel.close()

    asyncio.run(exercise())


def test_skipped_greeting_is_not_reemitted_with_next_response() -> None:
    async def exercise() -> None:
        with FixtureServer() as fixture:
            channel = configured_channel(fixture.base_url, greeting="slow")
            try:
                await channel.start()
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(channel.receive(), timeout=0.02)
                await channel.send("Start without a greeting")
                response = await channel.receive()
                assert [part.id for part in response.messages] == [
                    "assistant:response-1-1",
                    "assistant:response-1-2",
                ]
            finally:
                await channel.close()

    asyncio.run(exercise())


def test_greeting_grace_cancellation_does_not_finalize_partial_observation() -> None:
    async def exercise() -> None:
        with FixtureServer() as fixture:
            channel = configured_channel(fixture.base_url, greeting="slow")
            try:
                await channel.start()
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(channel.receive(), timeout=0.02)
                greeting = await channel.receive()
                assert [part.id for part in greeting.messages] == ["assistant:greeting-1"]
            finally:
                await channel.close()

    asyncio.run(exercise())


def test_cancelled_post_submission_echo_does_not_allow_a_blind_retry() -> None:
    async def exercise() -> None:
        with FixtureServer() as fixture:
            channel = configured_channel(fixture.base_url, echo="none", echo_window=0.8)
            try:
                await channel.start()
                await channel.receive()
                sending = asyncio.create_task(channel.send("One physical submission"))
                scope = channel._scope
                assert scope is not None
                # A new assistant bubble proves the UI submission already occurred.
                await (
                    scope.get_by_test_id("assistant-message")
                    .nth(2)
                    .wait_for(state="attached", timeout=1000)
                )
                sending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await sending
                with pytest.raises(ChannelError, match="previous submission"):
                    await channel.send("Do not retry")
            finally:
                await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("framed", [False, True])
def test_chromium_missing_input_classified_and_partial_browser_closed(framed: bool) -> None:
    async def exercise() -> None:
        with FixtureServer() as fixture:
            config = load_config(Path("examples/telecom/scenario.yaml"))
            channel_data = config.channel.model_dump(mode="json")
            channel_data["url"] = (
                f"{fixture.base_url}/?variant=missing_control&view={'frame' if framed else 'main'}"
            )
            if framed:
                channel_data["frame_locator"] = {"strategy": "test_id", "value": "chat-frame"}
            limits = config.execution.model_copy(update={"browser_action_timeout_seconds": 0.4})
            channel = PlaywrightChannel(ChannelConfig.model_validate(channel_data), limits)
            try:
                with pytest.raises(
                    ChannelError, match="browser startup or control resolution failed"
                ):
                    await channel.start()
            finally:
                await channel.close()
                await channel.close()

    asyncio.run(exercise())


def test_browser_failure_screenshot_is_committed_before_close(tmp_path: Path) -> None:
    async def exercise() -> None:
        with FixtureServer() as fixture:
            document = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
            document["channel"]["url"] = f"{fixture.base_url}/?variant=missing_control"
            document["execution"]["browser_action_timeout_seconds"] = 3
            document["privacy"]["allow_screenshots"] = True
            document["privacy"]["confirm_screenshot_target_safe"] = True
            document["storage"]["store_screenshots"] = True
            config = load_config_text(yaml.safe_dump(document))
            clock = RealClock()
            redactor = Redactor(config.privacy)
            backend = FilesystemStorage(
                tmp_path, redactor, timeout_seconds=5, project_config=config
            )
            run = Run.model_validate(
                {
                    "id": "browser-failure",
                    "scenario_id": config.scenario.id,
                    "persona_id": config.persona.id,
                    "started_at": clock.now(),
                    "lifecycle_status": "in_progress",
                    "metadata": {
                        "configuration_version": "1",
                        "framework_version": "0.1.0",
                        "persona_model": {"provider": "fake", "model": "fake"},
                        "evaluation_model": {"provider": "fake", "model": "fake"},
                        "usage": {"unavailable_reason": "no calls"},
                        "runtime_platform": "test",
                        "owner": {
                            "owner_uuid": "owner",
                            "hostname": "test",
                            "pid": 1,
                            "process_start_evidence": "start",
                        },
                    },
                }
            )
            channel = PlaywrightChannel(config.channel, config.execution)

            class UnusedPersona:
                async def initialize(self, *_args: object) -> None:
                    raise AssertionError("start failure must precede persona initialization")

                async def respond(self, *_args: object) -> None:
                    raise AssertionError("start failure must precede persona response")

            journal = RunJournal(run.id, backend, redactor, clock)
            result = await run_conversation(
                run=run,
                config_artifact=ConfigArtifact(source_content_hash="hash", effective_config={}),
                scenario=Scenario.model_validate(config.scenario.model_dump()),
                persona=PersonaDefinition.model_validate(config.persona.model_dump()),
                channel=channel,
                simulator=UnusedPersona(),
                storage=backend,
                journal=journal,
                clock=clock,
                wait=RealWaitController(),
                limits=config.execution,
                project_config=config,
            )
            assert result.termination_reason == "channel_error"
            artifacts = [
                event.data for event in result.committed_events if event.type == "artifact_recorded"
            ]
            assert len(artifacts) == 1 and artifacts[0].status == "captured"
            assert artifacts[0].path is not None
            image = (tmp_path / run.id / artifacts[0].path).read_bytes()
            assert image.startswith(b"\x89PNG\r\n\x1a\n")
            assert artifacts[0].content_hash == hashlib.sha256(image).hexdigest()
            assert b"\x89PNG" not in (tmp_path / run.id / "trace.jsonl").read_bytes()

    asyncio.run(exercise())


@pytest.mark.parametrize("outcome", ["success", "missing_control", "silent", "cancelled"])
def test_chromium_context_and_browser_close_once(
    outcome: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    context_closes = 0
    browser_closes = 0
    context_close = BrowserContext.close
    browser_close = Browser.close

    async def count_context_close(self: BrowserContext, *args: object, **kwargs: object) -> None:
        nonlocal context_closes
        context_closes += 1
        await context_close(self, *args, **kwargs)

    async def count_browser_close(self: Browser, *args: object, **kwargs: object) -> None:
        nonlocal browser_closes
        browser_closes += 1
        await browser_close(self, *args, **kwargs)

    monkeypatch.setattr(BrowserContext, "close", count_context_close)
    monkeypatch.setattr(Browser, "close", count_browser_close)

    async def exercise() -> None:
        with FixtureServer() as fixture:
            variant = {
                "success": "success",
                "missing_control": "missing_control",
                "silent": "silent_response",
                "cancelled": "silent_response",
            }[outcome]
            channel = configured_channel(fixture.base_url, variant=variant, response_timeout=0.25)
            if outcome == "missing_control":
                channel._limits = channel._limits.model_copy(
                    update={"browser_action_timeout_seconds": 0.4}
                )
            try:
                if outcome == "missing_control":
                    with pytest.raises(ChannelError):
                        await channel.start()
                else:
                    await channel.start()
                    await channel.receive()
                    await channel.send("Why did my bill increase?")
                    if outcome == "success":
                        await channel.receive()
                    elif outcome == "silent":
                        with pytest.raises(ChannelError, match="timed out"):
                            await channel.receive()
                    else:
                        receiving = asyncio.create_task(channel.receive())
                        await asyncio.sleep(0.02)
                        receiving.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await receiving
            finally:
                await channel.close()
                await channel.close()

    asyncio.run(exercise())
    assert context_closes == browser_closes == 1
