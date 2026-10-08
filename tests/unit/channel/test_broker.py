"""The broker public API never turns labels into action capabilities."""

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from syntheticcx.adapters.channel.broker import ChatBroker, NativeChatBroker
from syntheticcx.adapters.channel.options import eligibility
from syntheticcx.adapters.channel.ownership import OwnedScope, attest_root
from syntheticcx.adapters.channel.stateful import StatefulPlaywrightChannel
from syntheticcx.config.models import ExecutionLimits
from syntheticcx.errors import ChannelError


def test_shared_broker_api_and_runtime_initial_hook() -> None:
    assert ChatBroker is NativeChatBroker
    for name in (
        "start",
        "observe",
        "open",
        "bind_scope",
        "bind_reader",
        "bind_state",
        "submit_probe",
        "select_option",
        "send",
        "receive",
        "observe_initial",
        "close",
    ):
        assert inspect.iscoroutinefunction(getattr(ChatBroker, name))
    assert not inspect.iscoroutinefunction(ChatBroker.export_config)
    assert inspect.iscoroutinefunction(StatefulPlaywrightChannel.observe_initial)


def test_slow_context_cleanup_does_not_skip_browser_and_driver_cleanup():
    async def exercise():
        async def slow():
            await asyncio.sleep(10)

        broker = NativeChatBroker(
            "http://localhost:8765/",
            limits=ExecutionLimits(
                max_turns=3,
                journey_timeout_seconds=1,
                greeting_grace_seconds=0,
                browser_action_timeout_seconds=1,
                response_timeout_seconds=1,
                model_timeout_seconds=1,
                evaluator_timeout_seconds=1,
                persistence_timeout_seconds=1,
                close_timeout_seconds=0.06,
            ),
        )
        broker._context = SimpleNamespace(close=AsyncMock(side_effect=slow))
        broker._browser = SimpleNamespace(close=AsyncMock())
        broker._playwright = SimpleNamespace(stop=AsyncMock())
        with pytest.raises(ChannelError, match="resource cleanup"):
            await broker.close()
        broker._context.close.assert_awaited_once()
        broker._browser.close.assert_awaited_once()
        broker._playwright.stop.assert_awaited_once()
        with pytest.raises(ChannelError):
            await broker.close()
        broker._context.close.assert_awaited_once()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "label",
    ["Buy plans", "Pay invoice", "Login", "Continue", "Ativar plano", "Aceitar", "Enviar arquivo"],
)
def test_shared_positive_menu_policy_rejects_unsafe_labels(label: str) -> None:
    assert (
        eligibility(
            "Choose an information topic", label, {"safe": True, "visible": True, "enabled": True}
        )
        is not None
    )


def test_scope_catalog_does_not_swallow_an_origin_violation(monkeypatch):
    async def exercise():
        failure = ChannelError("channel.ownership", "target origin is not attested")
        monkeypatch.setattr(
            "syntheticcx.adapters.channel.broker.attest_root", AsyncMock(side_effect=failure)
        )
        monkeypatch.setattr(
            "syntheticcx.adapters.channel.broker.frame_origin_matches", AsyncMock(return_value=True)
        )
        broker = NativeChatBroker("http://localhost:8765/", page=SimpleNamespace())
        broker._check_parent = AsyncMock()
        proposal = SimpleNamespace(
            frame=SimpleNamespace(locator=lambda *_args: None),
            locator=SimpleNamespace(strategy="css", value="#chat"),
        )
        with pytest.raises(ChannelError) as caught:
            await broker._attestable_scope(proposal)
        assert caught.value is failure

    asyncio.run(exercise())


@pytest.mark.parametrize("failure_stage", ["metadata", "frame_name", "proof_mismatch", "pin_check"])
def test_root_attestation_releases_temporary_handles_on_failure(monkeypatch, failure_stage):
    async def exercise():
        metadata = {
            "valid": True,
            "dialog": False,
            "embed": failure_stage == "pin_check",
            "name": "",
            "validLabels": True,
            "id": "chat",
        }
        root = SimpleNamespace(evaluate=AsyncMock(return_value=metadata), dispose=AsyncMock())
        if failure_stage == "metadata":
            root.evaluate.side_effect = RuntimeError("inspection failed")
        frame_node = SimpleNamespace(
            evaluate=AsyncMock(return_value="Payment" if failure_stage == "frame_name" else "Chat"),
            dispose=AsyncMock(),
        )
        main = SimpleNamespace()
        frame = (
            SimpleNamespace(frame_element=AsyncMock(return_value=frame_node))
            if failure_stage in {"frame_name", "proof_mismatch"}
            else main
        )
        page = SimpleNamespace(url="http://localhost:8765/", main_frame=main)
        locator = SimpleNamespace(
            count=AsyncMock(return_value=1), element_handle=AsyncMock(return_value=root)
        )
        monkeypatch.setattr(
            "syntheticcx.adapters.channel.ownership.frame_origin_matches",
            AsyncMock(return_value=True),
        )
        monkeypatch.setattr(
            OwnedScope,
            "check",
            AsyncMock(side_effect=ChannelError("channel.ownership", "pinned root changed")),
        )
        with pytest.raises((ChannelError, RuntimeError)):
            await attest_root(
                page,
                frame,
                locator,
                page.url,
                expected_proof="opener_linked" if failure_stage == "proof_mismatch" else None,
            )
        root.dispose.assert_awaited_once()
        if frame is not main:
            frame_node.dispose.assert_awaited_once()

    asyncio.run(exercise())
