"""A late greeting must not masquerade as a reply to a synthetic probe."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml

from syntheticcx.config.models import ExecutionLimits
from syntheticcx.discovery import verify
from syntheticcx.discovery.explore import DiscoveryCandidate
from syntheticcx.errors import ChannelError
from syntheticcx.language import first_probe, second_probe


def candidate() -> DiscoveryCandidate:
    document: dict[str, Any] = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    return DiscoveryCandidate(channel=document["channel"], open_steps=1)


def test_portuguese_verification_sends_two_localized_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[str] = []

    class RespondingChannel:
        current_url = str(candidate().channel.url)

        def __init__(self, *_args: Any) -> None:
            pass

        async def start(self) -> None:
            pass

        async def send(self, message: str) -> None:
            sent.append(message)

        async def receive(self) -> Any:
            return type("Response", (), {"messages": [type("Message", (), {"content": "Olá"})()]})()

        async def close(self) -> None:
            pass

    monkeypatch.setattr(verify, "PlaywrightChannel", RespondingChannel)
    limits = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())["execution"]
    limits["greeting_grace_seconds"] = 0
    guide = asyncio.run(
        verify.verify_candidate(
            candidate(),
            ExecutionLimits.model_validate(limits),
            first_probe("pt-BR"),
            language="pt-BR",
        )
    )
    assert guide.validation.status == "verified"
    assert sent == [first_probe("pt-BR"), second_probe("pt-BR")]


def test_headed_verification_constructs_fresh_headed_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options: list[bool] = []
    progress: list[str] = []

    class FailingChannel:
        def __init__(self, *_args: Any, headed: bool = False) -> None:
            options.append(headed)

        async def start(self, *, on_progress: Any = None) -> None:
            raise ChannelError("channel.start", "target navigation returned HTTP 403")

        async def close(self) -> None:
            pass

    monkeypatch.setattr(verify, "PlaywrightChannel", FailingChannel)
    limits = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())["execution"]
    with pytest.raises(ChannelError, match="HTTP 403"):
        asyncio.run(
            verify.verify_candidate(
                candidate(),
                ExecutionLimits.model_validate(limits),
                "Synthetic probe",
                headed=True,
                on_progress=progress.append,
            )
        )
    assert options == [True]
    assert "starting fresh native Playwright verification browser (headed)" in progress


def test_late_greeting_does_not_verify_silent_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    class LateGreetingChannel:
        def __init__(self, *_args: Any) -> None:
            self.current_url = str(candidate().channel.url)
            self.submitted: list[str] = []
            self.receives = 0
            self.closed = False

        async def start(self, *, on_progress: Any = None) -> None:
            pass

        async def send(self, message: str, *, on_progress: Any = None) -> Any:
            self.submitted.append(message)
            return type("Receipt", (), {"evidence": {"echo": "absent"}})()

        async def receive(self, *, on_progress: Any = None) -> Any:
            self.receives += 1
            if self.receives == 1:
                raise ChannelError("channel.receive", "assistant response timed out")
            if self.receives == 2:
                return type(
                    "Response", (), {"messages": [type("Message", (), {"content": "hello"})()]}
                )()
            raise ChannelError("channel.receive", "assistant response timed out")

        async def close(self) -> None:
            self.closed = True

    channel = LateGreetingChannel()
    progress: list[str] = []
    monkeypatch.setattr(verify, "PlaywrightChannel", lambda *_args: channel)
    with pytest.raises(ChannelError, match="timed out"):
        limits = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())["execution"]
        asyncio.run(
            verify.verify_candidate(
                candidate(),
                ExecutionLimits.model_validate(limits),
                "Synthetic probe",
                on_progress=progress.append,
            )
        )
    assert len(channel.submitted) == 2
    assert channel.closed
    assert "sending synthetic verification probe 1/2" in progress
    assert "completed assistant response received for probe 1/2" in progress
    assert "sending synthetic verification probe 2/2" in progress
    assert "Synthetic probe" not in " ".join(progress)
