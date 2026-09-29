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


def candidate() -> DiscoveryCandidate:
    document: dict[str, Any] = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    return DiscoveryCandidate(channel=document["channel"], open_steps=1)


def test_late_greeting_does_not_verify_silent_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    class LateGreetingChannel:
        def __init__(self, *_args: Any) -> None:
            self.current_url = str(candidate().channel.url)
            self.submitted: list[str] = []
            self.receives = 0
            self.closed = False

        async def start(self) -> None:
            pass

        async def send(self, message: str) -> None:
            self.submitted.append(message)

        async def receive(self) -> Any:
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
