"""Verification must not send a probe or publish after changing origins."""

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


def test_origin_checked_again_after_greeting_before_probe(monkeypatch: Any) -> None:
    source = yaml.safe_load(Path("tests/fixtures/discovery-sample.yaml").read_text())
    source["channel"]["url"] = "https://example.com/"
    candidate = DiscoveryCandidate(channel=source["channel"], open_steps=0)
    limits = ExecutionLimits.model_validate(source["execution"])
    sent: list[str] = []

    class Channel:
        current_url = "https://example.com/"

        def __init__(self, *_args: Any) -> None:
            pass

        async def start(self) -> None:
            pass

        async def receive(self) -> None:
            self.current_url = "https://other.example/"

        async def send(self, message: str) -> None:
            sent.append(message)

        async def close(self) -> None:
            pass

    monkeypatch.setattr(verify, "PlaywrightChannel", Channel)
    with pytest.raises(ChannelError, match="selected target origin"):
        asyncio.run(verify.verify_candidate(candidate, limits, "synthetic greeting"))
    assert sent == []
