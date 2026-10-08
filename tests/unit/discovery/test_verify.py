"""A late greeting must not masquerade as a reply to a synthetic probe."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from syntheticcx.config.models import ExecutionLimits, ReplyOptionsConfig
from syntheticcx.discovery import verify
from syntheticcx.discovery.explore import DiscoveryCandidate
from syntheticcx.domain.models import (
    CompletedResponse,
    Message,
    ReplyOption,
    SelectionReceipt,
    SendReceipt,
)
from syntheticcx.errors import ChannelError
from syntheticcx.language import first_probe, second_probe


def candidate() -> DiscoveryCandidate:
    document: dict[str, Any] = yaml.safe_load(
        Path("tests/fixtures/discovery-sample.yaml").read_text()
    )
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
        response_bubble_attached = False

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
    with pytest.raises(ChannelError, match="before response deadline"):
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


@pytest.mark.parametrize("bubble_attached", [False, True])
def test_native_response_timeout_preserves_verification_observation_stage(
    monkeypatch, bubble_attached
):
    class Channel:
        current_url = str(candidate().channel.url)
        response_bubble_attached = bubble_attached

        async def start(self):
            pass

        async def send(self, _message):
            return None

        async def receive(self):
            raise ChannelError("channel.receive", "assistant response timed out")

        async def close(self):
            pass

    monkeypatch.setattr(verify, "PlaywrightChannel", lambda *_args: Channel())
    limits = ExecutionLimits.model_validate(
        yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())["execution"]
    ).model_copy(update={"greeting_grace_seconds": 0})
    expected = "completion was not observed" if bubble_attached else "did not observe a new bubble"
    with pytest.raises(ChannelError, match=expected):
        asyncio.run(verify.verify_candidate(candidate(), limits, "Synthetic probe"))


@pytest.mark.parametrize("failure", [None, "guard", "timeout", "receipt", "silent", "origin"])
def test_verification_selects_once_and_never_types_after_a_selection_failure(monkeypatch, failure):
    proposed = candidate()
    proposed = proposed.model_copy(
        update={
            "channel": proposed.channel.model_copy(
                update={
                    "reply_options": ReplyOptionsConfig(
                        scope_locator={"strategy": "css", "value": "#chat"},
                        option_locator={"strategy": "css", "value": "button.option"},
                    ),
                }
            )
        }
    )
    now = datetime.now(UTC)
    closed = []
    sends = []
    selections = []
    progress = []

    class Channel:
        current_url = str(proposed.channel.url)
        response_bubble_attached = False
        receives = 0

        async def start(self, **_kwargs):
            pass

        async def send(self, text, **_kwargs):
            sends.append(text)
            return SendReceipt(message_id="probe", submitted_at=now, evidence={"echo": "absent"})

        async def receive(self, **_kwargs):
            self.receives += 1
            if self.receives == 2 and failure == "silent":
                raise TimeoutError()
            return CompletedResponse(
                messages=[
                    Message(
                        id=f"reply-{self.receives}",
                        sequence=self.receives,
                        actor="assistant",
                        content="" if self.receives == 1 else "Plan details",
                        observed_at=now,
                        options=[
                            ReplyOption(id="current", label="Plans"),
                            ReplyOption(id="other", label="Coverage"),
                        ]
                        if self.receives == 1
                        else [],
                    )
                ],
                started_at=now,
                completed_at=now,
                completion_strategy="quiet_interval",
            )

        async def select_option(self, option_id):
            selections.append(option_id)
            if failure == "guard":
                raise ChannelError("channel.select_option", "reply option context changed")
            if failure == "timeout":
                raise TimeoutError()
            if failure == "origin":
                self.current_url = "https://elsewhere.example/"
            return SelectionReceipt(
                message_id="choice",
                submitted_at=now,
                evidence={"echo": "absent"},
                selected_option_id=option_id,
                selected_label="Plans",
                source_message_id="wrong" if failure == "receipt" else "reply-1",
            )

        async def close(self):
            closed.append(True)

    channel = Channel()
    monkeypatch.setattr(verify, "PlaywrightChannel", lambda *_args: channel)
    limits = ExecutionLimits.model_validate(
        yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())["execution"]
    ).model_copy(update={"greeting_grace_seconds": 0})
    operation = verify.verify_candidate(
        proposed, limits, "Synthetic probe", on_progress=progress.append
    )
    if failure is None:
        guide = asyncio.run(operation)
        assert "select_option" in guide.validation.observed_behaviors
    else:
        with pytest.raises(ChannelError):
            asyncio.run(operation)
    assert sends == ["Synthetic probe"]
    assert selections == ["current"]
    assert closed == [True]
    assert "Plans" not in "\n".join(progress) and "current" not in "\n".join(progress)
    assert "selecting one guarded informational option for verification 2/2" in progress
