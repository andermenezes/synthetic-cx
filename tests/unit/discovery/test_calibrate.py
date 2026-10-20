"""Calibration sends once, uses observed evidence, and fails closed without publication."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from playwright.async_api import Error as PlaywrightError

from syntheticcx.adapters.discovery import native
from syntheticcx.config.models import ChannelConfig, ExecutionLimits, PrivacyConfig
from syntheticcx.discovery import calibrate
from syntheticcx.discovery.explore import DiscoveryCandidate
from syntheticcx.errors import ChannelError
from syntheticcx.privacy import Redactor


def inputs() -> tuple[DiscoveryCandidate, ExecutionLimits]:
    document = yaml.safe_load(Path("tests/fixtures/discovery-sample.yaml").read_text())
    document["execution"]["greeting_grace_seconds"] = 0
    return (
        DiscoveryCandidate(channel=document["channel"], open_steps=1),
        ExecutionLimits.model_validate(document["execution"]),
    )


def entry(text: str, identity: str) -> dict[str, Any]:
    return {
        "text": text,
        "identity": identity,
        "visible": True,
        "contains_input": False,
        "tag": "p",
    }


@pytest.mark.parametrize("bad", ["missing", "echo", "identity", "controls"])
def test_reading_validation_rejects_invalid_assistant_selection(bad: str) -> None:
    candidate, _ = inputs()
    assistant = [entry("reply", "r1")]
    if bad == "missing":
        assistant = []
    elif bad == "echo":
        assistant[0]["text"] = "Olá"
    elif bad == "identity":
        assistant[0]["identity"] = None
    else:
        assistant[0]["contains_input"] = True
    with pytest.raises(ChannelError):
        calibrate._validate_messages([], assistant, {"entries": []}, "Olá", candidate.channel)


def test_reply_may_start_with_the_same_greeting_as_the_probe() -> None:
    candidate, _ = inputs()
    calibrate._validate_messages(
        [entry("Olá", "c1")],
        [entry("Olá! Como posso ajudar?", "r1")],
        {"entries": []},
        "Olá",
        candidate.channel,
    )


def test_baseline_greeting_cannot_be_used_as_new_reply() -> None:
    candidate, _ = inputs()
    with pytest.raises(ChannelError, match="new reply"):
        calibrate._validate_messages(
            [],
            [entry("Hello", "r1")],
            {"entries": [{"text": "Hello", "attributes": {"data-message-id": "r1"}}]},
            "Olá",
            candidate.channel,
        )


def test_new_identity_allows_reply_with_same_text_as_baseline_greeting() -> None:
    candidate, _ = inputs()
    baseline = {"entries": [{"text": "Hello", "attributes": {"data-message-id": "old"}}]}
    after = {"entries": [{"text": "Hello", "attributes": {"data-message-id": "new"}}]}
    assert calibrate._new_text_entries(baseline, after, "Olá")
    calibrate._validate_messages([], [entry("Hello", "new")], baseline, "Olá", candidate.channel)


@pytest.mark.parametrize("failure", [None, "send", "navigation", "timeout", "cancel"])
def test_calibration_corrects_without_resending_and_redacts_evidence(
    monkeypatch: Any,
    failure: str | None,
) -> None:
    candidate, limits = inputs()
    submitted: list[str] = []
    closed: list[bool] = []
    requests: list[Any] = []
    baseline = {"origin": "http://127.0.0.1:8000", "entries": [{"text": "greeting"}]}

    class Browser:
        current_url = str(candidate.channel.url)

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def start(self, **_kwargs: Any) -> None:
            pass

        async def inventory(self) -> Any:
            if not submitted:
                return baseline
            return {
                "origin": baseline["origin"],
                "entries": [{"text": 'reply secret-"escaped"', "attributes": {"data-id": "r1"}}],
            }

        async def send(self, text: str, **_kwargs: Any) -> None:
            submitted.append(text)
            if failure == "send":
                raise ChannelError("channel.send", "uncertain submission")
            if failure == "cancel":
                raise asyncio.CancelledError

        async def message_evidence(self, _config: Any) -> Any:
            if failure == "navigation":
                self.current_url = "https://other.example/"
            if failure == "timeout":
                raise TimeoutError
            return [], [entry("reply", "r1")]

        async def close(self) -> None:
            closed.append(True)

    class Provider:
        async def generate(self, request: Any) -> Any:
            requests.append(request)
            proposal = candidate.channel.model_dump(mode="json")
            proposal["url"] = "__SYNTHETICCX_TARGET_URL__"
            if len(requests) == 1:
                proposal["input_locator"] = {"strategy": "css", "value": "input"}
            return type(
                "Response", (), {"content": json.dumps({"channel_json": json.dumps(proposal)})}
            )()

    monkeypatch.setattr(calibrate, "NativeCalibrationBrowser", Browser)
    operation = calibrate.calibrate_candidate(
        candidate,
        limits,
        "Olá",
        Provider(),
        Redactor(PrivacyConfig(redaction_keys=["account"])),
        sensitive_context={"account": 'secret-"escaped"'},
        max_attempts=3,
    )
    if failure is None:
        assert asyncio.run(operation) == candidate
    elif failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(operation)
    else:
        with pytest.raises(ChannelError):
            asyncio.run(operation)
    assert submitted == ["Olá"] and closed == [True]
    if failure in {"send", "cancel"}:
        assert requests == []
        return
    assert len(requests) == 2
    prompt = requests[0].messages[-1].content
    assert "secret-" not in prompt
    assert str(candidate.channel.url) not in prompt


@pytest.mark.parametrize("left_origin", [False, True])
def test_observation_deadline_is_classified_without_resending(
    monkeypatch: Any, left_origin: bool
) -> None:
    candidate, limits = inputs()
    limits = limits.model_copy(update={"response_timeout_seconds": 0.01})
    submitted: list[str] = []
    closed: list[bool] = []

    class Browser:
        current_url = str(candidate.channel.url)

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def start(self, **_kwargs: Any) -> None:
            pass

        async def inventory(self) -> Any:
            if not submitted:
                return {"origin": "http://127.0.0.1:8000", "entries": []}
            if left_origin:
                self.current_url = "https://other.example/"
            await asyncio.Event().wait()

        async def send(self, text: str, **_kwargs: Any) -> None:
            submitted.append(text)

        async def close(self) -> None:
            closed.append(True)

    class Provider:
        async def generate(self, _request: Any) -> Any:
            pytest.fail("no model request is allowed without a stable reply")

    monkeypatch.setattr(calibrate, "NativeCalibrationBrowser", Browser)
    expected = "browser left the selected target origin" if left_origin else "no stable new reply"
    with pytest.raises(ChannelError, match=expected):
        asyncio.run(
            calibrate.calibrate_candidate(
                candidate, limits, "Olá", Provider(), Redactor(PrivacyConfig())
            )
        )
    assert submitted == ["Olá"] and closed == [True]


def test_refined_message_identity_must_be_unique() -> None:
    candidate, _ = inputs()
    with pytest.raises(ChannelError, match="duplicate identities"):
        calibrate._validate_messages(
            [],
            [entry("first", "same"), entry("second", "same")],
            {"entries": []},
            "Olá",
            ChannelConfig.model_validate(candidate.channel.model_dump()),
        )


@pytest.mark.parametrize("failure", ["json", "envelope", "channel", "schema", "url", "controls"])
def test_reading_proposal_errors_are_specific_and_value_free(failure: str) -> None:
    candidate, _ = inputs()
    proposal = candidate.channel.model_dump(mode="json")
    proposal["url"] = "__SYNTHETICCX_TARGET_URL__"
    expected = {
        "json": "contained invalid JSON",
        "envelope": "requires a channel_json string",
        "channel": "channel_json must be an object",
        "schema": "schema issues: message_identity",
        "url": "changed the target URL",
        "controls": "changed interaction controls: input_locator",
    }[failure]
    if failure == "schema":
        proposal["message_identity"] = "secret"
    if failure == "url":
        proposal["url"] = "https://secret.example/"
    if failure == "controls":
        proposal["input_locator"]["value"] = "secret"
    content = json.dumps({"channel_json": json.dumps(proposal)})
    if failure == "json":
        content = "secret"
    if failure == "envelope":
        content = json.dumps({"channel_json": {"secret": True}})
    if failure == "channel":
        content = json.dumps({"channel_json": json.dumps(["secret"])})
    with pytest.raises(ChannelError, match=expected) as captured:
        calibrate._reading_proposal(content, candidate.channel)
    assert "secret" not in str(captured.value)


def test_identity_feedback_is_redacted_bounded_and_keeps_ancestor_metadata() -> None:
    candidate, _ = inputs()
    selected = entry("reply", "secret")
    selected.update(
        attributes={"data-key": "secret", "id": str(candidate.channel.url)},
        ancestors=[{"tag": "div", "attributes": {"data-key": "secret"}}],
    )
    feedback = calibrate._reading_feedback(
        "assistant message selector has missing or blank identities",
        candidate.channel,
        [],
        [selected] * 120,
        Redactor(PrivacyConfig(redaction_keys=["account"])),
        {"account": "secret"},
        str(candidate.channel.url),
    )
    assert "secret" not in feedback and str(candidate.channel.url) not in feedback
    assert "ancestors" in feedback and "data-key" in feedback
    evidence = json.loads(feedback.split("evidence:\n", 1)[1])
    assert evidence["assistant"]["selected_count"] == 120
    assert evidence["assistant"]["duplicate_identities"] == 119
    assert len(evidence["assistant"]["samples"]) <= 6
    assert len(feedback) < 12_200


@pytest.mark.parametrize(
    "failure,expected,retryable",
    [
        ("Execution context was destroyed, secret", "execution context was replaced", True),
        (
            "JSHandles can be evaluated only in the context they were created! secret",
            "execution context was replaced",
            True,
        ),
        (
            "JSHandles can be evaluated only in context secret",
            "execution context was replaced",
            True,
        ),
        ("Frame was detached secret", "frame is unavailable", True),
        ("Failed to find frame for selector secret", "frame is unavailable", True),
        ("TypeError secret", "inspection script failed", False),
    ],
)
@pytest.mark.parametrize("phase", ["anchors", "inspection"])
def test_native_inspection_classifies_errors_without_raw_details(
    monkeypatch: Any, failure: str, expected: str, retryable: bool, phase: str
) -> None:
    candidate, limits = inputs()

    class Locator:
        async def element_handles(self) -> Any:
            if phase == "anchors":
                raise PlaywrightError(failure)
            return []

        async def evaluate_handle(self, *_args: Any) -> Any:
            raise PlaywrightError(failure)

    class Scope:
        def locator(self, *_args: Any) -> Locator:
            return Locator()

    browser = native.NativeCalibrationBrowser(candidate.channel, limits)
    browser._scope = Scope()  # type: ignore[assignment]
    browser._page = type("Page", (), {"url": str(candidate.channel.url)})()  # type: ignore[assignment]
    monkeypatch.setattr(native, "locate", lambda *_args: Locator())
    with pytest.raises(ChannelError, match=expected) as captured:
        asyncio.run(browser.inventory())
    assert captured.value.retryable is retryable
    assert "secret" not in str(captured.value)


@pytest.mark.parametrize("failure", ["recover", "missing", "script", "cancel", "navigation"])
def test_post_send_observation_recovery_never_resends(monkeypatch: Any, failure: str) -> None:
    candidate, limits = inputs()
    limits = limits.model_copy(update={"response_timeout_seconds": 1.6})
    sent: list[str] = []
    closed: list[bool] = []
    requests: list[Any] = []
    progress: list[str] = []

    class Browser:
        current_url = str(candidate.channel.url)
        observations = 0

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def start(self, **_kwargs: Any) -> None:
            pass

        async def inventory(self) -> Any:
            if sent:
                self.observations += 1
                if failure == "cancel":
                    raise asyncio.CancelledError
                if failure == "script":
                    raise ChannelError("discover calibration", "chat DOM inspection script failed")
                if failure == "navigation":
                    self.current_url = "https://other.example/"
                if failure != "recover" or self.observations == 1:
                    raise native.CalibrationObservationUnavailable(
                        "chat observation scope is unavailable"
                    )
            return {
                "origin": "http://127.0.0.1:8000",
                "entries": [{"text": "reply"}] if sent else [],
            }

        async def send(self, text: str, **_kwargs: Any) -> None:
            sent.append(text)

        async def message_evidence(self, _config: Any) -> Any:
            return [], [entry("reply", "r1")]

        async def close(self) -> None:
            closed.append(True)

    class Provider:
        async def generate(self, request: Any) -> Any:
            requests.append(request)
            proposal = candidate.channel.model_dump(mode="json")
            proposal["url"] = "__SYNTHETICCX_TARGET_URL__"
            return type(
                "Response", (), {"content": json.dumps({"channel_json": json.dumps(proposal)})}
            )()

    monkeypatch.setattr(calibrate, "NativeCalibrationBrowser", Browser)
    operation = calibrate.calibrate_candidate(
        candidate, limits, "Olá", Provider(), Redactor(PrivacyConfig()), on_progress=progress.append
    )
    if failure == "recover":
        assert asyncio.run(operation) == candidate
        assert len(requests) == 1
        assert "chat observation scope recovered; no probe resent" in progress
        assert "hint" not in requests[0].messages[-1].content
    else:
        expected = asyncio.CancelledError if failure == "cancel" else ChannelError
        with pytest.raises(expected) as captured:
            asyncio.run(operation)
        if failure == "navigation":
            assert captured.value.operation == "discover navigation"
        if failure == "missing":
            assert "scope is unavailable before response deadline" in str(captured.value)
        assert not requests
    assert sent == ["Olá"] and closed == [True]
