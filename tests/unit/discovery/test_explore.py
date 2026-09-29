"""Offline exploration contracts: bounded actions and no guide publication."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from syntheticcx.adapters.discovery.playwright_mcp import (
    _check_click,
    _check_hover,
    _check_page_origin,
    _origin,
)
from syntheticcx.config.models import ChannelConfig, PrivacyConfig
from syntheticcx.discovery.explore import explore
from syntheticcx.domain.models import LLMRequest, LLMResponse
from syntheticcx.errors import ChannelError, ModelError
from syntheticcx.privacy import Redactor

URL = "http://127.0.0.1:8000/?variant=success"
SNAPSHOT = "### Page\n- Page URL: " + URL + '\n### Snapshot\n- button "Open chat" [ref=e5]'
OPENED = SNAPSHOT + '\n- textbox "Message" [ref=e6]'
OPENED = SNAPSHOT + '\n- textbox "Message" [ref=e6]'
HOVERED = "### Page\n- Page URL: " + URL + '\n### Snapshot\n- button "Talk to an agent" [ref=e5]'


def candidate() -> dict[str, Any]:
    source: dict[str, Any] = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    channel = cast(dict[str, Any], source["channel"])
    channel["url"] = "__SYNTHETICCX_TARGET_URL__"
    return channel


class Browser:
    def __init__(self) -> None:
        self.clicks: list[str] = []
        self.hovers: list[str] = []
        self.chat_hint: str | None = None
        self.inspections = 0

    async def navigate(self, url: str) -> str:
        assert url == URL
        return SNAPSHOT

    async def snapshot(self) -> str:
        return OPENED

    async def click(self, target: str, element: str, snapshot: str) -> str:
        _check_click(snapshot, target, element, self.chat_hint)
        self.clicks.append(target)
        return OPENED

    async def inspect_chat(self) -> str:
        self.inspections += 1
        return '{"elements":[{"tag":"textarea","attributes":{"data-testid":"chat-input"}}]}'

    async def hover(self, target: str, element: str, snapshot: str) -> str:
        _check_hover(snapshot, target, element, self.chat_hint)
        self.hovers.append(target)
        return HOVERED


class Provider:
    def __init__(self, *decisions: dict[str, str]) -> None:
        self.decisions = list(decisions)
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return cast(
            LLMResponse, type("Response", (), {"content": json.dumps(self.decisions.pop(0))})()
        )


def decision(
    action: str, *, target: str = "", element: str = "", channel: str = ""
) -> dict[str, str]:
    return {"action": action, "target": target, "element": element, "channel_json": channel}


def run(
    browser: Browser,
    provider: Provider,
    *,
    chat_hint: str | None = None,
    on_progress: Any = None,
    max_discovery_attempts: int = 3,
) -> Any:
    browser.chat_hint = chat_hint
    return asyncio.run(
        explore(
            URL,
            browser,
            provider,
            Redactor(PrivacyConfig()),
            chat_hint=chat_hint,
            max_discovery_attempts=max_discovery_attempts,
            on_progress=on_progress,
        )
    )


def test_safe_open_then_typed_candidate() -> None:
    browser = Browser()
    provider = Provider(
        decision("open_chat", target="e5", element="Open chat"),
        decision("propose", channel=json.dumps(candidate())),
    )
    result = run(browser, provider)
    assert isinstance(result.channel, ChannelConfig)
    assert result.open_steps == 1
    assert str(result.channel.url) == URL
    assert browser.clicks == ["e5"]
    assert browser.inspections == 1
    assert len(provider.requests) == 2
    assert "chat-input" in provider.requests[1].messages[-1].content
    assert all(request.response_schema is not None for request in provider.requests)


def test_chat_inventory_is_redacted_before_model_request() -> None:
    class SecretBrowser(Browser):
        async def inspect_chat(self) -> str:
            return '{"elements":[{"attributes":{"aria-label":"canary-secret"}}]}'

    provider = Provider(
        decision("open_chat", target="e5", element="Open chat"),
        decision("propose", channel=json.dumps(candidate())),
    )
    asyncio.run(
        explore(
            URL,
            SecretBrowser(),
            provider,
            Redactor(PrivacyConfig(), resolved_secrets=["canary-secret"]),
        )
    )
    assert "canary-secret" not in provider.requests[1].messages[-1].content


def test_chat_hint_can_reveal_and_select_a_hover_tooltip() -> None:
    browser = Browser()
    hint = "Talk to an agent"
    provider = Provider(
        decision("hover", target="e5", element="chat launcher"),
        decision("open_chat", target="e5", element=hint),
        decision("propose", channel=json.dumps(candidate())),
    )

    result = run(browser, provider, chat_hint=hint)

    assert isinstance(result.channel, ChannelConfig)
    assert browser.hovers == ["e5"]
    assert browser.clicks == ["e5"]
    assert hint in provider.requests[0].messages[-1].content
    assert result.open_steps == 2


def test_verbose_schema_diagnostic_reports_safe_field_path_only() -> None:
    proposal = candidate()
    proposal["input_locator"] = {
        "strategy": "private-page-canary",
        "value": "private-page-canary",
    }
    progress: list[str] = []
    provider = Provider(decision("propose", channel=json.dumps(proposal)), decision("stop"))

    with pytest.raises(ChannelError):
        run(Browser(), provider, on_progress=progress.append, max_discovery_attempts=2)

    diagnostic = " ".join(progress)
    assert "input_locator.strategy [literal_error]" in diagnostic
    assert "use one of the documented locator or strategy values" in diagnostic
    assert "private-page-canary" not in diagnostic


def test_invalid_channel_proposal_gets_one_bounded_correction() -> None:
    invalid = candidate()
    invalid["input_locator"] = {"strategy": "unknown", "value": "message"}
    progress: list[str] = []
    provider = Provider(
        decision("propose", channel=json.dumps(invalid)),
        decision("propose", channel=json.dumps(candidate())),
    )

    result = run(Browser(), provider, on_progress=progress.append)

    assert isinstance(result.channel, ChannelConfig)
    assert len(provider.requests) == 2
    assert "requesting a bounded channel-candidate correction" in progress
    correction_prompt = provider.requests[1].messages[-1].content
    assert "previous channel proposal was rejected" in correction_prompt
    assert "input_locator.strategy [literal_error]" in correction_prompt


def test_invalid_channel_proposals_stop_at_configured_attempt_limit() -> None:
    invalid = candidate()
    invalid["input_locator"] = {"strategy": "unknown", "value": "message"}
    provider = Provider(*[decision("propose", channel=json.dumps(invalid))] * 3)
    with pytest.raises(ModelError, match="remained invalid after configured attempts"):
        run(Browser(), provider, max_discovery_attempts=3)
    assert len(provider.requests) == 3


def test_stop_refreshes_snapshot_before_reconsidering(monkeypatch: Any) -> None:
    async def no_wait(_seconds: float) -> None:
        pass

    monkeypatch.setattr("syntheticcx.discovery.explore.asyncio.sleep", no_wait)

    class DelayedBrowser(Browser):
        async def snapshot(self) -> str:
            return OPENED

    provider = Provider(decision("stop"), decision("propose", channel=json.dumps(candidate())))

    result = run(DelayedBrowser(), provider, max_discovery_attempts=2)

    assert isinstance(result.channel, ChannelConfig)
    assert len(provider.requests) == 2
    assert 'textbox "Message"' in provider.requests[1].messages[-1].content


def test_unchanged_snapshot_does_not_repeat_model_stop(monkeypatch: Any) -> None:
    async def no_wait(_seconds: float) -> None:
        pass

    monkeypatch.setattr("syntheticcx.discovery.explore.asyncio.sleep", no_wait)

    class UnchangedBrowser(Browser):
        async def snapshot(self) -> str:
            return SNAPSHOT

    provider = Provider(decision("stop"))
    with pytest.raises(ChannelError, match="could not be identified"):
        run(UnchangedBrowser(), provider, max_discovery_attempts=5)
    assert len(provider.requests) == 1


@pytest.mark.parametrize(
    "malformed", ["{", json.dumps({**candidate(), "url": "https://other.example"})]
)
def test_malformed_proposal_can_be_corrected(malformed: str) -> None:
    provider = Provider(
        decision("propose", channel=malformed),
        decision("propose", channel=json.dumps(candidate())),
    )
    result = run(Browser(), provider, max_discovery_attempts=2)
    assert isinstance(result.channel, ChannelConfig)
    assert "malformed or changed the target URL" in provider.requests[1].messages[-1].content


@pytest.mark.parametrize(
    ("snapshot", "target", "element"),
    [
        (SNAPSHOT, "css=.chat", "Open chat"),
        (SNAPSHOT, "e8", "Open chat"),
        (SNAPSHOT, "e5", "Purchase chat"),
        (SNAPSHOT.replace("Open chat", "Buy chat"), "e5", "Open chat"),
        (SNAPSHOT.replace("button", "textbox"), "e5", "Open chat"),
    ],
)
def test_untrusted_model_cannot_click_arbitrary_control(
    snapshot: str, target: str, element: str
) -> None:
    with pytest.raises(ChannelError):
        _check_click(snapshot, target, element)


@pytest.mark.parametrize(
    ("snapshot", "target", "element"),
    [
        (SNAPSHOT, "css=.button", "chat launcher"),
        (SNAPSHOT, "e9", "chat launcher"),
        (SNAPSHOT.replace("button", "textbox"), "e5", "chat launcher"),
        (SNAPSHOT.replace("Open chat", "Delete account"), "e5", "chat launcher"),
        (SNAPSHOT, "e5", "unrelated button"),
    ],
)
def test_hover_is_limited_to_safe_chat_controls(snapshot: str, target: str, element: str) -> None:
    with pytest.raises(ChannelError):
        _check_hover(snapshot, target, element, None)


def test_cross_origin_page_is_rejected() -> None:
    with pytest.raises(ChannelError, match="target origin"):
        _check_page_origin("### Page\n- Page URL: https://other.example/", _origin(URL))


@pytest.mark.parametrize(
    "bad",
    [
        decision("open_chat", target="e5", element="Open chat", channel="unexpected"),
        decision("propose", channel="{}"),
        decision("propose", channel=json.dumps({**candidate(), "url": "https://other.example"})),
        decision("stop"),
    ],
)
def test_invalid_decisions_do_not_publish_a_candidate(bad: dict[str, str]) -> None:
    decisions = (bad, decision("stop")) if bad["action"] == "propose" else (bad,)
    with pytest.raises((ChannelError, ModelError)):
        run(Browser(), Provider(*decisions), max_discovery_attempts=1)


def test_exploration_limit_prevents_unbounded_clicks() -> None:
    browser = Browser()
    provider = Provider(*[decision("open_chat", target="e5", element="Open chat")] * 3)
    with pytest.raises(ChannelError, match="limit"):
        run(browser, provider)
    assert len(browser.clicks) == 2


def test_snapshot_is_redacted_before_model_call() -> None:
    class SecretBrowser(Browser):
        async def navigate(self, url: str) -> str:
            return SNAPSHOT + " canary-secret"

    provider = Provider(decision("stop"))
    with pytest.raises(ChannelError):
        asyncio.run(
            explore(
                URL,
                SecretBrowser(),
                provider,
                Redactor(PrivacyConfig(), resolved_secrets=["canary-secret"]),
                max_discovery_attempts=1,
            )
        )
    assert "canary-secret" not in provider.requests[0].messages[-1].content


def test_page_url_and_sensitive_context_never_reach_model() -> None:
    class SecretBrowser(Browser):
        async def navigate(self, url: str) -> str:
            return SNAPSHOT.replace(URL, url) + " account-id-123"

    url = "http://127.0.0.1:8000/?token=hidden-token"
    provider = Provider(decision("stop"))
    with pytest.raises(ChannelError):
        asyncio.run(
            explore(
                url,
                SecretBrowser(),
                provider,
                Redactor(PrivacyConfig(redaction_keys=["account_number"])),
                sensitive_context={"knowledge": {"account_number": "account-id-123"}},
                max_discovery_attempts=1,
            )
        )
    prompt = provider.requests[0].messages[-1].content
    assert "hidden-token" not in prompt
    assert "account-id-123" not in prompt
    assert "__SYNTHETICCX_TARGET_URL__" in prompt


def test_bare_host_url_is_normalized_after_model_proposal() -> None:
    class HostBrowser(Browser):
        async def navigate(self, url: str) -> str:
            return SNAPSHOT.replace(URL, url)

    result = asyncio.run(
        explore(
            "https://example.com",
            HostBrowser(),
            Provider(decision("propose", channel=json.dumps(candidate()))),
            Redactor(PrivacyConfig()),
        )
    )
    assert str(result.channel.url) == "https://example.com/"


def test_model_timeout_is_bounded() -> None:
    class SlowProvider(Provider):
        async def generate(self, request: LLMRequest) -> LLMResponse:
            await asyncio.sleep(5)
            raise AssertionError("unreachable")

    with pytest.raises(ModelError, match="decision failed"):
        asyncio.run(
            explore(
                URL,
                Browser(),
                SlowProvider(),
                Redactor(PrivacyConfig()),
                model_timeout_seconds=0.01,
            )
        )
