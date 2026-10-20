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
    _observed_opener,
    _observed_send_control,
    _origin,
    _unique_safe_launcher,
)
from syntheticcx.config.models import ChannelConfig, LocatorConfig, PrivacyConfig
from syntheticcx.discovery.explore import _snapshot_excerpt, explore
from syntheticcx.domain.models import LLMRequest, LLMResponse
from syntheticcx.errors import ChannelError, ModelError
from syntheticcx.privacy import Redactor

URL = "http://127.0.0.1:8000/?variant=success"


def test_long_opened_page_excerpt_retains_chat_tail():
    snapshot = (
        "- Page URL: https://example.com/\n"
        + "FAQ material " * 3000
        + '\n- dialog\n- textbox "Chat input"'
    )
    excerpt = _snapshot_excerpt(snapshot, 16000, True)
    assert len(excerpt) <= 16000
    assert 'textbox "Chat input"' in excerpt
    assert "https://example.com/" in excerpt
    assert 'textbox "Chat input"' not in _snapshot_excerpt(snapshot, 16000, False)


SNAPSHOT = "### Page\n- Page URL: " + URL + '\n### Snapshot\n- button "Open chat" [ref=e5]'
OPENED = SNAPSHOT + '\n- textbox "Message" [ref=e6]'
ALREADY_OPEN = SNAPSHOT.replace('- button "Open chat" [ref=e5]', '- textbox "Message" [ref=e6]')
HOVERED = "### Page\n- Page URL: " + URL + '\n### Snapshot\n- button "Talk to an agent" [ref=e5]'


def candidate() -> dict[str, Any]:
    source: dict[str, Any] = yaml.safe_load(
        Path("tests/fixtures/discovery-sample.yaml").read_text()
    )
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


class AlreadyOpenBrowser(Browser):
    async def navigate(self, url: str) -> str:
        assert url == URL
        return ALREADY_OPEN


def test_explicit_id_hint_opens_unlabeled_button_before_model_proposal():
    class IdBrowser(Browser):
        async def navigate(self, url):
            return SNAPSHOT.replace('button "Open chat"', "button")

        async def open_id_hint(self, snapshot):
            assert 'button "' not in snapshot
            self.clicks.append("explicit-id")
            return OPENED, LocatorConfig(strategy="css", value='[id="ovc-btn-chat-widget"]')

    channel = candidate()
    channel["open_locator"] = None
    browser = IdBrowser()
    provider = Provider(decision("propose", channel=json.dumps(channel)))
    result = run(browser, provider, chat_hint="id=ovc-btn-chat-widget")
    assert result.channel.open_locator.value == '[id="ovc-btn-chat-widget"]'
    assert browser.clicks == ["explicit-id"] and browser.inspections == 1
    assert "already activated: True" in provider.requests[0].messages[1].content


def test_explicit_id_never_falls_back_to_model_click_for_unsupported_browser():
    provider = Provider()
    with pytest.raises(ChannelError, match="does not support"):
        run(Browser(), provider, chat_hint="id=ovc-btn-chat-widget")
    assert provider.requests == []


def test_redacted_id_hint_keeps_raw_dispatch_intent_separate_from_model_text():
    class IdBrowser(Browser):
        async def open_id_hint(self, snapshot):
            self.clicks.append("id")
            return OPENED, LocatorConfig(strategy="css", value='[id="chat-customer-123"]')

    browser = IdBrowser()
    provider = Provider(decision("propose", channel=json.dumps(candidate())))
    result = asyncio.run(
        explore(
            URL,
            browser,
            provider,
            Redactor(PrivacyConfig()),
            chat_hint="id=chat-<redacted:customer_id>",
            id_launcher_requested=True,
        )
    )
    assert result.channel.open_locator.value == '[id="chat-customer-123"]'
    assert browser.clicks == ["id"]
    assert "customer-123" not in provider.requests[0].model_dump_json()


def test_id_hint_reconsiders_controls_that_appear_after_initial_open(monkeypatch):
    class DelayedBrowser(Browser):
        async def open_id_hint(self, snapshot):
            self.clicks.append("id")
            return OPENED, LocatorConfig(strategy="css", value='[id="chat"]')

        async def snapshot(self):
            return OPENED + '\n- button "Send" [ref=e7]'

    async def sleep(seconds):
        pass

    monkeypatch.setattr(asyncio, "sleep", sleep)
    browser = DelayedBrowser()
    provider = Provider(decision("stop"), decision("propose", channel=json.dumps(candidate())))
    result = run(browser, provider, chat_hint="id=chat")
    assert result.channel.open_locator.value == '[id="chat"]'
    assert browser.clicks == ["id"]
    assert browser.inspections == 2 and len(provider.requests) == 2


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
    assert result.channel.open_locator == LocatorConfig(
        strategy="role", value="button", name="Open chat"
    )
    assert browser.inspections == 1
    assert len(provider.requests) == 2
    assert "chat-input" in provider.requests[1].messages[-1].content
    assert all(request.response_schema is not None for request in provider.requests)


@pytest.mark.parametrize("repeated", [False, True])
def test_auto_opened_launcher_is_not_clicked_again(repeated: bool) -> None:
    browser = Browser()
    open_again = decision("open_chat", target="e5", element="Open chat")
    provider = Provider(
        open_again,
        open_again if repeated else decision("propose", channel=json.dumps(candidate())),
    )
    if repeated:
        with pytest.raises(ChannelError, match="repeatedly requested"):
            run(browser, provider, chat_hint="Open chat", max_discovery_attempts=2)
    else:
        result = run(browser, provider, chat_hint="Open chat", max_discovery_attempts=2)
        assert isinstance(result.channel, ChannelConfig)
    assert browser.clicks == ["e5"]
    assert "Chat opener already activated: True" in provider.requests[0].messages[-1].content
    assert "No second click was made" in provider.requests[1].messages[-1].content


def test_control_only_proposal_defers_message_markup_to_native_calibration() -> None:
    proposal = candidate()
    for field in (
        "customer_message_locator",
        "assistant_message_locator",
        "message_identity",
        "response_completion",
    ):
        proposal.pop(field)
    result = run(AlreadyOpenBrowser(), Provider(decision("propose", channel=json.dumps(proposal))))
    assert result.channel.assistant_message_locator.value == ".syntheticcx-pending-assistant"
    assert (
        result.channel.input_locator
        == ChannelConfig.model_validate(
            {
                **candidate(),
                "url": URL,
            }
        ).input_locator
    )


def test_valid_proposal_opens_unique_safe_launcher_before_publication() -> None:
    browser = Browser()
    provider = Provider(
        decision("propose", channel=json.dumps(candidate())),
        decision("propose", channel=json.dumps(candidate())),
    )
    progress: list[str] = []

    result = run(browser, provider, on_progress=progress.append)

    assert isinstance(result.channel, ChannelConfig)
    assert browser.clicks == ["e5"]
    assert browser.inspections == 1
    assert result.open_steps == 1
    assert "opened unique safe chat launcher; refreshed snapshot and selector inventory" in progress
    assert "chat-input" in provider.requests[1].messages[-1].content


def test_opened_chat_uses_observed_opener_instead_of_missing_model_locator() -> None:
    proposal = candidate()
    proposal["open_locator"] = None
    browser = Browser()
    provider = Provider(
        decision("open_chat", target="e5", element="Open chat"),
        decision("propose", channel=json.dumps(proposal)),
    )
    result = run(browser, provider, max_discovery_attempts=2)
    assert result.channel.open_locator is not None
    assert result.channel.open_locator.strategy == "role"
    assert result.channel.open_locator.name == "Open chat"
    assert browser.clicks == ["e5"]
    assert len(provider.requests) == 2


def test_ambiguous_or_unsafe_launchers_are_not_automatically_clicked() -> None:
    assert _unique_safe_launcher(SNAPSHOT.replace("Open chat", "Buy chat"), None) is None
    assert _unique_safe_launcher(SNAPSHOT + '\n- button "Help chat" [ref=e7]', None) is None


def test_hint_identifies_unique_localized_launcher_without_relaxing_click_guard() -> None:
    localized = SNAPSHOT.replace("Open chat", "Pergunte à Aura")
    assert _unique_safe_launcher(localized, "Pergunte à Aura") == ("e5", "Pergunte à Aura")
    assert _unique_safe_launcher(localized, None) is None


def test_delayed_hinted_launcher_is_opened_before_first_model_decision(
    monkeypatch: Any,
) -> None:
    async def no_wait(_seconds: float) -> None:
        pass

    monkeypatch.setattr("syntheticcx.discovery.explore.asyncio.sleep", no_wait)
    localized = SNAPSHOT.replace("Open chat", "Pergunte à Aura")

    class DelayedHintBrowser(Browser):
        snapshots = 0

        async def navigate(self, url: str) -> str:
            assert url == URL
            return SNAPSHOT.replace("Open chat", "Página inicial")

        async def snapshot(self) -> str:
            self.snapshots += 1
            return (
                localized
                if self.snapshots >= 3
                else SNAPSHOT.replace("Open chat", f"Página inicial {self.snapshots}")
            )

    browser = DelayedHintBrowser()
    provider = Provider(decision("propose", channel=json.dumps(candidate())))
    progress: list[str] = []

    result = run(browser, provider, chat_hint="Pergunte à Aura", on_progress=progress.append)

    assert isinstance(result.channel, ChannelConfig)
    assert browser.snapshots == 3
    assert browser.clicks == ["e5"]
    assert result.open_steps == 1
    assert result.channel.open_locator == LocatorConfig(
        strategy="role", value="button", name="Pergunte à Aura", exact=False
    )
    assert len(provider.requests) == 1
    assert "opening unique safely labeled hinted chat launcher" in progress


def test_unrelated_page_changes_do_not_trigger_hint_stop_retry(monkeypatch: Any) -> None:
    async def no_wait(_seconds: float) -> None:
        pass

    monkeypatch.setattr("syntheticcx.discovery.explore.asyncio.sleep", no_wait)

    class UnrelatedBrowser(Browser):
        snapshots = 0

        async def snapshot(self) -> str:
            self.snapshots += 1
            return SNAPSHOT.replace("Open chat", f"Página inicial {self.snapshots}")

    provider = Provider(decision("stop"))
    with pytest.raises(ChannelError, match="could not be identified"):
        run(UnrelatedBrowser(), provider, chat_hint="Pergunte à Aura", max_discovery_attempts=3)
    assert len(provider.requests) == 1


def test_auto_open_derives_locator_when_model_omits_it() -> None:
    proposal = candidate()
    proposal["open_locator"] = None
    browser = Browser()
    provider = Provider(
        decision("propose", channel=json.dumps(proposal)),
        decision("propose", channel=json.dumps(proposal)),
    )
    result = run(browser, provider, max_discovery_attempts=1)
    assert result.channel.open_locator is not None
    assert browser.clicks == ["e5"]


def test_observed_hint_uses_partial_name_but_only_after_safe_ref_click() -> None:
    snapshot = SNAPSHOT.replace("Open chat", "Pergunte à Aura")
    opener = _observed_opener(snapshot, "e5", "Pergunte à Aura", "Pergunte à Aura")
    assert opener is not None
    assert (opener.strategy, opener.value, opener.name, opener.exact) == (
        "role",
        "button",
        "Pergunte à Aura",
        False,
    )
    with pytest.raises(ChannelError, match="unsafe or ambiguous"):
        _observed_opener(
            snapshot.replace("Pergunte à Aura", "Buy now"), "e5", "Buy now", "Pergunte à Aura"
        )


def test_new_unique_send_button_overrides_model_send_guess() -> None:
    opened_with_send = OPENED + '\n- button "Enviar mensagem" [ref=e7]'

    class SendButtonBrowser(Browser):
        async def click(self, target: str, element: str, snapshot: str) -> str:
            _check_click(snapshot, target, element, self.chat_hint)
            self.clicks.append(target)
            return opened_with_send

    proposal = candidate()
    proposal["send_strategy"] = {"type": "press_enter"}
    provider = Provider(
        decision("open_chat", target="e5", element="Open chat"),
        decision("propose", channel=json.dumps(proposal)),
    )
    progress: list[str] = []
    result = run(SendButtonBrowser(), provider, on_progress=progress.append)
    assert result.channel.send_strategy.type == "click"
    assert result.channel.send_strategy.locator == LocatorConfig(
        strategy="role", value="button", name="Enviar mensagem"
    )
    assert "using safely observed chat send control for native verification" in progress


def test_ambiguous_or_preexisting_send_buttons_are_not_derived() -> None:
    opened = OPENED + '\n- button "Enviar mensagem" [ref=e7]'
    assert _observed_send_control(SNAPSHOT, opened) == LocatorConfig(
        strategy="role", value="button", name="Enviar mensagem"
    )
    assert _observed_send_control(SNAPSHOT, opened + '\n- button "Enviar" [ref=e8]') is None
    assert (
        _observed_send_control(SNAPSHOT + '\n- button "Enviar mensagem" [ref=e9]', opened) is None
    )
    assert (
        _observed_send_control(SNAPSHOT, SNAPSHOT + '\n- button "Enviar mensagem" [ref=e7]') is None
    )


def test_click_without_new_chat_controls_does_not_publish_candidate(monkeypatch: Any) -> None:
    async def no_wait(_seconds: float) -> None:
        pass

    monkeypatch.setattr("syntheticcx.discovery.explore.asyncio.sleep", no_wait)

    class NoOpeningBrowser(Browser):
        async def click(self, target: str, element: str, snapshot: str) -> str:
            _check_click(snapshot, target, element, None)
            self.clicks.append(target)
            return snapshot

        async def snapshot(self) -> str:
            return SNAPSHOT

    browser = NoOpeningBrowser()
    provider = Provider(decision("propose", channel=json.dumps(candidate())))
    with pytest.raises(ChannelError, match="did not reveal new controls"):
        run(browser, provider)
    assert browser.clicks == ["e5"]
    assert browser.inspections == 0


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


def test_chat_hint_can_reveal_and_select_a_hover_tooltip(monkeypatch: Any) -> None:
    async def no_wait(_seconds: float) -> None:
        pass

    monkeypatch.setattr("syntheticcx.discovery.explore.asyncio.sleep", no_wait)
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
        run(AlreadyOpenBrowser(), provider, on_progress=progress.append, max_discovery_attempts=2)

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

    result = run(AlreadyOpenBrowser(), provider, on_progress=progress.append)

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
        run(AlreadyOpenBrowser(), provider, max_discovery_attempts=3)
    assert len(provider.requests) == 3


def test_stop_refreshes_snapshot_before_reconsidering(monkeypatch: Any) -> None:
    async def no_wait(_seconds: float) -> None:
        pass

    monkeypatch.setattr("syntheticcx.discovery.explore.asyncio.sleep", no_wait)

    class DelayedBrowser(Browser):
        async def snapshot(self) -> str:
            return ALREADY_OPEN

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
    result = run(AlreadyOpenBrowser(), provider, max_discovery_attempts=2)
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


def test_repeated_open_action_cannot_toggle_opened_chat() -> None:
    browser = Browser()
    provider = Provider(*[decision("open_chat", target="e5", element="Open chat")] * 4)
    with pytest.raises(ChannelError, match="repeatedly requested"):
        run(browser, provider)
    assert len(browser.clicks) == 1


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
            return ALREADY_OPEN.replace(URL, url)

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

    with pytest.raises(ModelError, match="decision timed out after 0.01 seconds"):
        asyncio.run(
            explore(
                URL,
                Browser(),
                SlowProvider(),
                Redactor(PrivacyConfig()),
                model_timeout_seconds=0.01,
            )
        )


@pytest.mark.parametrize(
    "provider_message,expected",
    [
        ("model request failed: timed out", "timed out"),
        ("model request failed (HTTP 429)", "HTTP 429"),
        ("model response is malformed", "response is malformed"),
        ("model response was refused", "response was refused"),
        ("model response was truncated", "response was truncated"),
        ("private-provider-canary", "details suppressed"),
    ],
)
def test_decision_failure_preserves_only_allowlisted_provider_diagnostics(
    provider_message: str,
    expected: str,
) -> None:
    class FailingProvider(Provider):
        async def generate(self, request: LLMRequest) -> LLMResponse:
            raise ModelError("model.generate", provider_message, retryable=True)

    with pytest.raises(ModelError, match=expected) as caught:
        run(AlreadyOpenBrowser(), FailingProvider())
    assert caught.value.retryable
    assert "private-provider-canary" not in str(caught.value)


@pytest.mark.parametrize(
    "content,expected",
    [
        ("{private-page-canary", "invalid JSON"),
        (json.dumps(decision("open_chat", target="e1", element="")), "invalid action schema"),
        ("private-page-canary" * 2000, "size limit"),
    ],
)
def test_decision_validation_failure_has_specific_value_free_diagnostic(
    content: str,
    expected: str,
) -> None:
    class InvalidProvider(Provider):
        async def generate(self, request: LLMRequest) -> LLMResponse:
            return cast(LLMResponse, type("Response", (), {"content": content})())

    with pytest.raises(ModelError, match=expected) as caught:
        run(AlreadyOpenBrowser(), InvalidProvider())
    assert "private-page-canary" not in str(caught.value)
