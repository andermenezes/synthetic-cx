"""Real browser discovery with persistent server history, never mocked channel echoes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

import pytest
import yaml
from playwright.async_api import BrowserType, Route

from syntheticcx.discovery import whatsapp as discovery
from syntheticcx.errors import ChannelError, ConfigurationError
from syntheticcx.language import first_probe

PHONE = "15551234567"
WEBSITE = "https://service.example/"
PROBE = "Owned repeatable discovery probe CANARY"
CONTACT_NAME = "PRIVATE NAMED BUSINESS CANARY"


class PersistentConversation:
    """Business-side history outlives both separately launched browser contexts."""

    def __init__(
        self,
        variant: str = "success",
        website_variant: str = "direct",
        *,
        named_empty: bool = False,
        drawer_phone: str = PHONE,
    ) -> None:
        self.variant = variant
        self.website_variant = website_variant
        self.named_empty = named_empty
        self.drawer_phone = drawer_phone
        self.drawer_phone_label = (
            f"+{drawer_phone[:-10]} {drawer_phone[-10:-7]} "
            f"{drawer_phone[-7:-4]}-{drawer_phone[-4:]}"
        )
        self.initial_histories: list[list[dict[str, str]]] = []
        self.drawer_openings: list[int] = []
        self.drawer_present_at_send: list[bool] = []
        self.launches = 0
        self.contexts: list[Any] = []
        self.submissions: list[str] = []
        self.website_visits: list[tuple[int, str]] = []
        self.history = (
            []
            if named_empty
            else [
                {"id": f"true_{PHONE}@c.us_OLD-CUSTOMER", "text": PROBE},
                {"id": f"false_{PHONE}@c.us_OLD-ASSISTANT", "text": "OLD SUCCESS CANARY"},
            ]
        )

    async def route(self, route: Route) -> None:
        url = urlsplit(route.request.url)
        if url.hostname == "service.example":
            self.website_visits.append((self.launches, url.path))
            if self.website_variant == "direct":
                body = (
                    '<html><body><a id="support-entry" aria-label="WhatsApp support" '
                    f'href="https://wa.me/{PHONE}">WhatsApp support</a></body></html>'
                )
            elif url.path == "/":
                launcher = '<a id="explore" href="/services">Explore</a>'
                if self.website_variant == "lazy":
                    body = (
                        "<body><p>Find your service options</p><script>"
                        "setTimeout(() => document.body.insertAdjacentHTML('beforeend', "
                        f"{json.dumps(launcher)}), 120)</script></body>"
                    )
                else:
                    body = f"<body><p>Find your service options</p>{launcher}</body>"
            elif url.path == "/services":
                body = (
                    "<body><p>Select how to continue with our team.</p>"
                    '<button id="next" onclick="document.querySelector(\'#options\').hidden=false">'
                    'Next</button><section id="options" hidden>'
                    f'<a id="handoff" href="https://wa.me/{PHONE}">Continue</a>'
                    "</section></body>"
                )
                if self.website_variant == "replay_drift" and self.launches == 2:
                    body = (
                        '<body><a id="alternative" href="https://wa.me/'
                        + PHONE
                        + '">Continue</a></body>'
                    )
            else:
                await route.abort()
                return
        elif url.hostname == "wa.me":
            body = (
                '<html><body><a aria-label="Continue to Chat" '
                f'href="https://web.whatsapp.com/send?phone={PHONE}">Continue to Chat</a>'
                "</body></html>"
            )
        elif url.hostname == "web.whatsapp.com" and url.path == "/fixture-drawer":
            self.drawer_openings.append(self.launches)
            await route.fulfill(status=204)
            return
        elif url.hostname == "web.whatsapp.com" and url.path == "/fixture-send":
            text = route.request.post_data_json["text"]
            self.submissions.append(text)
            self.drawer_present_at_send.append(route.request.post_data_json["drawer_present"])
            index = len(self.submissions)
            outgoing = {"id": f"true_{PHONE}@c.us_OUT-{index}", "text": text}
            incoming = {"id": f"false_{PHONE}@c.us_IN-{index}", "text": f"Fresh answer {index}"}
            if self.variant == "portability":
                incoming["text"] = "Portabilidade: deseja ver os planos?"
            appended = []
            if self.variant != "missing_echo":
                appended.append(outgoing)
            if self.variant != "silent" and not (self.variant == "failed_replay" and index == 2):
                appended.append(incoming)
            self.history.extend(appended)
            await route.fulfill(json=appended)
            return
        elif url.hostname == "web.whatsapp.com":
            if route.request.is_navigation_request():
                self.initial_histories.append([dict(message) for message in self.history])
            header = (
                '<div data-testid="conversation-info-header" role="button" tabindex="0">'
                '<span data-testid="conversation-info-header-chat-title-name">'
                f'<span data-testid="conversation-info-header-chat-title">{CONTACT_NAME}</span>'
                '</span><span id="contact-status">Conta comercial</span></div>'
                if self.named_empty
                else f'<span title="+{PHONE}">+{PHONE}</span>'
            )
            history = json.dumps(self.history)
            formatted_phone = self.drawer_phone_label
            body = f"""<!doctype html><html><body>
            <div id="main"><header>{header}</header>
            <div id="messages"></div>
            <footer><div role="textbox" aria-label="Type a message" contenteditable="true"></div>
            <button aria-label="Send">Send</button></footer></div>
            <script>
            const messages = document.querySelector('#messages');
            messages.dataset.testid = 'conversation-panel-messages';
            const opener = document.querySelector('[data-testid="conversation-info-header"]');
            if (opener) opener.addEventListener('click', () => {{
                if (document.querySelector('[data-testid="chat-info-drawer"]')) return;
                const drawer = document.createElement('aside');
                drawer.setAttribute('data-testid', 'chat-info-drawer');
                drawer.innerHTML = '<button aria-label="Close">Close</button>'
                    + '<div><span>{CONTACT_NAME}</span>'
                    + '<span>{formatted_phone}</span></div>'
                    + '<section><div data-testid="section-about-and-phone-number">'
                    + 'Recado e número de telefone</div>'
                    + '<div data-testid="container_with_separator"><div><div>'
                    + '<span data-testid="selectable-text">'
                    + '<span><span>{formatted_phone}</span></span></span>'
                    + '</div></div></div></section>'
                    + '<section><span>Business registration</span>'
                    + '<span>12345678999</span></section>';
                document.body.append(drawer);
                drawer.querySelector('button').onclick = () => drawer.remove();
                fetch('/fixture-drawer');
            }});
            function append(message) {{
                const row = document.createElement('div');
                row.setAttribute('data-id', message.id);
                row.className = message.id.startsWith('true_') ? 'message-out' : 'message-in';
                const text = document.createElement('span');
                text.className = 'selectable-text copyable-text';
                text.textContent = message.text;
                row.append(text); messages.append(row);
            }}
            {history}.forEach(append);
            const composer = document.querySelector('[role=textbox]');
            async function send() {{
                const text = composer.textContent; composer.textContent = '';
                const drawer_present = !!document.querySelector('[data-testid="chat-info-drawer"]');
                const status = document.querySelector('#contact-status');
                if (status) status.textContent = 'digitando…';
                const response = await fetch('/fixture-send', {{method:'POST',
                    headers:{{'Content-Type':'application/json'}},
                    body:JSON.stringify({{text, drawer_present}})}});
                (await response.json()).forEach(append);
                if (status) status.textContent = 'Conta comercial';
            }}
            composer.addEventListener('keydown', event => {{
                if (event.key === 'Enter' && !event.shiftKey) {{ event.preventDefault(); send(); }}
            }});
            document.querySelector('button').addEventListener('click', send);
            </script></body></html>"""
            if self.variant == "delayed_startup":
                payload = json.dumps(body).replace("<", "\\u003c")
                body = (
                    '<section id="main"><header>Loading conversation</header></section>'
                    f"<script>setTimeout(()=>{{document.open();document.write({payload});"
                    "document.close();},400);</script>"
                )
        else:
            await route.abort()
            return
        await route.fulfill(status=200, content_type="text/html", body=body)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = BrowserType.launch_persistent_context

        async def launch(browser: BrowserType, *args: Any, **kwargs: Any) -> Any:
            assert not self.contexts or all(context.pages == [] for context in self.contexts)
            context = await original(browser, *args, **kwargs)
            self.launches += 1
            self.contexts.append(context)
            await context.route("**/*", self.route)
            return context

        monkeypatch.setattr(BrowserType, "launch_persistent_context", launch)


class DiscoveryProvider:
    def __init__(self) -> None:
        self.closed = False
        self.requests: list[dict[str, Any]] = []

    async def generate(self, request: Any) -> Any:
        payload = json.loads(request.messages[-1].content)
        self.requests.append(payload)
        inventory = payload["inventory"]
        candidates = inventory["catalog"]["candidates"]
        labels = {item["id"]: item["description"] for item in candidates}
        choices = payload["action_choices"]
        priority = ["Explore", "Next", "Continue", "Continue to Chat", "WhatsApp support"]
        opens = [choice for choice in choices if choice["action"] in {"open", "follow_link"}]
        if opens:
            choice = min(
                opens,
                key=lambda item: (
                    priority.index(labels[item["target_id"]])
                    if labels[item["target_id"]] in priority
                    else len(priority)
                ),
            )
        else:
            choice = {"action": "wait"}
        decision = {
            "revision": inventory["revision"],
            **dict.fromkeys(
                [
                    "target_id",
                    "assistant_id",
                    "customer_id",
                    "input_id",
                    "send_id",
                    "option_id",
                    "send_mode",
                    "probe_id",
                    "stop_reason",
                ]
            ),
            **choice,
        }
        return SimpleNamespace(content=json.dumps(decision))

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    document = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    document.pop("channel")
    document["execution"].update(
        response_timeout_seconds=1.4,
        greeting_grace_seconds=0.02,
        browser_action_timeout_seconds=5,
        close_timeout_seconds=2,
        journey_timeout_seconds=30,
    )
    draft = tmp_path / "draft.yaml"
    draft.write_text(yaml.safe_dump(document))
    profile = tmp_path / "dedicated-profile"
    profile.mkdir()
    (profile / "user-owned-marker").write_text("profile-private-canary")
    monkeypatch.setenv("OPENAI_API_KEY", "credential-private-canary")
    monkeypatch.setenv("SYNTHETICCX_WHATSAPP_RECIPIENT", PHONE)
    provider = DiscoveryProvider()
    monkeypatch.setattr(discovery, "OpenAIProvider", lambda *_args, **_kwargs: provider)

    async def preflight(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(discovery, "_probe_role", preflight)
    return SimpleNamespace(draft=draft, profile=profile, provider=provider)


def invoke(setup: Any, **kwargs: Any) -> Any:
    return asyncio.run(
        discovery.discover_whatsapp_guide(
            setup.draft,
            kwargs.pop("url", WEBSITE),
            browser_profile=setup.profile,
            **kwargs,
        )
    )


def test_portability_reply_stops_discovery_before_replay(setup: Any, monkeypatch: Any) -> None:
    server = PersistentConversation(variant="portability")
    server.install(monkeypatch)
    with pytest.raises(ChannelError, match="forbidden in discovery"):
        invoke(setup, probe_message=PROBE)
    assert server.launches == 1
    assert server.submissions == [PROBE]
    assert all(context.pages == [] for context in server.contexts)
    assert setup.provider.closed


def test_verified_guide_requires_two_fresh_echoed_exchanges(setup: Any, monkeypatch: Any) -> None:
    server = PersistentConversation()
    server.install(monkeypatch)
    progress: list[str] = []
    guide = invoke(setup, probe_message=PROBE, on_progress=progress.append)
    assert server.launches == 2
    assert server.submissions == [PROBE, PROBE]
    assert guide.channel.navigation_steps
    assert guide.validation.replay_verified is True
    assert guide.validation.server_state_reset is False
    assert set(guide.validation.observed_behaviors) == {"open", "send", "receive", "completion"}
    serialized = guide.model_dump_json()
    for forbidden in (
        PHONE,
        PROBE,
        "OLD SUCCESS",
        "Fresh answer",
        "credential-private-canary",
        str(setup.profile),
        "profile-private-canary",
    ):
        assert forbidden not in serialized
        assert forbidden not in " ".join(progress)
    assert all(context.pages == [] for context in server.contexts)
    assert setup.provider.closed
    assert (setup.profile / "user-owned-marker").read_text() == "profile-private-canary"


def test_cli_recipient_override_verifies_and_publishes_without_leaking_number(
    setup: Any,
    monkeypatch: Any,
) -> None:
    from typer.testing import CliRunner

    from syntheticcx.cli import app

    monkeypatch.setenv("SYNTHETICCX_WHATSAPP_RECIPIENT", "15557654321")
    server = PersistentConversation(variant="delayed_startup")
    server.install(monkeypatch)
    output = setup.draft.parent / "guide.yaml"
    result = CliRunner().invoke(
        app,
        [
            "discover",
            str(setup.draft),
            "--channel",
            "whatsapp_web",
            "--url",
            WEBSITE,
            "--browser-profile",
            str(setup.profile),
            "--recipient",
            "+" + PHONE,
            "--probe-message",
            PROBE,
            "--output",
            str(output),
            "--verbose",
        ],
    )
    assert result.exit_code == 0, result.output
    assert server.submissions == [PROBE, PROBE]
    guide = yaml.safe_load(output.read_text())
    assert guide["validation"]["replay_verified"] is True
    published = "".join(path.read_text() for path in setup.draft.parent.glob("*.yaml"))
    for value in (PHONE, "+" + PHONE, PROBE):
        assert value not in published
        assert value not in result.output


def test_cli_named_empty_chat_attests_drawer_and_replays_persistent_history(
    setup: Any, monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    from typer.testing import CliRunner

    from syntheticcx.cli import app

    monkeypatch.setenv("SYNTHETICCX_WHATSAPP_RECIPIENT", "15557654321")
    server = PersistentConversation(named_empty=True)
    server.install(monkeypatch)
    output = setup.draft.parent / "guide.yaml"
    completed = setup.draft.with_name("draft.completed.yaml")
    result = CliRunner().invoke(
        app,
        [
            "discover",
            str(setup.draft),
            "--channel",
            "whatsapp_web",
            "--url",
            WEBSITE,
            "--browser-profile",
            str(setup.profile),
            "--recipient",
            "+" + PHONE,
            "--probe-message",
            PROBE,
            "--output",
            str(output),
            "--verbose",
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert server.launches == 2
    assert server.submissions == [PROBE, PROBE]
    # Each website handoff is transferred verbatim to a fresh input-owning tab.
    assert server.initial_histories[:2] == [[], []]
    expected_history = [
        {"id": f"true_{PHONE}@c.us_OUT-1", "text": PROBE},
        {"id": f"false_{PHONE}@c.us_IN-1", "text": "Fresh answer 1"},
    ]
    assert server.initial_histories[2:] == [expected_history, expected_history]
    assert server.drawer_openings[0] == 1
    assert server.drawer_present_at_send[0] is False
    guide = yaml.safe_load(output.read_text())
    assert guide["validation"]["replay_verified"] is True
    assert guide["validation"]["server_state_reset"] is False
    assert set(guide["validation"]["observed_behaviors"]) == {
        "open",
        "send",
        "receive",
        "completion",
    }
    assert completed.is_file()
    assert yaml.safe_load(completed.read_text())["channel_guide"] == output.name
    artifacts = output.read_text() + completed.read_text()
    safe_logs = result.output + caplog.text
    for value in (
        PHONE,
        server.drawer_phone_label,
        "12345678999",
        CONTACT_NAME,
        PROBE,
        "Fresh answer",
        "OUT-1",
        "IN-1",
        "credential-private-canary",
        "profile-private-canary",
    ):
        assert value not in artifacts
        assert value not in safe_logs
    assert setup.provider.closed
    assert all(context.pages == [] for context in server.contexts)


@pytest.mark.parametrize("boundary", ["api", "cli"])
def test_named_empty_chat_mismatching_drawer_phone_fails_without_sending_or_publication(
    setup: Any, monkeypatch: Any, boundary: str
) -> None:
    server = PersistentConversation(named_empty=True, drawer_phone="15557654321")
    server.install(monkeypatch)
    if boundary == "api":
        with pytest.raises(ChannelError):
            invoke(setup, recipient=PHONE, probe_message=PROBE)
    else:
        from typer.testing import CliRunner

        from syntheticcx.cli import app

        result = CliRunner().invoke(
            app,
            [
                "discover",
                str(setup.draft),
                "--channel",
                "whatsapp_web",
                "--url",
                WEBSITE,
                "--browser-profile",
                str(setup.profile),
                "--recipient",
                PHONE,
                "--probe-message",
                PROBE,
                "--output",
                str(setup.draft.parent / "guide.yaml"),
                "--verbose",
            ],
        )
        assert result.exit_code == 3, result.output
        assert "unexpected error" not in result.output
        for value in (PHONE, server.drawer_phone, server.drawer_phone_label, CONTACT_NAME, PROBE):
            assert value not in result.output
    assert server.launches == 1
    assert server.initial_histories == [[], []]
    assert server.drawer_openings == [1]
    assert server.submissions == []
    assert server.history == []
    assert not (setup.draft.parent / "guide.yaml").exists()
    assert not setup.draft.with_name("draft.completed.yaml").exists()
    assert setup.provider.closed
    assert all(context.pages == [] for context in server.contexts)


def test_localized_probe_is_owned_and_replayed_exactly(setup: Any, monkeypatch: Any) -> None:
    server = PersistentConversation()
    server.install(monkeypatch)
    invoke(setup, language="pt-BR")
    assert server.submissions == [first_probe("pt-BR"), first_probe("pt-BR")]


@pytest.mark.parametrize(
    ("variant", "launches", "sends"),
    [("failed_replay", 2, 2), ("missing_echo", 1, 1), ("silent", 1, 1)],
)
def test_failed_fresh_evidence_never_returns_guide_or_retries(
    setup: Any, monkeypatch: Any, variant: str, launches: int, sends: int
) -> None:
    server = PersistentConversation(variant)
    server.install(monkeypatch)
    with pytest.raises(ChannelError):
        invoke(setup, probe_message=PROBE, max_discovery_attempts=5)
    assert server.launches == launches
    assert server.submissions == [PROBE] * sends
    assert setup.provider.closed
    assert all(context.pages == [] for context in server.contexts)
    assert not list(setup.draft.parent.glob("*guide*"))


def test_cleanup_failure_prevents_verified_result(setup: Any, monkeypatch: Any) -> None:
    server = PersistentConversation()
    server.install(monkeypatch)

    async def failed_close() -> None:
        raise RuntimeError("private-provider-diagnostic")

    monkeypatch.setattr(setup.provider, "close", failed_close)
    with pytest.raises(ChannelError, match="cleanup exceeded discovery budget") as error:
        invoke(setup, probe_message=PROBE)
    assert "private-provider-diagnostic" not in str(error.value)
    assert server.submissions == [PROBE, PROBE]
    assert server.launches == 2
    assert all(context.pages == [] for context in server.contexts)


def test_preflight_and_cleanup_share_absolute_deadline(setup: Any, monkeypatch: Any) -> None:
    document = yaml.safe_load(setup.draft.read_text())
    document["execution"].update(journey_timeout_seconds=0.12, close_timeout_seconds=0.1)
    setup.draft.write_text(yaml.safe_dump(document))
    events: list[str] = []

    async def slow_preflight(*_args: Any, **_kwargs: Any) -> None:
        events.append("preflight")
        await asyncio.sleep(10)

    async def slow_close() -> None:
        events.append("close")
        await asyncio.sleep(10)

    monkeypatch.setattr(discovery, "_probe_role", slow_preflight)
    monkeypatch.setattr(setup.provider, "close", slow_close)
    server = PersistentConversation()
    server.install(monkeypatch)
    started = monotonic()
    with pytest.raises(ChannelError, match="budget exhausted"):
        invoke(setup)
    assert monotonic() - started < 0.3
    assert server.launches == 0
    assert events == ["preflight", "close"]


def test_private_website_query_is_rejected_before_any_business_contact(setup, monkeypatch):
    server = PersistentConversation()
    server.install(monkeypatch)
    with pytest.raises(ConfigurationError) as caught:
        invoke(setup, url=WEBSITE + "?session=PRIVATE_URL_TOKEN", probe_message=PROBE)
    assert "PRIVATE_URL_TOKEN" not in str(caught.value)
    assert server.submissions == []


@pytest.mark.parametrize("website_variant", ["intermediate", "lazy"])
def test_shared_explorer_follows_generic_page_widget_and_exact_fresh_path(
    setup: Any, monkeypatch: Any, website_variant: str
) -> None:
    server = PersistentConversation(website_variant=website_variant)
    server.install(monkeypatch)
    guide = invoke(setup, probe_message=PROBE)
    assert server.launches == 2
    assert server.submissions == [PROBE, PROBE]
    assert server.website_visits == [(1, "/"), (1, "/services"), (2, "/"), (2, "/services")]
    requests = setup.provider.requests
    offered = [
        item["description"]
        for request in requests
        for item in request["inventory"]["catalog"]["candidates"]
    ]
    assert "Explore" in offered and "Next" in offered and "Continue" in offered
    assert any(
        "Select how to continue" in request["inventory"].get("page_context", "")
        for request in requests
    )
    assert guide.validation.replay_verified
    assert all(context.pages == [] for context in server.contexts)


def test_navigation_replay_drift_never_scans_alternative_or_sends_again(
    setup: Any, monkeypatch: Any
) -> None:
    server = PersistentConversation(website_variant="replay_drift")
    server.install(monkeypatch)
    with pytest.raises(ChannelError):
        invoke(setup, probe_message=PROBE)
    assert server.launches == 2
    assert server.submissions == [PROBE]
    assert all(context.pages == [] for context in server.contexts)
    assert not list(setup.draft.parent.glob("*guide*"))
