"""Real Chromium: derive reading controls from dynamically created, nonstandard bubbles."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from syntheticcx.adapters.channel.playwright import PlaywrightChannel
from syntheticcx.adapters.discovery.native import (
    CalibrationObservationUnavailable,
    NativeCalibrationBrowser,
)
from syntheticcx.config.models import ChannelConfig, ExecutionLimits, PrivacyConfig
from syntheticcx.discovery.calibrate import calibrate_candidate
from syntheticcx.discovery.explore import DiscoveryCandidate
from syntheticcx.discovery.verify import verify_candidate
from syntheticcx.errors import ChannelError
from syntheticcx.privacy import Redactor

_CHAT = """<div role="dialog" hidden id="chat"><div id="messages"></div>
<textarea aria-label="Message"></textarea><button aria-label="Send message">Send</button></div>
<script>
const chat = document.querySelector('#chat');
const input = document.querySelector('textarea');
let count = 0;
function bubble(side, text) {
  const node = document.createElement('div');
  node.className = 'bubble'; node.dataset.side = side; node.dataset.key = `${side}-${++count}`;
  const span = document.createElement('span'); span.textContent = text;
  node.append(span); document.querySelector('#messages').append(node); return span;
}
document.querySelector('[aria-label="Send message"]').onclick = () => {
  bubble('user', input.value); input.value = '';
  if (new URLSearchParams(location.search).get('variant') === 'silent_response') return;
  setTimeout(() => {
    const reply = bubble('bot', 'Olá!');
    setTimeout(() => { reply.textContent = 'Olá! Como posso ajudar com sua conta?'; }, 80);
  }, 100);
};
</script>"""
_INDEX = (
    """<button aria-label="Open chat" id="open">Open chat</button>"""
    + _CHAT
    + """
<script>
document.querySelector('#open').onclick = () => {
  if (new URLSearchParams(location.search).get('view') === 'frame') {
    const frame = document.createElement('iframe'); frame.id = 'chat-frame';
    frame.src = '/frame.html' + location.search; document.body.append(frame);
  } else chat.hidden = false;
};
</script>"""
)
_FRAME = _CHAT + "<script>chat.hidden = false;</script>"


@pytest.mark.parametrize("view", ["main", "frame"])
@pytest.mark.parametrize(
    "composer", ["removed", "relabeled", "widget_replaced", "region_detached", "restored"]
)
def test_calibration_observes_reply_without_original_composer(
    tmp_path: Path, monkeypatch: Any, view: str, composer: str
) -> None:
    sends: list[PlaywrightChannel] = []
    original_send = PlaywrightChannel.send

    async def tracked_send(self: PlaywrightChannel, text: str, **kwargs: Any) -> Any:
        sends.append(self)
        return await original_send(self, text, **kwargs)

    monkeypatch.setattr(PlaywrightChannel, "send", tracked_send)
    mutation = {
        "removed": "input.remove();",
        "relabeled": "input.setAttribute('aria-label', 'New composer');",
        "widget_replaced": (
            "const replacement = chat.cloneNode(true); chat.replaceWith(replacement);"
        ),
        "region_detached": "chat.remove(); setTimeout(() => document.body.append(chat), 600);",
        "restored": "input.remove(); setTimeout(() => chat.append(input), 600);",
    }[composer]
    with fixture(tmp_path) as server:
        for name in ("index.html", "frame.html"):
            path = tmp_path / name
            document = path.read_text().replace("input.value = '';", "input.value = '';" + mutation)
            if composer == "region_detached":
                document = document.replace(
                    "document.querySelector('#messages').append(node)",
                    "chat.querySelector('#messages').append(node)",
                )
            path.write_text(document)
        initial, limits = candidate(f"{server.base_url}/?variant=success&view={view}", view)
        model = ObservedModel(initial)
        progress: list[str] = []

        async def run() -> DiscoveryCandidate:
            refined = await calibrate_candidate(
                initial,
                limits,
                "Olá",
                model,
                Redactor(PrivacyConfig()),
                on_progress=progress.append,
            )
            if composer == "restored":
                assert sends[0]._page is None  # Calibration has closed before verification.
                guide = await verify_candidate(refined, limits, "Olá", language="pt-BR")
                assert guide.validation.status == "verified"
            return refined

        refined = asyncio.run(run())
    assert refined.channel.message_identity.attribute_name == "data-key"
    assert model.calls == 1
    assert progress.count("calibration probe sent once; observing chat DOM changes") == 1
    assert progress[-1] == "closing native calibration browser"
    assert len(sends) == (3 if composer == "restored" else 1)
    assert all(browser._page is None for browser in sends)
    if composer == "restored":
        assert sends[0] is not sends[1] and sends[1] is sends[2]
    if composer == "region_detached":
        assert "chat observation scope recovered; no probe resent" in progress


def fixture(tmp_path: Path) -> Any:
    server = Path("examples/telecom/webchat_fixture/server.py").resolve()
    spec = importlib.util.spec_from_file_location("calibration_fixture", server)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    (tmp_path / "index.html").write_text(_INDEX)
    (tmp_path / "frame.html").write_text(_FRAME)
    return module.FixtureServer()


def candidate(url: str, view: str) -> tuple[DiscoveryCandidate, ExecutionLimits]:
    document = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    channel = document["channel"]
    # Calibration remains a private legacy sample, not a public runtime guide.
    channel["type"] = "webchat_playwright"
    channel.pop("ownership")
    channel.pop("frame_locator", None)
    channel.update(
        {
            "url": url,
            "open_locator": {"strategy": "role", "value": "button", "name": "Open chat"},
            "input_locator": {"strategy": "label", "value": "Message"},
            "send_strategy": {
                "type": "click",
                "locator": {"strategy": "role", "value": "button", "name": "Send message"},
            },
            "customer_message_locator": {"strategy": "css", "value": ".guessed-customer"},
            "assistant_message_locator": {"strategy": "css", "value": ".guessed-assistant"},
        }
    )
    if view == "frame":
        channel["frame_locator"] = {"strategy": "css", "value": "#chat-frame"}
    document["execution"].update(
        {
            "response_timeout_seconds": 4,
            "browser_action_timeout_seconds": 3,
            "greeting_grace_seconds": 0,
        }
    )
    return DiscoveryCandidate(channel=channel, open_steps=1), ExecutionLimits.model_validate(
        document["execution"]
    )


class ObservedModel:
    def __init__(self, initial: DiscoveryCandidate) -> None:
        self.initial = initial
        self.calls = 0

    async def generate(self, request: Any) -> Any:
        self.calls += 1
        evidence, _ = json.JSONDecoder().raw_decode(request.messages[-1].content)
        assert "Como posso ajudar" in request.messages[-1].content
        assert "data-key" in request.messages[-1].content
        assert "Como posso ajudar" not in json.dumps(evidence["baseline"])
        channel = self.initial.channel.model_dump(mode="json")
        channel.update(
            {
                "url": "__SYNTHETICCX_TARGET_URL__",
                "customer_message_locator": {
                    "strategy": "css",
                    "value": '.bubble[data-side="user"]',
                },
                "assistant_message_locator": {
                    "strategy": "css",
                    "value": '.bubble[data-side="bot"]',
                },
                "message_identity": {"strategy": "attribute", "attribute_name": "data-key"},
                "response_completion": {"strategy": "quiet_interval", "quiet_interval_seconds": 1},
            }
        )
        return type(
            "Response", (), {"content": json.dumps({"channel_json": json.dumps(channel)})}
        )()


@pytest.mark.parametrize("view", ["main", "frame"])
@pytest.mark.parametrize("mistake", ["child", "wrong_attribute", "duplicate"])
def test_identity_correction_explores_observed_attributes_without_resending(
    tmp_path: Path, monkeypatch: Any, view: str, mistake: str
) -> None:
    browsers: list[PlaywrightChannel] = []
    original_send = PlaywrightChannel.send

    async def tracked_send(self: PlaywrightChannel, text: str, **kwargs: Any) -> Any:
        browsers.append(self)
        return await original_send(self, text, **kwargs)

    monkeypatch.setattr(PlaywrightChannel, "send", tracked_send)

    class ExploringModel(ObservedModel):
        async def generate(self, request: Any) -> Any:
            response = await super().generate(request)
            channel = json.loads(json.loads(response.content)["channel_json"])
            if self.calls == 1:
                if mistake == "child":
                    channel["assistant_message_locator"]["value"] += " span"
                else:
                    channel["message_identity"]["attribute_name"] = (
                        "data-side" if mistake == "duplicate" else "data-not-present"
                    )
            else:
                feedback = request.messages[-1].content.split("evidence:\n", 1)[1]
                evidence = json.loads(feedback)
                selected = evidence["assistant"]
                if mistake == "child":
                    assert selected["missing_or_blank_identities"] == 1
                    ancestors = selected["samples"][0]["ancestors"]
                    bubble = next(node for node in ancestors if "data-key" in node["attributes"])
                    assert bubble["tag"] == "div"
                    channel["assistant_message_locator"]["value"] = (
                        f'{bubble["tag"]}.bubble[data-side="bot"]'
                    )
                else:
                    field = (
                        "duplicate_identities"
                        if mistake == "duplicate"
                        else "missing_or_blank_identities"
                    )
                    assert selected[field] > 0
                    choices = selected["observed_attribute_counts"]
                    identity = next(
                        option
                        for option in choices
                        if option["attribute"] == "data-key"
                        and option["missing_or_blank"] == option["duplicate_values"] == 0
                    )
                    channel["message_identity"]["attribute_name"] = identity["attribute"]
            return type(
                "Response", (), {"content": json.dumps({"channel_json": json.dumps(channel)})}
            )()

    with fixture(tmp_path) as server:
        if mistake == "duplicate":
            for name in ("index.html", "frame.html"):
                path = tmp_path / name
                path.write_text(
                    path.read_text().replace(
                        "const reply = bubble('bot', 'Olá!');",
                        "bubble('bot', 'Additional reply'); const reply = bubble('bot', 'Olá!');",
                    )
                )
        initial, limits = candidate(f"{server.base_url}/?variant=success&view={view}", view)
        model = ExploringModel(initial)
        progress: list[str] = []

        async def run() -> None:
            refined = await calibrate_candidate(
                initial,
                limits,
                "Olá",
                model,
                Redactor(PrivacyConfig()),
                on_progress=progress.append,
            )
            assert refined.channel.message_identity.attribute_name == "data-key"
            assert browsers[0]._page is None
            guide = await verify_candidate(refined, limits, "Olá", language="pt-BR")
            assert guide.validation.status == "verified"

        asyncio.run(run())
    assert model.calls == 2
    assert len(browsers) == 3
    assert browsers[0] is not browsers[1] and browsers[1] is browsers[2]
    assert all(browser._page is None for browser in browsers)
    expected = "duplicate identities" if mistake == "duplicate" else "missing or blank identities"
    assert any(expected in message for message in progress)


@pytest.mark.parametrize("last_proposal", ["missing_identity", "malformed"])
def test_identityless_widget_fails_truthfully_without_publication(
    tmp_path: Path, monkeypatch: Any, last_proposal: str
) -> None:
    browsers: list[PlaywrightChannel] = []
    original_send = PlaywrightChannel.send

    async def tracked_send(self: PlaywrightChannel, text: str, **kwargs: Any) -> Any:
        browsers.append(self)
        return await original_send(self, text, **kwargs)

    monkeypatch.setattr(PlaywrightChannel, "send", tracked_send)

    class IdentitylessModel(ObservedModel):
        async def generate(self, request: Any) -> Any:
            # No unique message identity exists: intentionally keep the rejected proposal.
            self.calls += 1
            if self.calls > 1:
                assert "missing or blank identities" in request.messages[-1].content
            channel = self.initial.channel.model_dump(mode="json")
            channel.update(
                url="__SYNTHETICCX_TARGET_URL__",
                customer_message_locator={"strategy": "css", "value": '.bubble[data-side="user"]'},
                assistant_message_locator={"strategy": "css", "value": '.bubble[data-side="bot"]'},
                message_identity={"strategy": "attribute", "attribute_name": "data-key"},
            )
            content = json.dumps({"channel_json": json.dumps(channel)})
            if self.calls == 2 and last_proposal == "malformed":
                content = "private malformed value"
            return type("Response", (), {"content": content})()

    with fixture(tmp_path) as server:
        (tmp_path / "index.html").write_text(
            _INDEX.replace("node.dataset.key = `${side}-${++count}`;", "")
        )
        initial, limits = candidate(f"{server.base_url}/?variant=success", "main")
        model = IdentitylessModel(initial)
        with pytest.raises(ChannelError, match="missing or blank identities") as captured:
            asyncio.run(
                calibrate_candidate(
                    initial, limits, "Olá", model, Redactor(PrivacyConfig()), max_attempts=2
                )
            )
    assert model.calls == 2 and len(browsers) == 1 and browsers[0]._page is None
    assert "private" not in str(captured.value)
    if last_proposal == "malformed":
        assert "invalid JSON; previous rejection:" in str(captured.value)
    assert not list(tmp_path.glob("*.yaml"))


@pytest.mark.parametrize("view,large_page", [("main", False), ("frame", False), ("main", True)])
def test_calibration_reads_actual_reply_then_verifies_in_fresh_browser(
    tmp_path: Path,
    view: str,
    large_page: bool,
) -> None:
    with fixture(tmp_path) as server:
        if large_page:
            noise = "<section>" + "<p>Unrelated page content</p>" * 1200 + "</section>"
            (tmp_path / "index.html").write_text(
                _INDEX.replace('<div role="dialog"', noise + "<div")
            )
        initial, limits = candidate(f"{server.base_url}/?variant=success&view={view}", view)
        model = ObservedModel(initial)
        progress: list[str] = []

        async def run() -> None:
            refined = await calibrate_candidate(
                initial,
                limits,
                "Olá",
                model,
                Redactor(PrivacyConfig()),
                on_progress=progress.append,
            )
            assert refined.channel.message_identity.attribute_name == "data-key"
            guide = await verify_candidate(
                refined, limits, "Olá", language="pt-BR", on_progress=progress.append
            )
            assert guide.validation.status == "verified"

        asyncio.run(run())
    assert model.calls == 1
    assert progress.count("calibration probe sent once; observing chat DOM changes") == 1
    assert "completed assistant response received for probe 2/2" in progress


def test_overlapping_actor_selectors_are_corrected_without_another_submission(
    tmp_path: Path,
) -> None:
    class OverlappingModel(ObservedModel):
        async def generate(self, request: Any) -> Any:
            response = await super().generate(request)
            if self.calls == 1:
                wrapper = json.loads(response.content)
                channel = json.loads(wrapper["channel_json"])
                channel["customer_message_locator"] = {"strategy": "css", "value": ".bubble"}
                response.content = json.dumps({"channel_json": json.dumps(channel)})
            return response

    with fixture(tmp_path) as server:
        initial, limits = candidate(f"{server.base_url}/?variant=success", "main")
        model = OverlappingModel(initial)
        progress: list[str] = []
        refined = asyncio.run(
            calibrate_candidate(
                initial,
                limits,
                "Olá",
                model,
                Redactor(PrivacyConfig()),
                max_attempts=2,
                on_progress=progress.append,
            )
        )
    assert model.calls == 2
    assert refined.channel.customer_message_locator.value == '.bubble[data-side="user"]'
    assert progress.count("calibration probe sent once; observing chat DOM changes") == 1


def test_silent_calibration_does_not_ask_model_or_publish(tmp_path: Path) -> None:
    with fixture(tmp_path) as server:
        initial, limits = candidate(f"{server.base_url}/?variant=silent_response", "main")
        limits = limits.model_copy(update={"response_timeout_seconds": 0.5})
        model = ObservedModel(initial)
        with pytest.raises(ChannelError, match="no stable new reply"):
            asyncio.run(
                calibrate_candidate(initial, limits, "Olá", model, Redactor(PrivacyConfig()))
            )
    assert model.calls == 0
    assert not list(tmp_path.glob("*.yaml"))


def test_calibration_reacquires_replaced_frame_document_without_resending(
    tmp_path: Path, monkeypatch: Any
) -> None:
    original_send = NativeCalibrationBrowser.send
    browsers: list[NativeCalibrationBrowser] = []

    async def replace_frame(self: NativeCalibrationBrowser, text: str, **kwargs: Any) -> Any:
        browsers.append(self)
        receipt = await original_send(self, text, **kwargs)
        assert self._page is not None
        await self._page.evaluate("""() => {
          const frame = document.querySelector('#chat-frame');
          const replacement = frame.cloneNode(false);
          const transcript = frame.contentDocument.querySelector('#chat').outerHTML;
          replacement.onload = () => {
            replacement.contentDocument.querySelector('#chat').outerHTML = transcript;
          };
          frame.remove();
          setTimeout(() => document.body.append(replacement), 200);
        }""")
        # The missing configured frame is classified without waiting on the composer.
        with pytest.raises(CalibrationObservationUnavailable, match="frame is unavailable"):
            await self.inventory()
        return receipt

    monkeypatch.setattr(NativeCalibrationBrowser, "send", replace_frame)
    with fixture(tmp_path) as server:
        initial, limits = candidate(f"{server.base_url}/?variant=success&view=frame", "frame")
        model = ObservedModel(initial)
        progress: list[str] = []
        refined = asyncio.run(
            calibrate_candidate(
                initial,
                limits,
                "Olá",
                model,
                Redactor(PrivacyConfig()),
                on_progress=progress.append,
            )
        )
    assert refined.channel.message_identity.attribute_name == "data-key"
    assert len(browsers) == model.calls == 1
    assert browsers[0]._page is None
    assert "chat observation scope recovered; no probe resent" in progress


@pytest.mark.parametrize("view", ["main", "frame"])
def test_missing_observation_region_fails_closed_after_single_submission(
    tmp_path: Path, monkeypatch: Any, view: str
) -> None:
    original_send = NativeCalibrationBrowser.send
    browsers: list[NativeCalibrationBrowser] = []

    async def remove_region(self: NativeCalibrationBrowser, text: str, **kwargs: Any) -> Any:
        browsers.append(self)
        receipt = await original_send(self, text, **kwargs)
        assert self._scope is not None
        await self._scope.locator("#chat").evaluate("node => node.remove()")
        return receipt

    monkeypatch.setattr(NativeCalibrationBrowser, "send", remove_region)
    with fixture(tmp_path) as server:
        initial, limits = candidate(f"{server.base_url}/?variant=success&view={view}", view)
        limits = limits.model_copy(update={"response_timeout_seconds": 0.4})
        model = ObservedModel(initial)
        with pytest.raises(ChannelError, match="scope is unavailable before response deadline"):
            asyncio.run(
                calibrate_candidate(initial, limits, "Olá", model, Redactor(PrivacyConfig()))
            )
    assert len(browsers) == 1 and browsers[0]._page is None
    assert model.calls == 0
    assert not list(tmp_path.glob("*.yaml"))


def test_post_send_cross_origin_frame_fails_before_reading_missing_region(
    tmp_path: Path, monkeypatch: Any
) -> None:
    other_root = tmp_path / "other"
    other_root.mkdir()
    original_send = NativeCalibrationBrowser.send
    browsers: list[NativeCalibrationBrowser] = []
    with fixture(other_root) as other_server, fixture(tmp_path) as server:
        (other_root / "frame.html").write_text("<body>Private unrelated page</body>")

        async def navigate_frame(self: NativeCalibrationBrowser, text: str, **kwargs: Any) -> Any:
            browsers.append(self)
            receipt = await original_send(self, text, **kwargs)
            assert self._page is not None
            frame = self._page.frames[-1]
            await frame.goto(f"{other_server.base_url}/frame.html")
            return receipt

        monkeypatch.setattr(NativeCalibrationBrowser, "send", navigate_frame)
        initial, limits = candidate(f"{server.base_url}/?variant=success&view=frame", "frame")
        model = ObservedModel(initial)
        with pytest.raises(ChannelError, match="selected target origin") as captured:
            asyncio.run(
                calibrate_candidate(initial, limits, "Olá", model, Redactor(PrivacyConfig()))
            )
    assert captured.value.operation == "discover navigation"
    assert len(browsers) == 1 and browsers[0]._page is None
    assert model.calls == 0
    assert not list(tmp_path.glob("*.yaml"))


def test_removed_composer_on_large_page_keeps_capture_near_original_chat(tmp_path: Path) -> None:
    class BoundedModel(ObservedModel):
        async def generate(self, request: Any) -> Any:
            assert "UNRELATED_FAR_SECRET" not in request.messages[-1].content
            assert '"hint"' not in request.messages[-1].content
            return await super().generate(request)

    with fixture(tmp_path) as server:
        noise = "<section><p>UNRELATED_FAR_SECRET</p>" + "<p>Noise</p>" * 1200 + "</section>"
        document = _INDEX.replace('<div role="dialog"', noise + "<div")
        (tmp_path / "index.html").write_text(
            document.replace("input.value = '';", "input.value = ''; input.remove();")
        )
        initial, limits = candidate(f"{server.base_url}/?variant=success", "main")
        model = BoundedModel(initial)
        refined = asyncio.run(
            calibrate_candidate(initial, limits, "Olá", model, Redactor(PrivacyConfig()))
        )
    assert model.calls == 1
    assert refined.channel.message_identity.attribute_name == "data-key"


def test_cross_origin_during_identity_inspection_is_not_a_model_correction(
    tmp_path: Path, monkeypatch: Any
) -> None:
    other_root = tmp_path / "identity-other"
    other_root.mkdir()
    original_evidence = NativeCalibrationBrowser.message_evidence
    inspected: list[NativeCalibrationBrowser] = []
    with fixture(other_root) as other_server, fixture(tmp_path) as server:
        (other_root / "frame.html").write_text("<body>Private unrelated transcript</body>")

        async def navigate_before_inspection(self: NativeCalibrationBrowser, config: Any) -> Any:
            inspected.append(self)
            assert self._page is not None
            await self._page.frames[-1].goto(f"{other_server.base_url}/frame.html")
            return await original_evidence(self, config)

        monkeypatch.setattr(
            NativeCalibrationBrowser, "message_evidence", navigate_before_inspection
        )
        initial, limits = candidate(f"{server.base_url}/?variant=success&view=frame", "frame")
        model = ObservedModel(initial)
        with pytest.raises(ChannelError, match="selected target origin") as captured:
            asyncio.run(
                calibrate_candidate(initial, limits, "Olá", model, Redactor(PrivacyConfig()))
            )
    assert captured.value.operation == "discover navigation"
    assert len(inspected) == model.calls == 1 and inspected[0]._page is None


@pytest.mark.parametrize("view", ["main", "frame"])
def test_idless_bubbles_use_node_identity_and_repair_decorated_customer_echo(
    tmp_path: Path, monkeypatch: Any, view: str
) -> None:
    browsers: list[PlaywrightChannel] = []
    original_send = PlaywrightChannel.send

    async def tracked_send(self: PlaywrightChannel, text: str, **kwargs: Any) -> Any:
        browsers.append(self)
        return await original_send(self, text, **kwargs)

    monkeypatch.setattr(PlaywrightChannel, "send", tracked_send)

    with fixture(tmp_path) as server:
        for name in ("index.html", "frame.html"):
            path = tmp_path / name
            path.write_text(
                path.read_text()
                .replace("node.dataset.key = `${side}-${++count}`;", "")
                .replace(
                    "node.append(span);",
                    "if (side === 'user') node.append('Você enviou:'); node.append(span);",
                )
            )
        initial, limits = candidate(f"{server.base_url}/?variant=success&view={view}", view)
        progress: list[str] = []

        # The fixture contains no key attribute; this provider only checks reply evidence.
        class Provider:
            async def generate(self, request: Any) -> Any:
                channel = initial.channel.model_dump(mode="json")
                channel.update(
                    url="__SYNTHETICCX_TARGET_URL__",
                    customer_message_locator={
                        "strategy": "css",
                        "value": '.bubble[data-side="user"]',
                    },
                    assistant_message_locator={
                        "strategy": "css",
                        "value": '.bubble[data-side="bot"]',
                    },
                    message_identity={"strategy": "dom_node"},
                    response_completion={"strategy": "quiet_interval", "quiet_interval_seconds": 1},
                )
                return type(
                    "Response", (), {"content": json.dumps({"channel_json": json.dumps(channel)})}
                )()

        async def run() -> None:
            refined = await calibrate_candidate(
                initial,
                limits,
                "Olá",
                Provider(),
                Redactor(PrivacyConfig()),
                on_progress=progress.append,
            )
            guide = await verify_candidate(refined, limits, "Olá", language="pt-BR")
            assert guide.validation.status == "verified"
            assert guide.channel.message_identity.strategy == "dom_node"
            assert "span" in guide.channel.customer_message_locator.value

        asyncio.run(run())
    assert len(browsers) == 3 and browsers[0] is not browsers[1]
    assert all(browser._page is None for browser in browsers)
    assert "customer echo selector validated from observed message structure" in progress


def test_node_identity_rejects_replaced_delivered_bubbles(tmp_path: Path) -> None:
    with fixture(tmp_path) as server:
        initial, limits = candidate(f"{server.base_url}/?variant=success", "main")
        config = ChannelConfig.model_validate(
            {
                **initial.channel.model_dump(mode="json"),
                "customer_message_locator": {
                    "strategy": "css",
                    "value": '.bubble[data-side="user"]',
                },
                "assistant_message_locator": {
                    "strategy": "css",
                    "value": '.bubble[data-side="bot"]',
                },
                "message_identity": {"strategy": "dom_node"},
                "response_completion": {"strategy": "quiet_interval", "quiet_interval_seconds": 1},
            }
        )

        async def run() -> None:
            channel = PlaywrightChannel(config, limits)
            try:
                await channel.start()
                await channel.send("Olá")
                first = await channel.receive()
                await channel.send("Olá")
                second = await channel.receive()
                assert first.messages[0].content == second.messages[0].content
                assert first.messages[0].id != second.messages[0].id
                assert channel._scope is not None
                await channel._scope.locator('.bubble[data-side="bot"]').first.evaluate(
                    "node => node.replaceWith(node.cloneNode(true))"
                )
                with pytest.raises(ChannelError, match="observed message disappeared"):
                    await channel.send("Olá")
            finally:
                await channel.close()
            assert channel._node_handles == {"customer": [], "assistant": []}

        asyncio.run(run())


@pytest.mark.parametrize("completion_failure", [False, True])
def test_outer_verification_timeout_preserves_message_observation_stage(
    tmp_path: Path,
    completion_failure: bool,
) -> None:
    with fixture(tmp_path) as server:
        initial, limits = candidate(f"{server.base_url}/?variant=success", "main")
        config = initial.channel.model_dump(mode="json")
        config["customer_message_locator"] = {
            "strategy": "css",
            "value": '.bubble[data-side="user"]',
        }
        config["message_identity"] = {"strategy": "attribute", "attribute_name": "data-key"}
        if completion_failure:
            config["assistant_message_locator"] = {
                "strategy": "css",
                "value": '.bubble[data-side="bot"]',
            }
        config["response_completion"] = {"strategy": "quiet_interval", "quiet_interval_seconds": 30}
        limits = limits.model_copy(update={"response_timeout_seconds": 0.5})
        progress: list[str] = []
        expected = (
            "completion was not observed" if completion_failure else "did not observe a new bubble"
        )
        with pytest.raises(ChannelError, match=expected):
            asyncio.run(
                verify_candidate(
                    DiscoveryCandidate(channel=config, open_steps=1),
                    limits,
                    "Olá",
                    on_progress=progress.append,
                )
            )
    assert not any("probe 2/2" in line for line in progress)
    assert any(expected in line for line in progress)
