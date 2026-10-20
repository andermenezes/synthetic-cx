"""Actual Chromium WhatsApp-shaped journeys with offline persona/model boundaries.

Routing replaces network delivery, not the channel or runtime. These tests establish
local integration and history isolation, not live WhatsApp selector compatibility.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml
from playwright.async_api import BrowserType

from syntheticcx.adapters.llm.preflight import CapabilityPreflight
from syntheticcx.application import run as application
from syntheticcx.domain.models import (
    LLMResponse,
    ModelIdentity,
    ModelInteraction,
    PersonaAction,
    TokenUsage,
)

ANSWER = "Your invoice is corrected. The refund is approved."


class GoalSeekingCustomer:
    def __init__(self, variant="resolved"):
        self.observations = []
        self.variant = variant
        self.delayed = False

    async def initialize(self, definition, context):
        return definition.fresh_state()

    async def respond(self, observation, state):
        self.observations.append(observation)
        if not self.delayed and (
            (self.variant == "hydrated_unresolved" and state.turn_number == 0)
            or (self.variant == "late_resolved" and state.turn_number == 3)
        ):
            self.delayed = True
            await asyncio.sleep(0.4)
        if state.turn_number == 0:
            return PersonaAction(
                message="Please correct my invoice and confirm my refund.",
                state=state,
                signal="continue",
            )
        if state.turn_number == 1:
            option = next(o for o in observation.current_options if o.label == "Plans")
            return PersonaAction(selected_option_id=option.id, state=state, signal="continue")
        if state.turn_number == 2:
            return PersonaAction(
                message="My account number is SYNTHETIC-001.", state=state, signal="continue"
            )
        # Deliberately believe completion in both cases: formal evaluation must disagree
        # when the assistant did not resolve the goal.
        return PersonaAction(state=state, signal="goal_believed_complete")

    async def close(self):
        pass


class EvidenceCheckingModel:
    def __init__(self):
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        payload = json.loads(request.messages[-1].content)
        assistant = [m for m in payload["transcript"] if m["actor"] == "assistant"]
        final = assistant[-1]
        passed = final["content"] == ANSWER
        now = datetime.now(UTC)
        return LLMResponse(
            content=json.dumps(
                {
                    "passed": passed,
                    "explanation": "The current-run answer resolves the goal."
                    if passed
                    else "The current-run answer explicitly does not resolve the goal.",
                    "evidence": [
                        {"event_id": final["event_id"], "message_id": final["message_id"]}
                    ],
                }
            ),
            interaction=ModelInteraction(
                id=request.id,
                identity=ModelIdentity(provider="openai", model="offline-evidence-checker"),
                started_at=now,
                ended_at=now,
                usage=TokenUsage(unavailable_reason="offline_model"),
            ),
        )

    async def close(self):
        pass


@pytest.mark.parametrize(
    "variant,menu_kind,exit_code,status",
    [
        ("resolved", "buttons", 0, "passed"),
        ("resolved", "list", 0, "passed"),
        ("unresolved", "buttons", 1, "failed"),
        ("unresolved", "list", 1, "failed"),
        ("late_resolved", "buttons", 0, "passed"),
        ("hydrated_unresolved", "buttons", 1, "failed"),
        ("limited", "buttons", 1, "failed"),
        ("abandon", "list", 1, "failed"),
        ("repetition", "list", 1, "failed"),
    ],
)
def test_whatsapp_access_does_not_replace_goal_evaluation(
    tmp_path, monkeypatch, variant, exit_code, status, menu_kind
):
    customer = GoalSeekingCustomer(variant)
    model = EvidenceCheckingModel()
    # One full goal journey exercises explicit precedence; the others retain env fallback.
    recipient = "+15551234567" if variant == "resolved" and menu_kind == "buttons" else None
    monkeypatch.setenv("OPENAI_API_KEY", "offline-whatsapp-model-secret")
    monkeypatch.setenv(
        "SYNTHETICCX_WHATSAPP_RECIPIENT", "15557654321" if recipient else "15551234567"
    )
    monkeypatch.setattr(application, "TinyTroupePersonaSimulator", lambda *a, **kw: customer)
    monkeypatch.setattr(application, "OpenAIProvider", lambda *a, **kw: model)

    async def preflight(*args):
        return CapabilityPreflight()

    monkeypatch.setattr(application, "preflight_models", preflight)
    original_launch = BrowserType.launch_persistent_context
    fixture = Path("tests/fixtures/whatsapp_web.html").read_text()
    surface_variant = "resolved" if variant == "resolved" else "unresolved"
    fixture = fixture.replace("window.variant='resolved'", f"window.variant='{surface_variant}'")
    if recipient is not None:
        # The first owned message must bootstrap from a named, empty business chat.
        fixture = fixture.replace(
            '<header><span title="+15551234567">Controlled Business</span></header>',
            """<header><div role="button" data-testid="conversation-info-header"
              onclick="openContactInfo()">
              <span data-testid="conversation-info-header-chat-title">Controlled Business</span>
              <span>Business account</span></div></header>
              <script>function openContactInfo() {
                const drawer=document.createElement('div');
                drawer.dataset.testid='chat-info-drawer';
                drawer.innerHTML='<button aria-label="Close">Close</button><section>'
                  +'<div data-testid="section-about-and-phone-number">About and phone number</div>'
                  +'<div data-testid="container_with_separator"><div><div>'
                  +'<span data-testid="selectable-text"><span>+1 555 123 4567</span></span>'
                  +'</div></div></div></section>';
                document.body.append(drawer);
                drawer.querySelector('button').onclick=()=>drawer.remove();
              }</script>""",
        )
        fixture = fixture.replace(
            '<div data-id="false_15551234567@c.us_history"><span class="selectable-text">'
            f"{ANSWER}</span></div>",
            "",
        )
    if variant == "late_resolved":
        fixture = fixture.replace(
            "else incoming('I cannot resolve your invoice. Please contact support.');",
            "else {incoming('I cannot resolve your invoice. Please contact support.');"
            f"setTimeout(()=>incoming('{ANSWER}'),200);}}",
        )
    if variant == "repetition":
        fixture = fixture.replace(
            "else incoming('I cannot resolve your invoice. Please contact support.');",
            "else incoming('Please repeat your account reference?');",
        )
    if variant == "hydrated_unresolved":
        history = (
            '<div data-id="false_15551234567@c.us_history"><span class="selectable-text">'
            f"{ANSWER}</span></div>"
        )
        fixture = fixture.replace(history, "")
        fixture = fixture.replace(
            "</script>",
            f"setTimeout(()=>messages.insertAdjacentHTML('afterbegin',"
            f"{json.dumps(history)}),200);</script>",
        )
    fixture = fixture.replace("window.menuKind='buttons'", f"window.menuKind='{menu_kind}'")
    contexts = []

    async def routed_launch(self, *args, **kwargs):
        context = await original_launch(self, *args, **kwargs)
        contexts.append(context)

        async def route_request(route):
            host = urlsplit(route.request.url).hostname
            if host == "telecom.example":
                body = (
                    '<a href="https://web.whatsapp.com/send?phone=15551234567">Contact WhatsApp</a>'
                )
            elif host == "web.whatsapp.com":
                body = fixture
            else:
                await route.abort()
                return
            await route.fulfill(status=200, content_type="text/html", body=body)

        await context.route("**/*", route_request)
        return context

    monkeypatch.setattr(BrowserType, "launch_persistent_context", routed_launch)
    document = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    document["scenario"]["goal"] = "Have the incorrect invoice corrected and the refund approved."
    document["scenario"]["success_criteria"][0]["parameters"]["expected_facts"] = [ANSWER]
    document["channel"] = {
        "type": "whatsapp_web_playwright",
        "url": "https://telecom.example/support",
        "navigation_steps": [
            {
                "action": "click",
                "locator": {"strategy": "role", "value": "link", "name": "Contact WhatsApp"},
            }
        ],
        "quiet_interval_seconds": 0.08,
    }
    document["storage"]["run_root"] = str(tmp_path / "runs")
    document["telemetry"] = {"enabled": False}
    document["reporting"] = {"markdown": {"enabled": True}}
    document["execution"].update(greeting_grace_seconds=0, journey_timeout_seconds=30)
    if variant == "limited":
        document["execution"]["max_turns"] = 2
    if variant == "abandon":
        document["persona"]["behavioral_rules"] = [
            {
                "id": "leave-after-two-turns",
                "priority": 0,
                "when": {"signal": "customer_turn_completed", "occurrence_at_least": 2},
                "effects": {"abandon": True},
                "fire": "once",
            }
        ]
    if variant == "repetition":
        document["scenario"]["information"][0]["aliases"].append("account reference")
        document["persona"]["behavioral_rules"] = [
            {
                "id": "leave-after-repeated-account-request",
                "priority": 0,
                "when": {"signal": "repeated_information"},
                "effects": {
                    "frustration_delta": 0.5,
                    "willingness_to_continue_delta": -1,
                    "abandonment_threshold": 0.1,
                },
                "fire": "once",
            }
        ]
    source = tmp_path / "journey.yaml"
    source.write_text(yaml.safe_dump(document))
    profile = tmp_path / "dedicated-profile"
    profile.mkdir()
    result = asyncio.run(
        application.execute_run(source, browser_profile=profile, recipient=recipient)
    )
    assert result.exit_code == exit_code, (result.exit_code, result.report, result.partial_reason)
    assert result.report is not None
    turns = 2 if variant in {"limited", "abandon"} else 3
    termination = {
        "limited": "max_turns",
        "abandon": "abandonment",
        "repetition": "abandonment",
    }.get(variant, "goal_believed_complete")
    assert result.report.termination_reason == termination
    assert result.report.turn_count == turns
    directory = Path(result.artifact_location)
    events = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]
    messages = [e["data"]["message"] for e in events if e["type"] == "message_observed"]
    incoming = [m for m in messages if m["actor"] == "assistant"]
    assert [m["content"] for m in incoming[:2]] == [
        "Choose an option for invoice help.",
        "What is your account reference?",
    ]
    expected_final = (
        "What is your account reference?"
        if turns == 2
        else ANSWER
        if status == "passed"
        else "I cannot resolve your invoice. Please contact support."
    )
    if variant == "repetition":
        expected_final = "Please repeat your account reference?"
    assert incoming[-1]["content"] == expected_final
    assert [m["metadata"].get("action_kind") for m in messages if m["actor"] == "customer"] == [
        None,
        "select_option",
        None,
    ][:turns]
    findings = [e["data"]["result"] for e in events if e["type"] == "evaluation_completed"]
    assert findings[0]["status"] == status
    assert findings[0]["evidence_message_ids"] == [incoming[-1]["id"]]
    assert customer.observations[0].messages == []
    assert len(model.requests) == 1
    if variant == "repetition":
        transitions = [
            e["data"]
            for e in events
            if e["type"] == "behavioral_state_changed"
            and e["data"]["signal"] == "repeated_information"
        ]
        assert len(transitions) == 1
        assert transitions[0]["after"]["frustration"] > transitions[0]["before"]["frustration"]
        assert transitions[0]["after"]["abandoned"] is True
        assert len(customer.observations) == 3
    transcript = json.loads(model.requests[0].messages[-1].content)["transcript"]
    assert [m["content"] for m in transcript if m["actor"] == "assistant"] == [
        m["content"] for m in incoming
    ]
    persisted = "".join(
        p.read_text()
        for p in directory.iterdir()
        if p.suffix in {".json", ".jsonl", ".yaml", ".md"}
    )
    assert "15551234567" not in persisted
    assert "offline-whatsapp-model-secret" not in persisted
    assert "WEBSITE PREFILLED TEXT" not in persisted
    assert (directory / "report.md").is_file()
    assert all(not context.pages for context in contexts)
    if variant == "late_resolved":
        boundaries = [
            e["data"]["kind"] for e in events if e["type"] == "behavior_observation_recorded"
        ]
        assert boundaries == [
            "greeting",
            "customer_turn_completed",
            "customer_turn_completed",
            "customer_turn_completed",
            "assistant_update",
        ]
        completed_signals = [
            e
            for e in events
            if e["type"] == "behavioral_signal_recorded"
            and e["data"]["signal"]["signal"] == "customer_turn_completed"
        ]
        assert len(completed_signals) == 3
        from syntheticcx.runtime.behavior_replay import replay_behavior

        replay = replay_behavior(directory)
        assert replay.completeness == "complete"
        assert replay.comparison_findings == ()
