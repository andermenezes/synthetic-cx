"""Real browser choices pass through graph dispatch and the redacted evidence journal."""

import asyncio
from pathlib import Path

import pytest
import yaml
from playwright.async_api import async_playwright

from syntheticcx.adapters.channel.options import ReplyOptions
from syntheticcx.adapters.channel.playwright import PlaywrightChannel, locate
from syntheticcx.config.loader import load_config
from syntheticcx.config.models import ChannelConfig
from syntheticcx.domain.events import MessageObservedEvent
from syntheticcx.domain.models import PersonaAction, PersonaDefinition, Scenario
from syntheticcx.privacy import Redactor
from syntheticcx.runtime.clock import RealClock, RealWaitController
from syntheticcx.runtime.graph import build_graph
from syntheticcx.runtime.journal import RunJournal
from syntheticcx.runtime.state import initial_state


@pytest.mark.parametrize("shadow", [False, True])
def test_browser_option_is_one_graph_turn_with_redacted_truthful_evidence(shadow):
    class Store:
        async def append_event(self, _run_id, _event):
            pass

    class Customer:
        decisions = 0

        async def initialize(self, definition, _context):
            return definition.fresh_state()

        async def respond(self, observation, state):
            self.decisions += 1
            assert "private-menu" not in observation.model_dump_json()
            if self.decisions == 1:
                choice = observation.messages[-1].options[0]
                assert choice.label == "Plans"
                return PersonaAction(selected_option_id=choice.id, state=state, signal="continue")
            assert not observation.messages[-1].options
            return PersonaAction(signal="goal_believed_complete", state=state)

    async def exercise():
        project = load_config(Path("examples/telecom/scenario.yaml"))
        private = yaml.safe_load(Path("tests/fixtures/discovery-sample.yaml").read_text())
        config = ChannelConfig.model_validate(
            {
                **private["channel"],
                "url": "http://127.0.0.1:8000/",
                "open_locator": None,
                "frame_locator": None,
                "input_locator": {"strategy": "css", "value": "#composer"},
                "customer_message_locator": {"strategy": "css", "value": ".customer"},
                "assistant_message_locator": {"strategy": "css", "value": ".assistant"},
                "message_identity": {"strategy": "dom_node"},
                "send_strategy": {"type": "press_enter"},
                "response_completion": {
                    "strategy": "quiet_interval",
                    "quiet_interval_seconds": 0.05,
                },
                "reply_options": {
                    "scope_locator": {"strategy": "css", "value": "#chat"},
                    "option_locator": {"strategy": "css", "value": "button.option"},
                },
            }
        )
        limits = project.execution.model_copy(
            update={
                "greeting_grace_seconds": 2,
                "response_timeout_seconds": 2,
                "echo_observation_seconds": 0.01,
            }
        )
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            channel = PlaywrightChannel(config, limits)
            try:
                page = await browser.new_page()
                await page.route(
                    str(config.url), lambda route: route.fulfill(body='<div id="host"></div>')
                )
                await page.goto(str(config.url))
                await page.evaluate(
                    """shadow => {
                  const host = document.getElementById('host');
                  const root = shadow ? host.attachShadow({mode: 'open'}) : host;
                  root.innerHTML = '<section id="chat"><input id="composer"></section>';
                }""",
                    shadow,
                )
                channel._scope = channel._page = page
                await channel._capture_history()
                channel._options = ReplyOptions(page, config, locate)
                await channel._options.bind()
                await page.locator("#chat").evaluate("""chat => {
                  window.optionClicks = 0; window.textSends = 0;
                  chat.querySelector('input').onkeydown = () => window.textSends++;
                  chat.insertAdjacentHTML('beforeend', '<article class="assistant">' +
                    '<p>Choose a topic for service information. private-menu</p>' +
                    '<button class="option" type="button">Plans</button>' +
                    '<button class="option" type="button">Coverage</button></article>');
                  chat.querySelector('button').onclick = () => {
                    window.optionClicks++;
                    chat.querySelectorAll('button').forEach(node => node.remove());
                    chat.insertAdjacentHTML('beforeend',
                      '<article class="customer">plans_key</article>' +
                      '<article class="assistant">Plan information private-menu</article>');
                  };
                }""")
                journal = RunJournal(
                    "r",
                    Store(),
                    Redactor(project.privacy, resolved_secrets=["private-menu"]),
                    RealClock(),
                )
                definition = PersonaDefinition.model_validate(project.persona.model_dump())
                graph = build_graph(
                    scenario=Scenario.model_validate(project.scenario.model_dump()),
                    persona=definition,
                    channel=channel,
                    simulator=Customer(),
                    journal=journal,
                    wait=RealWaitController(),
                    limits=limits,
                )
                result = await graph.ainvoke(initial_state(definition))
                assert result["turn_count"] == 1
                assert result["termination_reason"] == "goal_believed_complete"
                messages = [
                    event.data.message
                    for event in journal.committed_events
                    if isinstance(event, MessageObservedEvent)
                ]
                assert len(messages) == 3
                assert [option.label for option in messages[0].options] == ["Plans", "Coverage"]
                assert messages[1].actor == "customer" and messages[1].content == "Plans"
                assert messages[1].metadata["action_kind"] == "select_option"
                assert messages[1].metadata["source_assistant_message_id"] == messages[0].id
                assert "private-menu" not in "".join(
                    message.model_dump_json() for message in messages
                )
                assert await page.evaluate("window.optionClicks") == 1
                assert await page.evaluate("window.textSends") == 0
            finally:
                await channel.close()
                await browser.close()

    asyncio.run(exercise())
