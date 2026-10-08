"""Local native state transitions: menus first, no replay autoplay."""

import asyncio

import pytest
from playwright.async_api import async_playwright

from syntheticcx.adapters.channel.broker import NativeChatBroker
from syntheticcx.config.models import StatefulChannelConfig
from syntheticcx.errors import ChannelError

URL = "http://localhost:8765/"
HTML = """<input id="decoy"><section id="chat" role="dialog" aria-label="Chat assistant">
<article class="assistant"><p>Choose an information topic</p>
<button type="button" class="option">Plans</button>
<button type="button" class="option">Coverage</button></article></section>
<script>
let count = 0;
document.querySelector('#chat').addEventListener('click', event => {
  if (!event.target.matches('.option')) return;
  count++;
  document.querySelectorAll('.option').forEach(node => node.remove());
  if (count === 1) document.querySelector('#chat').insertAdjacentHTML('beforeend',
    '<article class="assistant"><p>Choose information about services</p>' +
    '<button type="button" class="option">Internet</button>' +
    '<button type="button" class="option">Support</button></article>');
  else document.querySelector('#chat').insertAdjacentHTML('beforeend',
    '<article class="assistant">How can I help?</article><input id="composer">');
});
document.querySelector('#chat').addEventListener('keydown', event => {
  if (event.key !== 'Enter' || !event.target.matches('#composer,#next')) return;
  event.target.remove();
  document.querySelector('#chat').insertAdjacentHTML('beforeend',
    '<article class="assistant">Here are the details.</article><input id="next">');
});
</script>"""


def contract() -> StatefulChannelConfig:
    def css(value):
        return {"strategy": "css", "value": value}

    return StatefulChannelConfig.model_validate(
        {
            "type": "webchat_playwright_stateful",
            "url": URL,
            "ownership": {"root_locator": css("#chat"), "proof": "chat_dialog"},
            "assistant_message_locator": css("article.assistant"),
            "message_identity": {"strategy": "dom_node"},
            "response_completion": {"strategy": "quiet_interval", "quiet_interval_seconds": 0.04},
            "initial_state_id": "menu1",
            "states": [
                {
                    "id": "menu1",
                    "kind": "menu",
                    "option_locator": css(".option"),
                    "option_labels": ["Plans", "Coverage"],
                },
                {
                    "id": "menu2",
                    "kind": "menu",
                    "option_locator": css(".option"),
                    "option_labels": ["Internet", "Support"],
                },
                {
                    "id": "composer1",
                    "kind": "composer",
                    "input_locator": css("#composer"),
                    "send_strategy": {"type": "press_enter"},
                },
                {
                    "id": "composer2",
                    "kind": "composer",
                    "input_locator": css("#next"),
                    "send_strategy": {"type": "press_enter"},
                },
            ],
            "transitions": [
                {
                    "id": "t1",
                    "from_state": "menu1",
                    "to_state": "menu2",
                    "action_kind": "select_option",
                    "option_label": "Plans",
                },
                {
                    "id": "t2",
                    "from_state": "menu2",
                    "to_state": "composer1",
                    "action_kind": "select_option",
                    "option_label": "Internet",
                },
                {
                    "id": "t3",
                    "from_state": "composer1",
                    "to_state": "composer2",
                    "action_kind": "probe",
                    "probe_id": "first",
                },
            ],
            "verification_recipe": [
                {"transition_id": "t1"},
                {"transition_id": "t2"},
                {"transition_id": "t3"},
            ],
        }
    )


def test_menu_menu_composer_and_verified_composer_replacement() -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.route(
                "**/*", lambda route: route.fulfill(status=200, content_type="text/html", body=HTML)
            )
            await page.goto(URL)
            broker = NativeChatBroker(contract=contract(), page=page)
            await broker.start()
            assert await page.evaluate("count") == 0
            with pytest.raises(ChannelError):
                await broker.select_option("guessed")
            initial = await broker.observe_initial()
            assert initial is not None
            assert initial.messages[0].content == "Choose an information topic"
            assert [option.label for option in initial.messages[0].options] == ["Plans", "Coverage"]
            first = initial.messages[0].options[0].id
            await broker.select_option(first)
            reply = await broker.receive()
            assert broker.current_state.id == "menu2"
            assert [option.label for option in reply.messages[0].options] == ["Internet", "Support"]
            with pytest.raises(ChannelError):
                await broker.select_option(first)
            await broker.select_option(reply.messages[0].options[0].id)
            assert (await broker.receive()).messages[0].content == "How can I help?"
            assert broker.current_state.id == "composer1"
            await broker.send("Tell me about coverage")
            assert (await broker.receive()).messages[0].content == "Here are the details."
            assert broker.current_state.id == "composer2"
            assert await page.locator("#decoy").input_value() == ""
            await broker.close()
            await browser.close()

    asyncio.run(exercise())


def test_mixed_state_allows_text_and_verified_self_loop_allows_another_turn():
    async def exercise():
        document = contract().model_dump(mode="json")
        document["initial_state_id"] = "mixed"
        document["states"] = [
            {
                "id": "mixed",
                "kind": "mixed",
                "input_locator": {"strategy": "css", "value": "#composer"},
                "send_strategy": {"type": "press_enter"},
                "option_locator": {"strategy": "css", "value": ".option"},
                "option_labels": ["Help"],
            },
            {
                "id": "plain",
                "kind": "composer",
                "input_locator": {"strategy": "css", "value": "#composer"},
                "send_strategy": {"type": "press_enter"},
            },
        ]
        document["transitions"] = [
            {
                "id": "text",
                "from_state": "mixed",
                "to_state": "plain",
                "action_kind": "probe",
                "probe_id": "first",
            },
            {
                "id": "again",
                "from_state": "plain",
                "to_state": "plain",
                "action_kind": "probe",
                "probe_id": "second",
            },
        ]
        document["verification_recipe"] = [{"transition_id": "text"}, {"transition_id": "again"}]
        markup = """<section id="chat" role="dialog" aria-label="Chat">
          <article class="assistant">How can I help?
            <button class="option" type="button">Help</button></article>
          <input id="composer"></section><script>let submissions=0, selections=0;
          document.querySelector('.option').onclick=()=>selections++;
          document.querySelector('#composer').onkeydown=e=>{if(e.key==='Enter') {
            e.preventDefault(); e.target.value=''; submissions++;
            document.querySelector('.option')?.remove();
            document.querySelector('#chat').insertAdjacentHTML('beforeend',
              '<article class="assistant">Here is information.</article>');
          }};</script>"""
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await page.route(
                    "**/*", lambda route: route.fulfill(body=markup, content_type="text/html")
                )
                await page.goto(URL)
                broker = NativeChatBroker(
                    contract=StatefulChannelConfig.model_validate(document), page=page
                )
                try:
                    await broker.start()
                    initial = await broker.observe_initial()
                    assert initial.messages[0].options[0].label == "Help"
                    for _ in range(2):
                        await broker.send("Please share information")
                        assert (await broker.receive()).messages[
                            0
                        ].content == "Here is information."
                    assert await page.evaluate("submissions") == 2
                    assert await page.evaluate("selections") == 0
                finally:
                    await broker.close()
            finally:
                await browser.close()

    asyncio.run(exercise())


def test_persona_can_choose_other_guarded_options_when_result_is_a_verified_state():
    async def exercise():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await page.route(
                    "**/*", lambda route: route.fulfill(body=HTML, content_type="text/html")
                )
                await page.goto(URL)
                broker = NativeChatBroker(contract=contract(), page=page)
                try:
                    await broker.start()
                    initial = await broker.observe_initial()
                    # The learning recipe selected Plans/Internet, not these choices.
                    await broker.select_option(initial.messages[0].options[1].id)
                    response = await broker.receive()
                    assert broker.current_state.id == "menu2"
                    await broker.select_option(response.messages[0].options[1].id)
                    await broker.receive()
                    assert broker.current_state.id == "composer1"
                    for _ in range(2):
                        await broker.send("Please share information")
                        await broker.receive()
                    assert broker.current_state.id == "composer2"
                    assert await page.evaluate("count") == 2
                finally:
                    await broker.close()
            finally:
                await browser.close()

    asyncio.run(exercise())


def test_late_initial_menu_and_options_only_followup_wait_for_owned_evidence():
    async def exercise():
        document = contract()
        markup = """<script>
          setTimeout(()=>{
            document.body.innerHTML=
              '<section id="chat" role="dialog" aria-label="Chat assistant">' +
              '<article class="assistant"></article></section>';
            setTimeout(()=>document.querySelector('article').innerHTML=
              '<button type="button" class="option">Plans</button>' +
              '<button type="button" class="option">Coverage</button>',100);
            document.querySelector('#chat').onclick=e=>{
              if(e.target.tagName!=='BUTTON') return;
              document.querySelectorAll('button').forEach(n=>n.remove());
              const reply=document.createElement('article'); reply.className='assistant';
              document.querySelector('#chat').append(reply);
              setTimeout(()=>reply.innerHTML=
                '<button class="option" type="button">Internet</button>' +
                '<button class="option" type="button">Support</button>',200);
            };
          },100);
        </script>"""
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await page.route(
                    "**/*", lambda route: route.fulfill(body=markup, content_type="text/html")
                )
                await page.goto(URL)
                broker = NativeChatBroker(contract=document, page=page)
                try:
                    await broker.start()
                    initial = await broker.observe_initial()
                    assert initial.messages[0].content == ""
                    await broker.select_option(initial.messages[0].options[0].id)
                    reply = await broker.receive()
                    assert reply.messages[0].content == ""
                    assert [option.label for option in reply.messages[0].options] == [
                        "Internet",
                        "Support",
                    ]
                    assert broker.current_state.id == "menu2"
                finally:
                    await broker.close()
            finally:
                await browser.close()

    asyncio.run(exercise())


def test_recorded_edge_drift_cannot_fall_back_to_a_different_known_state():
    async def exercise():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await page.route(
                    "**/*",
                    lambda route: route.fulfill(
                        body=HTML.replace("count === 1", "count === -1"), content_type="text/html"
                    ),
                )
                await page.goto(URL)
                broker = NativeChatBroker(contract=contract(), page=page)
                try:
                    await broker.start()
                    option = (await broker.observe_initial()).messages[0].options[0].id
                    await broker.select_option(option)
                    with pytest.raises(ChannelError, match="unknown or ambiguous"):
                        await broker.receive()
                    with pytest.raises(ChannelError):
                        await broker.select_option(option)
                    assert await page.evaluate("count") == 1
                finally:
                    await broker.close()
            finally:
                await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "mutation",
    [
        "document.querySelector('#chat').outerHTML = document.querySelector('#chat').outerHTML",
        "document.querySelector('.option').outerHTML = document.querySelector('.option').outerHTML",
        "document.querySelector('.option').setAttribute('data-url','/purchase')",
        "document.querySelector('.option').textContent = 'Buy plans'",
        "document.querySelector('.option').disabled = true",
    ],
)
def test_stale_or_unsafe_actual_node_has_no_dispatch(mutation: str) -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.route(
                "**/*", lambda route: route.fulfill(status=200, content_type="text/html", body=HTML)
            )
            await page.goto(URL)
            broker = NativeChatBroker(contract=contract(), page=page)
            await broker.start()
            initial = await broker.observe_initial()
            await page.evaluate(mutation)
            with pytest.raises(ChannelError):
                await broker.select_option(initial.messages[0].options[0].id)
            assert await page.evaluate("count") == 0
            await broker.close()
            await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("placement", ["shadow", "frame"])
def test_pinned_shadow_and_frame_menus_are_initial_context(placement: str) -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.route(
                "**/*",
                lambda route: route.fulfill(
                    status=200, content_type="text/html", body="<body></body>"
                ),
            )
            await page.goto(URL)
            markup = HTML.split("<script>")[0]
            config = contract()
            if placement == "shadow":
                await page.evaluate(
                    "html => { const host = document.createElement('chat-widget'); "
                    "document.body.append(host); "
                    "host.attachShadow({mode:'open'}).innerHTML = html; }",
                    markup,
                )
            else:
                await page.evaluate(
                    "html => { const frame = document.createElement('iframe'); "
                    "frame.id='chat-frame'; frame.title='Chat assistant'; "
                    "frame.srcdoc=html; document.body.append(frame); }",
                    markup,
                )
                await page.locator("#chat-frame").content_frame.locator("#chat").wait_for()
                from syntheticcx.config.models import LocatorConfig

                config = config.model_copy(
                    update={"frame_locator": LocatorConfig(strategy="css", value="#chat-frame")}
                )
            broker = NativeChatBroker(contract=config, page=page)
            await broker.start()
            result = await broker.observe_initial()
            assert result.messages[0].content == "Choose an information topic"
            assert len(result.messages[0].options) == 2
            assert await broker.observe_initial() is None
            await broker.close()
            await browser.close()

    asyncio.run(exercise())


def test_uncertain_native_click_latches_before_dispatch_without_retry() -> None:
    from unittest.mock import patch

    from playwright.async_api import ElementHandle
    from playwright.async_api import Error as PlaywrightError

    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.route(
                "**/*", lambda route: route.fulfill(status=200, content_type="text/html", body=HTML)
            )
            await page.goto(URL)
            broker = NativeChatBroker(contract=contract(), page=page)
            await broker.start()
            initial = await broker.observe_initial()
            original = ElementHandle.click

            async def uncertain(node, **kwargs):
                await original(node, **kwargs)
                raise PlaywrightError("connection lost after native dispatch")

            with patch.object(ElementHandle, "click", uncertain):
                with pytest.raises(ChannelError, match="uncertain"):
                    await broker.select_option(initial.messages[0].options[0].id)
            with pytest.raises(ChannelError):
                await broker.select_option(initial.messages[0].options[0].id)
            assert await page.evaluate("count") == 1
            await broker.close()
            await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("attach_reply", [False, True])
def test_response_diagnostic_precedes_equal_outer_budget(attach_reply: bool) -> None:
    async def exercise() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            markup = (
                '<section id="chat" role="dialog" aria-label="Chat assistant">'
                '<input id="composer"></section>'
            )
            if attach_reply:
                markup += """<script>
                  document.querySelector('#composer').addEventListener('keydown', event => {
                    if(event.key === 'Enter') document.querySelector('#chat').insertAdjacentHTML(
                      'beforeend', '<article class="assistant">Here are the details</article>');
                  });</script>"""
            await page.route(
                "**/*",
                lambda route: route.fulfill(status=200, content_type="text/html", body=markup),
            )
            await page.goto(URL)
            data = contract().model_dump(mode="json")
            data.update(
                initial_state_id="composer1",
                states=[data["states"][2]],
                transitions=[
                    {
                        "id": "t1",
                        "from_state": "composer1",
                        "to_state": "composer1",
                        "action_kind": "probe",
                        "probe_id": "first",
                    }
                ],
                verification_recipe=[{"transition_id": "t1"}],
                response_completion={
                    "strategy": "completion_locator",
                    "locator": {"strategy": "css", "value": ".done"},
                },
            )
            broker = NativeChatBroker(
                contract=StatefulChannelConfig.model_validate(data), page=page
            )
            broker.limits = broker.limits.model_copy(update={"response_timeout_seconds": 0.4})
            await broker.start()
            await broker.send("Tell me about coverage")
            diagnostic = "completion" if attach_reply else "bubble"
            with pytest.raises(ChannelError, match=f"assistant {diagnostic} not observed"):
                async with asyncio.timeout(0.4):
                    await broker.receive()
            with pytest.raises(ChannelError):
                await broker.send("No automatic retry")
            await broker.close()
            await browser.close()

    asyncio.run(exercise())
