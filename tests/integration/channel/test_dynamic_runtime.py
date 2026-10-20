"""Local unsampled native branches and transaction safety, never live sites."""

import asyncio
from unittest.mock import patch

import pytest
from playwright.async_api import ElementHandle, Locator, async_playwright
from playwright.async_api import Error as PlaywrightError

from syntheticcx.adapters.channel.bindings import NativeControlBinding
from syntheticcx.adapters.channel.dynamic import DynamicChatBroker
from syntheticcx.config.models import ExecutionLimits, InterfaceChannelConfig
from syntheticcx.errors import ChannelError

URL = "http://localhost:8765/"
HTML = """<input id="decoy"><section id="chat" role="dialog" aria-label="Chat assistant">
<article class="assistant"><p>Choose an option</p>
<button type="button">Plans</button><button type="button">Coverage</button></article></section>
<script>
let actions=0, branch=''; const chat=document.querySelector('#chat');
chat.onclick=e=>{
  if(e.target.tagName!=='BUTTON')return;
  actions++; const choice=e.target.textContent; branch+=choice+';';
  chat.querySelectorAll('button').forEach(n=>n.remove());
  if(actions===1) chat.insertAdjacentHTML('beforeend',
    '<article class="assistant"><p>Choose an option</p><button type="button">'+
    (choice==='Plans'?'Internet':'Locations')+'</button></article>');
  else chat.insertAdjacentHTML('beforeend',
    '<article class="assistant">How can I help?</article><input id="composer">');
};
chat.onkeydown=e=>{
  if(e.key!=='Enter')return;
  actions++; e.target.remove();
  chat.insertAdjacentHTML('beforeend',
    '<article class="assistant">Here are the details.</article><textarea></textarea>');
};
</script>"""


def config():
    return InterfaceChannelConfig.model_validate(
        {
            "type": "webchat_playwright_dynamic",
            "url": URL,
            "ownership": {
                "root_locator": {"strategy": "css", "value": "#chat"},
                "proof": "chat_dialog",
            },
            "assistant_message_locator": {"strategy": "css", "value": "article.assistant"},
            "message_identity": {"strategy": "dom_node"},
            "response_completion": {"strategy": "quiet_interval", "quiet_interval_seconds": 0.04},
        }
    )


def limits():
    return ExecutionLimits(
        max_turns=10,
        browser_action_timeout_seconds=0.4,
        response_timeout_seconds=2,
        greeting_grace_seconds=0,
        journey_timeout_seconds=30,
        model_timeout_seconds=2,
        evaluator_timeout_seconds=2,
        persistence_timeout_seconds=2,
        close_timeout_seconds=2,
    )


async def fixture(exercise, markup=HTML):
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        try:
            page = await browser.new_page()
            await page.route(
                "**/*",
                lambda route: route.fulfill(status=200, content_type="text/html", body=markup),
            )
            await page.goto(URL)
            broker = DynamicChatBroker(config(), limits(), page=page)
            try:
                await broker.start()
                await exercise(broker, page)
            finally:
                await broker.close()
        finally:
            await browser.close()


def test_optional_email_continue_is_offered_and_dispatched():
    async def exercise(broker, page):
        await broker.observe_initial()
        await page.evaluate("""() => { chat.onclick=()=>{
            actions++;
            chat.querySelectorAll('button').forEach(n=>n.remove());
            chat.insertAdjacentHTML('beforeend', '<article class="assistant"><p>' +
                'Se não quiser informar seu e-mail, clique em Continuar.</p>' +
                '<button type="button">Continuar</button></article>');
        }; }""")
        await broker.select_option((await broker.observe_controls())["options"][0].id)
        reply = await broker.receive()
        assert "e-mail" in reply.messages[-1].content
        options = (await broker.observe_controls())["options"]
        assert [option.label for option in options] == ["Continuar"]
        await broker.select_option(options[0].id)
        assert await page.evaluate("actions") == 2

    asyncio.run(fixture(exercise))


def test_options_only_ordinary_response_completes_and_publishes_current_choices():
    async def exercise(broker, page):
        await broker.observe_initial()
        await page.evaluate("""() => { chat.onclick=()=>{
            actions++;
            chat.querySelectorAll('button').forEach(n=>n.remove());
            chat.insertAdjacentHTML('beforeend', '<article class="assistant">' +
                '<button type="button">Yes</button><button type="button">No</button></article>');
        }; }""")
        await broker.select_option((await broker.observe_controls())["options"][0].id)
        reply = await broker.receive()
        assert [option.label for option in reply.messages[-1].options] == ["Yes", "No"]
        options = (await broker.observe_controls())["options"]
        assert [option.label for option in options] == ["Yes", "No"]
        await broker.select_option(options[1].id)
        assert await page.evaluate("actions") == 2

    asyncio.run(fixture(exercise))


def test_completed_reply_returned_with_ordinary_current_choice():
    async def exercise(broker, page):
        await broker.observe_initial()
        await page.evaluate("""() => { chat.onclick=()=>{
            chat.querySelectorAll('button').forEach(n=>n.remove());
            chat.insertAdjacentHTML('beforeend', '<article class="assistant">' +
                'Qual é seu email?</article>');
        }; }""")
        await broker.select_option((await broker.observe_controls())["options"][0].id)
        reply = await broker.receive()
        assert "email" in reply.messages[-1].content
        assert not reply.messages[-1].options
        with pytest.raises(ChannelError, match="no safe current controls"):
            await broker.observe_controls()

    asyncio.run(fixture(exercise))


def test_unsampled_branch_new_menu_and_replacements_exceed_learning_budget():
    async def exercise(broker, page):
        assert await page.evaluate("actions") == 0
        initial = await broker.observe_initial()
        controls = await broker.observe_controls()
        assert not controls["text_available"]
        assert [o.label for o in controls["options"]] == ["Plans", "Coverage"]
        assert controls["options"][1].id == initial.messages[0].options[1].id
        await broker.select_option(controls["options"][1].id)
        response = await broker.receive()
        assert [o.label for o in response.messages[0].options] == ["Locations"]
        await broker.select_option(response.messages[0].options[0].id)
        await broker.receive()
        for _ in range(4):
            assert (await broker.observe_controls())["text_available"]
            await broker.send("Please share details")
            await broker.receive()
        assert await page.evaluate("actions") == 6
        assert await page.evaluate("branch") == "Coverage;Locations;"
        assert await page.locator("#decoy").input_value() == ""
        assert broker._learning_edges == []
        assert broker._states == {}
        assert broker.current_state is None
        assert isinstance(broker._binding, NativeControlBinding)
        assert broker._learning_source is None
        assert not broker._learn_transition_allowed

    asyncio.run(fixture(exercise))


@pytest.mark.parametrize(
    "mutation",
    [
        "document.querySelector('#chat').outerHTML=document.querySelector('#chat').outerHTML",
        "document.querySelector('button').outerHTML=document.querySelector('button').outerHTML",
        "document.querySelector('button').setAttribute('data-url','/purchase')",
        "document.querySelector('button').textContent='Buy plans'",
        "document.querySelector('button').disabled=true",
        "document.querySelector('article').className='customer'",
        "document.body.append(document.querySelector('button'))",
    ],
)
def test_stale_unsafe_and_outside_root_options_never_dispatch(mutation):
    async def exercise(broker, page):
        option = (await broker.observe_initial()).messages[0].options[0].id
        await page.evaluate(mutation)
        with pytest.raises(ChannelError):
            await broker.select_option(option)
        assert await page.evaluate("actions") == 0

    asyncio.run(fixture(exercise))


def test_uncertain_click_cannot_retry():
    async def exercise(broker, page):
        option = (await broker.observe_initial()).messages[0].options[0].id
        original = ElementHandle.click

        async def uncertain(node, **kwargs):
            await original(node, **kwargs)
            raise PlaywrightError("connection lost")

        with patch.object(ElementHandle, "click", uncertain):
            with pytest.raises(ChannelError, match="uncertain"):
                await broker.select_option(option)
        with pytest.raises(ChannelError, match="uncertain"):
            await broker.select_option(option)
        assert await page.evaluate("actions") == 1

    asyncio.run(fixture(exercise))


def test_missing_controls_wait_then_specific_failure():
    async def exercise(broker, page):
        await broker.observe_initial()
        await page.locator("button").evaluate_all("nodes=>nodes.forEach(n=>n.remove())")
        with pytest.raises(ChannelError, match="no safe current controls before deadline"):
            await broker.observe_controls()

    asyncio.run(fixture(exercise))


def test_fill_replacement_is_uncertain_and_never_retargets():
    markup = HTML.replace("</section>", '<input id="composer"></section>')
    markup += """<script>document.querySelector('#composer').oninput=e=>{
      e.target.outerHTML='<textarea></textarea>'; };</script>"""

    async def exercise(broker, page):
        with pytest.raises(ChannelError, match="pinned current control changed"):
            await broker.send("Please share details")
        with pytest.raises(ChannelError, match="uncertain"):
            await broker.send("No retry")
        assert await page.evaluate("actions") == 0
        assert await page.locator("textarea").input_value() == ""

    asyncio.run(fixture(exercise, markup))


@pytest.mark.parametrize(
    "mutation",
    [
        "document.querySelector('textarea').setAttribute('name','credit-card')",
        "document.querySelector('textarea').disabled=true",
        "document.querySelector('textarea').outerHTML='<form><textarea></textarea></form>'",
        "document.querySelector('#chat').insertAdjacentHTML('beforeend','<input>')",
        "document.querySelector('#chat').insertAdjacentHTML('beforeend',"
        "'<button type=submit>Send</button>')",
    ],
)
def test_unsafe_or_ambiguous_current_composer_has_no_dispatch(mutation):
    async def exercise(broker, page):
        await broker.observe_initial()
        await page.locator("button").evaluate_all("nodes=>nodes.forEach(n=>n.remove())")
        await page.locator("#chat").evaluate(
            "node=>node.insertAdjacentHTML('beforeend','<textarea></textarea>')"
        )
        await page.evaluate(mutation)
        with pytest.raises(ChannelError):
            await broker.send("Please share details")
        assert await page.evaluate("actions") == 0

    asyncio.run(fixture(exercise))


def test_safe_current_send_click_and_pre_dispatch_replacement():
    markup = HTML.replace(
        "</section>", '<input id="composer"><button type="button" id="send">Send</button></section>'
    )
    markup += """<script>document.querySelector('#send').onclick=e=>{
      e.stopPropagation(); actions++;
      chat.insertAdjacentHTML('beforeend','<article class="assistant">Details</article>');
    };</script>"""

    async def exercise(broker, page):
        await broker.observe_initial()
        await page.locator("#composer").evaluate("node=>node.outerHTML='<textarea></textarea>'")
        await broker.send("Please share details")
        assert (await broker.receive()).messages[0].content == "Details"
        assert await page.evaluate("actions") == 1

    asyncio.run(fixture(exercise, markup))


def test_reader_corruption_and_origin_loss_remain_fatal():
    async def exercise(broker, page):
        await broker.observe_initial()
        await page.locator("article").evaluate("node=>node.setAttribute('data-role','customer')")
        with pytest.raises(ChannelError) as caught:
            await broker.observe_controls()
        assert caught.value.operation == "channel.reader"
        await page.goto("http://127.0.0.1:8765/")
        with pytest.raises(ChannelError) as caught:
            await broker.observe_controls()
        assert caught.value.operation == "channel.ownership"

    asyncio.run(fixture(exercise))


def test_late_replacement_control_waits_without_dispatch():
    async def exercise(broker, page):
        await broker.observe_initial()
        await page.evaluate("""() => {
          document.querySelectorAll('button').forEach(n=>n.remove());
          setTimeout(()=>chat.insertAdjacentHTML('beforeend','<textarea></textarea>'),100);
        }""")
        controls = await broker.observe_controls()
        assert controls["text_available"]
        assert controls["options"] == []
        assert await page.evaluate("actions") == 0

    asyncio.run(fixture(exercise))


@pytest.mark.parametrize("enable", [True, False])
def test_disabled_send_is_pinned_filled_once_and_never_falls_back_to_enter(enable):
    markup = """<section id="chat" role="dialog" aria-label="Chat assistant">
      <article class="assistant">How can I help?</article>
      <textarea></textarea><button id="send" type="button" disabled>Send</button></section>
      <script>
      let fills=0, clicks=0, enters=0;
      const input=document.querySelector('textarea'), send=document.querySelector('#send');
      input.oninput=()=>{fills++; ENABLE};
      input.onkeydown=e=>{if(e.key==='Enter') enters++;};
      send.onclick=()=>{clicks++; document.querySelector('#chat').insertAdjacentHTML('beforeend',
        '<article class="assistant">Here are the details.</article>');};
      </script>""".replace("ENABLE", "send.disabled=false;" if enable else "")

    async def exercise(broker, page):
        assert (await broker.observe_controls())["text_available"]
        assert await page.locator("#send").is_disabled()
        if enable:
            await broker.send("Please share details")
            assert (await broker.receive()).messages[0].content == "Here are the details."
        else:
            with pytest.raises(ChannelError, match="unsafe or unavailable"):
                await broker.send("Please share details")
            with pytest.raises(ChannelError, match="uncertain"):
                await broker.send("No retry")
        assert await page.evaluate("fills") == 1
        assert await page.evaluate("clicks") == int(enable)
        assert await page.evaluate("enters") == 0

    asyncio.run(fixture(exercise, markup))


def test_disabled_send_replaced_during_fill_is_not_retargeted():
    markup = """<section id="chat" role="dialog" aria-label="Chat assistant">
      <article class="assistant">How can I help?</article>
      <textarea></textarea><button id="send" type="button" disabled>Send</button></section>
      <script>
      let fills=0, clicks=0, enters=0;
      document.querySelector('textarea').oninput=()=>{
        fills++; document.querySelector('#send').outerHTML=
          '<button id="send" type="button" onclick="clicks++">Send</button>';
      };
      document.querySelector('textarea').onkeydown=e=>{if(e.key==='Enter') enters++;};
      </script>"""

    async def exercise(broker, page):
        with pytest.raises(ChannelError, match="pinned current control changed"):
            await broker.send("Please share details")
        with pytest.raises(ChannelError, match="uncertain"):
            await broker.send("No retry")
        assert await page.evaluate("[fills, clicks, enters]") == [1, 0, 0]

    asyncio.run(fixture(exercise, markup))


def live_handles(page):
    # Playwright's connection registry retains remote references until dispose,
    # unlike Python GC. Count both element and non-element JS handles.
    return {
        guid
        for guid, obj in page._impl_obj._connection._objects.items()
        if type(obj).__name__ in {"ElementHandle", "JSHandle"}
    }


def test_repeated_control_observations_and_revalidation_keep_handles_bounded():
    markup = HTML.replace(
        "</section>",
        '<textarea></textarea><input hidden><button hidden type="button">Help</button></section>',
    )

    async def exercise(broker, page):
        await broker.observe_controls()
        baseline = len(live_handles(page))
        for _ in range(20):
            await broker.observe_controls()
            await broker._validate_current_controls()
        assert len(live_handles(page)) == baseline
        # A replacement is pinned before dispatch and the superseded reference
        # is explicitly released, not added to observation history.
        await page.locator("textarea").evaluate("node=>node.outerHTML='<textarea></textarea>'")
        await broker.observe_controls()
        assert len(live_handles(page)) == baseline

    asyncio.run(fixture(exercise, markup))


def test_response_inspector_disposed_and_external_page_close_releases_owned_handles():
    async def exercise(broker, page):
        await broker.observe_initial()
        await broker.select_option((await broker.observe_controls())["options"][0].id)
        await broker.receive()
        # Only the persistent broker and option-owner histories remain, not a
        # second response-evidence inspector holding the same DOM nodes.
        retained = [broker.owned.root, *broker._known_nodes, *broker._options._known]
        expected = {handle._impl_obj._guid for handle in retained}
        assert live_handles(page) == expected, [
            (guid, await page._impl_obj._connection._objects[guid].evaluate("node=>node.outerHTML"))
            for guid in live_handles(page) - expected
        ]
        caller_handle = await page.locator("#decoy").element_handle()
        # Attested option root aliases the broker's root. Dispose that reference
        # exactly once, while preserving the independently acquired caller handle.
        root_guid = broker.owned.root._impl_obj._guid
        disposed = []
        original = ElementHandle.dispose

        async def counted(handle):
            disposed.append(handle._impl_obj._guid)
            await original(handle)

        with patch.object(ElementHandle, "dispose", counted):
            async with asyncio.timeout(broker.limits.close_timeout_seconds):
                await broker.close()
        assert disposed.count(root_guid) == 1
        assert live_handles(page) == {caller_handle._impl_obj._guid}
        assert not page.is_closed()
        assert await caller_handle.evaluate("node=>node.isConnected")
        await caller_handle.dispose()

    asyncio.run(fixture(exercise))


@pytest.mark.parametrize("iteration", range(20))
def test_response_timeout_releases_temporary_inspector_handles(iteration):
    async def exercise(broker, page):
        await broker.observe_initial()
        await page.evaluate("() => { chat.onclick = () => { actions++; }; }")
        await broker.select_option((await broker.observe_controls())["options"][0].id)
        baseline = live_handles(page)
        broker.limits = broker.limits.model_copy(update={"response_timeout_seconds": 0.2})
        with pytest.raises(ChannelError, match="assistant bubble not observed"):
            await broker.receive()
        assert live_handles(page) == baseline
        await broker.close()
        assert not live_handles(page)
        assert not page.is_closed()

    asyncio.run(fixture(exercise))


@pytest.mark.parametrize("stage", ["snapshot_acquisition", "option_inspection"])
def test_cancelled_receive_drains_handle_acquisition_before_cleanup(stage):
    async def exercise(broker, page):
        await broker.observe_initial()
        await page.evaluate("() => { chat.onclick = () => { actions++; }; }")
        await broker.select_option((await broker.observe_controls())["options"][0].id)
        baseline = live_handles(page)
        acquired = asyncio.Event()
        release = asyncio.Event()
        in_snapshot = False
        original_snapshot = broker._snapshot_owned
        original_handles = Locator.element_handles
        original_evaluate = ElementHandle.evaluate

        async def snapshot(actor):
            nonlocal in_snapshot
            in_snapshot = True
            try:
                return await original_snapshot(actor)
            finally:
                in_snapshot = False

        async def handles(locator):
            result = await original_handles(locator)
            if stage == "snapshot_acquisition" and in_snapshot and not acquired.is_set():
                acquired.set()
                await release.wait()
            return result

        async def evaluate(handle, expression, arg=None):
            result = await original_evaluate(handle, expression, arg)
            if (
                stage == "option_inspection"
                and "const owners = args.wrappers" in expression
                and not acquired.is_set()
            ):
                acquired.set()
                await release.wait()
            return result

        with (
            patch.object(broker, "_snapshot_owned", snapshot),
            patch.object(Locator, "element_handles", handles),
            patch.object(ElementHandle, "evaluate", evaluate),
        ):
            receive = asyncio.create_task(broker.receive())
            try:
                await asyncio.wait_for(acquired.wait(), timeout=1)
                receive.cancel()
                await asyncio.sleep(0)
                assert not receive.done()
                # A second cancellation must not abandon the in-flight collector.
                receive.cancel()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await receive
            finally:
                release.set()
                if not receive.done():
                    receive.cancel()
                    await asyncio.gather(receive, return_exceptions=True)
        assert live_handles(page) == baseline
        assert await page.evaluate("actions") == 1
        await broker.close()
        assert not live_handles(page)

    asyncio.run(fixture(exercise))
