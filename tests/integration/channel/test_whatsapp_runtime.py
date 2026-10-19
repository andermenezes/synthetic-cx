"""Consumer-visible behavior through real Chromium, routed to supported production hosts.

These controlled fixtures exercise native adapter actions, not live WhatsApp compatibility.
"""

import asyncio
import json
from pathlib import Path

import pytest
from playwright.async_api import BrowserType

from syntheticcx.adapters.channel.whatsapp import WhatsAppWebChannel
from syntheticcx.config.models import (
    ExecutionLimits,
    LocatorConfig,
    WebsiteNavigationStep,
    WhatsAppChannelConfig,
)
from syntheticcx.errors import ChannelError

PHONE = "15551234567"
SITE = "https://service.example/"
HTML = Path("tests/fixtures/whatsapp_web.html").read_text()


def config(*, discover: bool = False) -> WhatsAppChannelConfig:
    return WhatsAppChannelConfig.model_validate(
        {
            "type": "whatsapp_web_playwright",
            "url": SITE,
            "navigation_steps": []
            if discover
            else [
                {
                    "action": "click",
                    "locator": {
                        "strategy": "role",
                        "value": "link",
                        "name": "Contact WhatsApp",
                        "exact": True,
                    },
                }
            ],
            "quiet_interval_seconds": 0.07,
        }
    )


def limits() -> ExecutionLimits:
    return ExecutionLimits(
        max_turns=8,
        journey_timeout_seconds=30,
        greeting_grace_seconds=0,
        browser_action_timeout_seconds=8,
        echo_observation_seconds=0.3,
        response_timeout_seconds=1.5,
        model_timeout_seconds=2,
        evaluator_timeout_seconds=2,
        persistence_timeout_seconds=2,
        close_timeout_seconds=3,
    )


def route_fixture(
    monkeypatch,
    *,
    html: str = HTML,
    entry: str | None = None,
    popup: bool = False,
    landing: bool = False,
    dom_id: str | None = None,
    website_html: str | None = None,
    visits: list[str] | None = None,
):
    """Patch only browser launch: the real channel receives an actual routed context."""
    original = BrowserType.launch_persistent_context
    href = entry or (
        f"https://wa.me/{PHONE}" if landing else f"https://web.whatsapp.com/send?phone={PHONE}"
    )
    website = website_html or (
        f"<a {'target=_blank' if popup else ''} {'id=' + dom_id if dom_id else ''} "
        f'href="{href}">Contact WhatsApp</a>'
    )

    async def launch(browser_type, *args, **kwargs):
        context = await original(browser_type, *args, **kwargs)

        async def route(request):
            url = request.request.url
            if visits is not None:
                visits.append(url)
            if url.startswith(SITE):
                body = website
            elif url.startswith("https://web.whatsapp.com/"):
                body = html
            elif url.startswith(f"https://wa.me/{PHONE}"):
                body = f'<a href="https://web.whatsapp.com/send?phone={PHONE}">Use WhatsApp Web</a>'
            else:
                body = "Unsupported destination"
            await request.fulfill(status=200, content_type="text/html", body=body)

        await context.route("**/*", route)
        return context

    monkeypatch.setattr(BrowserType, "launch_persistent_context", launch)
    monkeypatch.setenv("SYNTHETICCX_WHATSAPP_RECIPIENT", PHONE)


@pytest.mark.parametrize("quoted_reply", [False, True])
def test_multi_turn_button_exchange_is_current_run_only(monkeypatch, tmp_path, quoted_reply):
    markup = HTML
    if quoted_reply:
        markup = markup.replace(
            "node.append(prose);",
            "if(actor==='customer'&&text==='Plans'){"
            "const quote=document.createElement('div');quote.dataset.testid='quoted-message';"
            "quote.innerHTML='<div role=button><span class=selectable-text>"
            "Previous assistant menu, not the selected reply</span></div>';node.append(quote);}"
            "node.append(prose);",
        )
    route_fixture(monkeypatch, html=markup)

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            assert channel.current_url == "https://web.whatsapp.com"
            assert await channel.observe_initial() is None
            assert channel._page is not None
            receipt = await channel.send("Please help correct my invoice.")
            assert receipt.evidence["observed_echo"] is True
            first = await channel.receive()
            assert [message.content for message in first.messages] == [
                "Choose an option for invoice help."
            ]
            assert PHONE not in first.model_dump_json() and "false_" not in first.model_dump_json()
            controls = await channel.observe_controls()
            plans = next(option for option in controls["options"] if option.label == "Plans")
            selection = await channel.select_option(plans.id)
            assert selection.source_message_id == first.messages[-1].id
            assert selection.selected_label == "Plans"
            second = await channel.receive()
            assert second.messages[0].content == "What is your account reference?"
            await channel.send("Account reference ABC123.")
            third = await channel.receive()
            assert third.messages[0].content == "Your invoice is corrected. The refund is approved."
            assert [
                first.messages[0].sequence,
                second.messages[0].sequence,
                third.messages[0].sequence,
            ] == [0, 1, 2]
            assert len({first.messages[0].id, second.messages[0].id, third.messages[0].id}) == 3
            assert await channel._page.evaluate("sent") == [
                "Please help correct my invoice.",
                "Plans",
                "Account reference ABC123.",
            ]
        finally:
            await channel.close()
        assert tmp_path.is_dir()

    asyncio.run(exercise())


def test_quoted_history_is_not_fresh_assistant_prose_or_a_reply_option(monkeypatch, tmp_path):
    markup = HTML.replace(
        "node.append(prose);",
        "if(actor==='assistant'){"
        "const quote=document.createElement('div');quote.dataset.testid='quoted-message';"
        "quote.innerHTML='<div role=button><span class=selectable-text>"
        "OLD QUOTED SUCCESS</span></div>';node.append(quote);}"
        "node.append(prose);",
    )
    route_fixture(monkeypatch, html=markup)

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            await channel.send("Owned probe")
            response = await channel.receive()
            assert [m.content for m in response.messages] == ["Choose an option for invoice help."]
            controls = await channel.observe_controls()
            assert [o.label for o in controls["options"]] == ["Plans", "Coverage"]
        finally:
            await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("native,radio_delay", [(False, 0), (True, 0), (True, 150)])
def test_list_opener_opens_without_sending_and_scoped_choice_submits_once(
    monkeypatch, tmp_path, native, radio_delay
):
    markup = HTML.replace(
        "window.menuKind='buttons'", f"window.menuKind='{'native-list' if native else 'list'}'"
    ).replace(
        "node.append(prose);",
        "node.append(prose);if(actor==='assistant'){"
        "for(let i=0;i<2;i++){const icon=document.createElement('button');"
        "icon.setAttribute('aria-label','Unrelated icon');node.append(icon);}}",
    )
    route_fixture(monkeypatch, html=markup)

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            await channel.send("Owned initial probe")
            await channel.receive()
            assert channel._page is not None
            await channel._page.evaluate("delay=>window.radioDelay=delay", radio_delay)
            controls = await channel.observe_controls()
            assert [option.label for option in controls["options"]] == ["Plans", "Coverage"]
            assert await channel._page.locator('[role="dialog"]').count() == 1
            assert await channel._page.evaluate("sent") == ["Owned initial probe"]
            if native:
                assert controls["text_available"] is False
                with pytest.raises(ChannelError, match="native list is open"):
                    await channel.send("Do not type behind the menu")
            again = await channel.observe_controls()
            assert again["options"] == controls["options"]
            await channel.select_option(controls["options"][0].id)
            response = await channel.receive()
            assert response.messages[0].content == "What is your account reference?"
            assert await channel._page.evaluate("sent") == ["Owned initial probe", "Plans"]
        finally:
            await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "change", ["preexisting", "wrong_heading", "dialog_replaced", "opener_removed"]
)
def test_native_list_cannot_borrow_or_reuse_unowned_controls(monkeypatch, tmp_path, change):
    markup = HTML.replace("window.menuKind='buttons'", "window.menuKind='native-list'")
    if change == "wrong_heading":
        markup = markup.replace(
            'data-testid="drawer-title-body">View options',
            'data-testid="drawer-title-body">Unrelated dialog',
        )
    route_fixture(monkeypatch, html=markup)

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            await channel.send("Owned initial probe")
            await channel.receive()
            assert channel._page is not None
            if change in {"preexisting", "wrong_heading"}:
                if change == "preexisting":
                    await channel._page.evaluate(
                        "document.body.insertAdjacentHTML('beforeend',"
                        "'<div role=dialog>Other</div>')"
                    )
                channel._limits = limits().model_copy(
                    update={"browser_action_timeout_seconds": 0.5}
                )
                with pytest.raises(ChannelError):
                    await channel.observe_controls()
            else:
                controls = await channel.observe_controls()
                if change == "dialog_replaced":
                    await channel._page.locator('[role="dialog"]').evaluate(
                        "n=>n.replaceWith(n.cloneNode(true))"
                    )
                else:
                    await channel._page.locator('button[title="View options"]').evaluate(
                        "n=>n.remove()"
                    )
                with pytest.raises(ChannelError):
                    await channel.select_option(controls["options"][0].id)
            assert await channel._page.evaluate("sent") == ["Owned initial probe"]
        finally:
            await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("strip_url", [False, True])
def test_handoff_transfers_exact_url_to_fresh_native_input_owner(monkeypatch, tmp_path, strip_url):
    entry = f"https://web.whatsapp.com/send?phone={PHONE}&text=UNAUTHORIZED-DRAFT"
    visits = []
    markup = HTML.replace(
        "<script>",
        '<script>if(window.name==="website-input-owner") '
        "for(const type of ['click','beforeinput','input','keydown']) "
        "window.addEventListener(type,e=>{e.preventDefault();"
        "e.stopImmediatePropagation();},true);",
        1,
    )
    if strip_url:
        markup = markup.replace(
            "</html>",
            "<script>if(location.pathname==='/') document.body.innerHTML="
            "'<p>Account overview, no conversation selected</p>';"
            "else history.replaceState({},'', '/');</script></html>",
        )
    route_fixture(
        monkeypatch,
        visits=visits,
        website_html=(
            '<script>window.name="website-input-owner";</script>'
            f'<a href="{entry}">Contact WhatsApp</a>'
        ),
        html=markup,
    )

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            assert channel._page is not None
            assert await channel._page.evaluate("window.name") == ""
            assert visits.count(entry) == 2
            assert (await channel.send("Owned probe")).evidence["observed_echo"] is True
            assert (await channel.receive()).messages
            assert await channel._page.evaluate("sent") == ["Owned probe"]
        finally:
            await channel.close()

    asyncio.run(exercise())


def test_failed_opaque_outgoing_without_tail_is_not_acknowledged_or_retried(monkeypatch, tmp_path):
    markup = _named_empty_chat(_contact_drawer()).replace(
        "node.dataset.id=(actor==='customer'?'true_':'false_')+'15551234567@c.us_'+(++counter);",
        "node.dataset.id='3EB0000000000000'+(++counter).toString(16).toUpperCase();"
        "node.dataset.testid='conv-msg-'+node.dataset.id;"
        "const container=document.createElement('div');container.dataset.testid='msg-container';"
        "const failure=document.createElement('div');failure.dataset.testid='fail-container';"
        "container.append(failure);node.append(container);",
    )
    route_fixture(monkeypatch, html=markup)

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            assert channel._page is not None
            await channel._page.evaluate("window.delay=10000")
            with pytest.raises(ChannelError, match="message was not sent; retry prohibited"):
                await channel.send("Owned failed probe")
            assert not channel._has_submitted
            with pytest.raises(ChannelError, match="retry prohibited"):
                await channel.send("Owned failed probe")
            assert await channel._page.evaluate("sent") == ["Owned failed probe"]
        finally:
            await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "pending",
    ["recipient", "message_identity", "composer_hidden", "composer_disabled"],
)
def test_start_waits_for_chat_hydration_before_clearing_or_sending(
    monkeypatch,
    tmp_path,
    pending,
):
    hydrated = HTML
    if pending == "message_identity":
        hydrated = hydrated.replace(f'title="+{PHONE}"', "")
    if pending in {"recipient", "message_identity"}:
        markup = (
            '<section id="main"><header>Loading business chat</header></section>'
            "<script>setTimeout(()=>{document.open();document.write("
            + json.dumps(hydrated).replace("<", "\\u003c")
            + ");document.close();},400);</script>"
        )
    else:
        if pending == "composer_hidden":
            markup = HTML.replace("<footer>", "<footer hidden>")
            reveal = "document.querySelector('footer').hidden=false"
        else:
            markup = HTML.replace(
                'contenteditable="true"', 'contenteditable="true" aria-disabled="true"'
            )
            reveal = "document.querySelector('[role=textbox]').removeAttribute('aria-disabled')"
        markup = markup.replace(
            "</html>", f"<script>setTimeout(()=>{{{reveal}}},400)</script></html>"
        )
    route_fixture(monkeypatch, html=markup)

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            assert await channel.observe_initial() is None
            receipt = await channel.send("Owned readiness probe")
            assert receipt.evidence["observed_echo"] is True
            response = await channel.receive()
            assert response.messages[0].content == "Choose an option for invoice help."
            assert await channel._page.evaluate("sent") == ["Owned readiness probe"]
        finally:
            await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("popup,landing", [(True, False), (False, True), (True, True)])
def test_supported_popup_and_landing_handoffs(monkeypatch, tmp_path, popup, landing):
    route_fixture(monkeypatch, popup=popup, landing=landing)

    async def exercise():
        channel_config = config()
        if landing:
            channel_config = channel_config.model_copy(
                update={
                    "navigation_steps": [
                        *channel_config.navigation_steps,
                        WebsiteNavigationStep(
                            action="click",
                            locator=LocatorConfig(
                                strategy="role", value="link", name="Use WhatsApp Web", exact=True
                            ),
                        ),
                    ]
                }
            )
        channel = WhatsAppWebChannel(channel_config, limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            assert await channel.observe_initial() is None
            assert channel._page is not None
            assert await channel._page.evaluate("sent") == []
        finally:
            await channel.close()

    asyncio.run(exercise())


def test_duplicate_text_is_distinct_and_delayed_batch_is_settled(monkeypatch, tmp_path):
    route_fixture(monkeypatch)

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            assert channel._page is not None
            await channel._page.evaluate("delay=10000")
            await channel.send("Hello")
            await channel._page.evaluate(
                "setTimeout(()=>incoming('Same reply'),10);"
                "setTimeout(()=>incoming('Same reply'),50)"
            )
            response = await channel.receive()
            assert [message.content for message in response.messages] == [
                "Same reply",
                "Same reply",
            ]
            assert response.messages[0].id != response.messages[1].id
            await channel._page.evaluate("incoming('A later reply')")
            with pytest.raises(ChannelError, match="fresh incoming"):
                await channel.send("Do not swallow that reply")
            later = await channel.observe_initial()
            assert later is not None and later.messages[0].content == "A later reply"
            assert later.messages[0].sequence == 2
        finally:
            await channel.close()

    asyncio.run(exercise())


def test_scoped_list_menu_selection(monkeypatch, tmp_path):
    route_fixture(monkeypatch)

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            assert channel._page is not None
            await channel._page.evaluate("menuKind='list'")
            await channel.send("Invoice help")
            source = await channel.receive()
            controls = await channel.observe_controls()
            assert [option.label for option in controls["options"]] == ["Plans", "Coverage"]
            receipt = await channel.select_option(controls["options"][0].id)
            assert receipt.source_message_id == source.messages[0].id
            assert (await channel.receive()).messages[
                0
            ].content == "What is your account reference?"
            assert await channel._page.evaluate("sent") == ["Invoice help", "Plans"]
        finally:
            await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("change", ["wrong_chat", "replaced_button", "new_message", "new_control"])
def test_stale_options_and_wrong_chat_cannot_dispatch(monkeypatch, tmp_path, change):
    route_fixture(monkeypatch)

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            await channel.send("Invoice help")
            await channel.receive()
            options = (await channel.observe_controls())["options"]
            assert channel._page is not None
            scripts = {
                "wrong_chat": "document.querySelector('#main header span').title='+15557654321'",
                "replaced_button": (
                    "const b=document.querySelector('#messages button');"
                    "b.replaceWith(b.cloneNode(true))"
                ),
                "new_message": "incoming('A newer instruction')",
                "new_control": (
                    "const b=document.createElement('button');b.textContent='Locations';"
                    "messages.lastElementChild.append(b)"
                ),
            }
            await channel._page.evaluate(scripts[change])
            with pytest.raises(ChannelError):
                await channel.select_option(options[0].id)
            assert await channel._page.evaluate("sent") == ["Invoice help"]
        finally:
            await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "entry",
    [
        "https://web.whatsapp.com/send?phone=15557654321",
        "https://evil.example/send?phone=15551234567",
        "whatsapp://send?phone=15551234567",
    ],
)
def test_wrong_or_unsupported_entry_fails_before_browser_action(monkeypatch, tmp_path, entry):
    visits: list[str] = []
    route_fixture(monkeypatch, entry=entry, visits=visits)

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        with pytest.raises(ChannelError) as error:
            await channel.start()
        assert PHONE not in str(error.value) and entry not in str(error.value)
        assert channel._context is None and channel._profile_lock is None
        assert visits == [SITE]

    asyncio.run(exercise())


def test_wrong_recipient_surface_rejected_before_send(monkeypatch, tmp_path):
    route_fixture(monkeypatch, html=HTML.replace(PHONE, "15557654321"))

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        with pytest.raises(ChannelError, match="recipient evidence"):
            await channel.start()
        assert channel._profile_lock is None

    asyncio.run(exercise())


def test_uncertain_send_never_retries_and_errors_hide_page_content(monkeypatch, tmp_path):
    route_fixture(monkeypatch)

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            assert channel._page is not None
            await channel._page.evaluate("uncertain=true")
            with pytest.raises(ChannelError, match="uncertain") as error:
                await channel.send("Sensitive account material")
            assert "Sensitive" not in str(error.value) and PHONE not in str(error.value)
            with pytest.raises(ChannelError, match="retry prohibited"):
                await channel.send("Sensitive account material")
            with pytest.raises(ChannelError):
                await channel.observe_controls()
            assert await channel._page.evaluate("sent") == ["Sensitive account material"]
        finally:
            await channel.close()

    asyncio.run(exercise())


def test_profile_is_exclusive_preserved_and_reusable(monkeypatch, tmp_path):
    route_fixture(monkeypatch)
    marker = tmp_path / "owned-account-marker"
    marker.write_text("retain")

    async def exercise():
        first = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        second = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await first.start()
            with pytest.raises(ChannelError, match="already in use"):
                await second.start()
        finally:
            await first.close()
        third = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await third.start()
            assert await third.observe_initial() is None
        finally:
            await third.close()
        assert marker.read_text() == "retain"
        assert (tmp_path / "Default").is_dir()

    asyncio.run(exercise())


def test_unowned_list_dialog_and_unauthorized_payment_are_unavailable(monkeypatch, tmp_path):
    route_fixture(monkeypatch)

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            await channel.send("Invoice help")
            await channel.receive()
            assert channel._page is not None
            await channel._page.evaluate(
                "document.querySelectorAll('#messages button').forEach("
                "(b,i)=>{b.textContent='Pay USD '+(10+i*10)})"
            )
            assert (await channel.observe_controls())["options"] == []
            await channel._page.evaluate(
                "document.querySelectorAll('#messages button').forEach(b=>b.remove());"
                "const b=document.createElement('button');b.textContent='View options';"
                "b.setAttribute('aria-haspopup','dialog');messages.lastElementChild.append(b)"
            )
            with pytest.raises(ChannelError, match="ownership is unsupported"):
                await channel.observe_controls()
            assert await channel._page.evaluate("sent") == ["Invoice help"]
        finally:
            await channel.close()

    asyncio.run(exercise())


def test_cancelled_dispatch_is_poisoned_without_resend(monkeypatch, tmp_path):
    route_fixture(monkeypatch)

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            assert channel._page is not None
            await channel._page.evaluate("uncertain=true")
            sending = asyncio.create_task(channel.send("One owned message"))
            for _ in range(100):
                if await channel._page.evaluate("sent.length"):
                    break
                await asyncio.sleep(0.005)
            assert await channel._page.evaluate("sent.length") == 1
            sending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await sending
            with pytest.raises(ChannelError, match="retry prohibited"):
                await channel.send("One owned message")
            assert await channel._page.evaluate("sent") == ["One owned message"]
        finally:
            await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("change", ["identity_loss", "disconnect", "no_reply"])
def test_missing_response_surface_is_bounded_and_never_historical(monkeypatch, tmp_path, change):
    route_fixture(monkeypatch)

    async def exercise():
        bounded_limits = limits().model_copy(update={"response_timeout_seconds": 0.2})
        channel = WhatsAppWebChannel(config(), bounded_limits, browser_profile=tmp_path)
        try:
            await channel.start()
            assert channel._page is not None
            await channel._page.evaluate("delay=10000")
            await channel.send("Owned current-run probe")
            if change == "identity_loss":
                await channel._page.evaluate(
                    "messages.lastElementChild.remove();incoming('Old success-looking answer')"
                )
            elif change == "disconnect":
                await channel._page.close()
            with pytest.raises(ChannelError):
                await channel.receive()
            with pytest.raises(ChannelError):
                await channel.send("Never retry the probe")
        finally:
            await channel.close()

    asyncio.run(exercise())


def test_display_name_alone_is_not_recipient_evidence(monkeypatch, tmp_path):
    markup = HTML.replace('title="+15551234567"', "")
    markup = markup.replace(
        '<div data-id="false_15551234567@c.us_history"><span class="selectable-text">'
        "Your invoice is corrected. The refund is approved.</span></div>",
        "",
    )
    route_fixture(monkeypatch, html=markup)

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        with pytest.raises(ChannelError, match="recipient evidence"):
            await channel.start()
        assert channel._context is None

    asyncio.run(exercise())


@pytest.mark.parametrize("switch_stage", ["fill", "enter"])
def test_recipient_switch_cannot_receive_private_draft_or_message(
    monkeypatch, tmp_path, switch_stage
):
    from playwright.async_api import ElementHandle

    route_fixture(monkeypatch)
    original_fill = ElementHandle.type
    original_press = ElementHandle.press
    private = "Private current-run account material"
    channel = None

    async def switch():
        await channel._page.evaluate("""() => {
          document.querySelector('#main header span').title='+15557654321';
          for (const node of document.querySelectorAll('#main [data-id]'))
            node.dataset.id=node.dataset.id.replace('15551234567','15557654321');
        }""")

    async def fill(node, value, **kwargs):
        if value == private and switch_stage == "fill":
            await switch()
        return await original_fill(node, value, **kwargs)

    async def press(node, key, **kwargs):
        if key == "Enter" and switch_stage == "enter":
            await switch()
        return await original_press(node, key, **kwargs)

    monkeypatch.setattr(ElementHandle, "type", fill)
    monkeypatch.setattr(ElementHandle, "press", press)

    async def exercise():
        nonlocal channel
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            with pytest.raises(ChannelError):
                await channel.send(private)
            assert await channel._page.evaluate("sent") == []
            assert await channel._page.locator('[role="textbox"]').inner_text() == ""
            with pytest.raises(ChannelError):
                await channel.send(private)
        finally:
            await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("terms_stage", ["open", "confirmation"])
@pytest.mark.parametrize("native", [False, True])
def test_list_terms_cannot_bypass_scenario_authorization(
    monkeypatch, tmp_path, terms_stage, native
):
    mode = "native-list" if native else "list"
    markup = HTML.replace("window.menuKind='buttons'", f"window.menuKind='{mode}'")
    if terms_stage == "open":
        markup = markup.replace(
            "document.body.append(dialog);",
            "dialog.insertAdjacentHTML('afterbegin','<p>Authorize payment of $100</p>');"
            "document.body.append(dialog);",
        )
    else:
        markup = markup.replace(
            "dialog.dataset.selected=label;",
            "dialog.dataset.selected=label;"
            "dialog.insertAdjacentHTML('afterbegin','<p>Authorize payment of $100</p>');",
        )
    route_fixture(monkeypatch, html=markup)

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            await channel.send("Please explain the invoice.")
            await channel.receive()
            if native and terms_stage == "open":
                with pytest.raises(ChannelError, match="no supported customer controls"):
                    await channel.observe_controls()
                assert await channel._page.evaluate("sent") == ["Please explain the invoice."]
                return
            controls = await channel.observe_controls()
            if terms_stage == "open":
                assert controls["options"] == []
            else:
                option = next(o for o in controls["options"] if o.label == "Plans")
                with pytest.raises(ChannelError):
                    await channel.select_option(option.id)
            assert await channel._page.evaluate("sent") == ["Please explain the invoice."]
        finally:
            await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("evidence", ["wrong", "conflicting"])
def test_start_never_waits_through_wrong_or_conflicting_recipient_evidence(
    monkeypatch,
    tmp_path,
    evidence,
):
    other = "15557654321"
    if evidence == "wrong":
        hydrated = HTML.replace(PHONE, other)
    else:
        hydrated = HTML.replace("</header>", f'<span title="+{other}">Other</span></header>')
    # A tempting later correction must not hide the ownership violation.
    hydrated = hydrated.replace(
        "</html>",
        "<script>addEventListener('beforeinput',()=>fetch('https://service.example/startup-input'));"
        "setTimeout(()=>{document.querySelectorAll('header [title]').forEach(n=>"
        f"n.title='+{PHONE}');document.querySelectorAll('[data-id]').forEach(n=>"
        f"n.dataset.id=n.dataset.id.replace('{other}','{PHONE}'));"
        "},600);</script></html>",
    )
    markup = (
        '<section id="main"><header>Loading business chat</header></section>'
        "<script>setTimeout(()=>{document.open();document.write("
        + json.dumps(hydrated).replace("<", "\\u003c")
        + ");document.close();},200);</script>"
    )
    visits = []
    route_fixture(monkeypatch, html=markup, visits=visits)

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        with pytest.raises(ChannelError) as caught:
            await channel.start()
        assert caught.value.operation == "channel.ownership"
        assert channel._context is None and channel._profile_lock is None
        assert not any("startup-input" in url for url in visits)

    asyncio.run(exercise())


def test_missing_evidence_after_binding_revokes_dispatch_instead_of_waiting(
    monkeypatch,
    tmp_path,
):
    route_fixture(monkeypatch)

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            await channel._page.evaluate("""() => {
              document.querySelector('header span').removeAttribute('title');
              document.querySelector('#messages').replaceChildren();
            }""")
            with pytest.raises(ChannelError) as caught:
                await channel.send("Private probe")
            assert caught.value.operation == "channel.ownership"
            await channel._page.evaluate(
                "(phone) => document.querySelector('header span').title='+'+phone",
                PHONE,
            )
            with pytest.raises(ChannelError) as caught:
                await channel.send("Private probe")
            assert caught.value.operation == "channel.ownership"
            assert await channel._page.evaluate("sent") == []
        finally:
            await channel.close()

    asyncio.run(exercise())


def _contact_drawer(phone: str = PHONE) -> str:
    formatted = "+1 555 123-4567" if phone == PHONE else phone
    return (
        '<aside data-testid="chat-info-drawer">'
        '<button aria-label="Close" onclick="this.closest(\'aside\').remove()">Close</button>'
        f"<div><span>{formatted}</span></div>"
        '<section><div data-testid="section-about-and-phone-number">'
        "About and phone number</div>"
        '<div data-testid="container_with_separator"><div><div>'
        '<span data-testid="selectable-text"><span>'
        f"<span>{formatted}</span>"
        "</span></span></div></div></div></section>"
        "<section><span>Registration 12345678</span></section>"
        "</aside>"
    )


def _named_empty_chat(drawer: str, *, setup: str = "", opening: str = "") -> str:
    markup = HTML.replace(
        '<header><span title="+15551234567">Controlled Business</span></header>',
        '<header><div data-testid="conversation-info-header" role="button" tabindex="0">'
        '<span data-testid="conversation-info-header-chat-title-name">'
        '<span data-testid="conversation-info-header-chat-title">Controlled Business</span>'
        '</span><span class="contact-status">Business account</span></div></header>',
    ).replace(
        '<div data-id="false_15551234567@c.us_history"><span class="selectable-text">'
        "Your invoice is corrected. The refund is approved.</span></div>",
        "",
    )
    script = (
        "<script>"
        "window.contactOpens=0;"
        "document.querySelector('[data-testid=conversation-info-header]').onclick=()=>{"
        "contactOpens++;"
        f"document.body.insertAdjacentHTML('beforeend',{json.dumps(drawer)});"
        f"{opening}"
        "};"
        "addEventListener('input',()=>fetch('https://service.example/startup-input'));"
        f"{setup}"
        "</script>"
    )
    return markup.replace("</html>", script + "</html>")


@pytest.mark.parametrize(
    "opaque,hydrate,grouped",
    [(False, False, False), (True, False, False), (True, True, False), (True, True, True)],
)
def test_named_empty_chat_first_owned_exchange_uses_live_scoped_phone(
    monkeypatch, tmp_path, opaque, hydrate, grouped
):
    markup = _named_empty_chat(
        _contact_drawer().replace(
            "About and phone number</div>",
            'About and phone number</div><div class="contact-details">Loading</div>',
        ),
        opening="document.querySelector('.contact-status').textContent='Online';",
    )
    if opaque:
        markup = markup.replace(
            "node.dataset.id=(actor==='customer'?'true_':'false_')+'15551234567@c.us_'+(++counter);",
            "node.dataset.id='3EB0000000000000'+(++counter).toString(16).toUpperCase();"
            "node.dataset.testid='conv-msg-'+node.dataset.id;"
            "const container=document.createElement('div');"
            "container.dataset.testid='msg-container';"
            "const tail=document.createElement('span');"
            "tail.dataset.testid=actor==='customer'?'tail-out':'tail-in';"
            "container.append(tail);node.append(container);",
        ).replace(
            '<div id="messages">',
            '<div id="messages"><div data-id="3EB0AAAAAAAAAAAA" '
            'data-testid="conv-msg-3EB0AAAAAAAAAAAA">'
            '<span data-testid="system_message">System notice, not a reply</span></div>',
        )
        if grouped:
            markup = markup.replace(
                "container.append(tail);node.append(container);",
                "const group=document.createElement('div');"
                "group.style.display='flex';group.style.flexDirection='column';"
                "group.style.alignItems=actor==='customer'?'flex-end':'flex-start';"
                "group.append(container);setTimeout(()=>node.append(group),100);",
            )
        elif hydrate:
            markup = markup.replace(
                "container.append(tail);node.append(container);",
                "container.append(tail);setTimeout(()=>node.append(container),100);",
            )
    route_fixture(
        monkeypatch,
        html=markup,
    )

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            page = channel._page
            assert page is not None
            assert await page.locator("#main [data-id]").count() == int(opaque)
            assert await page.evaluate("contactOpens") == 1
            assert await channel.observe_initial() is None
            # Status is not chat identity; ordinary typing/online updates must
            # not revoke the independently pinned title and contact phone.
            await page.evaluate("document.querySelector('.contact-status').textContent='Typing…'")
            await page.evaluate(
                "document.querySelector('.contact-details')?.replaceChildren("
                "document.createTextNode('Contact details loaded'))"
            )
            receipt = await channel.send("First owned empty-chat exchange")
            response = await channel.receive()
            assert receipt.evidence["observed_echo"] is True
            assert response.messages[0].content == "Choose an option for invoice help."
            controls = await channel.observe_controls()
            await channel.select_option(controls["options"][0].id)
            assert (await channel.receive()).messages[
                0
            ].content == "What is your account reference?"
            assert await page.evaluate("sent") == ["First owned empty-chat exchange", "Plans"]
            assert await page.locator('[data-testid="chat-info-drawer"]').count() == 0
            assert PHONE not in receipt.model_dump_json() + response.model_dump_json()
        finally:
            await channel.close()

    asyncio.run(exercise())


def _virtualized_history() -> str:
    return (
        '<div data-id="3EB0BBBBBBBBBBBB" data-testid="conv-msg-3EB0BBBBBBBBBBBB">'
        '<div data-virtualized="true" style="min-height:53px"><div></div></div></div>'
    )


@pytest.mark.parametrize("discovery", [False, True])
def test_discovery_never_advances_an_existing_portability_conversation(
    monkeypatch, tmp_path, discovery
):
    markup = HTML.replace(
        "Your invoice is corrected. The refund is approved.",
        "Portabilidade: deseja ver os planos?",
    )
    route_fixture(monkeypatch, html=markup)

    async def exercise():
        channel = WhatsAppWebChannel(
            config(), limits(), browser_profile=tmp_path, discovery_only=discovery
        )
        try:
            await channel.start()
            assert channel._page is not None
            if discovery:
                with pytest.raises(ChannelError, match="forbidden in discovery"):
                    await channel.send("Olá")
                assert await channel._page.evaluate("sent") == []
            else:
                assert (await channel.send("Olá")).evidence["observed_echo"] is True
        finally:
            await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "message", ["Owned message", "Oi! Quero saber dos planos 😉\nPode ajudar?"]
)
def test_native_editor_clear_removes_framework_prefill_and_preserves_emoji(
    monkeypatch, tmp_path, message
):
    markup = (
        _named_empty_chat(
            _contact_drawer(),
            setup=(
                "window.websiteDraft='UNAUTHORIZED WEBSITE PREFILL ';"
                "document.querySelector('[role=textbox]').addEventListener('keydown',e=>{"
                "if(e.key==='Backspace')window.websiteDraft='';});"
            ),
        )
        .replace(
            "const text=event.target.innerText;", "const text=websiteDraft+event.target.innerText;"
        )
        .replace(
            "node.append(prose);",
            "if(text.includes('😉')){"
            "const parts=text.split('😉');prose.replaceChildren(document.createTextNode(parts[0]));"
            "const emoji=document.createElement('img');emoji.className='emoji selectable-text';"
            "emoji.alt='😉';emoji.dataset.plainText='😉';prose.append(emoji,parts[1]);}"
            "node.append(prose);",
        )
    )
    route_fixture(monkeypatch, html=markup)

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            assert channel._page is not None
            assert await channel._page.evaluate("websiteDraft") == ""
            assert await channel._page.locator('[data-testid="chat-info-drawer"]').count() == 0
            receipt = await channel.send(message)
            assert receipt.evidence["observed_echo"] is True
            assert await channel._page.evaluate("sent") == [message]
            assert (await channel.receive()).messages
        finally:
            await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "change",
    ["title", "title_restored", "panel", "panel_restored", "root", "drawer", "recipient", "anchor"],
)
def test_closed_contact_proof_revokes_on_chat_change_before_dispatch(monkeypatch, tmp_path, change):
    markup = _named_empty_chat(_contact_drawer())
    if change == "anchor":
        markup = markup.replace(
            '<div id="messages">', '<div id="messages">' + _virtualized_history()
        )
    route_fixture(monkeypatch, html=markup)
    changes = {
        "title": "title.textContent='Another Business'",
        "title_restored": (
            "const text=title.textContent;title.textContent='Other';title.textContent=text"
        ),
        "panel": "panel.replaceWith(panel.cloneNode(true))",
        "panel_restored": "panel.remove();root.append(panel)",
        "anchor": "const n=panel.querySelector('[data-id]');n.replaceWith(n.cloneNode(true))",
        "root": "root.replaceWith(root.cloneNode(true))",
        "drawer": f"document.body.insertAdjacentHTML('beforeend',{json.dumps(_contact_drawer())})",
        "recipient": "root.querySelector('header').insertAdjacentHTML('beforeend',"
        "'<span title=\"+15557654321\">Other</span>')",
    }

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            assert channel._page is not None
            await channel._page.evaluate(
                "()=>{const root=document.querySelector('#main'),"
                "title=root.querySelector('[data-testid=conversation-info-header-chat-title]'),"
                "panel=root.querySelector('[data-testid=conversation-panel-messages]');"
                + changes[change]
                + ";}"
            )
            with pytest.raises(ChannelError):
                await channel.send("Private message after switch")
            assert await channel._page.evaluate("sent") == []
            assert not (await channel._page.locator("[role=textbox]").inner_text()).strip()
        finally:
            await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("close", ["animated", "missing", "ambiguous", "wrong_phone"])
def test_contact_info_close_is_scoped_and_requires_still_valid_proof(monkeypatch, tmp_path, close):
    drawer = _contact_drawer()
    if close == "animated":
        drawer = drawer.replace(
            "this.closest('aside').remove()",
            "const d=this.closest('aside');d.setAttribute('aria-hidden','true');"
            "setTimeout(()=>d.remove(),100)",
        )
    elif close == "missing":
        drawer = drawer.replace('aria-label="Close"', 'aria-label="Unrelated action"')
    elif close == "ambiguous":
        drawer = drawer.replace("</aside>", '<button aria-label="Close">Close</button></aside>')
    else:
        from playwright.async_api import ElementHandle

        original = ElementHandle.click

        async def click(node, **kwargs):
            if await node.get_attribute("aria-label") == "Close":
                await node.evaluate(
                    "n=>n.closest('aside').querySelector('section [data-testid=selectable-text]')"
                    ".textContent='+15557654321'"
                )
            return await original(node, **kwargs)

        monkeypatch.setattr(ElementHandle, "click", click)
    route_fixture(monkeypatch, html=_named_empty_chat(drawer))

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            if close == "animated":
                await channel.start()
                assert channel._page is not None
                assert await channel._page.locator('[data-testid="chat-info-drawer"]').count() == 0
                assert (await channel.send("Owned message after close")).evidence["observed_echo"]
            else:
                with pytest.raises(ChannelError):
                    await channel.start()
                assert channel._context is None
        finally:
            await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("initial", ["placeholder", "notification"])
def test_virtualized_history_does_not_block_start_or_become_fresh_evidence(
    monkeypatch, tmp_path, initial
):
    history = _virtualized_history()
    if initial == "notification":
        history = history.replace(
            '<div data-virtualized="true" style="min-height:53px"><div></div></div>',
            '<span data-testid="system_message">Old notification</span>',
        )
    markup = _named_empty_chat(_contact_drawer()).replace(
        '<div id="messages">', '<div id="messages">' + history
    )
    route_fixture(monkeypatch, html=markup)

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            page = channel._page
            assert page is not None
            assert "3EB0BBBBBBBBBBBB" in channel._seen
            assert await channel.observe_initial() is None
            if initial == "notification":
                await page.locator('[data-id="3EB0BBBBBBBBBBBB"]').evaluate(
                    """(node, markup)=>{const copy=document.createElement('div');
                    copy.innerHTML=markup;node.innerHTML=copy.firstElementChild.innerHTML;}""",
                    _virtualized_history(),
                )
            receipt = await channel.send("Fresh owned probe")
            assert receipt.evidence["observed_echo"] is True
            # Hydrating a pre-run placeholder after dispatch must not manufacture
            # a fresh answer, even when the old content claims business success.
            await page.locator('[data-id="3EB0BBBBBBBBBBBB"]').evaluate(
                """node => node.innerHTML = '<div data-testid="msg-container">' +
                '<span data-testid="tail-in"></span><span class="selectable-text">' +
                'OLD SUCCESS MUST NOT BE EVIDENCE</span></div>'"""
            )
            response = await channel.receive()
            assert [m.content for m in response.messages] == ["Choose an option for invoice help."]
            assert await page.evaluate("sent") == ["Fresh owned probe"]
        finally:
            await channel.close()

    asyncio.run(exercise())


def test_new_unread_virtualized_message_fails_closed(monkeypatch, tmp_path):
    route_fixture(monkeypatch, html=_named_empty_chat(_contact_drawer()))

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            assert channel._page is not None
            await channel._page.evaluate("window.delay=10000")
            await channel.send("Fresh owned probe")
            await channel._page.locator("#messages").evaluate(
                "(node, markup)=>node.insertAdjacentHTML('beforeend',markup)",
                _virtualized_history(),
            )
            with pytest.raises(ChannelError, match="unobserved message was virtualized"):
                await channel.receive()
            assert await channel._page.evaluate("sent") == ["Fresh owned probe"]
        finally:
            await channel.close()

    asyncio.run(exercise())


def test_unmarked_opaque_shell_is_not_treated_as_virtualized_history(monkeypatch, tmp_path):
    route_fixture(monkeypatch, html=_named_empty_chat(_contact_drawer()))

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            assert channel._page is not None
            channel._limits = limits().model_copy(update={"browser_action_timeout_seconds": 0.2})
            await channel._page.locator("#messages").evaluate(
                "(node, markup)=>node.insertAdjacentHTML('beforeend',markup)",
                _virtualized_history().replace('data-virtualized="true"', ""),
            )
            with pytest.raises(ChannelError, match="message direction did not become ready"):
                await channel._snapshot(baseline=True)
            assert await channel._page.evaluate("sent") == []
        finally:
            await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "evidence",
    [
        "wrong_phone",
        "malformed_phone",
        "later_corrected_phone",
        "title_switched_on_open",
        "duplicate_phone",
        "conflicting_phone",
        "duplicate_drawer",
        "preexisting_drawer",
        "wrong_header",
        "conflicting_header",
        "wrong_message",
        "malformed_message",
        "unsupported_message",
        "duplicate_opener",
    ],
)
def test_named_empty_chat_rejects_unowned_or_ambiguous_proof_before_input(
    monkeypatch, tmp_path, evidence
):
    from playwright.async_api import ElementHandle

    original_click = ElementHandle.click
    opened = []

    async def click(node, **kwargs):
        if await node.get_attribute("data-testid") == "conversation-info-header":
            opened.append(True)
        return await original_click(node, **kwargs)

    monkeypatch.setattr(ElementHandle, "click", click)
    drawer = _contact_drawer()
    setup = ""
    opening = ""
    other = "15557654321"
    if evidence == "wrong_phone":
        drawer = _contact_drawer(other)
    elif evidence == "malformed_phone":
        drawer = _contact_drawer("+1555-invalid-phone")
    elif evidence == "later_corrected_phone":
        drawer = _contact_drawer(other)
        opening = (
            "setTimeout(()=>document.querySelector("
            "'section [data-testid=selectable-text]').textContent='+1 555 123-4567',400);"
        )
    elif evidence == "title_switched_on_open":
        opening = (
            "document.querySelector('[data-testid=conversation-info-header-chat-title]')"
            ".textContent='Another Business';"
        )
    elif evidence in {"duplicate_phone", "conflicting_phone"}:
        extra = "+1 555 123-4567" if evidence == "duplicate_phone" else other
        drawer = drawer.replace(
            "</section>",
            '<div data-testid="container_with_separator">'
            f'<span data-testid="selectable-text">{extra}</span></div></section>',
            1,
        )
    elif evidence == "duplicate_drawer":
        drawer += drawer
    elif evidence == "preexisting_drawer":
        setup = f"document.body.insertAdjacentHTML('beforeend',{json.dumps(drawer)});"
    elif evidence in {"wrong_header", "conflicting_header"}:
        setup = (
            "document.querySelector('[data-testid=conversation-info-header-chat-title]')"
            f".setAttribute('title','+{other}');"
        )
        if evidence == "conflicting_header":
            setup += (
                "document.querySelector('header').insertAdjacentHTML('beforeend',"
                f"'<span title=\"+{PHONE}\">Other numeric evidence</span>');"
            )
    elif evidence in {"wrong_message", "malformed_message", "unsupported_message"}:
        raw = {
            "wrong_message": f"false_{other}@c.us_history",
            "malformed_message": "false_unsupported_history",
            "unsupported_message": "unsupported_history",
        }[evidence]
        setup = (
            "document.querySelector('#messages').insertAdjacentHTML('beforeend',"
            f"'<div data-id=\"{raw}\"></div>');"
        )
    elif evidence == "duplicate_opener":
        setup = (
            "const opener=document.querySelector('[data-testid=conversation-info-header]');"
            "opener.after(opener.cloneNode(true));"
        )
    visits = []
    route_fixture(
        monkeypatch,
        html=_named_empty_chat(drawer, setup=setup, opening=opening),
        visits=visits,
    )

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        with pytest.raises(ChannelError) as caught:
            await channel.start()
        assert PHONE not in str(caught.value) and other not in str(caught.value)
        assert not any("startup-input" in url for url in visits)
        no_open = {
            "preexisting_drawer",
            "wrong_header",
            "conflicting_header",
            "wrong_message",
            "malformed_message",
            "unsupported_message",
            "duplicate_opener",
        }
        assert len(opened) == (0 if evidence in no_open else 1)
        assert channel._context is None and channel._profile_lock is None

    asyncio.run(exercise())


def _contact_change(change: str) -> str:
    changes = {
        "phone_changed": "phone.textContent='+15557654321'",
        "phone_replaced": "phone.replaceWith(phone.cloneNode(true))",
        "phone_removed": "phone.remove()",
        "phone_ambiguous": "phone.after(phone.cloneNode(true))",
        "phone_ambiguous_restored": (
            "const duplicate=phone.cloneNode(true);phone.after(duplicate);duplicate.remove()"
        ),
        "drawer_replaced": "drawer.replaceWith(drawer.cloneNode(true))",
        "drawer_closed": "drawer.remove()",
        "drawer_ambiguous": "drawer.after(drawer.cloneNode(true))",
        "title_changed": "title.textContent='Another Business'",
        "title_replaced": "title.replaceWith(title.cloneNode(true))",
        "header_replaced": "header.replaceWith(header.cloneNode(true))",
        "opener_replaced": "opener.replaceWith(opener.cloneNode(true))",
        "chat_replaced": (
            "const next=root.cloneNode(true);"
            "next.querySelector('[contenteditable]').textContent='';root.replaceWith(next)"
        ),
        "phone_restored": (
            "const before=phone.textContent;"
            "phone.textContent='+15557654321';phone.textContent=before"
        ),
        "drawer_restored": "drawer.remove();document.body.append(drawer)",
        "title_restored": (
            "const before=title.textContent;"
            "title.textContent='Another Business';title.textContent=before"
        ),
        "conflicting_history": (
            "document.querySelector('#messages').insertAdjacentHTML('beforeend',"
            "'<div data-id=\"false_15557654321@c.us_other\"></div>')"
        ),
        "malformed_history": (
            "document.querySelector('#messages').insertAdjacentHTML('beforeend',"
            "'<div data-id=\"false_unsupported\"></div>')"
        ),
        "unsupported_history": (
            "document.querySelector('#messages').insertAdjacentHTML('beforeend',"
            "'<div data-id=\"unsupported_history\"></div>')"
        ),
    }
    return (
        "()=>{"
        "const root=document.querySelector('#main'),header=root.querySelector('header'),"
        "opener=header.querySelector('[data-testid=conversation-info-header]'),"
        "title=opener.querySelector('[data-testid=conversation-info-header-chat-title]'),"
        "drawer=document.querySelector('[data-testid=chat-info-drawer]'),"
        "phone=drawer.querySelector('section [data-testid=selectable-text]');"
        + changes[change]
        + ";}"
    )


@pytest.mark.parametrize(
    "change",
    [
        "phone_changed",
        "phone_replaced",
        "phone_removed",
        "phone_ambiguous",
        "phone_ambiguous_restored",
        "drawer_replaced",
        "drawer_closed",
        "drawer_ambiguous",
        "title_changed",
        "title_replaced",
        "header_replaced",
        "opener_replaced",
        "chat_replaced",
        "phone_restored",
        "drawer_restored",
        "title_restored",
        "conflicting_history",
        "malformed_history",
        "unsupported_history",
    ],
)
def test_named_chat_proof_loss_or_mutation_revokes_before_any_draft(monkeypatch, tmp_path, change):
    async def keep_open(self):
        pass

    # Exercise the open-proof phase independently of the owned close transition.
    monkeypatch.setattr(WhatsAppWebChannel, "_close_contact_proof", keep_open)
    route_fixture(monkeypatch, html=_named_empty_chat(_contact_drawer()))

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            page = channel._page
            assert page is not None
            await page.evaluate(_contact_change(change))
            with pytest.raises(ChannelError):
                await channel.send("Private empty-chat material")
            assert await page.evaluate("sent") == []
            assert await page.locator('[role="textbox"]').inner_text() == ""
            with pytest.raises(ChannelError):
                await channel.send("Private empty-chat material")
        finally:
            await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("stage", ["fill", "enter", "click"])
@pytest.mark.parametrize(
    "change",
    ["phone_changed", "phone_replaced", "drawer_closed", "title_changed", "chat_replaced"],
)
def test_named_chat_capture_guard_blocks_proof_change_at_native_dispatch(
    monkeypatch, tmp_path, stage, change
):
    from playwright.async_api import ElementHandle

    async def keep_open(self):
        pass

    monkeypatch.setattr(WhatsAppWebChannel, "_close_contact_proof", keep_open)
    route_fixture(monkeypatch, html=_named_empty_chat(_contact_drawer()))
    original_fill = ElementHandle.type
    original_press = ElementHandle.press
    original_click = ElementHandle.click
    private = "Private current-run named-chat material"
    channel = None
    armed = False

    async def mutate():
        assert channel is not None and channel._page is not None
        await channel._page.evaluate(_contact_change(change))

    async def fill(node, value, **kwargs):
        if armed and stage == "fill" and value == private:
            await mutate()
        return await original_fill(node, value, **kwargs)

    async def press(node, key, **kwargs):
        if armed and stage == "enter" and key == "Enter":
            await mutate()
        return await original_press(node, key, **kwargs)

    async def click(node, **kwargs):
        if armed and stage == "click":
            await mutate()
        return await original_click(node, **kwargs)

    monkeypatch.setattr(ElementHandle, "type", fill)
    monkeypatch.setattr(ElementHandle, "press", press)
    monkeypatch.setattr(ElementHandle, "click", click)

    async def exercise():
        nonlocal channel, armed
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            page = channel._page
            assert page is not None
            option = None
            if stage == "click":
                await channel.send("First owned exchange")
                await channel.receive()
                option = (await channel.observe_controls())["options"][0]
            armed = True
            with pytest.raises(ChannelError):
                if option is None:
                    await channel.send(private)
                else:
                    await channel.select_option(option.id)
            expected = ["First owned exchange"] if stage == "click" else []
            assert await page.evaluate("sent") == expected
            assert await page.locator('[role="textbox"]').inner_text() == ""
        finally:
            await channel.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "pending", ["title", "drawer", "phone", "drawer_replaced", "drawer_hidden"]
)
def test_named_empty_chat_waits_within_startup_bound_for_contact_phone(
    monkeypatch, tmp_path, pending
):
    setup = ""
    if pending == "title":
        drawer = _contact_drawer()
        opening = ""
        setup = (
            "const title=document.querySelector("
            "'[data-testid=conversation-info-header-chat-title]');"
            "title.textContent='';setTimeout(()=>title.textContent='Controlled Business',150);"
        )
    elif pending == "drawer":
        drawer = ""
        opening = (
            "setTimeout(()=>document.body.insertAdjacentHTML('beforeend',"
            + json.dumps(_contact_drawer())
            + "),150);"
        )
    elif pending == "drawer_replaced":
        drawer = '<aside data-testid="chat-info-drawer">Loading contact</aside>'
        opening = (
            "setTimeout(()=>document.querySelector('[data-testid=chat-info-drawer]').outerHTML="
            + json.dumps(_contact_drawer())
            + ",150);"
        )
    elif pending == "drawer_hidden":
        drawer = _contact_drawer().replace(
            '<aside data-testid="chat-info-drawer">',
            '<aside data-testid="chat-info-drawer" hidden>',
        )
        opening = (
            "setTimeout(()=>document.querySelector('[data-testid=chat-info-drawer]')"
            ".hidden=false,150);"
        )
    else:
        drawer = _contact_drawer("")
        opening = (
            "setTimeout(()=>document.querySelector("
            "'section [data-testid=selectable-text]').textContent='+1 555 123-4567',150);"
        )
    route_fixture(monkeypatch, html=_named_empty_chat(drawer, setup=setup, opening=opening))

    async def exercise():
        channel = WhatsAppWebChannel(config(), limits(), browser_profile=tmp_path)
        try:
            await channel.start()
            assert (await channel.send("Owned contact readiness probe")).evidence["observed_echo"]
            assert (await channel.receive()).messages[
                0
            ].content == "Choose an option for invoice help."
            assert await channel._page.evaluate("contactOpens") == 1
        finally:
            await channel.close()

    asyncio.run(exercise())
