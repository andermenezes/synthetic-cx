"""Local DOM-only native option tests; no external targets or probes."""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from playwright.async_api import Browser, ElementHandle, Page, Route, async_playwright
from playwright.async_api import Error as PlaywrightError

from syntheticcx.adapters.channel.options import ReplyOptions, inspect_reply_options
from syntheticcx.adapters.channel.playwright import PlaywrightChannel, locate
from syntheticcx.config.models import ChannelConfig, ExecutionLimits
from syntheticcx.domain.models import SendReceipt
from syntheticcx.errors import ChannelError


def channel_config() -> tuple[ChannelConfig, ExecutionLimits]:
    base = yaml.safe_load(Path("tests/fixtures/discovery-sample.yaml").read_text())
    data = base["channel"]
    data.update(
        open_locator=None,
        frame_locator=None,
        input_locator={"strategy": "css", "value": "#composer"},
        assistant_message_locator={"strategy": "css", "value": ".assistant"},
        customer_message_locator={"strategy": "css", "value": ".customer"},
        message_identity={"strategy": "attribute", "attribute_name": "data-id"},
        send_strategy={"type": "press_enter"},
        response_completion={"strategy": "quiet_interval", "quiet_interval_seconds": 0.08},
        reply_options={
            "scope_locator": {"strategy": "css", "value": "#chat"},
            "option_locator": {"strategy": "css", "value": ".option"},
        },
    )
    return ChannelConfig.model_validate(data), ExecutionLimits.model_validate(base["execution"])


SHELL = '<section id="chat"><input id="composer"></section>'
REPLY = (
    '<article class="assistant" data-id="a1"><p>Choose an information topic</p>'
    '<button type="button" class="option">Plans</button>'
    '<button type="button" class="option">Coverage</button></article>'
)


async def local_page(browser: Browser, config: ChannelConfig) -> Page:
    """Use only fulfilled local requests, with a real configured HTTP origin."""
    page = await browser.new_page()

    async def fulfill(route: Route) -> None:
        await route.fulfill(
            status=200, content_type="text/html", body="<!doctype html><body></body>"
        )

    await page.route("**/*", fulfill)
    await page.goto(str(config.url))
    return page


@pytest.mark.parametrize("remove_composer", [False, True])
def test_select_native_retained_option_with_canonical_echo_and_removed_controls(
    remove_composer: bool,
) -> None:
    async def exercise() -> None:
        config, limits = channel_config()
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await local_page(browser, config)
            await page.set_content(SHELL)
            channel = PlaywrightChannel(config, limits)
            channel._scope = page
            channel._page = page
            await channel._capture_history()
            channel._options = ReplyOptions(page, config, locate)
            await channel._options.bind()
            await page.locator("#chat").evaluate(
                "(node, html) => node.insertAdjacentHTML('beforeend', html)",
                REPLY.replace("<p>Choose an information topic</p>", "")
                if remove_composer
                else REPLY,
            )
            if remove_composer:
                await page.locator("#composer").evaluate("node => node.remove()")
            greeting = (await channel.receive()).messages[0]
            assert greeting.content == ("" if remove_composer else "Choose an information topic")
            assert [option.label for option in greeting.options] == ["Plans", "Coverage"]
            assert all(
                node["eligible"]
                for node in (await channel.inspect_reply_options(config))[0]["options"]
            )
            await page.evaluate("""() => {
              document.querySelector('.option').addEventListener('click', () => {
                document.querySelectorAll('.option').forEach(node => node.remove());
                document.querySelector('#chat').insertAdjacentHTML('beforeend',
                  '<article class="customer" data-id="c1">canonical_plan_key</article>' +
                  '<article class="assistant" data-id="a2">Here are the plan details</article>');
              });
            }""")
            receipt = await channel.select_option(greeting.options[0].id)
            assert receipt.selected_label == "Plans"
            assert receipt.source_message_id == greeting.id
            assert receipt.message_id == "customer:c1"
            assert receipt.evidence["echo"] == "observed"
            assert (await channel.receive()).messages[0].content == "Here are the plan details"
            with pytest.raises(ChannelError):
                await channel.select_option(greeting.options[0].id)
            await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "mutation",
    [
        "document.querySelector('.option').outerHTML = document.querySelector('.option').outerHTML",
        "document.querySelector('.option').disabled = true",
        "document.querySelector('.option').setAttribute('aria-disabled', 'true')",
        "document.querySelector('.option').setAttribute('aria-hidden', 'true')",
        "document.querySelector('.option').style.display = 'none'",
        "document.querySelector('.option').textContent = 'Billing'",
        "document.querySelector('#chat').append(document.querySelector('.option'))",
        "document.querySelector('body').append(document.querySelector('.option'))",
        "document.querySelector('#chat').outerHTML = document.querySelector('#chat').outerHTML",
        "document.querySelector('#composer').outerHTML = '<input id=\"composer\">'",
        "document.querySelector('.assistant').insertAdjacentHTML('beforeend', "
        '\'<button class="option" type="button">Plans</button>\')',
        "document.querySelector('#chat').insertAdjacentHTML('beforeend', "
        '\'<article class="assistant" data-id="a2">New reply</article>\')',
    ],
)
def test_at_click_changes_reject_without_activation(mutation: str) -> None:
    async def exercise() -> None:
        config, _ = channel_config()
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await local_page(browser, config)
            await page.set_content(SHELL.replace("</section>", REPLY + "</section>"))
            options = ReplyOptions(page, config, locate)
            await options.bind()
            choices = options.publish(await options.collect(), {0: "assistant:a1"})[0]
            await page.evaluate(mutation)
            with pytest.raises(ChannelError):
                await options.validate(choices[0].id)
            await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "control",
    [
        '<a class="option" href="/buy">Plans</a>',
        '<button class="option" type="submit">Plans</button>',
        '<button class="option" type="reset">Plans</button>',
        '<form><button class="option" type="button">Plans</button></form>',
        '<form><button class="option">Plans</button></form>',
        '<fieldset disabled><button class="option" type="button">Plans</button></fieldset>',
        '<button class="option" type="button" data-url="/buy">Plans</button>',
    ],
)
def test_read_only_inspection_rejects_structurally_unsafe_controls(control: str) -> None:
    async def exercise() -> None:
        config, _ = channel_config()
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await local_page(browser, config)
            await page.set_content(
                SHELL.replace(
                    "</section>",
                    '<article class="assistant" data-id="a1">Choose an information topic'
                    + control
                    + "</article></section>",
                )
            )
            entries = await inspect_reply_options(page, config)
            assert entries[0]["prose"] == "Choose an information topic"
            assert entries[0]["options"][0]["eligible"] is False
            await browser.close()

    asyncio.run(exercise())


def test_typeless_button_outside_forms_is_a_safe_option_control() -> None:
    """Widgets commonly omit type=button; absent form ancestry it stays safe."""

    async def exercise() -> None:
        config, _ = channel_config()
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await local_page(browser, config)
            await page.set_content(
                SHELL.replace(
                    "</section>",
                    '<article class="assistant" data-id="a1">Choose an information topic'
                    '<button class="option">Plans</button>'
                    "</article></section>",
                )
            )
            entries = await inspect_reply_options(page, config)
            assert entries[0]["options"][0]["eligible"] is True
            await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("enabled", [False, True])
def test_options_only_reply_and_legacy_textcontent(enabled: bool) -> None:
    async def exercise() -> None:
        config, limits = channel_config()
        if not enabled:
            config = config.model_copy(update={"reply_options": None})
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await local_page(browser, config)
            await page.set_content(SHELL)
            channel = PlaywrightChannel(config, limits)
            channel._scope = page
            await channel._capture_history()
            if enabled:
                channel._options = ReplyOptions(page, config, locate)
                await channel._options.bind()
            await page.locator("#chat").evaluate(
                "(node, html) => node.insertAdjacentHTML('beforeend', html)",
                REPLY.replace("<p>Choose an information topic</p>", ""),
            )
            message = (await channel.receive()).messages[0]
            assert message.content == ("" if enabled else "PlansCoverage")
            assert len(message.options) == (2 if enabled else 0)
            await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("framed", [False, True])
def test_inspection_keeps_shadow_and_frame_scope(framed: bool) -> None:
    async def exercise() -> None:
        config, _ = channel_config()
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await local_page(browser, config)
            if framed:
                await page.set_content('<iframe srcdoc=""></iframe>')
                frame = page.frames[1]
                await frame.set_content('<section id="chat"><div id="host"></div></section>')
                scope = page.locator("iframe").content_frame
            else:
                await page.set_content('<section id="chat"><div id="host"></div></section>')
                frame = page
                scope = page
            await frame.locator("#host").evaluate(
                "(node, html) => node.attachShadow({mode: 'open'}).innerHTML = html",
                '<input id="composer">' + REPLY,
            )
            inspected = await inspect_reply_options(scope, config)
            assert [node["label"] for node in inspected[0]["options"]] == ["Plans", "Coverage"]
            assert all(node["eligible"] for node in inspected[0]["options"])
            await browser.close()

    asyncio.run(exercise())


def test_option_membership_and_disabled_changes_extend_completion_window() -> None:
    async def exercise() -> None:
        config, limits = channel_config()
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await local_page(browser, config)
            await page.set_content(SHELL)
            channel = PlaywrightChannel(config, limits)
            channel._scope = page
            await channel._capture_history()
            channel._options = ReplyOptions(page, config, locate)
            await channel._options.bind()
            await page.locator("#chat").evaluate(
                "(node, html) => node.insertAdjacentHTML('beforeend', html)", REPLY
            )
            await page.evaluate("""() => {
              document.querySelectorAll('.option').forEach(node => node.disabled = true);
              setTimeout(() => document.querySelectorAll('.option').forEach(node =>
                node.disabled = false), 60);
              setTimeout(() => document.querySelector('.assistant').insertAdjacentHTML('beforeend',
                '<button type="button" class="option">Billing</button>'), 110);
            }""")
            message = (await channel.receive()).messages[0]
            assert [node.label for node in message.options] == ["Plans", "Coverage", "Billing"]
            await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("send_first", [False, True])
def test_send_and_selection_share_one_submission_lock(send_first: bool) -> None:
    async def exercise() -> None:
        config, limits = channel_config()
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await local_page(browser, config)
            await page.set_content(SHELL)
            channel = PlaywrightChannel(config, limits)
            channel._scope = page
            await channel._capture_history()
            channel._options = ReplyOptions(page, config, locate)
            await channel._options.bind()
            await page.locator("#chat").evaluate(
                "(node, html) => node.insertAdjacentHTML('beforeend', html)", REPLY
            )
            option = (await channel.receive()).messages[0].options[0]
            await page.evaluate("""() => {
              window.activations = 0;
              const echo = text => {
                window.activations++;
                const node = document.createElement('article');
                node.className = 'customer'; node.dataset.id = 'c1'; node.textContent = text;
                document.querySelector('#chat').append(node);
              };
              document.querySelector('.option').onclick = () => echo('canonical_key');
              document.querySelector('#composer').onkeydown = e => {
                if (e.key === 'Enter') echo(e.target.value);
              };
            }""")
            tasks = [channel.send("Free form"), channel.select_option(option.id)]
            if not send_first:
                tasks.reverse()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            assert sum(isinstance(result, SendReceipt) for result in results) == 1
            assert sum(isinstance(result, ChannelError) for result in results) == 1
            assert await page.evaluate("window.activations") == 1
            await browser.close()

    asyncio.run(exercise())


def test_failure_after_activation_is_uncertain_and_never_retried() -> None:
    async def exercise() -> None:
        config, limits = channel_config()
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await local_page(browser, config)
            await page.set_content(SHELL)
            channel = PlaywrightChannel(config, limits)
            channel._scope = page
            await channel._capture_history()
            channel._options = ReplyOptions(page, config, locate)
            await channel._options.bind()
            await page.locator("#chat").evaluate(
                "(node, html) => node.insertAdjacentHTML('beforeend', html)", REPLY
            )
            choice = (await channel.receive()).messages[0].options[0]
            capability = channel._options.registry[choice.id]
            click = capability.option.node.click
            await page.evaluate("""() => {
              window.activations = 0;
              document.querySelector('.option').onclick = () => window.activations++;
            }""")

            async def fail_after_click(**kwargs: object) -> None:
                await click()
                raise PlaywrightError("simulated lost acknowledgement")

            with patch.object(capability.option.node, "click", side_effect=fail_after_click):
                with pytest.raises(ChannelError):
                    await channel.select_option(choice.id)
            assert channel._submission_uncertain
            assert not channel._options.registry
            with pytest.raises(ChannelError):
                await channel.select_option(choice.id)
            with pytest.raises(ChannelError):
                await channel.send("fallback prohibited")
            assert await page.evaluate("window.activations") == 1
            await browser.close()

    asyncio.run(exercise())


def test_slotted_control_labels_never_enter_prose() -> None:
    async def exercise() -> None:
        config, _ = channel_config()
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await local_page(browser, config)
            await page.set_content(
                SHELL.replace(
                    "</section>",
                    '<article class="assistant" data-id="a1">Choose an information topic'
                    '<div id="option-host">Plans</div></article></section>',
                )
            )
            await page.locator("#option-host").evaluate("""node => {
              node.attachShadow({mode: 'open'}).innerHTML =
                '<button class="option" type="button"><slot></slot></button>';
            }""")
            result = await inspect_reply_options(page, config)
            assert result[0]["prose"] == "Choose an information topic"
            assert result[0]["options"][0]["label"] == "Plans"
            assert result[0]["options"][0]["eligible"]
            await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("navigation", ["parent", "frame"])
def test_navigated_parent_or_frame_origin_rejected_without_click(navigation: str) -> None:
    async def exercise() -> None:
        config, limits = channel_config()
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await local_page(browser, config)
            await page.set_content('<iframe srcdoc=""></iframe>')
            scope = page.locator("iframe").content_frame
            frame = page.frames[1]
            await frame.set_content(SHELL)
            channel = PlaywrightChannel(config, limits)
            channel._scope = scope
            channel._page = page
            await channel._capture_history()
            channel._options = ReplyOptions(scope, config, locate)
            await channel._options.bind()
            await frame.locator("#chat").evaluate(
                "(node, html) => node.insertAdjacentHTML('beforeend', html)", REPLY
            )
            choice = (await channel.receive()).messages[0].options[0]
            capability = channel._options.registry[choice.id]
            target = "http://127.0.0.2:9999/changed-origin"
            if navigation == "parent":
                await page.goto(target)
            else:
                await frame.goto(target)
                await frame.set_content(SHELL.replace("</section>", REPLY + "</section>"))
            with patch.object(capability.option.node, "click") as click:
                with pytest.raises(ChannelError, match="origin"):
                    await channel.select_option(choice.id)
                click.assert_not_called()
            assert not channel._submission_uncertain
            assert not channel._options.registry
            with pytest.raises(ChannelError, match="origin"):
                await channel.inspect_reply_options(config)
            with pytest.raises(ChannelError, match="origin"):
                await inspect_reply_options(scope, config)
            await browser.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "invalid_wrapper",
    [
        "root",
        "composer",
        "customer",
        "send",
        "nested",
        "log",
        "dialog",
        "status",
        "form",
        "toolbar",
        "feed",
        "list",
        "progressbar",
    ],
)
def test_chat_containers_and_mixed_actor_wrappers_rejected_without_activation(
    invalid_wrapper: str,
) -> None:
    async def exercise() -> None:
        config, limits = channel_config()
        config = config.model_copy(
            update={
                "send_strategy": config.send_strategy.model_validate(
                    {
                        "type": "click",
                        "locator": {"strategy": "css", "value": "#send"},
                    }
                ),
            }
        )
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await local_page(browser, config)
            await page.set_content(SHELL)
            channel = PlaywrightChannel(config, limits)
            channel._scope = page
            await channel._capture_history()
            channel._options = ReplyOptions(page, config, locate)
            await channel._options.bind()
            await page.locator("#chat").evaluate(
                "(node, html) => node.insertAdjacentHTML('beforeend', html)", REPLY
            )
            choice = (await channel.receive()).messages[0].options[0]
            capability = channel._options.registry[choice.id]
            await page.evaluate(
                """kind => {
              const wrapper = document.querySelector('.assistant');
              if (kind === 'root') {
                wrapper.classList.remove('assistant');
                document.querySelector('#chat').classList.add('assistant');
              } else if (kind === 'composer') {
                wrapper.append(document.querySelector('#composer'));
              } else if (kind === 'customer') {
                wrapper.insertAdjacentHTML('beforeend',
                  '<article class="customer" data-id="c1">Hello</article>');
              } else if (kind === 'send') {
                wrapper.insertAdjacentHTML('beforeend', '<button id="send">Send</button>');
              } else if (kind === 'nested') {
                wrapper.insertAdjacentHTML('beforeend',
                  '<article class="assistant" data-id="a2">Nested bubble</article>');
              } else if (kind === 'form') {
                wrapper.outerHTML = wrapper.outerHTML.replaceAll('article', 'form');
              } else {
                wrapper.setAttribute('role', kind);
              }
            }""",
                invalid_wrapper,
            )
            with patch.object(capability.option.node, "click") as click:
                with pytest.raises(ChannelError):
                    await channel.select_option(choice.id)
                click.assert_not_called()
            with pytest.raises(ChannelError):
                await channel.inspect_reply_options(config)
            assert not channel._submission_uncertain
            await browser.close()

    asyncio.run(exercise())


def test_mixed_informational_and_unsafe_siblings_publish_only_safe_topics() -> None:
    async def exercise() -> None:
        config, _ = channel_config()
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await local_page(browser, config)
            await page.set_content(SHELL.replace("</section>", REPLY + "</section>"))
            labels = [
                "Activate plan",
                "Withdraw funds",
                "pa\u200by now",
                "\u200b",
                "Ativar plano",
                "Sacar dinheiro",
                "Alterar plano",
                "pa\u200bgar",
            ]
            await page.locator(".assistant").evaluate(
                """(wrapper, labels) => {
              for (const label of labels) {
                const button = document.createElement('button');
                button.type = 'button'; button.className = 'option'; button.textContent = label;
                wrapper.append(button);
              }
            }""",
                labels,
            )
            options = ReplyOptions(page, config, locate)
            await options.bind()
            published = options.publish(await options.collect(), {0: "assistant:a1"})[0]
            assert [choice.label for choice in published] == ["Plans", "Coverage"]
            inspected = (await options.inspect())[0]["options"]
            assert [entry["eligible"] for entry in inspected] == [True, True] + [False] * len(
                labels
            )
            await browser.close()

    asyncio.run(exercise())


def test_origin_change_during_inspection_is_rejected_after_reads() -> None:
    async def exercise() -> None:
        config, _ = channel_config()
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await local_page(browser, config)
            await page.set_content(SHELL.replace("</section>", REPLY + "</section>"))
            options = ReplyOptions(page, config, locate)
            await options.bind()
            options.publish(await options.collect(), {0: "assistant:a1"})
            identity = options._identity
            reads = 0

            async def navigate_after_last_read(node: ElementHandle) -> int:
                nonlocal reads
                index = await identity(node)
                reads += 1
                if reads == 3:
                    await page.goto("http://127.0.0.2:9999/changed-origin")
                return index

            with patch.object(options, "_identity", side_effect=navigate_after_last_read):
                with pytest.raises(ChannelError, match="origin"):
                    await options.inspect()
            assert not options.registry
            await browser.close()

    asyncio.run(exercise())
