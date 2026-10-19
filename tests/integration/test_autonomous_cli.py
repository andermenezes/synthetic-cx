"""Whole native learning/replay/publication path against an authorized local menu."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml
from typer.testing import CliRunner

from syntheticcx.cli import app
from syntheticcx.config.loader import load_config
from syntheticcx.config.models import InterfaceChannelConfig
from syntheticcx.discovery.contracts import DiscoveryDecision

PAGE = """<!doctype html><html><body>
<section id="chat" role="dialog" aria-label="Support chat">
 <div id="transcript" role="log" aria-label="Chat messages">
  <div class="assistant" data-testid="assistant-message" data-actor="assistant">
   <span>Choose a topic</span><button type="button">Help</button>
  </div>
 </div>
</section>
<script>
let stage=0;
const delayed = false, replaceComposer = false;
const root=document.querySelector('#chat'), log=document.querySelector('#transcript');
function reply(text,label) {
 let message=document.createElement('div');
 message.className='assistant'; message.dataset.testid='assistant-message';
 message.dataset.actor='assistant'; message.innerHTML='<span>'+text+'</span>';
 if(label) {let b=document.createElement('button'); b.type='button';
 b.textContent=label; message.append(b);}
 log.append(message);
}
function composer(id) {
  let input=document.createElement('textarea'); input.id=id;
  input.setAttribute('aria-label','Chat message'); root.append(input);
  input.addEventListener('keydown',e=>{if(e.key==='Enter') {
   e.preventDefault();
   if(replaceComposer) input.remove(); else input.value='';
   fetch('/evidence?event=probe');
   setTimeout(()=>{
    reply('Here is information about our plans');
    if(replaceComposer) composer('next-message-input');
   }, delayed ? 180 : 0);
  }});
}
root.addEventListener('click',e=>{
 if(e.target.tagName!=='BUTTON') return;
 fetch('/evidence?event='+encodeURIComponent(e.target.textContent));
 e.target.disabled=true; e.target.hidden=true;
 setTimeout(()=>{
  if(stage++===0) reply('Select information','Plans');
  else {reply('Ask about plans'); composer('message-input');}
 }, delayed ? 180 : 0);
});
</script></body></html>"""


@pytest.mark.parametrize(
    "variant",
    [
        "immediate",
        "delayed",
        "composer_replacement",
        "delayed_root",
        "unproven_root",
        "premature_finish",
        "early_stop",
        "unmarked_exchange",
        "unmarked_routing_menu",
    ],
)
def test_native_menu_learning_exact_replay_and_publication(tmp_path, monkeypatch, variant):
    from syntheticcx.discovery import learn as learning_module
    from syntheticcx.discovery import run as discovery_run

    samples = []
    original_learn = learning_module.learn

    async def capture_sample(*args, **kwargs):
        learned = await original_learn(*args, **kwargs)
        samples.append(learned.config)
        return learned

    monkeypatch.setattr(learning_module, "learn", capture_sample)

    events = []
    page_loads = []
    markup = PAGE.replace(
        "const delayed = false, replaceComposer = false;",
        "const delayed = "
        + ("false" if variant == "immediate" else "true")
        + ", replaceComposer = "
        + ("true" if variant == "composer_replacement" else "false")
        + ";",
    )
    if variant == "unmarked_exchange":
        markup = """<!doctype html><section id="chat" role="dialog" aria-label="Support chat">
        <h2>Support</h2><button type="button" aria-haspopup="menu">Settings</button>
        <div id="messages"><div class="left">
          <p class="content">Hello, how can I help?</p></div></div>
        <input id="message-input" aria-label="Chat message">
        </section><script>let count=0;
        document.querySelector('input').onkeydown = event => {
          if(event.key !== 'Enter') return; event.preventDefault();
          const log=document.querySelector('#messages');
          const echo=document.createElement('div'); echo.className='right';
          const p=document.createElement('p'); p.className='content';
          p.textContent=event.target.value; echo.append(p); log.append(echo); event.target.value='';
          count++; fetch('/evidence?event=probe');
          setTimeout(()=>{const row=document.createElement('div'); row.className='left';
            const text=document.createElement('p'); text.className='content';
            text.innerHTML='Information about <strong>coverage</strong>.';
            row.append(text); log.append(row);}, 150);
        };</script>"""
    if variant == "unmarked_routing_menu":
        markup = """<!doctype html><section id="chat" role="dialog" aria-label="Support chat">
        <h2>Support</h2><button type="button" aria-haspopup="menu">Settings</button>
        <div id="messages"><div class="row start"><div class="bubble">
          <p class="content">Olá! Como posso te ajudar?</p><time>09:00</time>
        </div></div></div>
        </section><script>
        const log=document.querySelector('#messages'), root=document.querySelector('#chat');
        function row(side, html){
          const r=document.createElement('div'); r.className='row '+side;
          const b=document.createElement('div'); b.className='bubble';
          b.innerHTML=html+'<time>10:14</time>'; r.append(b); log.append(r); return r;
        }
        function addComposer(){
          if(document.querySelector('#message-input')) return;
          const input=document.createElement('input'); input.id='message-input';
          input.setAttribute('aria-label','Chat message'); root.append(input);
          input.addEventListener('keydown', event=>{
            if(event.key!=='Enter') return; event.preventDefault();
            row('end','<p class="content"></p>').querySelector('p').textContent =
              event.target.value;
            input.remove();
            fetch('/evidence?event=probe');
            setTimeout(()=>{ row('start',
              '<p class="content">Por favor escolha uma opção abaixo para direcionar ' +
              'seu atendimento:</p>' +
              '<div class="opts"><button>Contratei pelo site</button>' +
              '<button>Tenho inclusa no plano</button><button>Ainda não tenho</button>' +
              '<button>Copy</button><button type="submit">Enviar arquivo</button></div>');
            }, 150);
          });
        }
        addComposer();
        root.addEventListener('click', event=>{
          const target=event.target;
          if(target.tagName!=='BUTTON' || !target.closest('.opts')) return;
          fetch('/evidence?event='+encodeURIComponent(target.textContent));
          target.disabled=true;
          row('end','<p class="content"></p>').querySelector('p').textContent =
            target.textContent;
          setTimeout(()=>{
            row('start','<p class="content">Entendi, vou te ajudar com isso.</p>');
            addComposer();
          },150);
        });
        </script>"""
    if variant in {"delayed_root", "unproven_root"}:
        body, script = markup.split("<script>", 1)
        body = body.split("<body>", 1)[1]
        script = script.split("</script>", 1)[0]
        if variant == "unproven_root":
            body = body.replace('role="dialog"', 'role="region"')
        relationship = ' aria-controls="chat"' if variant == "delayed_root" else ""
        markup = (
            '<button id="launcher" type="button"' + relationship + ">Chat</button>"
            '<section id="faq">Unrelated FAQ region</section><script>'
            "document.querySelector('#launcher').addEventListener('click', () => {"
            "fetch('/evidence?event=opener'); setTimeout(() => {"
            "document.body.insertAdjacentHTML('beforeend', " + json.dumps(body) + ");"
            "const script=document.createElement('script'); script.textContent="
            + json.dumps(script)
            + "; document.body.append(script); }, 600);});</script>"
        )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            target = urlsplit(self.path)
            if target.path == "/evidence":
                events.append(parse_qs(target.query)["event"][0])
                self.send_response(204)
                self.end_headers()
                return
            if target.path == "/":
                page_loads.append(target.path)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(markup.encode())

        def log_message(self, *_args):
            pass

    class Planner:
        sequence = [
            "bind_scope",
            "bind_reader",
            "bind_state",
            "select_option",
            "bind_state",
            "select_option",
            "bind_state",
            "submit_probe",
            "bind_state",
            "finish",
        ]
        if variant == "premature_finish":
            sequence.insert(3, "finish")
        elif variant == "early_stop":
            sequence = ["bind_scope", "stop"]
        elif variant == "unmarked_exchange":
            sequence = [
                "bind_scope",
                "bind_state",
                "submit_probe",
                "bind_reader",
                "bind_state",
                "submit_probe",
                "bind_state",
                "submit_probe",
                "bind_state",
                "finish",
            ]
        elif variant == "unmarked_routing_menu":
            sequence = [
                "bind_scope",
                "bind_state",
                "submit_probe",
                "bind_reader",
                "bind_state",
                "select_option",
                "bind_state",
                "finish",
            ]
        actions = iter(sequence)

        def __init__(self):
            self.requests = []
            self.close = AsyncMock()

        async def generate(self, request):
            self.requests.append(request)
            inventory = json.loads(request.messages[-1].content)["inventory"]
            action = next(self.actions)
            candidates = inventory["catalog"]["candidates"]

            def candidate(kind):
                return next((item["id"] for item in candidates if item["kind"] == kind), None)

            fields = dict.fromkeys(DiscoveryDecision.model_fields)
            fields.update(action=action, revision=inventory["revision"])
            if action == "stop":
                fields["stop_reason"] = "unsupported_interface"
            if action == "bind_scope":
                fields["target_id"] = candidate("scope")
            elif action == "bind_reader":
                fields["assistant_id"] = candidate("assistant")
                if variant in {"unmarked_exchange", "unmarked_routing_menu"}:
                    fields["customer_id"] = candidate("customer")
            elif action == "bind_state":
                fields["input_id"] = candidate("input")
                if fields["input_id"] is not None:
                    fields["send_mode"] = "press_enter"
                else:
                    fields["option_id"] = candidate("option")
            elif action == "select_option":
                fields["option_id"] = inventory["available_options"][0]["id"]
            elif action == "submit_probe":
                fields["target_id"] = inventory["composer_id"]
                fields["probe_id"] = "first"
            return SimpleNamespace(content=json.dumps(fields))

    planner = Planner()
    monkeypatch.setattr(discovery_run, "OpenAIProvider", lambda *_args, **_kwargs: planner)
    monkeypatch.setattr(discovery_run, "_probe_role", AsyncMock())
    monkeypatch.setenv("OPENAI_API_KEY", "LOCAL-ONLY-TEST-CREDENTIAL")
    document = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    document.pop("channel")
    document["execution"].update(
        journey_timeout_seconds=60,
        browser_action_timeout_seconds=5,
        response_timeout_seconds=3,
        model_timeout_seconds=3,
    )
    source = tmp_path / "draft.yaml"
    source.write_text(yaml.safe_dump(document))
    output = tmp_path / "guide.yaml"
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    arguments = [
        "discover",
        str(source),
        "--url",
        f"http://127.0.0.1:{server.server_port}/",
        "--output",
        str(output),
        "--verbose",
    ]
    if variant in {"delayed_root", "unproven_root"}:
        arguments.extend(["--chat-hint", "id=launcher"])
    try:
        result = CliRunner().invoke(
            app,
            arguments,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    if variant == "unproven_root":
        assert result.exit_code == 3, result.output
        assert "owned interaction controls did not become ready" in result.output
        assert events == ["opener"]
        assert page_loads == ["/"]
        assert not planner.requests
        assert not output.exists()
        assert not (tmp_path / "draft.completed.yaml").exists()
        planner.close.assert_awaited_once()
        return
    if variant == "early_stop":
        assert result.exit_code == 3, result.output
        assert "learning stopped without verification: unsupported_interface" in result.output
        assert "discovery readiness: state=unbound; reader_bound=False" in result.output
        assert "assistant_candidates=1" in result.output
        assert not events
        assert page_loads == ["/"]
        assert len(planner.requests) == 2
        assert not output.exists()
        assert not (tmp_path / "draft.completed.yaml").exists()
        planner.close.assert_awaited_once()
        return
    assert result.exit_code == 0, result.output
    loaded = load_config(tmp_path / "draft.completed.yaml")
    assert isinstance(loaded.channel, InterfaceChannelConfig)
    published = yaml.safe_load(output.read_text())
    assert "version" not in published
    assert published["channel"]["type"] == "webchat_playwright_dynamic"
    assert not {"states", "transitions", "verification_recipe", "initial_state_id"} & set(
        published["channel"]
    )
    assert len(samples) == 1
    sample = samples[0]
    if variant == "unmarked_exchange":
        assert loaded.channel.reader_evidence == "observed_exchange"
        assert loaded.channel.customer_message_locator is not None
        assert len(sample.verification_recipe) == 3
        assert len(sample.states) == 1
        assert len(planner.requests) == 10
        assert events == ["probe"] * 6
        assert page_loads == ["/", "/"]
        assert "observing witnessed exchange for reader learning" in result.output
        planner.close.assert_awaited_once()

        async def runtime():
            from playwright.async_api import async_playwright

            from syntheticcx.adapters.channel.dynamic import DynamicChatBroker

            async with async_playwright() as pw:
                browser = await pw.chromium.launch()
                page = await browser.new_page()
                await page.route(
                    "**/*", lambda route: route.fulfill(body=markup, content_type="text/html")
                )
                await page.goto(str(loaded.channel.url))
                broker = DynamicChatBroker(
                    config=loaded.channel, limits=loaded.execution, page=page
                )
                try:
                    await broker.start()
                    await broker.observe_initial()
                    assert await page.evaluate("count") == 0
                    for text in ("Please explain coverage", "Please explain service hours"):
                        await broker.send(text)
                        response = await broker.receive()
                        assert response.messages[-1].content == "Information about coverage."
                    assert await page.evaluate("count") == 2
                finally:
                    await broker.close()
                    await browser.close()

        asyncio.run(runtime())
        return
    if variant == "unmarked_routing_menu":
        assert loaded.channel.reader_evidence == "observed_exchange"
        assert loaded.channel.customer_message_locator is not None
        assert len(sample.verification_recipe) == 2
        assert len(sample.states) == 2
        assert [step.action_kind for step in sample.transitions] == [
            "probe",
            "select_option",
        ]
        menu = next(state for state in sample.states if state.kind == "menu")
        assert menu.option_labels == [
            "Contratei pelo site",
            "Tenho inclusa no plano",
            "Ainda não tenho",
        ]
        assert len(planner.requests) == 8
        assert events == ["probe", "Contratei pelo site"] * 2
        assert page_loads == ["/", "/"]
        assert "LOCAL-ONLY-TEST-CREDENTIAL" not in output.read_text()
        planner.close.assert_awaited_once()

        async def runtime():
            from playwright.async_api import async_playwright

            from syntheticcx.adapters.channel.dynamic import DynamicChatBroker

            async with async_playwright() as pw:
                browser = await pw.chromium.launch()
                page = await browser.new_page()
                await page.route(
                    "**/*",
                    lambda route: route.fulfill(
                        body=markup, content_type="text/html; charset=utf-8"
                    ),
                )
                await page.goto(str(loaded.channel.url))
                broker = DynamicChatBroker(
                    config=loaded.channel, limits=loaded.execution, page=page
                )
                try:
                    await broker.start()
                    greeting = await broker.observe_initial()
                    assert greeting is not None
                    assert greeting.messages[-1].content == "Olá! Como posso te ajudar?"
                    for text in ("Please explain coverage", "Please explain service hours"):
                        await broker.send(text)
                        response = await broker.receive()
                        message = response.messages[-1]
                        assert message.content == (
                            "Por favor escolha uma opção abaixo para direcionar seu atendimento:"
                        )
                        assert [option.label for option in message.options] == [
                            "Contratei pelo site",
                            "Tenho inclusa no plano",
                            "Ainda não tenho",
                        ]
                        await broker.select_option(message.options[0].id)
                        routed = await broker.receive()
                        assert routed.messages[-1].content == "Entendi, vou te ajudar com isso."
                finally:
                    await broker.close()
                    await browser.close()

        asyncio.run(runtime())
        return
    assert len(sample.verification_recipe) == 3
    last = sample.transitions[-1]
    if variant == "composer_replacement":
        assert len(sample.states) == 4
        assert last.from_state != last.to_state
        states = {state.id: state for state in sample.states}
        assert states[last.from_state].input_locator.value == "#message-input"
        assert states[last.to_state].input_locator.value == "#next-message-input"
    else:
        assert len(sample.states) == 3
        assert last.from_state == last.to_state
    assert [step.action_kind for step in sample.transitions] == [
        "select_option",
        "select_option",
        "probe",
    ]
    assert len(planner.requests) == (11 if variant == "premature_finish" else 10)
    if variant == "premature_finish":
        assert "discovery decision rejected: premature_finish; attempt 1/3" in result.output
    assert page_loads == ["/", "/"]
    expected = ["Help", "Plans", "probe"]
    if variant == "delayed_root":
        expected.insert(0, "opener")
    assert events == expected * 2
    assert "LOCAL-ONLY-TEST-CREDENTIAL" not in output.read_text()
    planner.close.assert_awaited_once()
