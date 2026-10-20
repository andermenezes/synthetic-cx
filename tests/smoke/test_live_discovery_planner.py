"""Opt-in real planner against a synthetic loopback widget; never contacts SKY."""

import asyncio
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml

from syntheticcx.config.loader import load_config, load_discovery_draft
from syntheticcx.config.models import InterfaceChannelGuide
from syntheticcx.discovery.publish import publish_guide
from syntheticcx.discovery.run import discover_guide

PAGE = """<!doctype html><button id="fixture-launcher" type="button"
aria-controls="panel-root">Chat</button><script>
const launcher=document.querySelector('#fixture-launcher');
launcher.onclick=()=>{
 fetch('/event?kind=open');
 const panel=document.createElement('section'); panel.id='panel-root';
 panel.setAttribute('role','dialog'); panel.setAttribute('aria-label','Support');
 panel.innerHTML='<h2>Central de ajuda</h2>'+
   '<button type="button" aria-haspopup="menu">Configurações</button>'+
   '<div id="content">Carregando…</div>'; document.body.append(panel);
 setTimeout(()=>{
   document.querySelector('#content').innerHTML='<div id="messages">'+
     '<div class="row left"><p class="message">Olá! Como posso ajudar?</p></div></div>'+
     '<button type="button" aria-haspopup="menu">Anexar</button>'+
     '<input id="fixture-input" placeholder="Digite sua mensagem">'+
     '<button type="button">Enviar</button>';
   const input=document.querySelector('#fixture-input');
   function send() {
     if(!input.value.trim()) return;
     fetch('/event?kind=probe');
     const echo=document.createElement('div'); echo.className='row right';
     const text=document.createElement('p'); text.className='message'; text.textContent=input.value;
     echo.append(text); const badge=document.createElement('span'); badge.className='badge';
     badge.textContent='Entregue'; echo.append(badge);
     document.querySelector('#messages').append(echo); input.value='';
     setTimeout(()=>{ const reply=document.createElement('div'); reply.className='row left';
       reply.innerHTML='<p class="message">Posso ajudar com informações sobre '+
         '<strong>planos e cobertura</strong>.</p>';
       document.querySelector('#messages').append(reply); },250);
   }
   input.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();send();}};
   [...panel.querySelectorAll('button')].find(b=>b.textContent==='Enviar').onclick=send;
 },1500);
};</script>"""


def test_real_configured_planner_learns_unmarked_loopback_widget(tmp_path):
    if os.environ.get("SYNTHETICCX_RUN_DISCOVERY_PLANNER_SMOKE") != "1":
        pytest.skip(
            "set SYNTHETICCX_RUN_DISCOVERY_PLANNER_SMOKE=1 for real-model loopback discovery"
        )
    selected_path = Path(os.environ.get("SYNTHETICCX_PLANNER_SCENARIO", "sky_scenario.yaml"))
    selected = load_discovery_draft(selected_path)
    if not os.environ.get(selected.model.credential_environment_variable):
        pytest.fail("selected planner credential is required for the explicitly opted-in test")
    document = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())
    document.pop("channel")
    document["model"] = selected.model.model_dump(mode="json")
    document["execution"] = selected.execution.model_dump(mode="json")
    document["persona"]["communication"]["language"] = "pt-BR"
    source = tmp_path / "synthetic-draft.yaml"
    source.write_text(yaml.safe_dump(document, allow_unicode=True))
    events, loads = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            target = urlsplit(self.path)
            if target.path == "/event":
                events.append(parse_qs(target.query)["kind"][0])
                self.send_response(204)
                self.end_headers()
                return
            loads.append(target.path)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE.encode())

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    progress = []

    def report(stage):
        progress.append(stage)
        print("[real planner] " + stage, flush=True)

    try:
        guide = asyncio.run(
            discover_guide(
                source,
                f"http://127.0.0.1:{server.server_port}/",
                chat_hint="id=fixture-launcher",
                max_discovery_attempts=5,
                language="pt-BR",
                on_progress=report,
            )
        )
        completed = publish_guide(source, tmp_path / "guide.yaml", guide, in_place=False)
        assert completed.exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert isinstance(guide, InterfaceChannelGuide)
    assert load_config(completed).channel.type == "webchat_playwright_dynamic"
    assert not hasattr(guide.channel, "verification_recipe")
    assert events.count("probe") % 2 == 0
    steps = events.count("probe") // 2
    assert guide.channel.reader_evidence == "observed_exchange"
    assert 1 <= steps <= 3
    assert loads == ["/", "/"]
    assert events == (["open"] + ["probe"] * steps) * 2
    assert "observing witnessed exchange for reader learning; no probe resent" in progress
    print(
        f"Verified {steps}+{steps} real-planner learning/replay actions "
        f"with {selected.model.model}."
    )
