# synthetic-cx

Open-source framework for simulating synthetic customers and evaluating real conversational customer journeys.

## Local setup

SyntheticCX uses [uv](https://docs.astral.sh/uv/) and supports Python 3.11 through
3.13. The committed lockfile is the authoritative development environment.

```bash
uv sync --locked --group dev
uv run playwright install chromium
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

The browser installation uses the Chromium binary matching the Playwright
version in `uv.lock`. On Linux, install Chromium's system packages as well:
`uv run playwright install --with-deps chromium`. Browser tests run in the
normal pytest suite without credentials; the live-model smoke test is opt-in.

## Controlled telecom web chat

Start the local fixture in another terminal:

```bash
uv run python examples/telecom/webchat_fixture/server.py
```

It prints a URL such as `http://127.0.0.1:<port>/?variant=success` and stops
on Ctrl-C. The port is selected by the OS. The port `8000` in
`examples/telecom/scenario.yaml` is an example value; substitute the printed
port when configuring this target. Integration tests start and stop their own
ephemeral fixture servers. The CLI `run` command and full evaluation/reporting
composition are planned for later phases; for now, use
`uv run pytest tests/integration/channel` to exercise the browser channel
against the fixture.

The five variants are `success`, `repeated_question`, `streamed_response`,
`missing_control`, and `silent_response`. For example:

```text
http://127.0.0.1:<port>/?variant=success&view=main
http://127.0.0.1:<port>/?variant=repeated_question&view=frame
http://127.0.0.1:<port>/?variant=streamed_response&view=main&echo=none
http://127.0.0.1:<port>/?variant=missing_control&view=frame
http://127.0.0.1:<port>/?variant=silent_response&view=main
```

`view=main` (default) uses the main document; `view=frame` uses the
pre-existing iframe with test ID `chat-frame`. Both contain hidden history
before the `open-chat` button is clicked, followed by a new greeting. The
example's `chat-input`, `send-message`, `customer-message`,
`assistant-message`, and `response-complete` test IDs and `data-message-id`
values form the fixture contract. The browser tests verify role, label, text,
placeholder, test-ID, and explicit CSS locators; XPath is unsupported. They
exercise click and Enter submission in both views. Completion is explicitly
configured as a quiet interval, streaming-indicator disappearance, or a new
completion signal; these strategies are verified against the controlled
fixture, not arbitrary external widgets.

`echo=match` is the default. `echo=none` omits the customer DOM echo;
`echo=normalized`, `echo=mismatch`, `echo=mutating`, and `echo=duplicate` cover
identity and failure cases. A successful send makes the submitted customer
text canonical. A mismatch can occur *after* the UI has submitted a message
without a successful receipt; the adapter never blindly retries it. The
`actor_sequence_content_hash` identity fallback is limited to fixture-shaped
local URLs and tested with `identity=none`; stable `data-message-id` values are
preferred.

Screenshots default off. Capturing a failure screenshot requires all three
explicit settings: `privacy.allow_screenshots`,
`privacy.confirm_screenshot_target_safe`, and `storage.store_screenshots`.
The controlled fixture test opts in deliberately. Image pixels are not
sanitized by text redaction, and no raw image bytes go into the event trace.

## Model and persona verification

The offline suite checks OpenAI and OpenAI-compatible provider translation,
selected credentials/endpoints, controlled capability probing, and a pinned
TinyTroupe persona action through an in-memory SDK transport.

The reference OpenAI `gpt-5-mini` persona interaction is opt-in:

```bash
export OPENAI_API_KEY='replace-with-selected-credential'
SYNTHETICCX_RUN_LIVE_SMOKE=1 uv run pytest -q tests/smoke/test_live_tinytroupe.py
```

It skips when the flag or credential is absent, probes chat and structured
output before constructing the persona, and checks typed owned results without
asserting generated wording. The selected persona path does not use embeddings.
Compatible endpoints require an explicit URL/credential and a successful
chat/structured-output preflight; arbitrary endpoints are not assumed compatible.

## Build and verify a distribution

```bash
uv build
python3 -m venv /tmp/syntheticcx-wheel-check
/tmp/syntheticcx-wheel-check/bin/pip install dist/*.whl
/tmp/syntheticcx-wheel-check/bin/synthetic-cx --help
```
