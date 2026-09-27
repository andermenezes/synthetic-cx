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

The canonical version-1 configuration is `examples/telecom/scenario.yaml`.
The fixture chooses an ephemeral loopback port, so the example launcher starts
it, writes a **temporary copy** of the YAML with the actual URL and selected
persona, then runs the installed CLI. From the repository root, supply your
explicitly selected model credential and run each journey separately:

```bash
export OPENAI_API_KEY='your-credential'
uv run python examples/telecom/run_demo.py --variant success --persona standard
uv run python examples/telecom/run_demo.py --variant repeated_question --persona impatient
```

The first command targets a sufficient billing explanation. The second uses
the *same scenario* with an impatient persona and a repeated account-number
request, which can trigger a recorded state transition and abandonment. Model
behavior is nondeterministic: inspect actual outcomes rather than assuming a
particular generated message or a fixed exit code. A selected OpenAI-compatible
endpoint can be configured instead of OpenAI when its explicit credential and
capabilities pass preflight; the reference demo uses `gpt-5-mini`. Persona and
semantic evaluation model calls may transmit configured synthetic conversation
content to the selected endpoint. They do not silently fall back to another
credential source.

For manual fixture diagnosis, start the server separately:

```bash
uv run python examples/telecom/webchat_fixture/server.py
```

It prints a URL such as `http://127.0.0.1:<port>/?variant=success` and stops on
Ctrl-C. To run `synthetic-cx run <scenario.yaml>` directly, make a temporary YAML
copy and substitute that port in `channel.url`; the committed canonical YAML's
port `8000` is illustrative. The CLI validates before browser activity, probes
selected model capabilities, prints an acknowledged run ID, and returns `0`
for pass, `1` for evaluation failure, `2` for configuration/capability failure,
`3` for infrastructure/evaluator error, or `130` for user interruption.

Each invocation that reaches acknowledged run creation writes a separate
`.runs/<run_id>/` directory. Configuration validation and model-capability
preflight failures occur earlier and leave no run artifacts. Inspect
`metadata.json` for lifecycle and outcome, `report.json` for the final summary
and findings, `conversation.json` for ordered messages, `evaluation.json` for
criterion results, and `trace.jsonl` for committed event IDs referenced by
findings. `config.yaml` is a redacted effective snapshot, not a replayable
credential source. Enabled local telemetry writes supplemental
`artifacts/telemetry/spans.jsonl` with run/trace correlation; `trace.jsonl`
remains the authoritative evaluation record. Remote OTLP requires the optional
`telemetry` extra and explicit `telemetry.remote_export`/`otlp_endpoint` values.
Prompt capture and screenshots default off; recorded omissions explain missing
captures. On an incomplete run, inspect available artifacts without assuming a
final report exists. If a control is missing, check `channel.*_locator` and
the fixture variant; if a response stalls, check `response_timeout_seconds`
and the configured completion strategy. A capability failure before browser
startup means the chosen endpoint did not satisfy the persona/evaluator path.

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
