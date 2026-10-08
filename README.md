# SyntheticCX

> **Simulate customers. Test journeys. Understand failures.**

SyntheticCX is an open-source Python framework for exercising and evaluating a real conversational customer journey from a synthetic customer's point of view. Rather than asserting an exact chatbot reply, it combines a **scenario** (the customer's goal), a **persona** (how that customer behaves), a real web-chat interface, and evidence-based evaluation.

The MVP is designed for controlled, text-based web chats. It opens the configured site with Playwright, lets a TinyTroupe-backed customer persona interact through the visible chat controls, records the journey, and produces deterministic and semantic findings with supporting evidence.

**Current release:** `0.1.0` (pre-alpha) • **Python:** 3.11–3.13 • **License:** Apache-2.0

---

## What it is for

Use SyntheticCX to answer questions such as:

- Can a customer with a given goal complete a web-chat journey?
- Does a repeated request for information increase frustration or cause abandonment?
- Did the journey time out, exceed a turn limit, or fail to resolve the stated goal?
- What messages, state changes, browser actions, and evaluator findings support the result?

The framework tests the journey as delivered through the customer-facing channel, so failures can occur in the chat UI, browser integration, conversation, or evaluation—not only in the underlying model.

SyntheticCX results describe the configured synthetic simulation. They are **not** a measurement of a real customer population.

---

## Architecture

![SyntheticCX architecture](diagrams/architecture.svg)

*SyntheticCX separates its owned domain and runtime from replaceable external implementations.*

SyntheticCX owns its domain contracts—configuration, scenarios, personas, messages, traces, evaluation results, and protocols. Third-party libraries are contained behind adapters:

| Concern | MVP implementation |
| --- | --- |
| CLI | Typer (`synthetic-cx`) |
| Configuration and schemas | YAML + Pydantic v2 |
| Orchestration | LangGraph |
| Synthetic persona | TinyTroupe adapter |
| Customer channel | Async Playwright web-chat adapter |
| Models | OpenAI and explicitly configured OpenAI-compatible endpoints |
| Evaluation | Deterministic evaluators and an LLM-based semantic goal evaluator |
| Evidence and reports | Local filesystem, JSON, JSONL |
| Observability | Local OpenTelemetry spans and structured application logging; remote OTLP is opt-in |

The domain is deliberately independent of these libraries. Its extension boundaries are `ChannelAdapter`, `PersonaSimulator`, `Evaluator`, `LLMProvider`, and `StorageBackend`.

---

## Get started

### 1. Install the development environment

Clone the repository and run the following from its root:

```bash
uv sync --locked --group dev
uv run playwright install chromium
```

On Linux, install browser system dependencies as well:

```bash
uv run playwright install --with-deps chromium
```

SyntheticCX uses the committed `uv.lock` as its development-environment authority. The browser install must match the locked Playwright version.

### 2. Set the selected model credential

The reference configuration uses OpenAI and `gpt-5.4-mini`:

```bash
export OPENAI_API_KEY='your-credential'
```

Credentials are read from the environment, not from YAML. The framework redacts resolved credentials from persisted configuration, reports, logs, telemetry, and prompt captures.

### 3. Run the controlled telecom example

The included fixture provides a known web chat so you can verify the full browser path safely:

```bash
uv run python examples/telecom/run_demo.py --variant success --persona standard
```

For an explained failure based on an impatient customer receiving a repeated account-number request:

```bash
uv run python examples/telecom/run_demo.py --variant repeated_question --persona impatient
```

The launcher starts the fixture on an ephemeral loopback port, creates a temporary configuration with that port and selected persona, then invokes the CLI. It does not modify the canonical example.

> Model output is nondeterministic. Inspect the generated outcome and evidence; do not assume a fixed wording or verdict.

### 4. Inspect the run

Each run that reaches acknowledged creation receives its own directory under `.runs/`:

```text
.runs/
└── run-<id>/
    ├── metadata.json
    ├── config.yaml
    ├── conversation.json
    ├── trace.jsonl
    ├── evaluation.json
    ├── report.json
    └── artifacts/
        └── telemetry/spans.jsonl  # when local telemetry is enabled
```

Start with `report.json` for the summary and findings, then use `trace.jsonl` for the append-only evidence referenced by each finding. `config.yaml` is an effective **redacted** snapshot and cannot be used as a credential source.

### Customer impressions and optional Markdown reports

Add observation topics to request first-person feedback from the **same simulated customer** after the conversation. Topics are separate from `success_criteria` and are only shown during reflection—not while the persona is chatting.

```yaml
scenario:
  # Keep the existing scenario fields and success_criteria.
  observation_topics:
    - id: clarity
      description: Were the options and prices understandable?
    - id: usability
      description: What felt easy, confusing, or frustrating?

reporting:
  language: pt-BR  # Optional: en-US or pt-BR; defaults to effective persona language.
  markdown:
    enabled: true  # Default: false.
```

Topics independently enable one bounded reflection even when Markdown is off. Configure at most 10 unique IDs (`A–Z`, `a–z`, digits, `_`, `-`; up to 80 characters), each with a nonblank description up to 2,000 characters. Empty topics make no additional persona model call. Enabling Markdown without topics produces a report with customer feedback marked **not requested**.

Reflection happens after the channel closes and before the persona closes. It retains that customer's history, characteristics, writing voice, and final owned emotional state; it never sends another chat message or adds a turn. The request includes only customer-visible experience and topics, not hidden criteria, evaluator explanations, or verdicts. Feedback and its model records are excluded from evaluator input. These are **subjective impressions, not formal usability measurements**, and never determine pass/fail.

Healthy personas can reflect on partial experiences, including abandonment, turn limits, or channel failures. Cancellation, unavailable original sessions, missing experience, unsupported adapters, timeout, and invalid output are explicitly recorded as unavailable—never as dissatisfaction. There is no replacement persona or reflection retry. Reflection uses `execution.model_timeout_seconds` separately from the journey deadline; model/output failures are supplemental, while evidence/report persistence failures retain the normal storage-failure behavior. Each topic distinguishes observed from not observed and validates references against recorded conversation messages.

When enabled, `<storage.run_root>/<run_id>/report.md` uses fixed sections: context, execution summary, customer impressions, formal test conclusions, source conversation evidence, and limitations/artifacts. Markdown works independently of `storage.materialize_report` (which still controls `report.json`) and conversation/evaluation JSON flags. The template and feedback use `reporting.language`, falling back to the effective persona language including `run --language` overrides. Original topic wording, conversation quotations, and evaluator explanations are preserved and labeled as source material; they are not translated. No report-generation model is called. Untrusted content is escaped, and only redacted committed evidence is rendered.

Requested feedback is always in `trace.jsonl` and appears in `report.json` when enabled. New reports separately record `conversation_duration_seconds` (before cleanup/reflection/evaluation) and the existing full-trace `duration_seconds`; older traces have no separate conversation-duration measurement. Metadata is published only after enabled report artifacts are written.

---

## Run lifecycle

![SyntheticCX run lifecycle](diagrams/run-lifecycle.svg)

*The run preserves evidence across each meaningful stage and keeps termination separate from the evaluated outcome.*

A run validates configuration and checks model capabilities before it creates browser or run resources. During the conversation, the runtime persists messages, persona transitions, model interaction metadata, and errors as typed events. Conversation termination and final outcome are separate: a persona believing that its goal is complete is only a stopping signal; evaluators decide whether the run passes.

---

## Use the CLI

```bash
synthetic-cx run path/to/scenario.yaml
```

or from a checkout:

```bash
uv run synthetic-cx run path/to/scenario.yaml
```

Add `--headed` to `run` to display its native Chromium browser instead of running it headless. This requires a desktop display and does not change the scenario or its channel guide. Use `run --language pt-BR` (or `en-US`) to override the customer persona's spoken language for that run; the effective configuration artifact records the selection. The model is instructed to keep customer messages in that language, but model output is not guaranteed to comply perfectly.

Useful commands:

```bash
synthetic-cx --help
synthetic-cx --version
python -m syntheticcx --help
```

### Exit codes

| Code | Meaning |
| ---: | --- |
| `0` | All required evaluation criteria passed |
| `1` | The journey completed but one or more required criteria failed |
| `2` | Configuration or capability-preflight error; no browser run is created |
| `3` | Browser, model, evaluator, storage, or other infrastructure error |
| `130` | User interruption after bounded cleanup |

An infrastructure error takes precedence over evaluation findings. A journey timeout is an infrastructure error (`3`), even if an evaluator had already passed.

---

## Discover a web-chat channel

`synthetic-cx discover` uses bounded, model-planned native Playwright exploration to learn an interface guide. One browser retains state while discovering controls and completing interactions; **one separate fresh browser replays the same successful interactions** before publication. Ordinary `synthetic-cx run` resolves current safe controls natively inside the verified conversation root, without a browser-planning model. The private discovery sample graph is replay evidence, not runtime authority.

Install the Python `discovery` extra and Chromium. The CLI opens native Playwright directly; it does not launch a Playwright MCP server or require Node.js/`npx`:

```bash
uv sync --locked --group dev --extra discovery
uv run playwright install chromium
```

Prepare a version-1 scenario draft with its normal `scenario`, `persona`, `execution`, and `model` sections, but without `channel`. For a safe first test, start the controlled telecom fixture, copy `examples/telecom/scenario.yaml` to a draft, and remove its inline `channel` section. Use the fixture URL and ephemeral port it prints:

In one terminal, keep the fixture running:

```bash
uv run python - <<'PY'
from examples.telecom.webchat_fixture.server import FixtureServer

with FixtureServer() as fixture:
    print(f'Fixture URL: {fixture.base_url}', flush=True)
    input('Press Enter to stop the fixture...')
PY
```

In another terminal, prepare the draft and discover its channel (replace `<port>` with the printed port):

```bash
uv run python - <<'PY'
from pathlib import Path
import yaml

source = yaml.safe_load(Path('examples/telecom/scenario.yaml').read_text())
source.pop('channel')
Path('billing-draft.yaml').write_text(yaml.safe_dump(source, sort_keys=False))
PY
```

```bash
export OPENAI_API_KEY='your-credential'
uv run --extra discovery synthetic-cx discover billing-draft.yaml \
  --url 'http://127.0.0.1:<port>/?variant=success&view=frame' \
  --output billing-channel.yaml --verbose
# Add --headed to show native learning and fresh recipe-replay browsers.
# Add --chat-hint 'Talk to an agent' to guide label/tooltip discovery.
# For a Portuguese chat, add --language pt-BR to localize owned text probes.
uv run synthetic-cx run billing-draft.completed.yaml
# If verification required headed mode, use: uv run synthetic-cx run billing-draft.completed.yaml --headed
```

Credentials are read from the selected model's explicitly named environment variable. An optional discovery-only model can be supplied with `--discovery-model-config path/to/model.yaml`. Text comes from owned localized probes or an explicit `--probe-message`, not arbitrary model-written requests. When supplied, the explicit message is the only available text probe; the model can still choose safe menu options instead.

Discovery allows **at most three conversational actions while learning and replays those same one-to-three actions once in a fresh session**. An action is one text submission or one guarded informational option selection. Observations, waits and model corrections are separately bounded. Uncertain submissions are never retried, replaced with another option, or followed by a text fallback. Use only an authorized synthetic target.

The agent learns the interaction in this order:

1. Navigate, activate a safe launcher if needed, and wait for usable chat controls inside a proven conversation root.
2. Bind observed controls and readers. A composer without marked readers can submit its first owned probe to learn them from an actual exchange.
3. Observe a completed reply and explicitly bind the resulting composer, menu, mixed or terminal state. Continue exploring within the action budget, or finish after a complete verified interaction.
4. Close the learning browser and replay the exact recorded actions in one fresh browser, using native checks without model planning.
5. Publish the version-1 `InterfaceChannelGuide` and completed scenario only after replay and privacy validation succeed. The public guide contains fixed interface/readers and content-free validation evidence, not sample states, edges or a replay recipe.

Learning permits at most 24 model cycles, 24 observations and four explicit waits, plus one launcher activation per session. These limits share the configured journey deadline and operation timeouts; replay and cleanup retain reserved time. `--max-discovery-attempts` does not enlarge these limits.

The browser generates bounded, revision-scoped control/reader candidates. The model selects opaque IDs; owned code compiles locators, proves conversation ownership and executes actions against actual nodes. It cannot provide executable JavaScript, arbitrary CSS or unrestricted browser commands. Redacted bounded observations remain in memory, not in the guide or progress output. Origin checks, private handles, latest-reply option ownership and one-use capabilities remain enforced.

Supported languages are `en-US` and `pt-BR`. `--language` selects owned probe wording and is saved in the completed scenario. Without it, discovery uses the persona language, otherwise `en-US`. Explicit probe text is not translated and stays in memory for exact replay; neither it nor the private sample recipe is published in the guide. Normal runs retain the persona language unless overridden with `run --language`.

Discovery and normal runs are headless by default. `discover --headed` shows learning and fresh replay in separate browsers; no cookies/storage are shared. Headed success does not prove headless compatibility; use `run --headed` when needed. A `--chat-hint` helps identify a launcher but never bypasses safe-control checks.

`--max-discovery-attempts` bounds invalid-decision corrections, not browser retries. This includes premature `finish` requests: a completed interaction, learned reader and bound resulting state are required before finishing. The planner receives next actions and concrete current-ID choices supported by native evidence; native guards still authorize execution. Already bound controls are offered again only when learning a resulting state. Opening has its own one-activation-per-session bound; planning, observation and waiting share the overall discovery deadline. A claim of missing controls/readers is corrected when current evidence offers a composer or reader-learning path. Unsafe or ambiguous evidence still permits immediate stop, and rejected decisions never force a probe. The model must choose observed, safe controls and cannot infer missing authorization from a hint.

Add `--verbose` for safe stage progress and owned failure categories. Inventory, prompts, probe contents, credentials and raw browser exceptions are not printed. Origin is checked around conversational actions. Discovery does not authenticate, complete business goals, or repair guides automatically during ordinary runs.

### Discovery update status (2026-10-05)

The approved interface-bootstrap/native-runtime replacement is implemented; **local implementation, regression and review gates are complete; overall user/live acceptance remains open**. Runtime follows current eligible menus and safe replacement composers independently of the discovery sample, while retaining the pinned root/frame/origin, fixed readers, transaction guards and no-retry rules. Current typing availability and opaque option IDs are supplied to the same customer persona. Missing controls wait within existing deadlines and then fail with safe reason codes, never imply goal completion.

The final local suite passed **1,303 tests, with 4 opt-in skips and 3 dependency warnings**, in **405.99 seconds**, including parity for all nine schemas. Ruff lint, CI-scope mypy (94 files), formatting (175 files) and diff checks are clean. Functional, conventions and simplicity reviewers examined the replacement; disabled-until-filled send handling, handle accumulation, no-greeting fallback and completion-marker handle-leak findings were repaired with regression tests. This final suite also includes cancellation-safe handle cleanup with parameterized tests and the ownership origin-count race classification repair.

The user reported that prior live SKY discovery succeeded but customer runtime failed. That report motivated this redesign; it is not validation of the new implementation. **No live-target or configured-model validation of this redesign has been performed.** Earlier 1,213-test and synthetic `gpt-5.4-mini` loopback checkpoints belong to the superseded stateful implementation, not the replacement's final results.

The approved replacement supersedes the prior no-unvisited-branches/state-graph runtime policy. Remaining gates are overall user/live acceptance, authorized live discovery/replay/runtime validation, configured-model loopback validation and hosted Linux/Python 3.13 evidence. Passing local regressions does not establish live compatibility.

Internal documentation and historical checkpoints are indexed in [DOCUMENTATION_INDEX.md](/Users/andermenezes/Documents/SyntheticCX/DOCUMENTATION_INDEX.md). The current evidence and completion checklist are in [autonomous_discovery_progress.md](/Users/andermenezes/Documents/SyntheticCX/autonomous_discovery_progress.md).

### Discovery command options

| Option | Purpose |
| --- | --- |
| `--url URL` | Required page URL to explore. |
| `--output PATH` | Required destination for the verified channel guide; existing files are not overwritten. |
| `--headed` | Show native learning and fresh replay browsers; requires a desktop/display. |
| `--chat-hint TEXT` | Optional observed launcher label or exact `id=<DOM-ID>` hint; 1–256 characters. Does not bypass safe-control checks. |
| `--language en-US\|pt-BR` | Owned probe language, persisted in the completed scenario. Defaults to the persona language, otherwise `en-US`. |
| `--max-discovery-attempts N` | Up to 1–5 corrections per invalid model decision (default 3), never conversational retries. |
| `--verbose` | Show safe stage progress, readiness counts, allowlisted decision-rejection reasons and SyntheticCX error category/operation. Raw page content, prompts, probe text, credentials and raw exceptions are not printed. |
| `--probe-message TEXT` | Explicit owned text override, kept in memory for exact replay and never translated. |
| `--discovery-model-config PATH` | Optional separate `openai` or `openai_compatible` discovery model configuration, using its explicitly named credential environment variable. |

Example: append `--headed --chat-hint 'Talk to an agent' --verbose` to watch exploration and identify an observed launcher label.

On an interactive terminal, choose to save a completed copy or edit the draft in place; in-place editing creates a `.bak` backup. Non-interactive execution always creates `<draft-name>.completed.yaml`. The completed scenario refers to the guide by a relative `channel_guide` path. Guide loading validates the guide before model or browser activity, and run artifacts include the guide's source hash. Existing output paths are not overwritten; failed or ambiguous verification publishes no guide.

Discovery targets inspectable text/menu chats with independently verifiable ownership, actors, identity and completion. Ownership can come from a named chat dialog, marked embed, observed opener/root relationship, or a structurally identified chat panel after opening a launcher. The `chat_panel` proof requires the smallest rendered container containing one safe composer, a named chat control, and a named send control; it does not rely on a brand, domain, or hard-coded widget selector. The root's actual DOM identity and origin stay pinned while composer availability may change. Actors need native marker evidence or the witnessed-exchange evidence described below; the model cannot invent attribution for arbitrary regions or ambiguous messages. Inventories are bounded to 2,000 candidates/nodes within inspected scopes and expose at most 160 candidates with bounded descriptions. Pre-ownership inspection targets chat controls and root evidence, not every element of a large landing page. Insufficient evidence still fails safely. If the interface changes, explicitly rediscover and review the guide.

#### If discovery stops before finding chat

Model/provider failures, rejected decisions, missing ownership/reader evidence and fresh replay drift are distinct from successful browser interactions. A failed decision may occur after earlier actions; sanitized progress is not a transcript. Increasing correction limits cannot fix an unsafe menu, ambiguous composer or missing chat proof.

Scope candidates are offered only after native ownership and uniqueness checks; binding rechecks the selected node. With `--verbose`, an exact-ID launcher reports successful activation and learning reports how many proven chat roots are available. If an activated launcher has not produced a proven root yet, discovery observes again using its existing finite wait and journey budgets. Missing proof ends discovery without another launcher click, probe or guide publication. An ID-bearing page region alone is not chat ownership evidence.

Opening a panel is separate from interaction readiness. Native startup observation waits within the existing browser and journey deadlines for a lazily rendered launcher or root, then for one available safe composer or an assistant-owned menu. Visible native buttons and non-navigating `<a role="button">` launchers are supported. Accessible labels, titles, and the user-supplied text hint are rechecked at activation; ordinary links, forms, disabled controls and consequential actions are not launchers. Header/settings buttons and loading panels do not establish readiness, and the planner is not asked to stop while that bounded startup wait is in progress.

For composer chats without actor markers, the agent can learn readers after one owned probe. Native code retains the pre-send node baseline and exposes compiled, opaque reader candidates only when it observes a fresh exact customer echo and a distinct response outside that customer's newly appended message container. When the fresh response also appends control buttons, assistant reader candidates are restricted to containers that own those controls, so a learned reader can never exclude its reply's menu. The agent selects candidate IDs; native checks reject overlap, headers, controls, notifications, incomplete formatted fragments, customer delivery metadata and conflicting actor markers. Reader attribution and response completion share the original response deadline. During attribution no second submission, option selection, state binding or finish is allowed.

Interface guides record `reader_evidence: observed_exchange` for this path; marked readers use `marked`. Observed-exchange evidence requires a separated customer reader and verified send evidence. A fresh exchange row may contain separate customer and assistant branches; their common row is not treated as entirely customer-owned. Nested response fragments or footers are withheld when a corroborated enclosing response carries additional prose. The private discovery recipe starts with a composer probe, and fresh replay corroborates the readers before publication. Normal runtime applies the same per-action evidence checks without a discovery model or startup probing. Unmarked attribution requires a unique exact echo and separated message branches; absent/transformed echoes, indistinguishable actor structure and reused/virtualized message containers may remain unsupported. Unmarked menu-only startup requires additional ownership evidence; this composer-probe path does not authorize speculative menu clicks.

Readiness progress reports the state kind, reader status, input/assistant/option candidate counts, current usable option count, completed steps, pending result binding and remaining conversational actions. Stop reasons distinguish unsupported controls/interfaces, missing readers, ambiguous evidence, unsafe interfaces and exhausted budgets. These are model-reported reasons, not an independently verified diagnosis of the website. Raw labels, selector values and rejected model content are excluded from progress. A premature finish is corrected without browser action; an uncertain submission or response failure still terminates without retries.

After reader binding, option proposals come only from the latest assistant reply; header and historical buttons are excluded. Click-send proposals must satisfy the existing safe-send check and have a unique native locator. The opt-in real-planner regression uses the configured model on a synthetic loopback widget with delayed startup and unmarked messages, then performs fresh replay and publication. Run it with `SYNTHETICCX_RUN_DISCOVERY_PLANNER_SMOKE=1 .venv/bin/python -m pytest -q -s tests/smoke/test_live_discovery_planner.py`; it reads model/execution settings from `sky_scenario.yaml` (override with `SYNTHETICCX_PLANNER_SCENARIO`) and never contacts SKY.

A structurally valid candidate is not a verified guide. Fresh browser navigation, opener activation, state binding and every recorded completed response must succeed. Navigation restrictions, login/consent gates, cross-origin embeds and changed states can prevent replay. No alternative action or reset is attempted after uncertain submission, and no guide is published after failed replay.

For icon-only launchers without accessible text, use an explicit DOM ID hint:

```bash
uv run --extra discovery synthetic-cx discover sky_scenario.yaml \
  --url 'https://www.skymais.com.br/faq' --output site-sky.yaml \
  --headed --chat-hint 'id=ovc-btn-chat-widget' --verbose \
  --max-discovery-attempts 5 --language pt-BR
```

`id=<DOM-ID>` is an exact-ID hint, not arbitrary CSS. IDs must start with a letter and contain only letters, digits, `_`, `:`, or `-`, up to 128 characters. Explicit IDs authorize only unique, visible, enabled safe launcher controls outside forms. Normal Playwright actionability is required—never forced or JavaScript-dispatched clicks. Hints do not authorize conversational options, transactions or unrelated page controls.

Authorized headed validation on 2026-10-06 completed discovery, exact fresh replay, and normal persona runs against both SKY and Vivo. Neither final runtime run had model/channel errors; Vivo reached its five-turn limit, and SKY stopped when the persona believed its goal was complete. Both independent evaluations marked the information goal unmet. Working integration is not guaranteed customer-goal success, headless compatibility, or permanent support for a changing public site.

Verified copies, commands, reports, limitations, and the project review are in [`verification/2026-10-06/`](verification/2026-10-06/README.md). Your original scenario and guide files are not replaced.

For Vivo discovery, use the canonical origin and a fresh output path:

```bash
uv run --extra discovery synthetic-cx discover vivo_scenario.yaml \
  --url 'https://vivo.com.br/para-voce' --output site-vivo.yaml \
  --headed --chat-hint 'Pergunte à Aura' --verbose \
  --max-discovery-attempts 5 --language pt-BR
```

If the guide, completed-scenario output, or backup already exists, preserve it and discover from a new draft copy with a new guide path. An `output or backup already exists` error is a publication conflict, not a chat-control failure.

Inspection traverses nested **open Shadow DOM** generically. CSS/semantic locators pierce open roots; XPath does not. Inventories and node traversal are bounded, incomplete scans cannot establish uniqueness, and root/frame identity remains pinned. Closed roots, inaccessible/cross-origin frames, uncertain actor attribution and insufficient completion evidence remain unsupported. Runtime may resolve a new safe composer within the pinned root; it never searches globally or repairs the fixed reader/root contract.

Malformed decisions receive bounded, value-free corrections without browser actions. Provider failures/refusals are not treated as schema-correction retries. Reading must distinguish actual assistant/customer nodes from shared transcript/status/control containers. Completion requires new witnessed assistant evidence, not merely a successful click or an old greeting. Unknown ownership, identity or state evidence fails closed.

#### Conversational menu options

Option-enabled channels represent an assistant reply as **prose plus offered options**, or as an options-only reply. The same customer persona chooses either typed text or one current option; it cannot do both in a turn. A successful selection counts as one customer turn and is recorded as a selected option, not a typed message or supplied account information.

The current guide is version-1 `InterfaceChannelGuide` with channel type `webchat_playwright_dynamic`. Old version-1 text/option guides and version-2 stateful guides are rejected: regenerate through discovery; there is no conversion. A verified interface can begin with a menu-only chat root. Detached/global button bars remain unsupported. Startup menus are presented to the persona even with zero greeting grace; discovery probes and recipes never autoplay during customer startup.

Ordinary current assistant-owned choices are offered to the scenario-driven persona without an informational-label or prompt-keyword whitelist. The browser still checks reply ownership, actual node identity, visibility, enabled state, uniqueness and current capability identity before one normal click. Utility controls, links, forms and controls outside the pinned widget are not conversation options. Stale, replaced or consumed options cannot be retried or substituted with typed labels after an uncertain click. These guards cannot prove the semantics of opaque website event handlers; run only on authorized targets.

Consequential actions require explicit scenario `action_authorizations`; assistant prose or a persona goal alone does not grant permission. Sandbox payments must declare `environment: sandbox`, `max_amount`, and ISO currency. Payment controls with an unparseable amount, an amount above the limit or a conflicting currency are withheld. This MVP does not authorize live payments. Other categories (`purchase`, `account_change`, `consent`, `authentication`, `destructive`) can be explicitly authorized. Timestamps (`<time>`) are delivery metadata and never enter message content. New discovery catalogs currently offer quiet-interval completion; inline interface configuration also supports completion-marker and streaming-indicator rules.

Compound consequential choices require every detected category's permission. Payment bounds use the selected control's single monetary amount, or one unambiguous contextual amount if the control has none; ambiguous grouping and conflicting currency evidence are rejected. Consequential detection is conservative native classification, not proof of arbitrary website handler semantics or universal language understanding. Prose extraction intentionally excludes control labels, including embedded links; links are not followed or treated as current reply options.

Multiple assistant bubbles in one completed response are all supplied together to the persona. If the persona emits multiple external actions in a single batch, the runtime records a `persona_output_ambiguity_observed` event, executes only the first external action, and discards the rest; no second click/message is dispatched from the same batch. An empty or malformed action batch still fails with a specific model diagnostic.

The TinyTroupe adapter supplies the scenario goal and a reachable, non-executing customer-support peer in the pinned library's actual prompt fields. Real chat delivery stays in SyntheticCX. Before any dispatch, an internal-only THINK/DONE batch, unavailable option ID, or unavailable text action may receive one model-only correction using the same current observation and persona, within the original model timeout. This never manufactures a decision or repeats a browser action; a second invalid decision, malformed/empty batch, or provider failure still ends with a sanitized model diagnostic.

Learning may select a currently issued guarded informational option. Every recorded action must yield a completed response and a witnessed resulting state; fresh replay must verify the exact sequence. Publication records only actually observed behaviors, not coverage of all branches.

**Approved runtime boundary:** customer runtime may follow an unsampled branch, new conversational choices or a replacement composer when current native evidence authorizes them within the fixed interface. It does not match results against sample states or edges. Runtime uses its own turn/journey budget and may exceed discovery's three-action limit. Unsafe, stale or outside-root controls, unauthorized consequential choices, ownership loss and fixed-reader failures still terminate safely; an uncertain action cannot be retried or replaced. This is current-control resolution, not selector self-healing.

---

## Configure a journey

A version-1 YAML document contains all configuration inline. Unknown fields, unsupported versions, invalid limits, and invalid adapter choices are rejected before external activity.

The canonical example is [`examples/telecom/scenario.yaml`](examples/telecom/scenario.yaml). Its configured port is illustrative; use `run_demo.py` or change `channel.url` to the address of your own controlled target.

```yaml
version: "1"

scenario:
  id: unexpected-bill-increase
  description: A subscriber needs to understand a higher mobile invoice.
  goal: Understand why the current invoice is higher than the previous invoice.
  starting_context:
    current_invoice: 150.00
    previous_invoice: 100.00
  knowledge:
    account_number: SYNTHETIC-001
  information:
    - id: account_number
      aliases: [account number, account ID]
  success_criteria:
    - id: billing-explanation
      evaluator_type: semantic_goal_completion
      result_kind: boolean
      evidence_policy: required
      parameters:
        expected_facts:
          - The invoice increased by 50 because the customer changed to a more expensive plan.

persona:
  id: standard-customer
  name: Alex Customer
  traits:
    digital_literacy: 0.7
  communication:
    tone: polite
    style: concise
    language: en-US
  initial_state:
    frustration: 0.1
    willingness_to_continue: 0.9
  behavioral_rules:
    - id: repeated-information-frustration
      priority: 10
      event: repeated_information
      frustration_delta: 0.2
      willingness_to_continue_delta: -0.1

channel:
  type: webchat_playwright_dynamic
  url: http://127.0.0.1:8000/?variant=success&view=frame
  open_locator: { strategy: test_id, value: open-chat }
  frame_locator: { strategy: test_id, value: chat-frame }
  ownership:
    root_locator: { strategy: test_id, value: chat-widget }
    proof: chat_embed
  customer_message_locator: { strategy: test_id, value: customer-message }
  assistant_message_locator: { strategy: test_id, value: assistant-message }
  message_identity: { strategy: attribute, attribute_name: data-message-id }
  response_completion:
    strategy: completion_locator
    locator: { strategy: test_id, value: response-complete }

execution:
  max_turns: 5
  journey_timeout_seconds: 120
  greeting_grace_seconds: 2
  browser_action_timeout_seconds: 10
  response_timeout_seconds: 20
  model_timeout_seconds: 30
  evaluator_timeout_seconds: 30
  persistence_timeout_seconds: 5
  close_timeout_seconds: 5

model:
  provider: openai
  model: gpt-5.4-mini

telemetry: { enabled: true, remote_export: false }
storage: { run_root: .runs }
privacy:
  redaction_keys: [account_number]
  capture_prompts: false
  allow_screenshots: false
  confirm_screenshot_target_safe: false
  allow_video: false
```

### Scenario

A scenario describes the customer's objective independently from the persona.

| Field | Purpose |
| --- | --- |
| `id`, `description`, `goal` | Stable identifier, customer situation, and objective |
| `starting_context` | Customer-visible context supplied to the simulator |
| `knowledge` | Facts the customer may know or reveal |
| `information` | IDs and phrase aliases used for deterministic repeated-information detection |
| `success_criteria` | Evaluator-only pass/fail rules; never supplied to the synthetic persona |

### Persona and behavior

A persona has static `traits`, a communication style, fresh `initial_state` for every run, and deterministic behavioral rules. Normalized traits and state values are in `[0, 1]`.

Describe **who the customer is** with optional `persona.description`, and **how they write** with optional `communication.writing_description`. Both accept up to 2,000 characters and must be nonblank when supplied. Add these fields to an existing persona; keep its required `id`, `communication.tone`, `communication.style`, and `initial_state`:

```yaml
persona:
  id: busy-customer
  description: >
    A busy customer with limited technical knowledge who wants
    a practical solution and dislikes repeating information.
  communication:
    language: pt-BR
    tone: informal
    style: conversational
    message_length: short
    grammar_error_probability: 0.15
    writing_description: >
      Writes like a WhatsApp conversation: uses "vc" and "pq",
      often omits accents and punctuation, and asks one question at a time.
  initial_state:
    frustration: 0.1
    willingness_to_continue: 0.9
```

`message_length` is optional: `short` favors one or two concise sentences, `medium` a compact paragraph, and `long` several sentences or paragraphs when useful. These are soft preferences, not word/token limits. `grammar_error_probability` defaults to `0` and accepts `[0,1]`: it guides grammatical-error propensity, separately from existing typo/spelling-error probabilities, without guaranteeing measured error frequencies. The adapter does not truncate or rewrite messages or introduce errors into structured actions, identifiers, or factual values. Settings guide generation; model compliance is not guaranteed.

Narratives cannot override structured language/tone/style/length/error settings, customer action rules, current behavioral state, or scenario facts. Omitted/null descriptions and message length plus grammar probability `0` preserve legacy persona prompts. Narrative background does not replace deterministic `behavioral_rules` or change evaluator success criteria.

Use synthetic data in descriptions. The selected persona model receives the original prose. Effective configuration is persisted under the normal privacy policy even when prompt capture is disabled. Add `description` and/or `writing_description` to `privacy.redaction_keys` to redact those complete fields and exact echoes in persisted evidence/captured prompts; free-text sensitive substrings still require configured patterns, known secrets, or whole-field redaction. Automatic PII detection is not provided.

The MVP supports the `repeated_information` behavioral event. Matching rules execute by `priority` and then `id`; deltas accumulate before values are clamped. When `willingness_to_continue` reaches a rule's `abandonment_threshold`, the persona abandons and no further customer message is sent.

### Evaluators

Version 1 supports these evaluator types:

| Evaluator | Result | Configuration |
| --- | --- | --- |
| `turn_count` | Boolean | `parameters.max_turns` |
| `journey_timeout` | Boolean | No parameters |
| `repeated_information` | Boolean | Optional `parameters.max_repeats` (default `0`) |
| `semantic_goal_completion` | Boolean or scored | `parameters.expected_facts`; scored results also require `threshold` |

A required criterion gates the run. Optional criteria are diagnostic only. Evidence policies are `required`, `optional`, or `none`; required evidence must reference committed trace events or messages.

### Web-chat channel

The canonical MVP channel type is `webchat_playwright_dynamic`. Its interface contract fixes `open_locator`, optional `frame_locator`, attested conversation-root `ownership`, message readers, message identity and assistant-response completion. Safe composer/send controls and eligible latest-reply options are resolved from the current owned DOM rather than configured sample states. The example uses a controlled same-origin frame.

Runtime exposes ordinary current chat choices to the scenario-driven persona without an informational-menu keyword whitelist. Consequential choices require explicit scenario authorization. For example, sandbox payment tests must declare a maximum amount and currency:

```yaml
scenario:
  id: sandbox-payment
  description: Complete a bounded test payment
  goal: Pay the test invoice in the sandbox
  action_authorizations:
    - action: payment
      environment: sandbox
      max_amount: 25.00
      currency: USD
```

Payment choices without a parseable amount, above the configured maximum, or with an explicitly conflicting currency are not offered. Live payments are not authorized by this MVP schema. Other consequential categories (`purchase`, `account_change`, `consent`, `authentication`, `destructive`) can be explicitly authorized by a scenario. Native ownership, current-node validation and no-retry-after-uncertain-dispatch safeguards still apply.

Locator strategies: `role`, `label`, `placeholder`, `text`, `test_id`, and `css`. XPath is deliberately unsupported. Prefer accessible locators or `test_id` values.

| Feature | Supported values |
| --- | --- |
| Send | `press_enter`, or `click` with a locator |
| Message identity | Stable DOM `attribute`; browser-reference `dom_node`; fixture-only `actor_sequence_content_hash` fallback |
| Response completion | `quiet_interval`, `streaming_indicator_disappears`, or `completion_locator` |
| Embedded chat | Optional `frame_locator` |

The submitted customer text is canonical after a successful send. In marked-reader channels, a matching DOM echo is supplemental evidence; no echo is allowed, and a material mismatch is a channel error. Interface guides using `reader_evidence: observed_exchange` require a unique exact echo and separated fresh response evidence on each action. Stream fragments are combined into a finalized message, while distinct assistant bubbles remain ordered messages in one observed assistant turn.

For append-only chats without message IDs, `message_identity: { strategy: dom_node }` tracks actual browser node references in memory. It does not inject IDs into the page or use content hashes. Repeated identical replies remain distinct, and streaming changes retain the same identity. Removing/replacing observed nodes, reordering them, or changing finalized text fails closed. Node tracking is bounded to 120 nodes per actor and is released with the browser session. Prefer stable attributes when available; virtualized/re-rendered transcripts may not support this strategy.

The public `InterfaceChannelGuide` uses `version: "1"` and `webchat_playwright_dynamic`, with content-free fresh-replay validation. The private sample graph is used only during discovery/replay; it is not loaded as runtime authority or published. Old guide shapes require rediscovery, even if they also say version 1. Inline journey channels use the same dynamic interface contract; old text/stateful channel shapes are not accepted as current run configuration.

### Model providers

The model configuration selects one provider and is also used by the TinyTroupe persona. A separate `evaluation_model` may override it for semantic evaluation.

```yaml
# Reference OpenAI provider
model:
  provider: openai
  model: gpt-5.4-mini
  credential_environment_variable: OPENAI_API_KEY # default
  temperature: 0
  seed: 18372
```

```yaml
# Explicit OpenAI-compatible provider
model:
  provider: openai_compatible
  model: local-model-name
  base_url: http://127.0.0.1:8001/v1
  credential_environment_variable: LOCAL_MODEL_API_KEY
  capabilities:
    chat_completions: true
    structured_output: true
    embeddings: false
    token_usage: false
    model_version: false
```

Custom endpoints must declare an explicit `base_url` and credential environment variable. HTTP is accepted only for loopback addresses; other endpoints require HTTPS. SyntheticCX never forwards ambient `OPENAI_API_KEY` credentials to a custom endpoint. Before browser startup it probes the capabilities needed by the configured persona and evaluators.

### Execution limits

All external operations are bounded. Configure the whole-journey deadline and finite timeouts for browser actions, greeting, response completion, persona/model work, evaluators, persistence, and cleanup. `max_turns` counts successfully submitted customer messages; an initial assistant greeting does not count.

### Storage, privacy, and telemetry

`storage.run_root` defaults to `.runs`. Every acknowledged event is flushed to the append-only `trace.jsonl`; JSON files are derived views. Repeated invocations create new directories.

Local telemetry is controlled by `telemetry.enabled`. Remote OTLP export is disabled by default and requires the optional `telemetry` dependency extra plus both `remote_export: true` and an `otlp_endpoint`.

Privacy defaults favor minimal retention:

- Persisted text is sanitized using `privacy.redaction_keys`, selected credentials, and optional `synthetic_id_patterns`.
- Prompt capture is off by default. When `capture_prompts: true`, redacted input messages are stored as `captured_prompt` model-interaction evidence; model output is not captured.
- Screenshots are off by default. Storage requires **all** of `privacy.allow_screenshots`, `privacy.confirm_screenshot_target_safe`, and `storage.store_screenshots`.
- Video requires both `privacy.allow_video` and `storage.store_video`.

Image pixels are not text-redacted, so enable screenshot capture only for an explicitly safe synthetic target.

---

## Understand the output

| Artifact | What to inspect |
| --- | --- |
| `metadata.json` | Lifecycle, timestamps, termination reason, final outcome, model metadata, capture omissions, and artifact inventory |
| `config.yaml` | Effective redacted configuration and source hash |
| `conversation.json` | Ordered customer and assistant messages |
| `trace.jsonl` | Durable source of truth: typed, sequenced events |
| `evaluation.json` | Criterion snapshots and evaluator findings |
| `report.json` | Outcome, metrics, findings, evidence references, and artifact omissions |
| `report.md` | Optional localized template, same-persona subjective feedback, formal conclusions and source evidence |
| `artifacts/telemetry/spans.jsonl` | Local, run-correlated supplementary span records when telemetry is enabled |

`trace.jsonl` is authoritative. The message `sequence` is conversation-wide and distinct from trace-event sequence. If a provider does not report token usage or a model version, the metadata records that information as unavailable rather than reporting zero or a misleading aggregate.

---

## Controlled fixture resources

The telecom fixture under `examples/telecom/webchat_fixture/` supplies five deterministic UI variants:

| Variant | Use |
| --- | --- |
| `success` | Sufficient billing explanation |
| `repeated_question` | Repeated account-number request |
| `streamed_response` | Fragmented response completion |
| `missing_control` | Browser locator/control failure |
| `silent_response` | Response timeout path |

Run it manually for browser diagnosis:

```bash
uv run python examples/telecom/webchat_fixture/server.py
```

It prints a loopback URL. To invoke the CLI directly, make a temporary copy of the YAML and substitute the printed port in `channel.url`.

The fixture supports `view=main` (default) and `view=frame` (`chat-frame` iframe), plus controlled customer-echo modes. It is a supported test target; arbitrary third-party widgets require their own validated configuration and are not guaranteed compatible.

---

## Validate the checkout

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests/contract tests/smoke/test_live_tinytroupe.py
```

Build and test distribution archives:

```bash
uv build --clear
uv run --no-sync python scripts/verify_distribution.py dist
```

The live TinyTroupe/OpenAI smoke test is intentionally opt-in and validates the persona/model boundary—not a full browser journey:

```bash
export OPENAI_API_KEY='your-credential'
SYNTHETICCX_RUN_LIVE_SMOKE=1 uv run pytest -q tests/smoke/test_live_tinytroupe.py
```

To exercise the actual discovery planner on a synthetic local widget, select an existing scenario draft with model/execution settings and enable its separate smoke:

```bash
SYNTHETICCX_PLANNER_SCENARIO=billing-draft.yaml \
SYNTHETICCX_RUN_DISCOVERY_PLANNER_SMOKE=1 \
uv run --extra discovery pytest -q -s tests/smoke/test_live_discovery_planner.py
```

This check calls the selected model API and launches local Chromium. It tests delayed startup, unmarked reader learning, exact fresh replay and publication/loading, using synthetic data. It does not use the draft's target site or contact SKY. Without `SYNTHETICCX_PLANNER_SCENARIO`, it reads model/execution settings from `sky_scenario.yaml`; that local file is not shipped with the repository. The regular suite skips all four opt-in smokes. Legacy MCP smokes exercise retained compatibility tooling and are not evidence for the current native CLI discovery path.

---

## Scope and limitations

SyntheticCX 0.1.0 is a pre-alpha MVP. It does **not** currently provide a hosted UI, HTTP API, population simulation, dashboard, selector self-healing, voice/IVR/WhatsApp/API channels, arbitrary-widget compatibility guarantees, or a fully local/offline model adapter. Web-chat discovery is an optional, explicit setup command; it does not repair guides during runs. Its tested browser target is Chromium on Linux CI; headed Chromium can help local diagnosis.

Runs are best-effort reproducible: model, sampling parameters, requested seed, configuration, trace, and available model metadata are stored, but LLM behavior can still vary. Use only authorized, controlled targets and synthetic data. Do not run uncontrolled journeys against production systems.

---

## Further reference

- Repository overview and commands: [`README.md`](README.md)
- Canonical configuration: [`examples/telecom/scenario.yaml`](examples/telecom/scenario.yaml)
- Generated contracts: [`schemas/`](schemas/)
- License: [`LICENSE`](LICENSE)
