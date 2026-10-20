import asyncio
import json
from datetime import UTC, datetime
from functools import wraps
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from syntheticcx.config.models import ChatState, PrivacyConfig
from syntheticcx.discovery.budget import DiscoveryBudget
from syntheticcx.discovery.contracts import DiscoveryDecision
from syntheticcx.discovery.learn import _action_choices, _validate, learn
from syntheticcx.domain.models import CompletedResponse, Message, SendReceipt
from syntheticcx.errors import ChannelError, ModelError
from syntheticcx.privacy import Redactor


def run_async(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


def decision(action, **kwargs):
    return {
        **dict.fromkeys(DiscoveryDecision.model_fields),
        "action": action,
        "revision": "r1",
        "stop_reason": "unsupported_interface" if action == "stop" else None,
        **kwargs,
    }


def state():
    return ChatState(
        id="composer",
        kind="composer",
        input_locator={"strategy": "test_id", "value": "input"},
        send_strategy={"type": "press_enter"},
    )


def completed():
    now = datetime.now(UTC)
    return CompletedResponse(
        messages=[
            Message(id="reply", sequence=1, actor="assistant", content="hello", observed_at=now)
        ],
        started_at=now,
        completed_at=now,
        completion_strategy="quiet_interval",
    )


def browser():
    inventory = {
        "revision": "r1",
        "catalog": {
            "revision": "r1",
            "candidates": [{"id": "input1", "kind": "input", "description": "input"}],
        },
        "ownership_bound": True,
        "reader_bound": True,
        "state": state().model_dump(include={"id", "kind", "option_labels"}),
        "available_options": [],
        "composer_id": "input1",
    }
    return SimpleNamespace(
        current_state=state(),
        observe=AsyncMock(return_value=inventory),
        submit_probe=AsyncMock(
            return_value=SendReceipt(message_id="sent", submitted_at=datetime.now(UTC))
        ),
        receive=AsyncMock(return_value=completed()),
        bind_state=AsyncMock(return_value=state()),
        export_config=lambda *args: args,
        open=AsyncMock(),
    )


def provider(*decisions):
    return SimpleNamespace(
        generate=AsyncMock(
            side_effect=[SimpleNamespace(content=json.dumps(item)) for item in decisions]
        )
    )


def redactor():
    return Redactor(PrivacyConfig(redaction_keys=[], synthetic_id_patterns=[]))


def test_binding_choices_include_click_send_and_new_menu_without_option_capabilities():
    observation = browser().observe.return_value
    observation["catalog"]["candidates"].extend(
        [{"id": "send1", "kind": "send"}, {"id": "menu1", "kind": "option"}]
    )
    choices = _action_choices(observation, ["bind_state"], {"first": "PRIVATE-PROBE"})
    assert all(
        choice["input_id"] == "input1" and choice["option_id"] == "menu1" for choice in choices
    )
    assert {
        "action": "bind_state",
        "input_id": "input1",
        "send_mode": "click",
        "send_id": "send1",
        "option_id": "menu1",
    } in choices
    for choice in choices:
        _validate(
            DiscoveryDecision.model_validate(decision(**choice)), observation, {"first": "Hi"}
        )
    assert "PRIVATE-PROBE" not in json.dumps(choices)
    observation["catalog"]["candidates"] = [{"id": "menu1", "kind": "option"}]
    assert _action_choices(observation, ["bind_state"], {}) == [
        {
            "action": "bind_state",
            "input_id": None,
            "send_mode": None,
            "send_id": None,
            "option_id": "menu1",
        }
    ]
    observation["catalog"]["candidates"] = [
        {"id": "input1", "kind": "input"},
        {"id": "send1", "kind": "send"},
    ]
    assert {
        "action": "bind_state",
        "input_id": "input1",
        "send_mode": "click",
        "send_id": "send1",
        "option_id": None,
    } in _action_choices(observation, ["bind_state"], {})


@run_async
async def test_three_actions_are_maximum_even_if_model_requests_four():
    actions = []
    for _ in range(4):
        actions += [
            decision("submit_probe", target_id="input1", probe_id="first"),
            decision("bind_state", input_id="input1", send_mode="press_enter"),
        ]
    broker = browser()
    with pytest.raises(ModelError):
        await learn(
            broker,
            provider(*actions),
            redactor(),
            budget=DiscoveryBudget.create(30),
            probes={"first": "Hello"},
            max_discovery_attempts=1,
        )
    assert broker.submit_probe.await_count == 3


@run_async
async def test_delayed_root_waits_without_reopening_or_model_decisions(monkeypatch):
    broker = browser()
    bound = broker.observe.return_value
    missing = {
        **bound,
        "ownership_bound": False,
        "opener_opened": True,
        "catalog": {"revision": "r1", "candidates": []},
    }
    available = {
        **missing,
        "catalog": {
            "revision": "r1",
            "candidates": [{"id": "root1", "kind": "scope", "description": "root"}],
        },
    }
    broker.observe.side_effect = [missing, missing, available, bound, bound, bound]
    broker.bind_scope = AsyncMock()
    planner = provider(
        decision("bind_scope", target_id="root1"),
        decision("submit_probe", target_id="input1", probe_id="first"),
        decision("bind_state", input_id="input1", send_mode="press_enter"),
        decision("finish"),
    )
    sleep = AsyncMock()
    monkeypatch.setattr("syntheticcx.discovery.learn.asyncio.sleep", sleep)
    budget = DiscoveryBudget.create(30)
    progress = []
    await learn(
        broker,
        planner,
        redactor(),
        budget=budget,
        probes={"first": "Hello"},
        on_progress=progress.append,
    )
    assert sleep.await_count == budget.waits == 2
    assert budget.opened
    assert planner.generate.await_count == 4
    broker.open.assert_not_awaited()
    broker.submit_probe.assert_awaited_once()
    assert "waiting for independently proven chat root after opening" in progress


@run_async
async def test_missing_root_exhausts_wait_budget_without_dispatch(monkeypatch):
    broker = browser()
    broker.observe.return_value = {
        **broker.observe.return_value,
        "ownership_bound": False,
        "opener_opened": True,
        "catalog": {"revision": "r1", "candidates": []},
    }
    sleep = AsyncMock()
    monkeypatch.setattr("syntheticcx.discovery.learn.asyncio.sleep", sleep)
    planner = provider()
    budget = DiscoveryBudget.create(30)
    with pytest.raises(ChannelError, match="no independently proven chat root"):
        await learn(broker, planner, redactor(), budget=budget, probes={"first": "Hello"})
    assert sleep.await_count == budget.waits == 4
    assert broker.observe.await_count == 5
    planner.generate.assert_not_awaited()
    broker.open.assert_not_awaited()
    broker.submit_probe.assert_not_awaited()


@run_async
async def test_uncertain_submission_is_not_retried():
    broker = browser()
    broker.submit_probe.side_effect = ChannelError("send", "uncertain")
    with pytest.raises(ChannelError):
        await learn(
            broker,
            provider(decision("submit_probe", target_id="input1", probe_id="first")),
            redactor(),
            budget=DiscoveryBudget.create(30),
            probes={"first": "Hello"},
        )
    assert broker.submit_probe.await_count == 1
    broker.receive.assert_not_awaited()


@run_async
async def test_stale_and_malformed_output_are_bounded_without_dispatch():
    broker = browser()
    model = provider(
        decision("submit_probe", revision="stale", target_id="input1", probe_id="first"),
        {"action": "submit_probe"},
    )
    with pytest.raises(ModelError):
        await learn(
            broker,
            model,
            redactor(),
            budget=DiscoveryBudget.create(30),
            probes={"first": "Hello"},
            max_discovery_attempts=2,
        )
    broker.submit_probe.assert_not_awaited()


@run_async
async def test_correction_budget_resets_after_a_valid_decision():
    broker = browser()
    invalid = decision("observe", revision="stale")
    model = provider(invalid, decision("observe"), invalid, decision("stop"))
    with pytest.raises(ChannelError, match="learning stopped"):
        await learn(
            broker,
            model,
            redactor(),
            budget=DiscoveryBudget.create(30),
            probes={"first": "Hello"},
            max_discovery_attempts=2,
        )
    assert model.generate.await_count == 4
    broker.submit_probe.assert_not_awaited()


@run_async
async def test_oversized_decisions_are_rejected_before_dispatch():
    broker = browser()
    model = SimpleNamespace(generate=AsyncMock(return_value=SimpleNamespace(content=" " * 32769)))
    with pytest.raises(ModelError, match="invalid discovery decision"):
        await learn(
            broker,
            model,
            redactor(),
            budget=DiscoveryBudget.create(30),
            probes={"first": "Hello"},
            max_discovery_attempts=1,
        )
    broker.submit_probe.assert_not_awaited()


@run_async
async def test_provider_refusal_has_no_correction_retry():
    broker = browser()
    model = SimpleNamespace(generate=AsyncMock(side_effect=ModelError("model", "refused")))
    with pytest.raises(ModelError):
        await learn(
            broker, model, redactor(), budget=DiscoveryBudget.create(30), probes={"first": "Hello"}
        )
    assert model.generate.await_count == 1


@run_async
async def test_explicit_probe_is_not_in_model_prompt():
    broker = browser()
    broker.observe.return_value["catalog"]["candidates"][0]["description"] = (
        "private override secret-token"
    )
    safe = Redactor(
        PrivacyConfig(redaction_keys=[], synthetic_id_patterns=[]),
        resolved_secrets=["private override", "secret-token"],
    )
    model = provider(decision("stop"))
    with pytest.raises(ChannelError):
        await learn(
            broker,
            model,
            safe,
            budget=DiscoveryBudget.create(30),
            probes={"explicit": "private override"},
        )
    prompt = model.generate.call_args.args[0].messages[1].content
    assert "private override" not in prompt
    assert "secret-token" not in prompt


@run_async
async def test_success_records_only_completed_and_post_bound_transition():
    broker = browser()
    model = provider(
        decision("submit_probe", target_id="input1", probe_id="explicit"),
        decision("bind_state", input_id="input1", send_mode="press_enter"),
        decision("finish"),
    )
    result = await learn(
        broker,
        model,
        redactor(),
        budget=DiscoveryBudget.create(30),
        probes={"explicit": "private override"},
    )
    states, transitions, recipe = result.config
    assert transitions[0].probe_id == "explicit"
    assert "private override" not in str(transitions[0].model_dump())
    assert recipe[0].transition_id == transitions[0].id
    assert states == [state()]
    assert result.probes == {"explicit": "private override"}


@run_async
async def test_finish_before_poststate_binding_fails_closed():
    broker = browser()
    model = provider(
        decision("submit_probe", target_id="input1", probe_id="first"),
        decision("finish"),
        decision("finish"),
    )
    with pytest.raises(ModelError):
        await learn(
            broker,
            model,
            redactor(),
            budget=DiscoveryBudget.create(30),
            probes={"first": "Hello"},
            max_discovery_attempts=2,
        )
    broker.submit_probe.assert_awaited_once()
    broker.bind_state.assert_not_awaited()


@run_async
async def test_only_one_owned_opener_is_dispatched():
    broker = browser()
    broker.observe.return_value["ownership_bound"] = False
    broker.observe.return_value["catalog"]["candidates"].append(
        {"id": "open1", "kind": "opener", "description": "chat"}
    )
    model = provider(decision("open", target_id="open1"), decision("open", target_id="open1"))
    with pytest.raises(ModelError):
        await learn(
            broker,
            model,
            redactor(),
            budget=DiscoveryBudget.create(30),
            probes={"first": "Hello"},
            max_discovery_attempts=1,
        )
    broker.open.assert_awaited_once()


@run_async
@pytest.mark.parametrize("after_response", [False, True])
async def test_premature_finish_recovers_without_an_extra_submission(after_response):
    broker = browser()
    submit = decision("submit_probe", target_id="input1", probe_id="first")
    decisions = [submit, decision("finish")] if after_response else [decision("finish"), submit]
    model = provider(
        *decisions,
        decision("bind_state", input_id="input1", send_mode="press_enter"),
        decision("finish"),
    )
    progress = []
    budget = DiscoveryBudget.create(30)
    result = await learn(
        broker,
        model,
        redactor(),
        budget=budget,
        probes={"first": "Hello"},
        on_progress=progress.append,
    )
    assert len(result.config[1]) == 1
    assert budget.actions == 1
    broker.submit_probe.assert_awaited_once()
    broker.receive.assert_awaited_once()
    payloads = [
        json.loads(call.args[0].messages[1].content) for call in model.generate.call_args_list
    ]
    assert "finish" not in payloads[0]["available_actions"]
    assert "finish" in payloads[-1]["available_actions"]
    assert "completed transition" in payloads[1 if not after_response else 2]["correction"]
    assert any("decision rejected: premature_finish" in item for item in progress)
    if after_response:
        assert payloads[1]["readiness"]["poststate_binding_pending"]
        assert "submit_probe" not in payloads[1]["available_actions"]


@run_async
async def test_repeated_premature_finish_exhausts_corrections_without_dispatch():
    broker = browser()
    model = provider(decision("finish"), decision("finish"))
    budget = DiscoveryBudget.create(30)
    with pytest.raises(ModelError, match="invalid discovery decision"):
        await learn(
            broker,
            model,
            redactor(),
            budget=budget,
            probes={"first": "Hello"},
            max_discovery_attempts=2,
        )
    assert model.generate.await_count == 2
    assert budget.actions == 0
    broker.submit_probe.assert_not_awaited()
    broker.receive.assert_not_awaited()


@run_async
async def test_supported_empty_composer_has_probe_action_without_forcing_it():
    broker = browser()
    broker.observe.return_value["reader_bound"] = False
    model = provider(decision("stop", stop_reason="unsafe_interface"))
    progress = []
    with pytest.raises(ChannelError, match="unsafe_interface"):
        await learn(
            broker,
            model,
            redactor(),
            budget=DiscoveryBudget.create(30),
            probes={"first": "Hello"},
            on_progress=progress.append,
        )
    payload = json.loads(model.generate.call_args.args[0].messages[1].content)
    assert "submit_probe" in payload["available_actions"]
    assert "finish" not in payload["available_actions"]
    assert payload["readiness"]["state_kind"] == "composer"
    assert not payload["readiness"]["reader_bound"]
    assert any("state=composer; reader_bound=False; inputs=1" in item for item in progress)
    assert model.generate.await_count == 1
    broker.submit_probe.assert_not_awaited()


@run_async
@pytest.mark.parametrize("reason", ["no_supported_controls", "missing_reader_evidence"])
async def test_absence_stop_with_scoped_input_recovers_to_reader_learning(reason):
    broker = reader_learning_browser()
    # Reproduce the real logs: proven scope, offered input, no bound state/reader.
    broker.current_state = None
    before = {
        **broker.observe.return_value,
        "reader_bound": False,
        "state": None,
        "composer_id": None,
    }
    after = {
        **before,
        "state": state().model_dump(include={"id", "kind", "option_labels"}),
        "composer_id": "input1",
    }
    witness = {
        **after,
        "catalog": {
            "candidates": after["catalog"]["candidates"]
            + [
                {
                    "id": "reply1",
                    "kind": "assistant",
                    "description": "reply",
                    "evidence": "observed_exchange",
                },
                {
                    "id": "echo1",
                    "kind": "customer",
                    "description": "echo",
                    "evidence": "observed_exchange",
                },
            ]
        },
    }
    bound = {**after, "reader_bound": True}
    broker.observe.side_effect = [before, before, after, witness, bound, bound]
    broker.bind_state.side_effect = lambda **_kwargs: (
        setattr(broker, "current_state", state()) or state()
    )
    model = provider(
        decision("stop", stop_reason=reason),
        decision("bind_state", input_id="input1", send_mode="press_enter"),
        decision("submit_probe", target_id="input1", probe_id="first"),
        decision("bind_reader", assistant_id="reply1", customer_id="echo1"),
        decision("bind_state", input_id="input1", send_mode="press_enter"),
        decision("finish"),
    )
    progress = []
    result = await learn(
        broker,
        model,
        redactor(),
        budget=DiscoveryBudget.create(30),
        probes={"first": "Hello"},
        on_progress=progress.append,
    )
    assert len(result.config[1]) == 1
    broker.submit_probe.assert_awaited_once()
    assert any("premature_absence_stop" in item for item in progress)
    payload = json.loads(model.generate.call_args_list[1].args[0].messages[1].content)
    assert {
        "action": "bind_state",
        "input_id": "input1",
        "send_mode": "press_enter",
        "send_id": None,
        "option_id": None,
    } in payload["action_choices"]
    assert "Bind the scoped composer" in payload["correction"]


@run_async
async def test_repeated_absence_stop_exhausts_corrections_without_forced_submission():
    broker = browser()
    model = provider(
        decision("stop", stop_reason="no_supported_controls"),
        decision("stop", stop_reason="no_supported_controls"),
    )
    with pytest.raises(ModelError):
        await learn(
            broker,
            model,
            redactor(),
            budget=DiscoveryBudget.create(30),
            probes={"first": "Hello"},
            max_discovery_attempts=2,
        )
    broker.submit_probe.assert_not_awaited()
    broker.bind_state.assert_not_awaited()


@run_async
async def test_bound_controls_are_not_reoffered_until_poststate_binding():
    broker = browser()
    broker.observe.return_value["catalog"]["candidates"].append(
        {"id": "reader1", "kind": "assistant", "description": "reader"}
    )
    model = provider(
        decision("submit_probe", target_id="input1", probe_id="first"),
        # Reproduce the real model carrying its previous reader into a state decision.
        decision("bind_state", input_id="input1", send_mode="press_enter", assistant_id="reader1"),
        decision("bind_state", input_id="input1", send_mode="press_enter"),
        decision("finish"),
    )
    progress = []
    await learn(
        broker,
        model,
        redactor(),
        budget=DiscoveryBudget.create(30),
        probes={"first": "Hello"},
        on_progress=progress.append,
    )
    payloads = [
        json.loads(call.args[0].messages[1].content) for call in model.generate.call_args_list
    ]
    assert "bind_state" not in payloads[0]["available_actions"]
    assert "bind_reader" not in payloads[0]["available_actions"]
    assert "bind_state" in payloads[1]["available_actions"]
    assert "unused_controls" in payloads[2]["correction"]
    assert any("unused_controls" in item for item in progress)
    assert {"action": "finish"} in payloads[3]["action_choices"]
    assert "bind_state" not in payloads[3]["available_actions"]
    broker.bind_state.assert_awaited_once()
    broker.submit_probe.assert_awaited_once()


@run_async
async def test_missing_controls_cannot_bind_a_terminal_state_to_finish():
    broker = browser()
    broker.current_state = None
    broker.observe.return_value.update(
        catalog={"candidates": []}, reader_bound=False, composer_id=None
    )
    model = provider(decision("bind_state"), decision("stop", stop_reason="no_supported_controls"))
    with pytest.raises(ChannelError, match="no_supported_controls"):
        await learn(
            broker, model, redactor(), budget=DiscoveryBudget.create(30), probes={"first": "Hello"}
        )
    payload = json.loads(model.generate.call_args_list[0].args[0].messages[1].content)
    assert set(payload["available_actions"]) == {"observe", "wait", "stop"}
    assert payload["readiness"]["catalog_counts"] == dict.fromkeys(
        ("input", "send", "assistant", "customer", "option"),
        0,
    )
    broker.bind_state.assert_not_awaited()
    broker.submit_probe.assert_not_awaited()


@run_async
@pytest.mark.parametrize("reason", [None, "CANARY-REJECTED-SECRET"])
async def test_invalid_stop_reason_is_corrected_without_echoing_it(reason):
    broker = browser()
    model = provider(
        decision("stop", stop_reason=reason), decision("stop", stop_reason="unsafe_interface")
    )
    progress = []
    with pytest.raises(ChannelError, match="unsafe_interface") as caught:
        await learn(
            broker,
            model,
            redactor(),
            budget=DiscoveryBudget.create(30),
            probes={"first": "Hello"},
            on_progress=progress.append,
        )
    assert model.generate.await_count == 2
    assert "CANARY-REJECTED-SECRET" not in str(caught.value)
    assert "CANARY-REJECTED-SECRET" not in str(progress)
    assert "CANARY-REJECTED-SECRET" not in model.generate.call_args.args[0].messages[1].content
    broker.open.assert_not_awaited()
    broker.submit_probe.assert_not_awaited()


@run_async
async def test_response_deadline_does_not_resubmit():
    broker = browser()

    async def slow_response():
        await asyncio.sleep(1)
        return completed()

    broker.receive.side_effect = slow_response
    with pytest.raises(TimeoutError):
        await learn(
            broker,
            provider(decision("submit_probe", target_id="input1", probe_id="first")),
            redactor(),
            budget=DiscoveryBudget.create(30),
            probes={"first": "Hello"},
            max_discovery_attempts=1,
            response_timeout_seconds=0.001,
        )
    broker.submit_probe.assert_awaited_once()


def reader_learning_browser():
    broker = browser()
    before = {**broker.observe.return_value, "reader_bound": False}
    after = {
        **before,
        "catalog": {
            "candidates": before["catalog"]["candidates"]
            + [
                {
                    "id": "reply1",
                    "kind": "assistant",
                    "description": "reply",
                    "evidence": "observed_exchange",
                },
                {
                    "id": "echo1",
                    "kind": "customer",
                    "description": "echo",
                    "evidence": "observed_exchange",
                },
            ]
        },
    }
    broker.observe.side_effect = [
        before,
        after,
        after,
        after,
        broker.observe.return_value,
        broker.observe.return_value,
    ]
    broker.bind_reader = AsyncMock()
    broker.wait_for_reader_candidates = AsyncMock()
    return broker


@run_async
async def test_reader_learning_blocks_other_actions_without_resending():
    broker = reader_learning_browser()
    model = provider(
        decision("submit_probe", target_id="input1", probe_id="first"),
        decision("submit_probe", target_id="input1", probe_id="first"),
        decision("finish"),
        decision("bind_reader", assistant_id="reply1", customer_id="echo1"),
        decision("bind_state", input_id="input1", send_mode="press_enter"),
        decision("finish"),
    )
    result = await learn(
        broker,
        model,
        redactor(),
        budget=DiscoveryBudget.create(30),
        probes={"first": "Hello"},
        max_discovery_attempts=5,
    )
    assert len(result.config[1]) == 1
    broker.submit_probe.assert_awaited_once()
    broker.wait_for_reader_candidates.assert_awaited_once()
    broker.bind_reader.assert_awaited_once()
    broker.receive.assert_awaited_once()
    context = json.loads(model.generate.call_args_list[1].args[0].messages[1].content)
    assert set(context["available_actions"]) == {"observe", "wait", "bind_reader", "stop"}
    assert context["readiness"]["reader_learning_pending"]


@run_async
async def test_reader_model_delay_spends_the_original_response_deadline():
    broker = reader_learning_browser()
    model = provider(decision("submit_probe", target_id="input1", probe_id="first"))
    first = model.generate.side_effect

    async def generate(request):
        if model.generate.await_count == 1:
            return next(first)
        await asyncio.sleep(0.1)
        return SimpleNamespace(
            content=json.dumps(decision("bind_reader", assistant_id="reply1", customer_id="echo1"))
        )

    model.generate.side_effect = generate
    with pytest.raises(TimeoutError):
        await learn(
            broker,
            model,
            redactor(),
            budget=DiscoveryBudget.create(30),
            probes={"first": "Hello"},
            response_timeout_seconds=0.01,
        )
    broker.submit_probe.assert_awaited_once()
    broker.receive.assert_not_awaited()
    broker.bind_reader.assert_not_awaited()


@run_async
async def test_reader_rejections_are_bounded_and_never_repeat_the_probe():
    broker = reader_learning_browser()
    broker.bind_reader.side_effect = ChannelError("channel.reader_evidence", "unproven")
    model = provider(
        decision("submit_probe", target_id="input1", probe_id="first"),
        decision("bind_reader", assistant_id="reply1", customer_id="echo1"),
        decision("bind_reader", assistant_id="reply1", customer_id="echo1"),
    )
    with pytest.raises(ModelError, match="reader corroboration attempts exhausted"):
        await learn(
            broker,
            model,
            redactor(),
            budget=DiscoveryBudget.create(30),
            probes={"first": "Hello"},
            max_discovery_attempts=2,
        )
    broker.submit_probe.assert_awaited_once()
    broker.receive.assert_not_awaited()
    broker.bind_state.assert_not_awaited()


@run_async
async def test_reader_wait_is_clipped_to_the_original_response_deadline():
    from time import monotonic

    broker = reader_learning_browser()
    model = provider(
        decision("submit_probe", target_id="input1", probe_id="first"), decision("wait")
    )
    started = monotonic()
    with pytest.raises(TimeoutError):
        await learn(
            broker,
            model,
            redactor(),
            budget=DiscoveryBudget.create(30),
            probes={"first": "Hello"},
            response_timeout_seconds=0.01,
        )
    assert monotonic() - started < 0.5
    broker.submit_probe.assert_awaited_once()
    broker.receive.assert_not_awaited()
