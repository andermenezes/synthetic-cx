"""Contract validation independent of providers, browser, and persistence."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from syntheticcx.config.loader import load_config
from syntheticcx.config.schema import export_schemas
from syntheticcx.domain.artifacts import (
    ConfigArtifact,
    ConversationArtifact,
    EvaluationArtifact,
    MetadataArtifact,
    Run,
)
from syntheticcx.domain.events import TypedTraceEvent
from syntheticcx.domain.models import (
    CompletedResponse,
    CustomerScenarioContext,
    Message,
    PersonaAction,
    PersonaDefinition,
    PersonaState,
    ReplyOption,
    Report,
    SelectionReceipt,
    SendReceipt,
    TokenUsage,
)

NOW = datetime(2026, 9, 23, tzinfo=UTC)


def message(identifier: str = "m1", sequence: int = 0) -> Message:
    return Message(
        id=identifier, sequence=sequence, actor="assistant", content="Hello", observed_at=NOW
    )


def test_customer_projection_is_detached_and_excludes_hidden_criteria() -> None:
    config = load_config(Path("examples/telecom/scenario.yaml"))
    scenario = config.scenario.model_copy(deep=True)
    scenario.success_criteria[0].parameters["hidden"] = "HIDDEN_MARKER"
    context = CustomerScenarioContext.from_scenario(scenario)
    assert "HIDDEN_MARKER" not in context.model_dump_json()
    assert "success_criteria" not in context.model_dump()
    context.knowledge["new"] = "customer-only"
    assert "new" not in scenario.knowledge


def test_persona_state_is_fresh_for_each_run() -> None:
    config = load_config(Path("examples/telecom/scenario.yaml"))
    definition = PersonaDefinition.model_validate(config.persona.model_dump())
    first, second = definition.fresh_state(), definition.fresh_state()
    assert first == second
    assert first is not second
    assert first is not definition.initial_state
    with pytest.raises(ValidationError):
        first.frustration = 0.9


@pytest.mark.parametrize("case", ["actor", "duplicate", "order", "empty", "time"])
def test_completed_response_rejects_invalid_boundaries(case: str) -> None:
    payload = {
        "messages": [message().model_dump(), message("m2", 1).model_dump()],
        "started_at": NOW,
        "completed_at": NOW,
        "completion_strategy": "completion_locator",
    }
    if case == "actor":
        payload["messages"][0]["actor"] = "customer"
    elif case == "duplicate":
        payload["messages"][1]["id"] = "m1"
    elif case == "order":
        payload["messages"].reverse()
    elif case == "empty":
        payload["messages"] = []
    else:
        payload["completed_at"] = datetime(2025, 1, 1, tzinfo=UTC)
    with pytest.raises(ValidationError):
        CompletedResponse.model_validate(payload)


def test_terminal_actions_cannot_send_another_message() -> None:
    state = PersonaState()
    action = PersonaAction(state=state, signal="goal_believed_complete")
    assert action.message is None
    with pytest.raises(ValidationError):
        PersonaAction(state=state, signal="goal_believed_complete", message="another turn")


def test_options_are_assistant_owned_unique_and_support_options_only_replies() -> None:
    option = ReplyOption(id="opaque:1", label="Billing information")
    payload = message().model_dump()
    payload.update(content="", options=[option.model_dump()])
    assistant = Message.model_validate(payload)
    assert assistant.content == "" and assistant.options == [option]
    assert message().options == []
    for actor in ("customer", "system"):
        with pytest.raises(ValidationError, match="assistant"):
            Message.model_validate({**payload, "actor": actor})
    with pytest.raises(ValidationError, match="unique"):
        Message.model_validate({**payload, "options": [option, option]})


def test_selection_action_and_receipt_contracts() -> None:
    state = PersonaState()
    selected = PersonaAction(selected_option_id="opaque:1", state=state, signal="continue")
    assert selected.message is None
    with pytest.raises(ValidationError, match="mutually exclusive"):
        PersonaAction(
            message="Billing", selected_option_id="opaque:1", state=state, signal="continue"
        )
    with pytest.raises(ValidationError, match="terminal"):
        PersonaAction(selected_option_id="opaque:1", state=state, signal="goal_believed_complete")
    receipt = SelectionReceipt(
        message_id="customer-1",
        submitted_at=NOW,
        selected_option_id="opaque:1",
        selected_label="Billing information",
        source_message_id="assistant-1",
    )
    assert isinstance(receipt, SendReceipt)
    assert SelectionReceipt.model_validate_json(receipt.model_dump_json()) == receipt


def test_reply_options_config_is_optional_and_has_a_fixed_policy() -> None:
    from syntheticcx.config.models import ChannelConfig, ReplyOptionsConfig

    # Legacy options are an internal adapter contract, not the public interface.
    channel = ChannelConfig.model_validate(
        {
            "type": "webchat_playwright",
            "url": "http://127.0.0.1:8000/?variant=success",
            "input_locator": {"strategy": "test_id", "value": "chat-input"},
            "send_strategy": {"type": "press_enter"},
            "customer_message_locator": {"strategy": "test_id", "value": "customer-message"},
            "assistant_message_locator": {"strategy": "test_id", "value": "assistant-message"},
            "message_identity": {"strategy": "attribute", "attribute_name": "data-message-id"},
            "response_completion": {"strategy": "quiet_interval", "quiet_interval_seconds": 0.1},
        }
    )
    assert channel.reply_options is None
    options = ReplyOptionsConfig(
        scope_locator={"strategy": "css", "value": ".conversation"},
        option_locator={"strategy": "role", "value": "button"},
    )
    assert (
        ChannelConfig.model_validate(
            {**channel.model_dump(), "reply_options": options}
        ).reply_options
        == options
    )
    with pytest.raises(ValidationError):
        ReplyOptionsConfig.model_validate({**options.model_dump(), "policy": "unrestricted"})


def test_public_channel_is_dynamic_without_controls_states_or_recipe() -> None:
    from syntheticcx.config.models import InterfaceChannelConfig, ProjectConfig

    config = load_config(Path("examples/telecom/scenario.yaml"))
    assert isinstance(config.channel, InterfaceChannelConfig)
    assert ProjectConfig.model_validate_json(config.model_dump_json()) == config
    channel = config.channel.model_dump()
    for key in (
        "input_locator",
        "send_strategy",
        "reply_options",
        "states",
        "transitions",
        "initial_state_id",
        "verification_recipe",
    ):
        assert key not in channel
        with pytest.raises(ValidationError):
            InterfaceChannelConfig.model_validate({**channel, key: []})


def test_naive_timestamps_are_rejected() -> None:
    payload = message().model_dump()
    payload["observed_at"] = datetime(2026, 9, 23)
    with pytest.raises(ValidationError):
        Message.model_validate(payload)


def test_missing_usage_is_explicit_not_zero() -> None:
    with pytest.raises(ValidationError):
        TokenUsage()
    assert TokenUsage(unavailable_reason="provider omitted usage").input_tokens is None
    assert TokenUsage(input_tokens=0, output_tokens=0).unavailable_reason is None


def event_payloads() -> dict[str, object]:
    config = load_config(Path("examples/telecom/scenario.yaml"))
    result = {
        "evaluator_id": "goal",
        "criterion_id": "billing-explanation",
        "status": "passed",
        "required": True,
        "explanation": "billing explained",
    }
    return {
        "run_lifecycle_changed": {"current": "in_progress"},
        "conversation_started": {"greeting_observed": False, "grace_seconds": 2},
        "message_observed": {"message": message().model_dump()},
        "persona_state_changed": {
            "before": PersonaState(),
            "after": PersonaState(frustration=0.2),
            "trigger_ids": ["event1"],
            "applied_rule_ids": ["rule1"],
        },
        "model_interaction_completed": {
            "id": "call1",
            "identity": {"provider": "openai", "model": "reference"},
            "started_at": NOW,
            "ended_at": NOW,
            "usage": {"unavailable_reason": "not reported"},
        },
        "evaluation_completed": {
            "result": result,
            "criterion": config.scenario.success_criteria[0],
        },
        "operation_failed": {
            "error": {"category": "channel", "operation": "receive", "message": "timeout"},
        },
        "artifact_recorded": {
            "media_type": "image/png",
            "status": "omitted",
            "reason": "disabled",
        },
    }


def test_all_eight_event_variants_round_trip_and_reject_untyped_payloads() -> None:
    adapter = TypeAdapter(TypedTraceEvent)
    payloads = event_payloads()
    assert len(payloads) == 8
    for kind, data in payloads.items():
        envelope = {"id": "event1", "sequence": 0, "timestamp": NOW, "type": kind, "data": data}
        event = adapter.validate_python(envelope)
        assert adapter.validate_json(adapter.dump_json(event)) == event
        envelope["data"] = {"arbitrary": "not an event payload"}
        with pytest.raises(ValidationError):
            adapter.validate_python(envelope)


def test_artifact_schemas_match_generated_output(tmp_path: Path) -> None:
    paths = export_schemas(tmp_path)
    for path in paths:
        assert path.read_text() == (Path("schemas") / path.name).read_text()
    assert json.loads((tmp_path / "conversation.schema.json").read_text())["type"] == "array"
    conversation = ConversationArtifact([message()])
    assert ConversationArtifact.model_validate_json(conversation.model_dump_json()) == conversation


def test_run_metadata_report_and_snapshot_round_trip() -> None:
    identity = {"provider": "openai", "model": "reference"}
    metadata = {
        "configuration_version": "1",
        "framework_version": "0.1.0",
        "persona_model": identity,
        "evaluation_model": identity,
        "usage": {"unavailable_reason": "no calls yet"},
        "runtime_platform": "test",
        "owner": {
            "owner_uuid": "owner-1",
            "hostname": "test-host",
            "pid": 1,
            "process_start_evidence": "process-start-marker",
        },
    }
    run = Run.model_validate(
        {
            "id": "run-1",
            "scenario_id": "billing",
            "persona_id": "standard",
            "started_at": NOW,
            "lifecycle_status": "in_progress",
            "metadata": metadata,
        }
    )
    assert Run.model_validate_json(run.model_dump_json()) == run
    artifact = MetadataArtifact.model_validate(
        run.model_dump(exclude={"events", "messages", "evaluations"})
    )
    assert MetadataArtifact.model_validate_json(artifact.model_dump_json()) == artifact
    incomplete_finalization = artifact.model_dump()
    incomplete_finalization["lifecycle_status"] = "finalized"
    with pytest.raises(ValidationError, match="finalized runs require"):
        MetadataArtifact.model_validate(incomplete_finalization)
    report = Report(
        run_id="run-1",
        outcome="passed",
        termination_reason="goal_reached",
        turn_count=1,
        duration_seconds=1.5,
        repetition_count=0,
        artifact_location=".runs/run-1",
    )
    assert Report.model_validate_json(report.model_dump_json()) == report
    snapshot = ConfigArtifact(
        source_content_hash="source-hash",
        effective_config={"nested": {"list": [{"value": 1}]}},
    )
    assert ConfigArtifact.model_validate_json(snapshot.model_dump_json()) == snapshot
    evaluation = EvaluationArtifact(criteria=[], results=[])
    assert EvaluationArtifact.model_validate_json(evaluation.model_dump_json()) == evaluation
