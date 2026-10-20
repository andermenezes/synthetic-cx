"""Privacy boundary regressions, using synthetic sensitive values only."""

import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

import pytest

from syntheticcx.config.loader import load_config
from syntheticcx.config.models import PrivacyConfig
from syntheticcx.domain.events import MessageObserved, MessageObservedEvent
from syntheticcx.domain.models import ErrorDetail, Message
from syntheticcx.errors import ConfigurationError
from syntheticcx.privacy import Redactor
from syntheticcx.privacy.prompt_capture import PromptCapture


def test_nested_fields_are_redacted_without_mutating_input() -> None:
    value = {"rows": [{"account_number": {"raw": "sensitive"}, "count": 2}], "ok": True}
    redactor = Redactor(PrivacyConfig(redaction_keys=["account_number"]))
    result = redactor.redact_value(value)
    assert result == {
        "rows": [{"account_number": "<redacted:account_number>", "count": 2}],
        "ok": True,
    }
    assert value["rows"][0]["account_number"] == {"raw": "sensitive"}
    assert redactor.redact_value(result) == result


def test_offered_options_and_click_labels_share_the_message_redaction_boundary():
    now = datetime.now(UTC)
    redactor = Redactor(PrivacyConfig(), resolved_secrets=["private-option"])
    message = Message.model_validate(
        {
            "id": "reply",
            "sequence": 0,
            "actor": "assistant",
            "content": "",
            "observed_at": now,
            "options": [
                {"id": "opaque-one", "label": "Information private-option"},
                {"id": "opaque-two", "label": "Support private-option"},
            ],
        }
    )
    safe = redactor.redact_model(message)
    assert "private-option" not in safe.model_dump_json()
    assert [option.id for option in safe.options] == ["opaque-one", "opaque-two"]
    assert "private-option" in message.options[0].label
    click = Message(
        id="choice",
        sequence=1,
        actor="customer",
        content=message.options[0].label,
        observed_at=now,
        metadata={
            "action_kind": "select_option",
            "option_id": "opaque-one",
            "source_assistant_message_id": "reply",
        },
    )
    safe_click = redactor.redact_model(click)
    assert "private-option" not in safe_click.model_dump_json()
    assert safe_click.content == safe.options[0].label
    assert safe_click.metadata["action_kind"] == "select_option"


def test_option_labels_are_redacted_inside_nested_persona_prompt_capture():
    config = load_config(Path("examples/telecom/scenario.yaml"))
    capture = PromptCapture(
        Redactor(config.privacy, resolved_secrets=['private-"option"']), config.scenario
    )
    stimulus = json.dumps(
        {"options": [{"id": "opaque-option", "label": 'Information private-"option"'}]}
    )
    result = capture.sanitize([{"role": "user", "content": stimulus}])
    assert "private-" not in result
    assert "opaque-option" in result


@pytest.mark.parametrize(
    ("secrets", "source", "expected"),
    [
        (["abc", "abcdef"], "abcdef", "<redacted:secret>"),
        (["abc", "cde"], "abcde", "<redacted:secret>"),
        (["aaa"], "aaaa", "<redacted:secret>"),
        (["", "key", "key"], "key key", "<redacted:secret> <redacted:secret>"),
        (["🔒credential"], "x🔒credentialx", "x<redacted:secret>x"),
    ],
)
def test_exact_secret_spans_do_not_leave_overlap_suffixes(
    secrets: list[str], source: str, expected: str
) -> None:
    redactor = Redactor(PrivacyConfig(), resolved_secrets=secrets)
    assert redactor.redact_text(source) == expected
    assert redactor.redact_text(expected) == expected


def test_regex_and_secret_overlap_and_placeholder_stability() -> None:
    redactor = Redactor(
        PrivacyConfig(synthetic_id_patterns=[r"ID-\d+", r"\d+XYZ", "redacted"]),
        resolved_secrets=["ID-123"],
    )
    result = redactor.redact_text("ID-123XYZ ID-456")
    assert result == "<redacted:secret> <redacted:synthetic_id>"
    assert redactor.redact_text(result) == result


def test_arbitrary_placeholder_like_input_is_not_exempt() -> None:
    redactor = Redactor(PrivacyConfig(), resolved_secrets=["hidden-value"])
    assert "hidden-value" not in redactor.redact_text("<redacted:hidden-value>")


def test_secret_spanning_reserved_token_is_redacted() -> None:
    secret = "prefix<redacted:secret>suffix"
    redactor = Redactor(PrivacyConfig(), resolved_secrets=[secret])
    assert redactor.redact_text(secret) == "<redacted:secret>"


def test_secret_conflicting_with_reserved_output_fails_safely() -> None:
    with pytest.raises(ConfigurationError, match="conflicts with reserved"):
        Redactor(PrivacyConfig(), resolved_secrets=["redacted"])


@pytest.mark.parametrize(
    ("patterns", "source", "expected"),
    [
        (["aba"], "ababa", "<redacted:synthetic_id>"),
        (["^abc"], "keyabc", "<redacted:secret>abc"),
        ([r"prefix.*suffix"], "prefix<redacted:secret>suffix", "<redacted:synthetic_id>"),
        ([r"(?<=>)abc"], "keyabc", "<redacted:secret><redacted:synthetic_id>"),
        ([r"^|$"], "ordinary", "ordinary"),
    ],
)
def test_regex_context_overlaps_and_fixed_point(
    patterns: list[str], source: str, expected: str
) -> None:
    redactor = Redactor(PrivacyConfig(synthetic_id_patterns=patterns), resolved_secrets=["key"])
    assert redactor.redact_text(source) == expected
    assert redactor.redact_text(expected) == expected


def test_sensitive_field_names_do_not_leak_in_placeholders() -> None:
    redactor = Redactor(
        PrivacyConfig(redaction_keys=["hidden-value", "ID-123"]),
        resolved_secrets=["hidden-value"],
    )
    result = redactor.redact_value({"hidden-value": "ignored"})
    assert result == {"<redacted:secret>": "<redacted:field>"}
    assert redactor.redact_value(result) == result


def test_collision_raises_safe_error_instead_of_losing_data() -> None:
    redactor = Redactor(PrivacyConfig(), resolved_secrets=["first-secret", "second-secret"])
    with pytest.raises(ConfigurationError) as caught:
        redactor.redact_value({"first-secret": 1, "second-secret": 2})
    assert "duplicate mapping keys" in str(caught.value)
    assert "first-secret" not in str(caught.value)
    assert "second-secret" not in str(caught.value)


def test_policy_is_snapshotted_and_model_output_is_detached() -> None:
    privacy = PrivacyConfig(redaction_keys=["account_number"])
    secrets = ["persona-key", "judge-key"]
    redactor = Redactor(privacy, resolved_secrets=secrets)
    privacy.redaction_keys.clear()
    secrets.clear()
    message = Message(
        id="message-1",
        sequence=0,
        actor="assistant",
        content="persona-key judge-key",
        observed_at=datetime(2026, 9, 24, tzinfo=UTC),
        metadata={"account_number": "account-value"},
    )
    event = MessageObservedEvent(
        id="event-1",
        sequence=0,
        timestamp=message.observed_at,
        type="message_observed",
        data=MessageObserved(message=message),
    )
    sanitized = redactor.redact_model(event)
    assert type(sanitized) is MessageObservedEvent
    assert sanitized.id == event.id
    for secret in ["persona-key", "judge-key", "account-value"]:
        assert secret not in sanitized.model_dump_json()
    assert event.data.message.content == "persona-key judge-key"
    assert redactor.redact_model(sanitized) == sanitized


def test_invalid_sanitized_model_does_not_expose_validation_input() -> None:
    redactor = Redactor(PrivacyConfig(), resolved_secrets=["channel"])
    error = ErrorDetail(category="channel", operation="receive", message="private-text")
    with pytest.raises(ConfigurationError) as caught:
        redactor.redact_model(error)
    rendered = "".join(traceback.format_exception(caught.value))
    assert "private-text" not in rendered
    assert "input_value" not in rendered
    assert caught.value.__suppress_context__


def test_safe_error_and_simulated_textual_boundaries_share_redaction() -> None:
    redactor = Redactor(
        PrivacyConfig(synthetic_id_patterns=[r"ACCOUNT-\d+"]), resolved_secrets=["api-key"]
    )
    error = ErrorDetail(category="model", operation="generate", message="api-key ACCOUNT-123")
    safe_error = redactor.redact_model(error)
    outputs = [
        safe_error.model_dump_json(),
        json.dumps(redactor.redact_value({"log": error.message, "span": {"error": error.message}})),
        redactor.redact_text(error.message),
    ]
    assert all("api-key" not in output and "ACCOUNT-123" not in output for output in outputs)


@pytest.mark.parametrize("encoded_layers", [0, 1, 2, 3])
def test_persona_sensitive_prose_is_redacted_in_captured_prompts(encoded_layers: int) -> None:
    config = load_config(Path("examples/telecom/scenario.yaml"))
    narrative = 'private-persona-"João\\profile"'
    writing = 'private-writing-"voice\\style"'
    persona = config.persona.model_copy(
        update={
            "description": narrative,
            "communication": config.persona.communication.model_copy(
                update={"writing_description": writing}
            ),
        }
    )
    redactor = Redactor(PrivacyConfig(redaction_keys=["description", "writing_description"]))
    capture = PromptCapture(redactor, config.scenario, persona=persona)
    content = narrative + " " + writing
    for _ in range(encoded_layers):
        content = json.dumps(content, ensure_ascii=True)
    prompt = [{"role": "user", "content": content}]
    result = capture.sanitize(prompt)
    assert "private-persona" not in result and "private-writing" not in result
    assert "private-persona" in prompt[0]["content"]
    assert persona.description == narrative


def test_sensitive_persona_values_protect_echoes_and_snapshot_without_capture() -> None:
    context = {"persona": {"description": 'private-"narrative"'}}
    policy = PrivacyConfig(redaction_keys=["description"])
    redactor = Redactor(policy, sensitive_context=context)
    policy.redaction_keys.clear()
    context["persona"]["description"] = "changed after construction"
    assert redactor.redact_text('private-"narrative"') == "<redacted:field>"
    assert redactor.redact_text("changed after construction") == "changed after construction"
    assert redactor.redact_text("ordinary writing preference") == "ordinary writing preference"
    assert redactor.redact_text(redactor.redact_text('private-"narrative"')) == "<redacted:field>"


@pytest.mark.parametrize("key", ["communication", "persona"])
def test_sensitive_parent_fields_protect_descendant_prompt_values_and_echoes(key: str) -> None:
    config = load_config(Path("examples/telecom/scenario.yaml"))
    writing = 'PRIVATE-VOICE-"escaped\\text"'
    persona = config.persona.model_copy(
        update={
            "communication": config.persona.communication.model_copy(
                update={"writing_description": writing}
            ),
        }
    )
    context = {"persona": persona.model_dump(mode="json")}
    policy = PrivacyConfig(redaction_keys=[key])
    redactor = Redactor(policy, sensitive_context=context)
    assert "PRIVATE-VOICE" not in json.dumps(redactor.redact_value(context))
    assert redactor.redact_text(writing) == "<redacted:field>"
    # PromptCapture also protects parent-selected fields with an unbound redactor.
    capture = PromptCapture(Redactor(policy), config.scenario, persona=persona)
    prompt = [{"role": "user", "content": json.dumps({"echo": writing})}]
    safe = capture.sanitize(prompt)
    assert "PRIVATE-VOICE" not in safe
    assert json.loads(prompt[0]["content"])["echo"] == writing


@pytest.mark.parametrize("narrative", ["secret", "redacted", "description", "field"])
def test_sensitive_narrative_placeholder_substrings_are_valid_and_idempotent(
    narrative: str,
) -> None:
    config = load_config(Path("examples/telecom/scenario.yaml"))
    persona = config.persona.model_copy(update={"description": narrative})
    redactor = Redactor(
        PrivacyConfig(redaction_keys=["description"]),
        sensitive_context={"persona": persona.model_dump(mode="json")},
    )
    assert redactor.redact_text(narrative) == "<redacted:field>"
    assert redactor.redact_text("<redacted:field>") == "<redacted:field>"
    capture = PromptCapture(redactor, config.scenario, persona=persona)
    first = capture.sanitize([{"role": "user", "content": narrative}])
    assert "<redacted:description>" in first
    assert capture.sanitize(json.loads(first)) == first


def test_contextual_field_labels_preserve_legacy_provenance_for_matching_credentials() -> None:
    redactor = Redactor(
        PrivacyConfig(redaction_keys=["account_number"]), resolved_secrets=["secret-marker"]
    )
    assert redactor.redact_text("secret-marker") == "<redacted:secret>"
    assert (
        redactor.redact_text_with_field_values(
            "The plan changed for secret-marker.", {"account_number": "secret-marker"}
        )
        == "The plan changed for <redacted:account_number>."
    )


@pytest.mark.parametrize("encoded_layers", [0, 1, 2])
@pytest.mark.parametrize("ensure_ascii", [False, True])
def test_constructor_sensitive_context_is_redacted_in_escaped_observations(
    encoded_layers: int,
    ensure_ascii: bool,
) -> None:
    persona = 'CANARY-PERSONA-"João\\profile"'
    knowledge = 'CANARY-KNOWLEDGE-"私人\\account"'
    context = {"persona": {"description": persona}, "knowledge": {"private_fact": knowledge}}
    redactor = Redactor(
        PrivacyConfig(redaction_keys=["description", "private_fact"]),
        sensitive_context=context,
    )
    content = f"Echo {persona} and {knowledge}"
    for _ in range(encoded_layers):
        content = json.dumps(content, ensure_ascii=ensure_ascii)[1:-1]
    observation = {"assistant_prose": content, "options": [{"id": "current", "label": content}]}
    payload = json.dumps(redactor.redact_value(observation), ensure_ascii=ensure_ascii)
    safe = redactor.redact_prompt_text(payload, observation)
    assert "CANARY-PERSONA" not in safe
    assert "CANARY-KNOWLEDGE" not in safe
    assert "current" in safe
    assert redactor.redact_prompt_text(safe, observation) == safe
    assert observation["assistant_prose"] == content


def test_constructor_context_prompt_placeholders_remain_atomic() -> None:
    redactor = Redactor(
        PrivacyConfig(redaction_keys=["description"]),
        sensitive_context={"description": "field"},
    )
    prompt = "field <redacted:field> <redacted:secret>"
    safe = redactor.redact_prompt_text(prompt, {"unrelated": "ordinary"})
    assert safe == "<redacted:field> <redacted:field> <redacted:secret>"
    assert redactor.redact_prompt_text(safe, {}) == safe
