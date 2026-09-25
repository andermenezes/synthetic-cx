"""Privacy boundary regressions, using synthetic sensitive values only."""

import json
import traceback
from datetime import UTC, datetime

import pytest

from syntheticcx.config.models import PrivacyConfig
from syntheticcx.domain.events import MessageObserved, MessageObservedEvent
from syntheticcx.domain.models import ErrorDetail, Message
from syntheticcx.errors import ConfigurationError
from syntheticcx.privacy import Redactor


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
