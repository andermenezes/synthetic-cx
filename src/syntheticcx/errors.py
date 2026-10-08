"""Errors raised by SyntheticCX operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

ErrorCategory = Literal[
    "configuration", "channel", "model", "evaluation", "persistence", "cancelled"
]

ChannelFailureReason = Literal[
    "channel.state",
    "channel.reader",
    "channel.reader_evidence",
    "channel.ownership",
    "channel.observe",
    "channel.options",
    "channel.select_option",
    "channel.controls",
]
ModelFailureReason = Literal[
    "provider_timeout",
    "provider_connection",
    "provider_http_4xx",
    "provider_http_5xx",
    "provider_error",
    "invalid_structured_response",
    "persona_timeout",
    "persona_action_malformed",
    "persona_action_unsupported",
    "persona_action_missing",
    "persona_action_ambiguous",
    "persona_option_unavailable",
    "persona_text_unavailable",
    "persona_adapter_error",
    "unknown_model_failure",
]
MODEL_FAILURE_REASONS: frozenset[str] = frozenset(get_args(ModelFailureReason))
CHANNEL_FAILURE_REASONS: frozenset[str] = frozenset(get_args(ChannelFailureReason))


@dataclass(frozen=True)
class SyntheticCXError(Exception):
    """An error with its category, operation, message, and retry behavior."""

    category: ErrorCategory
    operation: str
    message: str
    retryable: bool = False

    def __str__(self) -> str:
        return f"{self.operation}: {self.message}"


class ConfigurationError(SyntheticCXError):
    """Configuration could not be validated or prepared."""

    def __init__(self, operation: str, message: str) -> None:
        super().__init__("configuration", operation, message)


class ChannelError(SyntheticCXError):
    """A channel adapter operation failed."""

    def __init__(self, operation: str, message: str, *, retryable: bool = False) -> None:
        super().__init__("channel", operation, message, retryable)


def safe_channel_error(exc: Exception, operation: str, message: str) -> ChannelError:
    """Preserve known native failure boundaries, never their exception messages."""
    if isinstance(exc, ChannelError) and exc.operation in CHANNEL_FAILURE_REASONS:
        return ChannelError(exc.operation, "channel operation failed", retryable=exc.retryable)
    return ChannelError(operation, message)


class ModelError(SyntheticCXError):
    """A model provider operation failed."""

    reason_code: ModelFailureReason | None

    def __init__(
        self,
        operation: str,
        message: str,
        *,
        retryable: bool = False,
        reason_code: ModelFailureReason | None = None,
    ) -> None:
        super().__init__("model", operation, message, retryable)
        object.__setattr__(self, "reason_code", reason_code)


class EvaluationError(SyntheticCXError):
    """An evaluator operation failed."""

    def __init__(self, operation: str, message: str, *, retryable: bool = False) -> None:
        super().__init__("evaluation", operation, message, retryable)


class PersistenceError(SyntheticCXError):
    """An evidence persistence operation failed."""

    def __init__(self, operation: str, message: str, *, retryable: bool = False) -> None:
        super().__init__("persistence", operation, message, retryable)
