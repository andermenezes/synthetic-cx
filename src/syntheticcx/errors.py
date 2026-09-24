"""Errors raised by SyntheticCX operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ErrorCategory = Literal[
    "configuration", "channel", "model", "evaluation", "persistence", "cancelled"
]


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


class ModelError(SyntheticCXError):
    """A model provider operation failed."""

    def __init__(self, operation: str, message: str, *, retryable: bool = False) -> None:
        super().__init__("model", operation, message, retryable)


class EvaluationError(SyntheticCXError):
    """An evaluator operation failed."""

    def __init__(self, operation: str, message: str, *, retryable: bool = False) -> None:
        super().__init__("evaluation", operation, message, retryable)


class PersistenceError(SyntheticCXError):
    """An evidence persistence operation failed."""

    def __init__(self, operation: str, message: str, *, retryable: bool = False) -> None:
        super().__init__("persistence", operation, message, retryable)
