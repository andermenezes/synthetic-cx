"""Deterministic information-ID matching over successfully submitted text."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from syntheticcx.config.models import ScenarioInformation
from syntheticcx.domain.models import CompletedResponse, SendReceipt


def normalize_phrase(value: str) -> tuple[str, ...]:
    """Case-fold Unicode and treat punctuation and spacing as word separators."""
    words = "".join(
        " " if character.isspace() or unicodedata.category(character).startswith("P") else character
        for character in value.casefold()
    )
    return tuple(words.split())


def _contains(words: tuple[str, ...], alias: tuple[str, ...]) -> bool:
    return any(
        words[index : index + len(alias)] == alias for index in range(len(words) - len(alias) + 1)
    )


@dataclass(frozen=True)
class RepeatedInformation:
    """One repeated request, linked to the canonical original and later request."""

    information_id: str
    original_message_id: str
    request_message_id: str


_REQUEST_WORDS = frozenset(
    {
        "ask",
        "asking",
        "can",
        "could",
        "give",
        "need",
        "please",
        "provide",
        "repeat",
        "share",
        "tell",
        "what",
        "which",
        "would",
    }
)


class RepetitionTracker:
    """Track IDs supplied only after a successful customer send receipt."""

    def __init__(self, information: list[ScenarioInformation]) -> None:
        self._aliases = {
            item.id: tuple(normalize_phrase(alias) for alias in item.aliases)
            for item in information
        }
        self._supplied: dict[str, str] = {}

    def record_send(self, content: str, receipt: SendReceipt) -> None:
        """Retain the first canonical supplied message ID for each information ID."""
        words = normalize_phrase(content)
        for identifier, aliases in self._aliases.items():
            if identifier not in self._supplied and any(
                _contains(words, alias) for alias in aliases
            ):
                self._supplied[identifier] = receipt.message_id

    def detect(self, response: CompletedResponse) -> list[RepeatedInformation]:
        """Emit at most one match per ID across one completed assistant turn."""
        matches: list[RepeatedInformation] = []
        for identifier, original_id in self._supplied.items():
            aliases = self._aliases[identifier]
            for message in response.messages:
                words = normalize_phrase(message.content)
                is_request = "?" in message.content or bool(_REQUEST_WORDS.intersection(words))
                if is_request and any(_contains(words, alias) for alias in aliases):
                    matches.append(RepeatedInformation(identifier, original_id, message.id))
                    break
        return matches
