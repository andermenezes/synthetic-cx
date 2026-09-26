"""Message identity must survive streaming without re-emitting history."""

import pytest

from syntheticcx.adapters.channel.cursor import Bubble, MessageCursor
from syntheticcx.config.models import MessageIdentityConfig
from syntheticcx.errors import ChannelError


def bubble(ordinal: int, content: str, identifier: str | None = None) -> Bubble:
    return Bubble(actor="assistant", ordinal=ordinal, content=content, attribute_id=identifier)


def test_attribute_cursor_ignores_history_buffers_updates_and_rejects_mutation() -> None:
    cursor = MessageCursor(
        MessageIdentityConfig(strategy="attribute", attribute_name="data-message-id")
    )
    old = bubble(0, "Old answer", "old")
    cursor.observe("assistant", [old], baseline=True)
    partial = bubble(1, "Your bill", "new")
    assert cursor.observe("assistant", [old, partial]) == [partial]
    final = bubble(1, "Your bill increased", "new")
    assert cursor.observe("assistant", [old, final]) == [final]
    assert cursor.finalize("assistant", [final]) == ["assistant:new"]
    assert cursor.observe("assistant", [old, final]) == []
    with pytest.raises(ChannelError, match="finalized message content changed"):
        cursor.observe("assistant", [old, bubble(1, "Changed", "new")])


def test_fallback_uses_final_content_only_and_rejects_ambiguous_identity() -> None:
    cursor = MessageCursor(MessageIdentityConfig(strategy="actor_sequence_content_hash"))
    cursor.observe("assistant", [bubble(0, "old")], baseline=True)
    cursor.observe("assistant", [bubble(0, "old"), bubble(1, "Partial")])
    final = bubble(1, "Final")
    cursor.observe("assistant", [bubble(0, "old"), final])
    assert cursor.finalize("assistant", [final])[0].startswith("assistant:")
    with pytest.raises(ChannelError, match="duplicate message identity"):
        cursor.observe("assistant", [bubble(0, "old"), bubble(0, "another")])
