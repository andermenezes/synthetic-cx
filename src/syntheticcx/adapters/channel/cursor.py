"""Per-run message identity and observation cursors for browser snapshots."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from typing import Literal

from syntheticcx.config.models import MessageIdentityConfig
from syntheticcx.errors import ChannelError

Actor = Literal["customer", "assistant"]


@dataclass(frozen=True)
class Bubble:
    """One atomic DOM snapshot entry; text may still be streaming."""

    actor: Actor
    ordinal: int
    content: str
    attribute_id: str | None


class MessageCursor:
    """Track baseline, pending updates, and delivered bubbles independently by actor."""

    def __init__(self, identity: MessageIdentityConfig) -> None:
        self._identity = identity
        self._known: dict[Actor, dict[str, tuple[int, str]]] = {
            "customer": {},
            "assistant": {},
        }
        self._delivered: dict[Actor, set[str]] = {"customer": set(), "assistant": set()}

    def _key(self, bubble: Bubble) -> str:
        if self._identity.strategy in {"attribute", "dom_node"}:
            if not bubble.attribute_id or not bubble.attribute_id.strip():
                raise ChannelError("channel.observe", "message has no stable identity")
            return bubble.attribute_id
        # Ordinal is stable while text streams; content enters only the final ID.
        return str(bubble.ordinal)

    def observe(
        self, actor: Actor, bubbles: list[Bubble], *, baseline: bool = False
    ) -> list[Bubble]:
        """Record snapshots, rejecting ambiguous identity and altered delivered text."""
        known = self._known[actor]
        seen: set[str] = set()
        pending: list[Bubble] = []
        for bubble in bubbles:
            if bubble.actor != actor or bubble.ordinal < 0:
                raise ChannelError("channel.observe", "message actor or position is invalid")
            key = self._key(bubble)
            if key in seen:
                raise ChannelError("channel.observe", "duplicate message identity")
            seen.add(key)
            previous = known.get(key)
            if previous is not None and previous[0] != bubble.ordinal:
                raise ChannelError("channel.observe", "message identity changed position")
            if (
                key in self._delivered[actor]
                and previous is not None
                and previous[1] != bubble.content
            ):
                raise ChannelError("channel.observe", "finalized message content changed")
            known[key] = (bubble.ordinal, bubble.content)
            if baseline:
                self._delivered[actor].add(key)
            elif key not in self._delivered[actor]:
                pending.append(bubble)
        if any(key not in seen for key in known):
            raise ChannelError("channel.observe", "observed message disappeared")
        return pending

    def finalize(self, actor: Actor, bubbles: list[Bubble]) -> list[str]:
        """Acknowledge completed bubbles and return stable actor-qualified IDs."""
        ids: list[str] = []
        for bubble in bubbles:
            key = self._key(bubble)
            if key in self._delivered[actor] or self._known[actor].get(key) != (
                bubble.ordinal,
                bubble.content,
            ):
                raise ChannelError("channel.observe", "message finalization is ambiguous")
            self._delivered[actor].add(key)
            if self._identity.strategy in {"attribute", "dom_node"}:
                ids.append(f"{actor}:{key}")
            else:
                normalized = unicodedata.normalize("NFKC", bubble.content).casefold().strip()
                digest = hashlib.sha256(f"{actor}\0{key}\0{normalized}".encode()).hexdigest()
                ids.append(f"{actor}:{digest}")
        return ids
