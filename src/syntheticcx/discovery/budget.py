"""Finite counters: dispatch consumes an action even when its outcome is uncertain."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from time import monotonic
from typing import TypeVar

from syntheticcx.errors import ChannelError

T = TypeVar("T")

OPERATION_LIMITS = {"actions": 3, "cycles": 24, "observations": 24, "waits": 4}


@dataclass
class DiscoveryBudget:
    deadline: float
    reserve_seconds: float = 0
    actions: int = 0
    cycles: int = 0
    observations: int = 0
    waits: int = 0
    opened: bool = False

    @classmethod
    def create(cls, seconds: float, *, reserve_seconds: float = 0) -> DiscoveryBudget:
        return cls(monotonic() + seconds, reserve_seconds)

    def check(self) -> None:
        if monotonic() >= self.deadline - self.reserve_seconds:
            raise ChannelError("discover budget", "discovery time budget exhausted")

    async def run(self, operation: Awaitable[T], timeout_seconds: float) -> T:
        remaining = self.deadline - self.reserve_seconds - monotonic()
        return await asyncio.wait_for(operation, timeout=max(0, min(timeout_seconds, remaining)))

    def consume(self, kind: str) -> None:
        self.check()
        if kind == "open":
            if self.opened:
                raise ChannelError("discover budget", "chat opener already consumed")
            self.opened = True
            return
        current = getattr(self, kind)
        if current >= OPERATION_LIMITS[kind]:
            raise ChannelError("discover budget", "discovery operation budget exhausted")
        setattr(self, kind, current + 1)
