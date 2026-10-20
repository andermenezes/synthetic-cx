"""UTC clocks and bounded waits, including a deterministic manual clock."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable
from datetime import UTC, datetime, timedelta
from typing import TypeVar

T = TypeVar("T")


class RealClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class ManualClock:
    """Advance a logical UTC clock and wake sleepers without real delays."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError("manual clock start must be timezone aware")
        self._now = start.astimezone(UTC)
        self._changed = asyncio.Condition()

    def now(self) -> datetime:
        return self._now

    async def advance(self, seconds: float) -> None:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("advance must be finite and nonnegative")
        async with self._changed:
            self._now += timedelta(seconds=seconds)
            self._changed.notify_all()

    async def wait_until(self, deadline: datetime) -> None:
        async with self._changed:
            await self._changed.wait_for(lambda: self._now >= deadline)


class RealWaitController:
    async def run(self, operation: Awaitable[T], timeout: float) -> T:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        return await asyncio.wait_for(operation, timeout)


class ManualWaitController:
    """Race an operation against logical time; cancel and join the losing task."""

    def __init__(self, clock: ManualClock) -> None:
        self._clock = clock

    async def run(self, operation: Awaitable[T], timeout: float) -> T:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        operation_task = asyncio.ensure_future(operation)
        deadline_task = asyncio.create_task(
            self._clock.wait_until(self._clock.now() + timedelta(seconds=timeout))
        )
        try:
            await asyncio.wait({operation_task, deadline_task}, return_when=asyncio.FIRST_COMPLETED)
            if operation_task.done():
                return await operation_task
            operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)
            raise TimeoutError("operation exceeded its logical deadline")
        except asyncio.CancelledError:
            operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)
            raise
        finally:
            deadline_task.cancel()
            await asyncio.gather(deadline_task, return_exceptions=True)
