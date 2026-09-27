"""Bounded handling for blocking library work.

Python cannot forcibly stop a running worker thread.  Timed-out work is marked
unusable and its eventual result is discarded; callers must not reuse it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Thread
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class BoundedResult(Generic[T]):
    """The observable result of one bounded blocking operation."""

    value: T | None
    timed_out: bool
    cancelled: bool


async def run_blocking_bounded(
    operation: Callable[[], T], *, timeout_seconds: float
) -> BoundedResult[T]:
    """Run blocking work without blocking the event loop beyond its deadline."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    # asyncio.run() joins its default executor during shutdown. A timed-out
    # TinyTroupe call must not keep the CLI alive while its late work finishes.
    result: Future[T] = Future()

    def worker() -> None:
        if not result.set_running_or_notify_cancel():
            return
        try:
            result.set_result(operation())
        except BaseException as exc:
            result.set_exception(exc)

    Thread(target=worker, name="syntheticcx-blocking", daemon=True).start()
    task = asyncio.wrap_future(result)
    try:
        value = await asyncio.wait_for(asyncio.shield(task), timeout=timeout_seconds)
    except TimeoutError:
        task.add_done_callback(_discard_late_result)
        return BoundedResult(value=None, timed_out=True, cancelled=False)
    except asyncio.CancelledError:
        task.add_done_callback(_discard_late_result)
        return BoundedResult(value=None, timed_out=False, cancelled=True)
    return BoundedResult(value=value, timed_out=False, cancelled=False)


def _discard_late_result(task: asyncio.Future[T]) -> None:
    """Consume a late result or exception so it cannot escape as unhandled work."""
    if not task.cancelled():
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass
