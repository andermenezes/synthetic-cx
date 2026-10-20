import asyncio
import threading
import time

import pytest

from syntheticcx._compat.blocking import run_blocking_bounded


def test_timeout_returns_promptly_and_late_result_is_discarded() -> None:
    release = threading.Event()

    def blocked() -> str:
        release.wait(timeout=1)
        return "late result"

    result = asyncio.run(run_blocking_bounded(blocked, timeout_seconds=0.01))
    release.set()

    assert result.timed_out is True
    assert result.value is None


def test_non_positive_timeout_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        asyncio.run(run_blocking_bounded(lambda: "unused", timeout_seconds=0))


def test_expired_worker_does_not_hold_up_asyncio_run_shutdown() -> None:
    release = threading.Event()
    started = time.monotonic()
    result = asyncio.run(run_blocking_bounded(lambda: release.wait(1), timeout_seconds=0.01))
    release.set()
    assert result.timed_out
    assert time.monotonic() - started < 0.5
