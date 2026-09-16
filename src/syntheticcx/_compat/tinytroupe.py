"""Private, owned boundary around TinyTroupe.

Only primitives cross this seam.  Runtime persona integration belongs to Phase 6.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from threading import RLock
from typing import Any

_CONFIGURATION_LOCK = RLock()


@dataclass(frozen=True)
class TinyTroupeMetadata:
    """Safe TinyTroupe facts that do not expose library types."""

    available: bool
    version: str | None


@dataclass(frozen=True)
class TinyTroupePersonaProbe:
    """Evidence that a private TinyTroupe persona was constructed successfully."""

    name: str


def metadata() -> TinyTroupeMetadata:
    """Return installed TinyTroupe metadata without raising when unavailable."""
    try:
        return TinyTroupeMetadata(available=True, version=version("tinytroupe"))
    except PackageNotFoundError:
        return TinyTroupeMetadata(available=False, version=None)


def import_tinytroupe() -> Any:
    """Import TinyTroupe only within the private compatibility seam."""
    return import_module("tinytroupe")


@contextmanager
def configured_tinytroupe(overrides: dict[str, Any]) -> Iterator[None]:
    """Apply private process-global settings and restore them after one operation."""
    tinytroupe = import_tinytroupe()
    configuration = tinytroupe.config_manager._config
    with _CONFIGURATION_LOCK:
        snapshot = dict(configuration)
        try:
            tinytroupe.config_manager.update_multiple(overrides)
            yield
        finally:
            configuration.clear()
            configuration.update(snapshot)


def construct_persona(
    name: str, *, overrides: dict[str, Any] | None = None
) -> TinyTroupePersonaProbe:
    """Construct one TinyTroupe persona without exposing its object beyond this seam."""
    if not name.strip():
        raise ValueError("name must not be empty")

    with configured_tinytroupe(overrides or {}):
        agent_module = import_module("tinytroupe.agent")
        person = agent_module.TinyPerson(name=name)
        person.__class__.clear_agents()
    return TinyTroupePersonaProbe(name=name)


def snapshot_configuration(configuration: dict[str, Any]) -> dict[str, Any]:
    """Copy owned configuration while holding the lock used for global mutations."""
    with _CONFIGURATION_LOCK:
        return dict(configuration)


def restore_configuration(target: dict[str, Any], snapshot: dict[str, Any]) -> None:
    """Restore an owned configuration mapping exactly under the process-local lock."""
    with _CONFIGURATION_LOCK:
        target.clear()
        target.update(snapshot)
