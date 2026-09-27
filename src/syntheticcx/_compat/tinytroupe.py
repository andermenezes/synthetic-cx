"""Private, owned boundary around TinyTroupe.

Only owned values cross this seam; TinyPerson stays private across turns.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from threading import RLock
from typing import Any
from uuid import uuid4

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
        previous = agent_module.TinyPerson.all_agents.get(name)
        try:
            agent_module.TinyPerson(name=name)
        finally:
            if previous is None:
                agent_module.TinyPerson.all_agents.pop(name, None)
            else:
                agent_module.TinyPerson.all_agents[name] = previous
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


class TinySession:
    """One TinyPerson, private custom client and global-state scope per blocking call."""

    def __init__(self, client: Any, *, model: str, timeout_seconds: float) -> None:
        self._client = client
        self._person: Any = None
        self._closed = False
        self._name = f"syntheticcx-{uuid4()}"
        self._api_type = f"syntheticcx-{uuid4()}"
        self._overrides = {
            "model": model,
            "timeout": timeout_seconds,
            "max_attempts": 1,
            "waiting_time": 0,
            "cache_api_calls": False,
            "enable_continuous_contextual_semantic_memory_retrieval": False,
            "enable_memory_consolidation": False,
            "action_generator_max_attempts": 1,
            "action_generator_enable_regeneration": False,
            "action_generator_enable_quality_checks": False,
            "action_generator_enable_direct_correction": False,
            "action_generator_continue_on_failure": False,
        }

    @contextmanager
    def _scope(self) -> Iterator[Any]:
        """Keep the lock in the worker until its own cleanup, even if an await times out."""
        with _CONFIGURATION_LOCK:
            tinytroupe = import_tinytroupe()
            clients = import_module("tinytroupe.clients")
            agent_module = import_module("tinytroupe.agent")
            configuration = tinytroupe.config_manager._config
            snapshot = snapshot_configuration(configuration)
            previous_client = clients._api_type_to_client.get(self._api_type)
            previous_override = clients._api_type_override
            previous_person = agent_module.TinyPerson.all_agents.get(self._name)
            loggers = [logging.getLogger(name) for name in ("tinytroupe", "openai", "httpx")]
            logging_states = [logger.disabled for logger in loggers]
            try:
                for logger in loggers:
                    logger.disabled = True
                tinytroupe.config_manager.update_multiple(self._overrides)
                clients.register_client(self._api_type, self._client)
                clients.force_api_type(self._api_type)
                if self._person is not None:
                    agent_module.TinyPerson.all_agents[self._name] = self._person
                yield agent_module
            finally:
                if previous_person is None:
                    agent_module.TinyPerson.all_agents.pop(self._name, None)
                else:
                    agent_module.TinyPerson.all_agents[self._name] = previous_person
                if previous_client is None:
                    clients._api_type_to_client.pop(self._api_type, None)
                else:
                    clients._api_type_to_client[self._api_type] = previous_client
                clients.force_api_type(previous_override)
                restore_configuration(configuration, snapshot)
                for logger, disabled in zip(loggers, logging_states):
                    logger.disabled = disabled

    def initialize(self, description: str, goal: str) -> None:
        if self._closed:
            raise ValueError("persona session is closed")
        with self._scope() as agent_module:
            faculty = import_module("tinytroupe.agent.mental_faculty").CustomMentalFaculty(
                name="SyntheticCX customer decision",
                actions_configs={
                    "GOAL_COMPLETE": {
                        "description": (
                            "Explicitly declare that YOUR customer goal is satisfied; do not talk."
                        ),
                        "function": None,
                    },
                    "ABANDON": {
                        "description": "Explicitly stop this customer journey; do not talk.",
                        "function": None,
                    },
                },
            )
            person = agent_module.TinyPerson(name=self._name, mental_faculties=[faculty])
            person.define("description", description)
            person.define("goal", goal)
            self._person = person

    def respond(self, stimulus: str) -> list[dict[str, Any]]:
        if self._closed or self._person is None:
            raise ValueError("persona has not been initialized")
        with self._scope():
            self._person.listen(stimulus, communication_display=False)
            actions = self._person.act(
                until_done=True, return_actions=True, communication_display=False
            )
            if not isinstance(actions, list):
                raise ValueError("persona returned no action list")
            return actions

    def close(self) -> None:
        with self._scope():
            if self._closed:
                return
            self._client.close()
            self._person = None
            self._closed = True
