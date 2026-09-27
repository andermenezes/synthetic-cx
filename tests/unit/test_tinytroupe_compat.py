import inspect
from importlib import import_module

from syntheticcx._compat import tinytroupe


def test_configuration_snapshot_restores_owned_values() -> None:
    configuration = {"model": "one", "temperature": 0.1}
    snapshot = tinytroupe.snapshot_configuration(configuration)
    configuration.update({"model": "two", "new": True})

    tinytroupe.restore_configuration(configuration, snapshot)

    assert configuration == {"model": "one", "temperature": 0.1}


def test_metadata_handles_available_or_unavailable_dependency() -> None:
    result = tinytroupe.metadata()

    assert isinstance(result.available, bool)
    assert result.version is None or isinstance(result.version, str)


def test_construct_persona_restores_tinytroupe_global_configuration() -> None:
    tinytroupe_module = tinytroupe.import_tinytroupe()
    original_model = tinytroupe_module.config_manager.get("model")

    result = tinytroupe.construct_persona(
        "syntheticcx-compat-probe", overrides={"model": "compatibility-model"}
    )

    assert result.name == "syntheticcx-compat-probe"
    assert tinytroupe_module.config_manager.get("model") == original_model


def test_pinned_persona_and_custom_client_extension_surface() -> None:
    """Detect an upstream change to the surfaces required for the Phase 6 persona adapter."""
    person = import_module("tinytroupe.agent").TinyPerson
    clients = import_module("tinytroupe.clients")
    from tinytroupe.agent.mental_faculty import CustomMentalFaculty

    # TinyTroupe decorators hide both method signatures behind (*args, **kwargs).
    # The action shape itself needs fake-client execution in checkpoint 6c.
    assert callable(person.act)
    assert callable(person.listen)
    assert "actions_configs" in inspect.signature(CustomMentalFaculty).parameters
    assert callable(clients.register_client)
    assert callable(clients.client)


def test_compatibility_probe_does_not_clear_other_agents() -> None:
    people = import_module("tinytroupe.agent").TinyPerson.all_agents
    existing = object()
    people["unrelated-agent"] = existing
    try:
        tinytroupe.construct_persona("syntheticcx-probe-isolated")
        assert people["unrelated-agent"] is existing
        assert "syntheticcx-probe-isolated" not in people
    finally:
        people.pop("unrelated-agent", None)
