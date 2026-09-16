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
