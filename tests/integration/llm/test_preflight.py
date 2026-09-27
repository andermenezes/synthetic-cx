"""Credential-free HTTP protocol tests for bounded provider preflight."""

import asyncio
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from syntheticcx.adapters.llm.preflight import CapabilityPreflight, preflight_models
from syntheticcx.config.models import ProjectConfig
from syntheticcx.config.settings import prepare_settings
from syntheticcx.errors import ConfigurationError

_CREDENTIALS = {"LOCAL_MODEL_TOKEN": "selected-secret", "OPENAI_API_KEY": "ambient-secret"}
_CAPABILITIES = {
    "chat_completions": True,
    "structured_output": True,
    "embeddings": False,
    "token_usage": True,
    "model_version": True,
}


@dataclass
class Endpoint:
    base_url: str
    requests: list[dict[str, Any]] = field(default_factory=list)
    status: int = 200
    structured_status: int = 200
    structured_content: str = '{"ok":true}'
    usage: bool = True
    version: bool = True
    delay_seconds: float = 0


@contextmanager
def serve() -> Iterator[Endpoint]:
    endpoint = Endpoint(base_url="")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["content-length"])
            request = json.loads(self.rfile.read(length))
            endpoint.requests.append(
                {
                    "path": self.path,
                    "headers": dict(self.headers),
                    "body": request,
                }
            )
            structured = "response_format" in request
            status = endpoint.structured_status if structured else endpoint.status
            if endpoint.delay_seconds:
                time.sleep(endpoint.delay_seconds)
            response: dict[str, Any]
            if status != 200:
                response = {"error": "secret-from-provider-body"}
            else:
                response = {
                    "id": "probe-response",
                    "object": "chat.completion",
                    "created": 1,
                    "model": request["model"] if endpoint.version else None,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": endpoint.structured_content if structured else "ack",
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 4}
                    if endpoint.usage
                    else None,
                }
            data = json.dumps(response).encode()
            try:
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The bounded client may have already cancelled its request.

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    endpoint.base_url = f"http://127.0.0.1:{server.server_port}/v1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield endpoint
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def project(
    endpoint: Endpoint,
    *,
    evaluation: Endpoint | None = None,
    declared: dict[str, bool] | None = None,
    semantic: bool = True,
    timeout: float = 1,
) -> ProjectConfig:
    document = yaml.safe_load(Path("examples/telecom/scenario.yaml").read_text())

    def compatible(address: Endpoint, model: str, variable: str) -> dict[str, Any]:
        return {
            "provider": "openai_compatible",
            "model": model,
            "base_url": address.base_url,
            "credential_environment_variable": variable,
            "capabilities": declared or _CAPABILITIES,
        }

    document["model"] = compatible(endpoint, "local-persona", "LOCAL_MODEL_TOKEN")
    if evaluation is not None:
        document["evaluation_model"] = compatible(evaluation, "local-evaluator", "EVAL_MODEL_TOKEN")
    if not semantic:
        document["scenario"]["success_criteria"] = [
            {
                "id": "turns",
                "evaluator_type": "turn_count",
                "result_kind": "boolean",
                "evidence_policy": "none",
                "parameters": {"max_turns": 5},
            }
        ]
    document["execution"]["model_timeout_seconds"] = timeout
    return cast(ProjectConfig, ProjectConfig.model_validate(document))


def run(config: ProjectConfig, environment: dict[str, str] | None = None) -> CapabilityPreflight:
    prepared = prepare_settings(config, environment or _CREDENTIALS)
    return asyncio.run(preflight_models(config, prepared))


def test_preflight_checks_chat_and_strict_structure_before_browser_start() -> None:
    with serve() as endpoint:
        channel_started = False

        async def start_channel_after_preflight() -> None:
            nonlocal channel_started
            config = project(endpoint)
            prepared = prepare_settings(config, _CREDENTIALS)
            await preflight_models(config, prepared)
            channel_started = True

        asyncio.run(start_channel_after_preflight())
        assert channel_started
        assert len(endpoint.requests) == 2  # Inherited evaluator role reuses the same probe.
        for request in endpoint.requests:
            assert request["path"] == "/v1/chat/completions"
            assert request["headers"]["Authorization"] == "Bearer selected-secret"
            assert request["body"]["model"] == "local-persona"
        schema = endpoint.requests[1]["body"]["response_format"]["json_schema"]
        assert schema["strict"] is True
        assert schema["schema"]["additionalProperties"] is False
        assert schema["schema"]["required"] == ["ok"]


def test_preflight_metadata_reports_only_observed_capabilities() -> None:
    with serve() as endpoint:
        endpoint.usage = False
        endpoint.version = False
        config = project(endpoint)
        result = run(config)
        for role in ("persona", "evaluation"):
            verified = result.verified_capabilities[role]
            assert verified.chat_completions and verified.structured_output
            assert not verified.embeddings
            assert not verified.token_usage and not verified.model_version
            assert result.unavailable_capabilities[role] == [
                "embeddings",
                "token_usage",
                "model_version",
            ]
        assert "selected-secret" not in result.model_dump_json()


def test_valid_false_boolean_still_proves_structured_output() -> None:
    with serve() as endpoint:
        endpoint.structured_content = '{"ok":false}'
        result = run(project(endpoint))
        assert result.verified_capabilities["persona"].structured_output


def test_optional_capability_must_be_declared_as_well_as_observed() -> None:
    with serve() as endpoint:
        declared = {**_CAPABILITIES, "token_usage": False, "model_version": False}
        result = run(project(endpoint, declared=declared))
        assert not result.verified_capabilities["persona"].token_usage
        assert not result.verified_capabilities["persona"].model_version
        assert "token_usage" in result.unavailable_capabilities["persona"]
        assert "model_version" in result.unavailable_capabilities["persona"]


@pytest.mark.parametrize(
    "failure", ["declaration", "chat", "structured", "wrong_shape", "malformed", "timeout"]
)
def test_required_capability_failure_prevents_browser_activity(failure: str) -> None:
    with serve() as endpoint:
        capabilities = dict(_CAPABILITIES)
        if failure == "declaration":
            capabilities["structured_output"] = False
        elif failure == "chat":
            endpoint.status = 404
        elif failure == "structured":
            endpoint.structured_status = 400
        elif failure == "wrong_shape":
            endpoint.structured_content = '{"ok":"not a boolean"}'
        elif failure == "malformed":
            endpoint.structured_content = "{not-json"
        else:
            endpoint.delay_seconds = 0.1
        config = project(
            endpoint, declared=capabilities, timeout=0.01 if failure == "timeout" else 1
        )
        started = False

        async def bootstrap() -> None:
            nonlocal started
            settings = prepare_settings(config, _CREDENTIALS)
            await preflight_models(config, settings)
            started = True

        with pytest.raises(ConfigurationError) as error:
            asyncio.run(bootstrap())
        assert not started
        assert "secret-from-provider-body" not in str(error.value)
        assert "selected-secret" not in str(error.value)
        if failure == "declaration":
            assert endpoint.requests == []
        else:
            assert endpoint.requests


def test_separate_evaluation_model_uses_its_own_credential_and_model() -> None:
    with serve() as persona, serve() as evaluation:
        config = project(persona, evaluation=evaluation)
        result = run(
            config,
            {**_CREDENTIALS, "EVAL_MODEL_TOKEN": "evaluation-secret"},
        )
        assert set(result.verified_capabilities) == {"persona", "evaluation"}
        assert len(persona.requests) == len(evaluation.requests) == 2
        for request in persona.requests:
            assert request["headers"]["Authorization"] == "Bearer selected-secret"
            assert request["body"]["model"] == "local-persona"
        for request in evaluation.requests:
            assert request["headers"]["Authorization"] == "Bearer evaluation-secret"
            assert request["body"]["model"] == "local-evaluator"


def test_unused_evaluation_model_is_not_probed() -> None:
    with serve() as persona, serve() as evaluation:
        config = project(persona, evaluation=evaluation, semantic=False)
        result = run(config, {**_CREDENTIALS, "EVAL_MODEL_TOKEN": "evaluation-secret"})
        assert set(result.verified_capabilities) == {"persona"}
        assert evaluation.requests == []
