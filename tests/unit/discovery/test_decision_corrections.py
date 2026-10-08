"""Malformed output is corrected within the user budget, without any browser action."""

import asyncio
import json

import pytest

from syntheticcx.config.models import PrivacyConfig
from syntheticcx.discovery.explore import explore
from syntheticcx.errors import ModelError
from syntheticcx.privacy import Redactor


@pytest.mark.parametrize("invalid", ["json", "missing", "unknown", "wrong_kind"])
def test_invalid_decision_is_corrected_without_extra_open_or_submission(invalid):
    requests = []
    progress = []

    class Browser:
        async def navigate(self, _url):
            return '- textbox "Message" [ref=e1]\n- button "Send" [ref=e2]'

        async def inspect_chat(self):
            return "{}"

        async def click(self, *_args):
            pytest.fail("schema corrections cannot click a control")

    class Provider:
        async def generate(self, request):
            requests.append(request)
            catalog, _ = json.JSONDecoder().raw_decode(
                request.messages[-1].content.split("Current candidate catalog:\n")[1]
            )
            payload = dict(
                action="propose",
                revision=catalog["revision"],
                target_id=None,
                input_id="input-1",
                frame_id=None,
                send_id="send-1",
                send_mode="click",
            )
            if len(requests) == 1:
                if invalid == "json":
                    return type("Response", (), {"content": "{PRIVATE_VALUE"})()
                if invalid == "missing":
                    payload.pop("input_id")
                elif invalid == "unknown":
                    payload["input_id"] = "PRIVATE_VALUE"
                else:
                    payload["input_id"] = "send-1"
            return type("Response", (), {"content": json.dumps(payload)})()

    result = asyncio.run(
        explore(
            "https://example.test",
            Browser(),
            Provider(),
            Redactor(PrivacyConfig()),
            max_discovery_attempts=2,
            on_progress=progress.append,
        )
    )
    assert result.channel.input_locator.value == "textbox"
    assert len(requests) == 2
    assert "Discovery feedback:" in requests[1].messages[-1].content
    assert "PRIVATE_VALUE" not in requests[1].messages[-1].content
    assert "PRIVATE_VALUE" not in "".join(progress)
    if invalid == "missing":
        assert "check input_id" in "".join(progress)


def test_provider_failure_is_not_retried_as_malformed_output():
    class Browser:
        async def navigate(self, _url):
            return '- textbox "Message" [ref=e1]'

        async def inspect_chat(self):
            return "{}"

    class Provider:
        calls = 0

        async def generate(self, _request):
            self.calls += 1
            raise ModelError("generate", "model response was refused")

    provider = Provider()
    with pytest.raises(ModelError, match="refused"):
        asyncio.run(explore("https://example.test", Browser(), provider, Redactor(PrivacyConfig())))
    assert provider.calls == 1
