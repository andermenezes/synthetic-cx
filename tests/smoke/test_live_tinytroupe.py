import os

import pytest


def test_live_smoke_requires_explicit_opt_in_and_selected_credential() -> None:
    """Keep credentialed provider work out of ordinary local and CI checks."""
    if os.getenv("SYNTHETICCX_RUN_LIVE_SMOKE") != "1":
        pytest.skip("set SYNTHETICCX_RUN_LIVE_SMOKE=1 to enable the live smoke test")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required after explicit live-smoke opt-in")

    pytest.skip("Live TinyTroupe interaction is implemented in Phase 6, not Phase 1.")
