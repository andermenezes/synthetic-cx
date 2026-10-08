"""Run the canonical billing scenario against a temporary local fixture target."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from webchat_fixture.server import FixtureServer

from syntheticcx.config.models import ProjectConfig

CANONICAL = Path(__file__).with_name("scenario.yaml")


def configured_example(
    fixture_url: str, *, variant: str, persona: str, run_root: Path
) -> dict[str, object]:
    """Derive one strict v1 document without altering the authoritative example."""
    source = yaml.safe_load(CANONICAL.read_text(encoding="utf-8"))
    # The named iframe is the fixture's independently attested chat boundary.
    source["channel"]["url"] = f"{fixture_url}/?variant={variant}&view=frame"
    source["storage"]["run_root"] = str(run_root)
    if persona == "impatient":
        source["persona"]["id"] = "impatient-customer"
        source["persona"]["name"] = "Alex Impatient"
        source["persona"]["initial_state"]["willingness_to_continue"] = 0.3
        source["persona"]["behavioral_rules"][0].update(
            frustration_delta=0.5,
            willingness_to_continue_delta=-0.3,
            abandonment_threshold=0.1,
        )
    ProjectConfig.model_validate(source)
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("success", "repeated_question"), default="success")
    parser.add_argument("--persona", choices=("standard", "impatient"), default="standard")
    parser.add_argument("--run-root", type=Path, default=Path(".runs"))
    arguments = parser.parse_args()

    with (
        FixtureServer() as fixture,
        tempfile.TemporaryDirectory(prefix="syntheticcx-demo-") as temporary,
    ):
        config = configured_example(
            fixture.base_url,
            variant=arguments.variant,
            persona=arguments.persona,
            run_root=arguments.run_root,
        )
        path = Path(temporary) / "scenario.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, "-m", "syntheticcx", "run", str(path)],
            check=False,
        )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
