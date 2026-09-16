# synthetic-cx

Open-source framework for simulating synthetic customers and evaluating real conversational customer journeys.

## Local setup

SyntheticCX uses [uv](https://docs.astral.sh/uv/) and supports Python 3.11 through
3.13. The committed lockfile is the authoritative development environment.

```bash
uv sync --locked --group dev
uv run pytest
uv run ruff check .
uv run mypy src
```

## Build and verify a distribution

```bash
uv build
python3 -m venv /tmp/syntheticcx-wheel-check
/tmp/syntheticcx-wheel-check/bin/pip install dist/*.whl
/tmp/syntheticcx-wheel-check/bin/synthetic-cx --help
```
