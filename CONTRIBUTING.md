# Contributing

## Development Setup

```bash
git clone https://github.com/debu-sinha/inspect-mlflow.git
cd inspect-mlflow
uv sync --locked --group dev --all-extras --group docs --python 3.10
uv run --no-sync pre-commit install
```

## Running Tests

```bash
uv run --no-sync pytest tests/ -v
uv run --no-sync pytest tests/ --cov=inspect_mlflow --cov-report=term-missing

# Run specific test modules
uv run --no-sync pytest tests/test_comparison.py -v    # comparison module
uv run --no-sync pytest tests/test_tracking.py -v      # tracking hook
uv run --no-sync pytest tests/test_tracing.py -v       # tracing hook
```

## Linting

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format .
uv run --no-sync mypy inspect_mlflow/
```

## Pre-commit

Pre-commit hooks run automatically on `git commit`. To run manually:

```bash
uv run --no-sync pre-commit run --all-files
```

## Integration Testing

The normal test suite includes `tests/test_end_to_end.py`. It runs the installed
Inspect hooks in a fresh process against SQLite MLflow, exercises model and tool
events, checks run metrics/artifacts/traces, and imports the trace with Scout.
It uses a deterministic Inspect model and requires no external API credentials.
Keep all extras installed so optional integrations are exercised as well.

To test with a real MLflow server and OpenAI API:

```bash
mlflow server --port 5556
export MLFLOW_TRACKING_URI="http://127.0.0.1:5556"
export MLFLOW_INSPECT_TRACING="true"
export OPENAI_API_KEY="sk-..."

uv run --no-sync python -c "
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate

task = Task(
    dataset=[Sample(input='What is 2+2?', target='4')],
    solver=generate(),
    scorer=match(),
)
eval(task, model='openai/gpt-4o-mini')
"
```

Open http://127.0.0.1:5556 to see runs and traces.

## Building Docs

```bash
uv run --no-sync sphinx-build -W --keep-going -b html docs/source docs/build
open docs/build/index.html
```

## Pull Requests

- Keep PRs focused on a single change
- Include tests for new functionality
- Run `uv run --no-sync pre-commit run --all-files` before pushing
- All CI checks must pass before merge
