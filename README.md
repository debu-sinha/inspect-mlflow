# inspect-mlflow

![logo](https://raw.githubusercontent.com/debu-sinha/inspect-mlflow/main/docs/images/logo.png)

[![CI](https://github.com/debu-sinha/inspect-mlflow/actions/workflows/ci.yml/badge.svg)](https://github.com/debu-sinha/inspect-mlflow/actions/workflows/ci.yml)
[![CodeQL](https://github.com/debu-sinha/inspect-mlflow/actions/workflows/codeql.yml/badge.svg)](https://github.com/debu-sinha/inspect-mlflow/actions/workflows/codeql.yml)
[![PyPI version](https://img.shields.io/pypi/v/inspect-mlflow.svg)](https://pypi.org/project/inspect-mlflow/)
[![Downloads](https://img.shields.io/pypi/dm/inspect-mlflow.svg)](https://pypi.org/project/inspect-mlflow/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://readthedocs.org/projects/inspect-mlflow/badge/?version=latest)](https://inspect-mlflow.readthedocs.io/)
[![GitHub stars](https://img.shields.io/github/stars/debu-sinha/inspect-mlflow?style=social)](https://github.com/debu-sinha/inspect-mlflow)

MLflow integration for [Inspect AI](https://inspect.aisi.org.uk/). Provides experiment tracking, execution tracing, and artifact logging for Inspect AI evaluations.

## Install

```bash
pip install inspect-mlflow
```

## Quick Start

Hooks auto-register via entry points when the package is installed. No code changes needed.

```bash
# Start MLflow server
mlflow server --port 5000

# Set env vars
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_INSPECT_TRACING="true"

# Run evals as usual. Both hooks activate automatically.
inspect eval my_task.py --model openai/gpt-4o
```

Then open http://localhost:5000 to see runs and traces.

## What it does

This package provides two hooks that run automatically during Inspect AI evaluations. Both hooks use the `MlflowClient` API for full isolation from user MLflow state (no global `mlflow.start_run` calls). Thread-safe for concurrent sample processing.

### Tracking Hook

Activated when `MLFLOW_TRACKING_URI` is set. Creates hierarchical MLflow runs with full evaluation telemetry.

**What gets logged:**

- Parent run per eval invocation with nested child runs per task
- Task configuration as parameters (model, dataset, solver, temperature, top_p, max_tokens)
- Per-sample scores as step metrics (accuracy, timing per sample)
- Aggregate metrics (total_samples, completed_samples, match/accuracy, match/stderr)
- Model token usage (input/output/total tokens per model)
- Real-time event counting (total_model_calls, total_tool_calls)
- Eval artifacts: per-sample results JSON + full eval log JSON
- Trace assessments: eval scores logged as MLflow assessments via `mlflow.log_feedback()`, visible in the Traces UI assessment column

**Task run showing 17 metrics and parameters from a tool-using eval:**

![Task run detail](https://raw.githubusercontent.com/debu-sinha/inspect-mlflow/main/docs/images/screenshot-02-task-run.png)

**Traces table with assessment column showing eval scores (match: AVG 1.0):**

![Trace assessments](https://raw.githubusercontent.com/debu-sinha/inspect-mlflow/main/docs/images/screenshot-assessments.png)

### Tracing Hook

Activated when `MLFLOW_INSPECT_TRACING=true` is also set. Maps eval execution to MLflow trace spans, giving you a visual debugging view of every model call, tool invocation, and scoring step.

**Span hierarchy:**

```
eval_run:98h4b4KN (CHAIN)
  task:task (CHAIN)
    sample:keAdeL1U (CHAIN)
      solvers (from SpanBeginEvent)
        use_tools (solver span)
          model:openai/gpt-4o-mini (LLM) - 5,167 tokens
          tool:calculator (TOOL) - args: {"expression": "47 * 89"}, result: "4183"
          model:openai/gpt-4o-mini (LLM) - 5,263 tokens
        generate (solver span)
          model:openai/gpt-4o-mini (LLM) - 182 tokens
      scorers (from SpanBeginEvent)
        match (scorer span)
          score (EVALUATOR) - value: C
    sample:HWl2wp2B (CHAIN)
      ...
```

**Each span type captures different data:**

| Span Type | Data Captured |
|-----------|------|
| CHAIN | eval run, task, and sample lifecycle with scores and timing |
| LLM | model name, input/output token counts, temperature, cache status, response text |
| TOOL | function name, arguments, result, working time, errors |
| EVALUATOR | score value, explanation, target |

**Traces list showing 3 eval runs (simple math + tool-using calculator eval):**

![Traces list](https://raw.githubusercontent.com/debu-sinha/inspect-mlflow/main/docs/images/screenshot-04-traces-list.png)

**Full span tree showing solver/scorer hierarchy with tool calls:**

![Span tree](https://raw.githubusercontent.com/debu-sinha/inspect-mlflow/main/docs/images/inspect-tracing-04-timeline.png)

**LLM span detail with model name, token counts, and response:**

![LLM detail](https://raw.githubusercontent.com/debu-sinha/inspect-mlflow/main/docs/images/inspect-tracing-05-model-expanded.png)

### Autolog

Autolog enables MLflow provider integrations at run start.
Supported providers are: `openai`, `anthropic`, `langchain`, `litellm`,
`mistral`, `groq`, `cohere`, `gemini`, `bedrock`.
Each provider is enabled only when both the MLflow flavor module and provider SDK are installed.

## Configuration

Configuration is loaded from environment variables. When `pydantic-settings` is installed (`pip install inspect-mlflow[config]`), settings are typed and validated with the `INSPECT_MLFLOW_` prefix. Without it, standard `os.getenv()` is used.

| Env var | Required | Default | Description |
|---------|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | Yes | - | MLflow server URL |
| `MLFLOW_EXPERIMENT_NAME` | No | `inspect_ai` | Experiment name |
| `MLFLOW_INSPECT_TRACING` | No | `false` | Enable execution tracing |
| `MLFLOW_INSPECT_LOG_ARTIFACTS` | No | `true` | Log eval artifacts |
| `INSPECT_MLFLOW_LOG_ARTIFACTS` | No | `true` | Same as above (new prefix, takes priority) |
| `INSPECT_MLFLOW_AUTOLOG_ENABLED` | No | `true` | Enable MLflow provider autolog integrations |
| `INSPECT_MLFLOW_AUTOLOG_MODELS` | No | `openai,anthropic,langchain,litellm` | CSV or JSON array of providers to autolog |

## Examples

### Basic eval (tracking + tracing)

```python
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate

# No special imports needed. Hooks auto-register on install.

task = Task(
    dataset=[
        Sample(input="What is 2 + 2?", target="4"),
        Sample(input="What is 3 * 5?", target="15"),
        Sample(input="What is 10 - 7?", target="3"),
    ],
    solver=generate(),
    scorer=match(),
)

logs = eval(task, model="openai/gpt-4o-mini")
# MLflow now has: runs with metrics + traces with span tree
```

### Eval with tool calls

```python
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import tool


@tool
def calculator():
    """Perform arithmetic calculations."""

    async def run(expression: str) -> str:
        """Evaluate a math expression.

        Args:
            expression: A math expression to evaluate, e.g. "47 * 89"
        """
        allowed = {"__builtins__": {}}
        return str(eval(expression, allowed))

    return run


task = Task(
    dataset=[
        Sample(
            input="Use the calculator to compute 47 * 89.",
            target="4183",
        ),
        Sample(
            input="Use the calculator to compute 1024 / 16.",
            target="64",
        ),
    ],
    solver=[use_tools([calculator()]), generate()],
    scorer=match(),
)

logs = eval(task, model="openai/gpt-4o-mini")
# Traces now include TOOL spans for each calculator() call
# with function name, arguments, and result
```

## Development

```bash
git clone https://github.com/debu-sinha/inspect-mlflow.git
cd inspect-mlflow
uv sync --group dev
uv run pre-commit install
uv run pytest tests/ -v
```

See [CONTRIBUTING.md](https://github.com/debu-sinha/inspect-mlflow/blob/main/CONTRIBUTING.md) for integration testing and PR guidelines.

## Related

- [Documentation](https://inspect-mlflow.readthedocs.io/) - Full API reference and usage guide
- [Inspect AI](https://inspect.aisi.org.uk/) - AI evaluation framework by UK AI Security Institute
- [MLflow](https://mlflow.org/) - ML experiment tracking and model management
- [Inspect AI hooks docs](https://inspect.aisi.org.uk/extensions.html#sec-hooks) - How hooks work
- [Issue #3547](https://github.com/UKGovernmentBEIS/inspect_ai/issues/3547) - Original proposal
- [Vector Institute inspect-mlflow](https://github.com/VectorInstitute/inspect-mlflow) - Related extension whose features are being consolidated here

## Contributors

- **Debu Sinha** - Creator and maintainer
- **Vector Institute / National Research Council of Canada (NRC)** - Autolog provider support, contributed on behalf of the Canadian AI Safety Institute (CAISI). Consolidated from [VectorInstitute/inspect-mlflow](https://github.com/VectorInstitute/inspect-mlflow).
