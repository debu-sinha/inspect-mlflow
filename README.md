# inspect-mlflow

MLflow integration for [Inspect AI](https://inspect.aisi.org.uk/). Provides experiment tracking, execution tracing, and artifact logging for Inspect AI evaluations.

## Install

```bash
pip install inspect-mlflow
```

## What it does

**Tracking hook** (activated when `MLFLOW_TRACKING_URI` is set):
- Hierarchical MLflow runs: parent run per eval, nested child runs per task
- Task configuration logged as parameters (model, dataset, solver, generate config)
- Per-sample scores as step metrics
- Model token usage (input/output/total per model)
- Real-time event counting (model calls, tool calls)
- Eval artifacts: per-sample results table + full eval log JSON

**Tracing hook** (activated when `MLFLOW_INSPECT_TRACING=true` is also set):
- Maps eval execution to MLflow trace spans
- Span hierarchy: eval_run -> task -> sample -> model calls, tool calls, scoring
- LLM spans with token counts, temperature, cache status, response text
- Tool spans with function name, arguments, result, errors
- Score spans with value, explanation, target
- Preserves Inspect AI's internal span hierarchy (solver/scorer nesting)

## Usage

```bash
# Start MLflow server
mlflow server --port 5000

# Set env vars
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_INSPECT_TRACING="true"  # optional, enables tracing

# Run evals as usual. Hooks auto-activate.
inspect eval my_task.py --model openai/gpt-4o
```

No code changes needed. The hooks register automatically when Inspect AI loads.

## Configuration

| Env var | Required | Default | Description |
|---------|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | Yes | - | MLflow server URL |
| `MLFLOW_EXPERIMENT_NAME` | No | `inspect_ai` | Experiment name |
| `MLFLOW_INSPECT_TRACING` | No | `false` | Enable execution tracing |
| `MLFLOW_INSPECT_LOG_ARTIFACTS` | No | `true` | Log eval artifacts |

## Related

- [Inspect AI](https://inspect.aisi.org.uk/) - AI evaluation framework
- [MLflow](https://mlflow.org/) - ML experiment tracking and model management
- [Inspect AI hooks docs](https://inspect.aisi.org.uk/extensions.html#sec-hooks) - How hooks work
