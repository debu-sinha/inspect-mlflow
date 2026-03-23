# Changelog

## 0.3.0 (2026-03-22)

- Log Inspect AI eval scores as MLflow trace assessments via `mlflow.log_feedback()`
- Scores appear in MLflow Traces UI assessment column with scorer name, value, and rationale
- Assessment source: CODE/inspect_ai (deterministic, not LLM-based)
- 42 unit tests

## 0.2.0 (2026-03-20)

- Scout import source: pull MLflow traces into Inspect Scout transcript databases
- Converts LLM spans to ModelEvents, tool spans to ToolEvents, score spans to ScoreEvents
- 40 unit tests

## 0.1.0 (2026-03-20)

Initial release.

- MLflow tracking hook: hierarchical runs, per-sample metrics, model usage, artifact logging
- MLflow tracing hook: execution span tree with model calls, tool calls, scoring
- Auto-registration via Inspect AI entry points
- 33 unit tests
