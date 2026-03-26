# Changelog

## 0.7.0 (2026-03-26)

- Evaluation comparison and regression detection: `compare_evals()` aligns samples by (id, epoch), computes score deltas, runs significance tests (bootstrap CI for continuous, McNemar's for binary), and reports effect size (Cohen's d)
- Statistical tests implemented with NumPy only (no scipy dependency)
- Sample filtering, regression threshold, and win rate tracking
- 100+ unit tests

## 0.6.0 (2026-03-26)

- Structured artifact tables: tasks, samples, messages, sample_scores, events, model_usage logged as `inspect/*.json` via `client.log_table()`. Contributed by **Farnaz Kohankhaki** (Vector Institute / NRC Canada). PR #10.
- Artifact manager refactor: extraction logic in `artifacts/tables.py`, logging in `artifacts/manager.py`
- Fallback to full eval log when task-end samples are absent
- 65 unit tests

## 0.5.0 (2026-03-25)

- LLM provider autolog: openai, anthropic, langchain, litellm, mistral, groq, cohere, gemini, bedrock. Dependency gating ensures providers are only enabled when both the MLflow flavor and provider SDK are installed. Contributed by **Farnaz Kohankhaki** (Vector Institute / NRC Canada). PR #6.
- Autolog config: `INSPECT_MLFLOW_AUTOLOG_ENABLED`, `INSPECT_MLFLOW_AUTOLOG_MODELS` (CSV or JSON array)
- 60 unit tests

## 0.4.2 (2026-03-24)

- Migrated tracking hook to MlflowClient API (no global mlflow.start_run calls)
- Thread-safe counters with threading.Lock
- Async logging enabled
- Tests rewritten with real SQLite store (no mocks on MLflow API)
- Python requirement lowered to 3.10+

## 0.4.0 (2026-03-24)

- Typed configuration via pydantic-settings with dataclass fallback
- Both MLFLOW_ and INSPECT_MLFLOW_ env var prefixes supported
- Lazy settings loading
- ReadTheDocs documentation with config page

## 0.3.1 (2026-03-23)

- Trace assessments via mlflow.log_feedback()

## 0.3.0 (2026-03-22)

- Log Inspect AI eval scores as MLflow trace assessments via mlflow.log_feedback()
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
