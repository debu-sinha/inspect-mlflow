# Changelog

## 0.8.0 (2026-06-13)

### Added

- **Cost and latency tracking per evaluation** (closes #22). The tracking hook now surfaces three new families of metrics derived from inspect_ai data already on `EvalLog`:
  - `usage/<model>/total_cost_usd` and `cost/total_usd` when inspect_ai's `ModelUsage.total_cost` is populated by the provider integration.
  - `latency/per_sample_mean_seconds`, `latency/per_sample_p50_seconds`, `latency/per_sample_p95_seconds`, and `latency/per_sample_working_mean_seconds` aggregated from `EvalSample.total_time` / `working_time`.
  - `latency/total_seconds` from `EvalStats.started_at` and `completed_at`.
- **Cost and latency deltas in `compare_evals`**. `ComparisonResult` gains six new optional fields: `baseline_total_cost_usd`, `candidate_total_cost_usd`, `cost_delta_usd`, `baseline_latency_p95_seconds`, `candidate_latency_p95_seconds`, `latency_p95_delta_seconds`. `ComparisonResult.summary()` includes new `Cost:` and `Latency p95:` lines when the underlying logs carry the data.
- New `inspect_mlflow.util.percentile()` and `inspect_mlflow.util.parse_iso8601()` helpers (public).

### Behavior

- Cost metrics are **omitted (not zero)** when the underlying inspect_ai provider does not surface a cost estimate. This avoids the misleading appearance that an eval was free when the cost simply was not computed.
- All new fields on `ComparisonResult` default to `None`. Existing consumers see no behavior change.

### Tests

- 10 new tests: 4 in `test_tracking.py` covering per-model cost logging, cost omission when `total_cost` is `None`, per-sample latency percentiles, and `latency/total_seconds`; 6 in `test_comparison.py` covering cost / latency delta population, None handling, and the new `summary()` lines.
- 112 / 112 unit tests pass against `inspect-ai==0.3.236` and `mlflow==3.13.0`.

### Verified

- End-to-end run against `openai/gpt-4o-mini` confirmed the new latency metrics land in the MLflow UI alongside existing token usage metrics. Screenshot: `docs/screenshots/v0.8.0-run-detail.png`.

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
