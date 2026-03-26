Tracking Hook
=============

Activated when ``MLFLOW_TRACKING_URI`` is set.

The tracking hook creates hierarchical MLflow runs mirroring the evaluation structure.
Uses ``MlflowClient`` API for full isolation from user MLflow state. Thread-safe
for concurrent sample processing.

Features
--------

- Parent run per eval invocation with nested child runs per task
- Task configuration logged as parameters
- Per-sample scores as step metrics
- Model token usage (input/output/total per model)
- Real-time event counting (model calls, tool calls)
- Eval artifacts: per-sample results JSON + full eval log JSON
- Additional rich table artifacts under ``inspect/*.json`` (tasks, samples,
  messages, sample scores, events, model usage)
- Trace assessments: eval scores logged via ``mlflow.log_feedback()``
- Optional provider autolog integration for LLM SDKs
- Async logging for reduced hook latency
- Thread-safe counters for concurrent samples

Configuration
-------------

.. list-table::
   :header-rows: 1

   * - Env var
     - Required
     - Default
     - Description
   * - ``MLFLOW_TRACKING_URI``
     - Yes
     - --
     - MLflow server URL
   * - ``MLFLOW_EXPERIMENT_NAME``
     - No
     - ``inspect_ai``
     - Experiment name
   * - ``MLFLOW_INSPECT_LOG_ARTIFACTS``
     - No
     - ``true``
     - Log eval artifacts
   * - ``INSPECT_MLFLOW_LOG_ARTIFACTS``
     - No
     - ``true``
     - Same as above (new prefix, takes priority)
   * - ``INSPECT_MLFLOW_AUTOLOG_ENABLED``
     - No
     - ``true``
     - Enable MLflow provider autolog integrations
   * - ``INSPECT_MLFLOW_AUTOLOG_MODELS``
     - No
     - ``openai,anthropic,langchain,litellm``
     - CSV or JSON array of providers to autolog

Supported provider integrations: ``openai``, ``anthropic``, ``langchain``, ``litellm``,
``mistral``, ``groq``, ``cohere``, ``gemini``, ``bedrock``.
Providers are enabled only when both the MLflow flavor module and provider SDK are present.

Artifacts
---------

With artifact logging enabled, the tracking hook writes the following artifacts:

- ``inspect/tasks.json``
- ``inspect/samples.json``
- ``inspect/messages.json``
- ``inspect/sample_scores.json``
- ``inspect/events.json``
- ``inspect/model_usage.json``
- ``sample_results/*.json``
- ``eval_logs/*.json``

API Reference
-------------

.. automodule:: inspect_mlflow.tracking
   :members:
   :undoc-members:
