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

API Reference
-------------

.. automodule:: inspect_mlflow.tracking
   :members:
   :undoc-members:
