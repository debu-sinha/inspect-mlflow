Configuration
=============

Settings are loaded from environment variables. When ``pydantic-settings`` is
installed (``pip install inspect-mlflow[config]``), settings are typed and
validated. Without it, a ``dataclass`` fallback reads the same environment
variables.

Both ``MLFLOW_`` and ``INSPECT_MLFLOW_`` prefixes are supported. The
``INSPECT_MLFLOW_`` prefix takes priority when both are set.

.. list-table::
   :header-rows: 1

   * - Env var
     - Default
     - Description
   * - ``MLFLOW_TRACKING_URI``
     - --
     - MLflow server URL (required for tracking hook)
   * - ``MLFLOW_EXPERIMENT_NAME``
     - ``inspect_ai``
     - Experiment name
   * - ``MLFLOW_INSPECT_TRACING``
     - ``false``
     - Enable execution tracing
   * - ``MLFLOW_INSPECT_LOG_ARTIFACTS``
     - ``true``
     - Log eval artifacts
   * - ``INSPECT_MLFLOW_LOG_ARTIFACTS``
     - ``true``
     - Same as above (new prefix, takes priority)
   * - ``INSPECT_MLFLOW_AUTOLOG_ENABLED``
     - ``true``
     - Enable MLflow provider autolog integrations
   * - ``INSPECT_MLFLOW_AUTOLOG_MODELS``
     - ``openai,anthropic,langchain,litellm``
     - CSV or JSON array of providers to autolog

Autolog support map includes ``openai``, ``anthropic``, ``langchain``, ``litellm``,
``mistral``, ``groq``, ``cohere``, ``gemini``, and ``bedrock``.
Each provider is enabled only when both the corresponding MLflow flavor module and
provider SDK are available in the environment.

API Reference
-------------

.. automodule:: inspect_mlflow.config
   :members:
   :undoc-members:
