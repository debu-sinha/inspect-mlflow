Configuration
=============

Settings are loaded from environment variables. When ``pydantic-settings`` is
installed (``pip install inspect-mlflow[config]``), settings are typed and
validated. Without it, a ``dataclass`` fallback reads the same environment
variables.

Both ``MLFLOW_`` and ``INSPECT_MLFLOW_`` prefixes are supported. The
``INSPECT_MLFLOW_`` prefix takes priority when both are set.

The corresponding overrides are ``INSPECT_MLFLOW_TRACKING_URI``,
``INSPECT_MLFLOW_EXPERIMENT_NAME``, and ``INSPECT_MLFLOW_TRACING_ENABLED``.
Boolean values accept ``true/false``, ``1/0``, ``yes/no``, and ``on/off``;
invalid values raise a configuration error.

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
   * - ``INSPECT_MLFLOW_PARENT_RUN_ID``
     - --
     - Log into this existing run instead of creating a parent run
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

When ``INSPECT_MLFLOW_PARENT_RUN_ID`` is set, the tracking hook nests its task runs
under that run instead of creating a parent of its own. It takes the experiment from
the supplied run rather than from ``MLFLOW_EXPERIMENT_NAME``, and leaves the run for
its owner to terminate.

Autolog support map includes ``openai``, ``anthropic``, ``langchain``, ``litellm``,
``mistral``, ``groq``, ``cohere``, ``gemini``, and ``bedrock``.
Each provider is enabled only when both the corresponding MLflow flavor module and
provider SDK are available in the environment.

API Reference
-------------

See :doc:`api` for the complete API reference.
