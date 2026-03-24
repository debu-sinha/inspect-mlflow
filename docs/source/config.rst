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

API Reference
-------------

.. automodule:: inspect_mlflow.config
   :members:
   :undoc-members:
