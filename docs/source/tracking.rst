Tracking Hook
=============

Activated when ``MLFLOW_TRACKING_URI`` is set.

The tracking hook creates hierarchical MLflow runs mirroring the evaluation structure.

Features
--------

- Parent run per eval invocation with nested child runs per task
- Task configuration logged as parameters
- Per-sample scores as step metrics
- Model token usage (input/output/total per model)
- Real-time event counting (model calls, tool calls)
- Eval artifacts: per-sample results JSON + full eval log JSON
- Trace assessments: eval scores logged via ``mlflow.log_feedback()``

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

API Reference
-------------

.. automodule:: inspect_mlflow.tracking
   :members:
   :undoc-members:
