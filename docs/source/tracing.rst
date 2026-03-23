Tracing Hook
============

Activated when both ``MLFLOW_TRACKING_URI`` and ``MLFLOW_INSPECT_TRACING=true`` are set.

Maps evaluation execution to MLflow trace spans, giving users a visual debugging view
of every model call, tool invocation, and scoring step.

Span Types
----------

.. list-table::
   :header-rows: 1

   * - Span Type
     - Data Captured
   * - CHAIN
     - Eval run, task, and sample lifecycle with scores and timing
   * - LLM
     - Model name, token counts, temperature, cache status, response text
   * - TOOL
     - Function name, arguments, result, working time, errors
   * - EVALUATOR
     - Score value, explanation, target

Trace Assessments
-----------------

Eval scores are automatically logged as MLflow trace assessments via ``mlflow.log_feedback()``.
Scores appear in the MLflow Traces UI assessment column with the scorer name, value, and rationale.

Configuration
-------------

.. list-table::
   :header-rows: 1

   * - Env var
     - Required
     - Default
     - Description
   * - ``MLFLOW_INSPECT_TRACING``
     - Yes (in addition to MLFLOW_TRACKING_URI)
     - ``false``
     - Enable execution tracing

API Reference
-------------

.. automodule:: inspect_mlflow.tracing
   :members:
   :undoc-members:
