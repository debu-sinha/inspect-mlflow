Scout Import
============

Import MLflow traces into Inspect Scout transcript databases for safety scanner analysis.

Requires the optional ``scout`` extra:

.. code-block:: bash

   pip install 'inspect-mlflow[scout]'

Usage
-----

.. code-block:: python

   from inspect_mlflow.scout import import_mlflow_traces
   from inspect_scout import transcripts_db

   async with transcripts_db("./safety-analysis") as db:
       await db.insert(import_mlflow_traces(
           experiment_name="my-evals",
           tracking_uri="http://localhost:5000",
       ))

Data Mapping
------------

.. list-table::
   :header-rows: 1

   * - MLflow Span
     - Scout Event
   * - LLM span
     - ModelEvent (model, tokens, timing)
   * - TOOL span
     - ToolEvent (function, arguments, result)
   * - EVALUATOR span
     - ScoreEvent (value, explanation, target)

API Reference
-------------

.. autofunction:: inspect_mlflow.scout.import_mlflow_traces
