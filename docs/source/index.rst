inspect-mlflow
==============

MLflow integration for `Inspect AI <https://inspect.aisi.org.uk/>`_.
Provides experiment tracking, execution tracing, trace assessments, and Scout analysis
for Inspect AI evaluations.

.. code-block:: bash

   pip install inspect-mlflow

Set environment variables and run evals as usual. Hooks auto-register via entry points.

.. code-block:: bash

   export MLFLOW_TRACKING_URI="http://localhost:5000"
   export MLFLOW_INSPECT_TRACING="true"
   inspect eval my_task.py --model openai/gpt-4o

.. toctree::
   :maxdepth: 2
   :caption: Contents

   tracking
   tracing
   scout
   api

API Reference
-------------

.. automodule:: inspect_mlflow
   :members:
   :undoc-members:
