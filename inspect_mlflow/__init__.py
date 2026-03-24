"""MLflow integration for Inspect AI.

Provides experiment tracking, execution tracing, and Scout analysis
for Inspect AI evaluations via MLflow.

Install and use:

    pip install inspect-mlflow

    # Set env vars
    export MLFLOW_TRACKING_URI="http://localhost:5000"
    export MLFLOW_INSPECT_TRACING="true"  # optional, enables tracing

    # Run evals as usual. Hooks auto-activate.
    inspect eval my_task.py
"""

__version__ = "0.4.0"

import contextlib

from inspect_mlflow.tracing import MlflowTracingHooks
from inspect_mlflow.tracking import MlflowTrackingHooks

__all__ = [
    "MlflowTracingHooks",
    "MlflowTrackingHooks",
    "import_mlflow_traces",
]

with contextlib.suppress(ImportError):
    from inspect_mlflow.scout import import_mlflow_traces
