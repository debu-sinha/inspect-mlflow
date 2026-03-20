"""Hook registration for Inspect AI.

This module is the entry point target declared in pyproject.toml.
Inspect AI imports it automatically when the package is installed,
which triggers the @hooks decorators and registers the hooks.
"""

from inspect_mlflow.tracing import MlflowTracingHooks  # noqa: F401
from inspect_mlflow.tracking import MlflowTrackingHooks  # noqa: F401
