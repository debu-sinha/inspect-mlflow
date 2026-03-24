"""Configuration for inspect-mlflow hooks.

Uses pydantic-settings when available for typed, validated config with the
INSPECT_MLFLOW_ prefix. Falls back to os.getenv() when pydantic-settings
is not installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings

    class MLflowSettings(BaseSettings):
        """Settings for the MLflow integration hooks."""

        model_config = {"env_prefix": "INSPECT_MLFLOW_"}

        tracking_uri: str | None = Field(
            default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI"),
        )
        experiment_name: str = Field(
            default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_NAME", "inspect_ai"),
        )
        tracing_enabled: bool = Field(
            default_factory=lambda: os.getenv("MLFLOW_INSPECT_TRACING", "").lower() == "true",
        )
        log_artifacts: bool = Field(default=True)

except ImportError:

    @dataclass
    class MLflowSettings:  # type: ignore[no-redef]
        """Fallback settings using os.getenv() when pydantic-settings is not installed."""

        tracking_uri: str | None = None
        experiment_name: str = "inspect_ai"
        tracing_enabled: bool = False
        log_artifacts: bool = True

        def __post_init__(self) -> None:
            self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            self.experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "inspect_ai")
            self.tracing_enabled = os.getenv("MLFLOW_INSPECT_TRACING", "").lower() == "true"
            # Support both old (MLFLOW_INSPECT_) and new (INSPECT_MLFLOW_) prefixes
            _artifacts = os.getenv(
                "INSPECT_MLFLOW_LOG_ARTIFACTS",
                os.getenv("MLFLOW_INSPECT_LOG_ARTIFACTS", "true"),
            )
            self.log_artifacts = _artifacts.lower() != "false"


def load_settings() -> MLflowSettings:
    return MLflowSettings()
