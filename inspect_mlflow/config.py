"""Configuration for inspect-mlflow hooks.

Uses pydantic-settings when available for typed, validated config with the
INSPECT_MLFLOW_ prefix. Falls back to os.getenv() when pydantic-settings
is not installed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Annotated, Any

DEFAULT_AUTOLOG_MODELS = ["openai", "anthropic", "langchain", "litellm"]


def _parse_autolog_models(value: Any) -> list[str]:
    """Parse autolog model setting from CSV, JSON-array string, or list."""
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return DEFAULT_AUTOLOG_MODELS.copy()

        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                parsed_list = [str(item).strip().lower() for item in parsed if str(item).strip()]
                return parsed_list or DEFAULT_AUTOLOG_MODELS.copy()

        parsed_list = [item.strip().lower() for item in raw.split(",") if item.strip()]
        return parsed_list or DEFAULT_AUTOLOG_MODELS.copy()

    if isinstance(value, list):
        parsed_list = [str(item).strip().lower() for item in value if str(item).strip()]
        return parsed_list or DEFAULT_AUTOLOG_MODELS.copy()

    return DEFAULT_AUTOLOG_MODELS.copy()

try:
    from pydantic import Field, field_validator
    from pydantic_settings import BaseSettings, NoDecode

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
        autolog_enabled: bool = Field(default=True)
        autolog_models: Annotated[list[str], NoDecode] = Field(
            default_factory=lambda: DEFAULT_AUTOLOG_MODELS.copy()
        )

        @field_validator("autolog_models", mode="before")
        @classmethod
        def parse_autolog_models(cls, value: Any) -> list[str]:
            return _parse_autolog_models(value)

except ImportError:

    @dataclass
    class MLflowSettings:  # type: ignore[no-redef]
        """Fallback settings using os.getenv() when pydantic-settings is not installed."""

        tracking_uri: str | None = None
        experiment_name: str = "inspect_ai"
        tracing_enabled: bool = False
        log_artifacts: bool = True
        autolog_enabled: bool = True
        autolog_models: list[str] = field(default_factory=lambda: DEFAULT_AUTOLOG_MODELS.copy())

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
            _autolog_enabled = os.getenv("INSPECT_MLFLOW_AUTOLOG_ENABLED", "true")
            self.autolog_enabled = _autolog_enabled.lower() != "false"
            self.autolog_models = _parse_autolog_models(
                os.getenv("INSPECT_MLFLOW_AUTOLOG_MODELS")
            )


def load_settings() -> MLflowSettings:
    return MLflowSettings()
