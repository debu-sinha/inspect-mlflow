"""Configuration for inspect-mlflow hooks.

Uses pydantic-settings when available for typed, validated config with the
``INSPECT_MLFLOW_`` prefix. Falls back to os.getenv() when pydantic-settings
is not installed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Annotated, Any

DEFAULT_AUTOLOG_MODELS = ["openai", "anthropic", "langchain", "litellm"]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y", "on", "t"}:
        return True
    if value in {"false", "0", "no", "n", "off", "f"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


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
            default_factory=lambda: _env_bool("MLFLOW_INSPECT_TRACING", False),
        )
        log_artifacts: bool = Field(
            default_factory=lambda: _env_bool("MLFLOW_INSPECT_LOG_ARTIFACTS", True)
        )
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
            self.tracking_uri = os.getenv(
                "INSPECT_MLFLOW_TRACKING_URI", os.getenv("MLFLOW_TRACKING_URI")
            )
            self.experiment_name = os.getenv(
                "INSPECT_MLFLOW_EXPERIMENT_NAME", os.getenv("MLFLOW_EXPERIMENT_NAME", "inspect_ai")
            )
            self.tracing_enabled = (
                _env_bool("INSPECT_MLFLOW_TRACING_ENABLED", False)
                if "INSPECT_MLFLOW_TRACING_ENABLED" in os.environ
                else _env_bool("MLFLOW_INSPECT_TRACING", False)
            )
            # Support both old (MLFLOW_INSPECT_) and new (INSPECT_MLFLOW_) prefixes
            self.log_artifacts = (
                _env_bool("INSPECT_MLFLOW_LOG_ARTIFACTS", True)
                if "INSPECT_MLFLOW_LOG_ARTIFACTS" in os.environ
                else _env_bool("MLFLOW_INSPECT_LOG_ARTIFACTS", True)
            )
            self.autolog_enabled = _env_bool("INSPECT_MLFLOW_AUTOLOG_ENABLED", True)
            self.autolog_models = _parse_autolog_models(os.getenv("INSPECT_MLFLOW_AUTOLOG_MODELS"))


def load_settings() -> MLflowSettings:
    return MLflowSettings()
