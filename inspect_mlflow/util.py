"""Shared utilities for MLflow hooks."""

from __future__ import annotations

import contextlib
from typing import Any


def safe_log_params(mlflow: Any, params: dict[str, Any]) -> None:
    """Log params, truncating values that exceed MLflow's 500-char limit."""
    for key, value in params.items():
        str_val = str(value)
        if len(str_val) > 500:
            str_val = str_val[:497] + "..."
        with contextlib.suppress(Exception):
            mlflow.log_param(key, str_val)


def score_to_numeric(value: Any) -> float | None:
    """Convert a Score value to a numeric value for MLflow metrics."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        mapping = {"C": 1.0, "I": 0.0, "P": 0.5, "correct": 1.0, "incorrect": 0.0}
        return mapping.get(value)
    return None


def truncate(text: Any, max_len: int = 200) -> str:
    """Truncate text to max_len characters, adding ellipsis if truncated."""
    s = str(text) if text is not None else ""
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s
