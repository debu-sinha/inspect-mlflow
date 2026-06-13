"""Shared utilities for MLflow hooks."""

from __future__ import annotations

import contextlib
from datetime import datetime
from typing import Any


def percentile(sorted_values: list[float], q: float) -> float:
    """Return the q-th percentile (0.0-1.0) of a pre-sorted list.

    Uses linear interpolation between adjacent ranks (numpy's default
    method). The input is expected to be sorted ascending. Raises
    ValueError on an empty input rather than returning a misleading
    sentinel, since callers should guard the empty case explicitly.

    Args:
        sorted_values: Values sorted ascending. Must be non-empty.
        q: Quantile in [0.0, 1.0]. q=0.5 returns the median.

    Returns:
        The interpolated percentile value.
    """
    if not sorted_values:
        raise ValueError("Cannot compute percentile of an empty sequence.")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"Quantile q must be in [0.0, 1.0], got {q}.")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = q * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def parse_iso8601(value: Any) -> datetime | None:
    """Parse an ISO 8601 datetime string from inspect_ai logs.

    inspect_ai serializes EvalStats.started_at and EvalStats.completed_at
    as ISO 8601 strings with timezone offsets (e.g. "2026-06-13T14:32:11.123+00:00").
    Returns None on parse failure rather than raising, since this is used
    in metric-aggregation paths where a malformed timestamp should drop
    the metric rather than fail the entire eval logging.

    Args:
        value: An ISO 8601 string, a datetime object, or None.

    Returns:
        A datetime, or None if parsing failed.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    with contextlib.suppress(ValueError, TypeError):
        return datetime.fromisoformat(value)
    return None


def safe_log_params(mlflow: Any, params: dict[str, Any]) -> None:
    """Log params, truncating values that exceed MLflow's 500-char limit."""
    for key, value in params.items():
        str_val = str(value)
        if len(str_val) > 500:
            str_val = str_val[:497] + "..."
        with contextlib.suppress(Exception):
            mlflow.log_param(key, str_val)


def score_to_numeric(value: Any) -> float | None:
    """Convert a Score value to a numeric value for MLflow metrics.

    Handles Inspect AI score conventions:
    - int/float: returned as-is
    - bool: True -> 1.0, False -> 0.0
    - str: "C"/"correct" -> 1.0, "I"/"incorrect" -> 0.0, "P"/"partial" -> 0.5
    - other: None (metric skipped)
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        mapping = {
            "C": 1.0,
            "correct": 1.0,
            "I": 0.0,
            "incorrect": 0.0,
            "P": 0.5,
            "partial": 0.5,
        }
        return mapping.get(value)
    return None


def truncate(text: Any, max_len: int = 200) -> str:
    """Truncate text to max_len characters, adding ellipsis if truncated."""
    s = str(text) if text is not None else ""
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s
