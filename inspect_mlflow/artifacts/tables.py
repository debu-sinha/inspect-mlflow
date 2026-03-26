"""Artifact table extraction and shaping helpers for MLflow tracking."""

from __future__ import annotations

import contextlib
import json
from collections import defaultdict
from typing import Any

from inspect_mlflow.util import score_to_numeric, truncate


def extract_inspect_table_rows(
    *,
    eval_id: str,
    task_name: str,
    log: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Build inspect table rows from eval log content."""
    tables: dict[str, list[dict[str, Any]]] = {
        "tasks": [],
        "samples": [],
        "messages": [],
        "sample_scores": [],
        "events": [],
        "model_usage": [],
    }

    eval_spec = obj_get(log, "eval")
    dataset = obj_get(eval_spec, "dataset")
    tables["tasks"].append(
        {
            "task_name": task_name,
            "eval_id": eval_id,
            "task_file": obj_get(eval_spec, "task_file"),
            "task_version": obj_get(eval_spec, "task_version"),
            "task_id": obj_get(eval_spec, "task_id"),
            "solver": to_string(obj_get(eval_spec, "solver")),
            "model": to_string(obj_get(eval_spec, "model")),
            "dataset": to_string(obj_get(dataset, "name")),
            "dataset_samples": obj_get(dataset, "samples"),
        }
    )

    samples = obj_get(log, "samples")
    if not isinstance(samples, list):
        return tables

    for sample in samples:
        sample_id = obj_get(sample, "id")
        scores = obj_get(sample, "scores")
        usage_map = obj_get(sample, "model_usage")
        usage_totals = sum_usage_map(usage_map)
        events = obj_get(sample, "events")

        sample_row: dict[str, Any] = {
            "task_name": task_name,
            "eval_id": eval_id,
            "sample_id": sample_id,
            "input": to_json(obj_get(sample, "input")),
            "target": to_json(obj_get(sample, "target")),
            "output": get_sample_output_text(sample),
            "scores": to_json(scores_to_dict(scores)),
            "events_count": len(events) if isinstance(events, list) else None,
            "total_time": obj_get(sample, "total_time"),
            "working_time": obj_get(sample, "working_time"),
            "error": to_json(obj_get(sample, "error")),
        }
        for key, value in usage_totals.items():
            sample_row[f"usage_{key}"] = value
        tables["samples"].append(sample_row)

        tables["sample_scores"].extend(
            extract_sample_score_rows(
                eval_id=eval_id,
                task_name=task_name,
                sample_id=sample_id,
                scores=scores,
            )
        )
        tables["messages"].extend(
            extract_message_rows(
                eval_id=eval_id,
                task_name=task_name,
                sample_id=sample_id,
                sample=sample,
            )
        )
        tables["events"].extend(
            extract_event_rows(
                eval_id=eval_id,
                task_name=task_name,
                sample_id=sample_id,
                sample=sample,
            )
        )
        tables["model_usage"].extend(
            extract_model_usage_rows(
                eval_id=eval_id,
                task_name=task_name,
                sample_id=sample_id,
                sample=sample,
            )
        )

    return tables


def extract_sample_score_rows(
    *,
    eval_id: str,
    task_name: str,
    sample_id: Any,
    scores: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if scores is None or not hasattr(scores, "items"):
        return rows

    for scorer_name, score in scores.items():
        raw_value = obj_get(score, "value")
        if raw_value is None:
            raw_value = score
        explanation = obj_get(score, "explanation")
        rows.append(
            {
                "task_name": task_name,
                "eval_id": eval_id,
                "sample_id": sample_id,
                "scorer": str(scorer_name),
                "raw_value": to_string(raw_value),
                "numeric_value": score_to_numeric(raw_value),
                "explanation": truncate(explanation, 500) if explanation else None,
            }
        )
    return rows


def extract_message_rows(
    *,
    eval_id: str,
    task_name: str,
    sample_id: Any,
    sample: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    messages = obj_get(sample, "messages")
    if not isinstance(messages, list):
        return rows

    for idx, message in enumerate(messages):
        rows.append(
            {
                "task_name": task_name,
                "eval_id": eval_id,
                "sample_id": sample_id,
                "message_index": idx,
                "role": obj_get(message, "role"),
                "source": obj_get(message, "source"),
                "content": to_json(obj_get(message, "content")),
                "tool_calls": to_json(obj_get(message, "tool_calls")),
                "tool_call_id": obj_get(message, "tool_call_id"),
                "model": obj_get(message, "model"),
                "stop_reason": obj_get(message, "stop_reason"),
            }
        )
    return rows


def extract_event_rows(
    *,
    eval_id: str,
    task_name: str,
    sample_id: Any,
    sample: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    events = obj_get(sample, "events")
    if not isinstance(events, list):
        return rows

    for idx, event in enumerate(events):
        event_type = obj_get(event, "event") or obj_get(event, "type")
        row: dict[str, Any] = {
            "task_name": task_name,
            "eval_id": eval_id,
            "sample_id": sample_id,
            "event_index": idx,
            "event_type": to_string(event_type),
            "timestamp": to_json(obj_get(event, "timestamp")),
        }

        if event_type == "model":
            row["model"] = obj_get(event, "model")
            output = obj_get(event, "output")
            row["completion"] = to_json(obj_get(output, "completion"))
            usage = usage_to_dict(obj_get(output, "usage"))
            for key, value in usage.items():
                row[f"usage_{key}"] = value
        elif event_type == "tool":
            row["tool_function"] = obj_get(event, "function")
            row["tool_arguments"] = to_json(obj_get(event, "arguments"))
            row["tool_result"] = to_json(obj_get(event, "result"))
            row["tool_error"] = to_json(obj_get(event, "error"))
        elif event_type == "error":
            row["error"] = to_json(obj_get(event, "error"))

        rows.append(row)
    return rows


def extract_model_usage_rows(
    *,
    eval_id: str,
    task_name: str,
    sample_id: Any,
    sample: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    usage_map = obj_get(sample, "model_usage")

    if isinstance(usage_map, dict) and usage_map:
        for model_name, usage in usage_map.items():
            usage_dict = usage_to_dict(usage)
            if not usage_dict:
                continue
            rows.append(
                {
                    "task_name": task_name,
                    "eval_id": eval_id,
                    "sample_id": sample_id,
                    "model": to_string(model_name),
                    **usage_dict,
                }
            )
        return rows

    usage_from_events = extract_usage_from_events(sample)
    for model_name, usage_dict in usage_from_events.items():
        rows.append(
            {
                "task_name": task_name,
                "eval_id": eval_id,
                "sample_id": sample_id,
                "model": model_name,
                **usage_dict,
            }
        )
    return rows


def extract_usage_from_events(sample: Any) -> dict[str, dict[str, int]]:
    events = obj_get(sample, "events")
    if not isinstance(events, list):
        return {}

    usage_by_model: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for event in events:
        event_type = obj_get(event, "event") or obj_get(event, "type")
        if event_type != "model":
            continue
        output = obj_get(event, "output")
        usage = usage_to_dict(obj_get(output, "usage"))
        if not usage:
            continue
        model_name = (
            to_string(obj_get(event, "model"))
            or to_string(obj_get(output, "model"))
            or "unknown"
        )
        model_usage = usage_by_model[model_name]
        for key, value in usage.items():
            model_usage[key] = int(model_usage.get(key, 0)) + int(value)

    return {
        model: {key: int(value) for key, value in usage.items()}
        for model, usage in usage_by_model.items()
    }


def sum_usage_map(usage_map: Any) -> dict[str, int]:
    if not isinstance(usage_map, dict):
        return {}

    totals: dict[str, int] = defaultdict(int)
    for usage in usage_map.values():
        usage_dict = usage_to_dict(usage)
        for key, value in usage_dict.items():
            totals[key] += int(value)
    return dict(totals)


def scores_to_dict(scores: Any) -> dict[str, Any]:
    if scores is None or not hasattr(scores, "items"):
        return {}

    out: dict[str, Any] = {}
    for scorer_name, score in scores.items():
        value = obj_get(score, "value")
        if value is None:
            value = score
        out[str(scorer_name)] = value
    return out


def get_sample_output_text(sample: Any) -> str | None:
    output = obj_get(sample, "output")
    if output is None:
        return None

    choices = obj_get(output, "choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        message = obj_get(first_choice, "message")
        message_text = obj_get(message, "text") or obj_get(message, "content")
        if message_text is not None:
            return truncate(message_text, 500)
        choice_text = obj_get(first_choice, "text") or obj_get(first_choice, "completion")
        if choice_text is not None:
            return truncate(choice_text, 500)

    output_text = obj_get(output, "completion") or obj_get(output, "text")
    if output_text is not None:
        return truncate(output_text, 500)

    serialized = to_json(output)
    return truncate(serialized, 500) if serialized is not None else None


def usage_to_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}

    usage_data: dict[str, Any] = {}
    if isinstance(usage, dict):
        usage_data = dict(usage)
    elif hasattr(usage, "model_dump"):
        with contextlib.suppress(Exception):
            dumped = usage.model_dump(exclude_none=True)
            if isinstance(dumped, dict):
                usage_data = dumped

    known_keys = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_tokens_cache_read",
        "output_tokens_cache_write",
        "reasoning_tokens",
    )
    for key in known_keys:
        with contextlib.suppress(Exception):
            value = getattr(usage, key)
            if value is not None and key not in usage_data:
                usage_data[key] = value

    result: dict[str, int] = {}
    for key, value in usage_data.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            result[str(key)] = int(value)
    return result


def rows_to_columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    columns: dict[str, list[Any]] = {}
    for row in rows:
        for key in row:
            columns.setdefault(str(key), [])
    for row in rows:
        for key in columns:
            columns[key].append(row.get(key))
    return columns


def obj_get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    with contextlib.suppress(Exception):
        return getattr(obj, key)
    return None


def to_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def to_json(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        with contextlib.suppress(Exception):
            dumped = value.model_dump(mode="json")
            return json.dumps(dumped, default=str)
    with contextlib.suppress(Exception):
        return json.dumps(value, default=str)
    return str(value)
