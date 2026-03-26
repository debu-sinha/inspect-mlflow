"""MLflow Tracking hook for Inspect AI.

Logs evaluation runs, task configurations, sample scores, and model usage
to an MLflow tracking server. Creates a parent run per eval run with nested
child runs per task.

Uses MlflowClient API to avoid contaminating global mlflow state, so user
code that calls mlflow.start_run() independently will not conflict.

Activated automatically when MLFLOW_TRACKING_URI is set.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
from collections import defaultdict
from typing import Any

import mlflow
from inspect_ai.event._model import ModelEvent
from inspect_ai.event._tool import ToolEvent
from inspect_ai.hooks import (
    Hooks,
    ModelUsageData,
    RunEnd,
    RunStart,
    SampleEnd,
    SampleEvent,
    TaskEnd,
    TaskStart,
    hooks,
)
from inspect_ai.log import EvalSpec
from mlflow.tracking import MlflowClient

from inspect_mlflow._autolog import enable_autolog
from inspect_mlflow.config import MLflowSettings, load_settings
from inspect_mlflow.util import score_to_numeric, truncate

_logger = logging.getLogger(__name__)


def _safe_log_param(client: MlflowClient, run_id: str, key: str, value: Any) -> None:
    str_val = str(value)
    if len(str_val) > 500:
        str_val = str_val[:497] + "..."
    with contextlib.suppress(Exception):
        client.log_param(run_id, key, str_val)


def _safe_log_params(client: MlflowClient, run_id: str, params: dict[str, Any]) -> None:
    for key, value in params.items():
        _safe_log_param(client, run_id, key, value)


@hooks(name="mlflow_tracking", description="MLflow Tracking")
class MlflowTrackingHooks(Hooks):
    """Tracks Inspect AI evaluations in MLflow with hierarchical runs.

    Uses MlflowClient API for isolation from user mlflow state.
    """

    def __init__(self) -> None:
        self._client: MlflowClient | None = None
        self._experiment_id: str | None = None
        self._parent_run_id: str | None = None
        self._task_run_ids: dict[str, str] = {}
        self._tasks: dict[str, EvalSpec] = {}
        self._sample_counts: dict[str, int] = {}
        self._model_usage: dict[str, dict[str, float]] = {}
        self._event_counts: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()
        self._settings: MLflowSettings | None = None
        self._autolog_enabled = False

    @property
    def settings(self) -> MLflowSettings:
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    @property
    def client(self) -> MlflowClient:
        if self._client is None:
            self._client = MlflowClient()
        return self._client

    def enabled(self) -> bool:
        return load_settings().tracking_uri is not None

    async def on_run_start(self, data: RunStart) -> None:
        self._settings = load_settings()
        self._client = MlflowClient()
        self._autolog_enabled = False

        if self.settings.tracking_uri:
            with contextlib.suppress(Exception):
                mlflow.set_tracking_uri(self.settings.tracking_uri)

        experiment = self._client.get_experiment_by_name(self.settings.experiment_name)
        if experiment is None:
            self._experiment_id = self._client.create_experiment(self.settings.experiment_name)
        else:
            self._experiment_id = experiment.experiment_id

        with contextlib.suppress(Exception):
            mlflow.set_experiment(self.settings.experiment_name)

        # Enable async logging for reduced hook latency
        with contextlib.suppress(Exception):
            mlflow.config.enable_async_logging(True)

        if self.settings.autolog_enabled:
            self._enable_autolog(self.settings.autolog_models)

        run = self._client.create_run(
            experiment_id=self._experiment_id,
            run_name=f"inspect-{data.run_id[:8]}",
            tags={
                "inspect.run_id": data.run_id,
                "inspect.task_count": str(len(data.task_names)),
                "inspect.tasks": ", ".join(data.task_names),
            },
        )
        self._parent_run_id = run.info.run_id
        _logger.debug("Started parent run %s", self._parent_run_id)

    async def on_run_end(self, data: RunEnd) -> None:
        # End each task run by its specific run_id (not global stack)
        for _eval_id, run_id in list(self._task_run_ids.items()):
            try:
                self.client.set_terminated(run_id, status="FAILED")
            except Exception:
                _logger.debug("Failed to terminate task run %s", run_id, exc_info=True)

        if self._parent_run_id:
            status = "FAILED" if data.exception else "FINISHED"
            try:
                self.client.set_terminated(self._parent_run_id, status=status)
            except Exception:
                _logger.debug("Failed to terminate parent run", exc_info=True)
            self._parent_run_id = None

        if self._autolog_enabled:
            self._disable_autolog()

        self._task_run_ids.clear()
        self._tasks.clear()
        self._sample_counts.clear()
        self._model_usage.clear()
        self._event_counts.clear()

    def _enable_autolog(self, models: list[str]) -> None:
        enabled_any = enable_autolog(models)
        self._autolog_enabled = enabled_any
        if enabled_any:
            _logger.info("MLflow autolog enabled for: %s", models)

    def _disable_autolog(self) -> None:
        try:
            mlflow.autolog(disable=True)
        except Exception:
            _logger.debug("Could not disable autolog", exc_info=True)
        self._autolog_enabled = False

    async def on_task_start(self, data: TaskStart) -> None:
        self._tasks[data.eval_id] = data.spec
        with self._lock:
            self._sample_counts[data.eval_id] = 0

        if not self._parent_run_id:
            _logger.debug("No parent run, skipping task %s", data.spec.task)
            return

        assert self._experiment_id is not None, "experiment_id not set; on_run_start not called"
        run = self.client.create_run(
            experiment_id=self._experiment_id,
            run_name=data.spec.task,
            tags={
                "mlflow.parentRunId": self._parent_run_id,
                "inspect.eval_id": data.eval_id,
                "inspect.run_id": data.run_id,
                "inspect.task": data.spec.task,
                "inspect.model": data.spec.model,
            },
        )
        task_run_id = run.info.run_id
        self._task_run_ids[data.eval_id] = task_run_id

        _safe_log_params(
            self.client,
            task_run_id,
            {
                "task": data.spec.task,
                "model": data.spec.model,
                "task_version": str(data.spec.task_version),
                "dataset.name": data.spec.dataset.name or "",
                "dataset.samples": str(data.spec.dataset.samples or ""),
                "solver": data.spec.solver or "",
            },
        )

        if data.spec.task_args_passed:
            _safe_log_params(
                self.client,
                task_run_id,
                {f"task_arg.{k}": v for k, v in data.spec.task_args_passed.items()},
            )

        config = data.spec.model_generate_config
        gen_params: dict[str, Any] = {}
        if config.temperature is not None:
            gen_params["temperature"] = config.temperature
        if config.top_p is not None:
            gen_params["top_p"] = config.top_p
        if config.max_tokens is not None:
            gen_params["max_tokens"] = config.max_tokens
        if gen_params:
            _safe_log_params(self.client, task_run_id, gen_params)

        if data.spec.tags:
            _safe_log_param(self.client, task_run_id, "tags", ", ".join(data.spec.tags))

    async def on_task_end(self, data: TaskEnd) -> None:
        eval_id = data.eval_id
        task_run_id = self._task_run_ids.get(eval_id)
        if not task_run_id:
            return

        log = data.log

        if log.results and log.results.scores:
            for eval_score in log.results.scores:
                scorer_name = eval_score.name
                for metric_name, metric in eval_score.metrics.items():
                    metric_key = f"{scorer_name}/{metric_name}"
                    if isinstance(metric.value, (int, float)):
                        with contextlib.suppress(Exception):
                            self.client.log_metric(task_run_id, metric_key, float(metric.value))

        if log.results:
            try:
                self.client.log_metric(task_run_id, "total_samples", log.results.total_samples)
                self.client.log_metric(
                    task_run_id, "completed_samples", log.results.completed_samples
                )
            except Exception:
                pass

        if log.stats and log.stats.model_usage:
            for model_name, usage in log.stats.model_usage.items():
                prefix = f"usage/{model_name}"
                try:
                    self.client.log_metric(
                        task_run_id, f"{prefix}/input_tokens", usage.input_tokens
                    )
                    self.client.log_metric(
                        task_run_id, f"{prefix}/output_tokens", usage.output_tokens
                    )
                    self.client.log_metric(
                        task_run_id, f"{prefix}/total_tokens", usage.total_tokens
                    )
                except Exception:
                    pass

        with self._lock:
            event_counts = self._event_counts.get(eval_id, {})
        if event_counts:
            try:
                self.client.log_metric(
                    task_run_id, "total_model_calls", event_counts.get("model_calls", 0)
                )
                self.client.log_metric(
                    task_run_id, "total_tool_calls", event_counts.get("tool_calls", 0)
                )
            except Exception:
                pass

        if self.settings.log_artifacts:
            self._log_eval_artifacts(task_run_id, log)

        status = "FINISHED" if log.status == "success" else "FAILED"
        try:
            self.client.set_terminated(task_run_id, status=status)
        except Exception:
            _logger.debug("Failed to terminate task run %s", task_run_id, exc_info=True)

        self._task_run_ids.pop(eval_id, None)
        self._tasks.pop(eval_id, None)
        with self._lock:
            self._event_counts.pop(eval_id, None)

    def _log_eval_artifacts(self, run_id: str, log: Any) -> None:
        try:
            self._log_inspect_tables(run_id, log)
        except Exception:
            _logger.debug("Failed to log inspect table artifacts", exc_info=True)
        try:
            self._log_sample_table(run_id, log)
        except Exception:
            _logger.debug("Failed to log sample results artifact", exc_info=True)
        try:
            self._log_eval_json(run_id, log)
        except Exception:
            _logger.debug("Failed to log eval log artifact", exc_info=True)

    def _log_inspect_tables(self, run_id: str, log: Any) -> None:
        eval_id = self._obj_get(self._obj_get(log, "eval"), "eval_id") or "unknown"
        task_name = self._obj_get(self._obj_get(log, "eval"), "task") or "unknown"

        tables = self._extract_inspect_table_rows(
            eval_id=str(eval_id),
            task_name=str(task_name),
            log=log,
        )
        for name, rows in tables.items():
            if not rows:
                continue
            with contextlib.suppress(Exception):
                self.client.log_table(
                    run_id=run_id,
                    data=self._rows_to_columns(rows),
                    artifact_file=f"inspect/{name}.json",
                )

    def _extract_inspect_table_rows(
        self, *, eval_id: str, task_name: str, log: Any
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

        eval_spec = self._obj_get(log, "eval")
        dataset = self._obj_get(eval_spec, "dataset")
        tables["tasks"].append(
            {
                "task_name": task_name,
                "eval_id": eval_id,
                "task_file": self._obj_get(eval_spec, "task_file"),
                "task_version": self._obj_get(eval_spec, "task_version"),
                "task_id": self._obj_get(eval_spec, "task_id"),
                "solver": self._to_string(self._obj_get(eval_spec, "solver")),
                "model": self._to_string(self._obj_get(eval_spec, "model")),
                "dataset": self._to_string(self._obj_get(dataset, "name")),
                "dataset_samples": self._obj_get(dataset, "samples"),
            }
        )

        samples = self._obj_get(log, "samples")
        if not isinstance(samples, list):
            return tables

        for sample in samples:
            sample_id = self._obj_get(sample, "id")
            scores = self._obj_get(sample, "scores")
            usage_map = self._obj_get(sample, "model_usage")
            usage_totals = self._sum_usage_map(usage_map)
            events = self._obj_get(sample, "events")

            sample_row: dict[str, Any] = {
                "task_name": task_name,
                "eval_id": eval_id,
                "sample_id": sample_id,
                "input": self._to_json(self._obj_get(sample, "input")),
                "target": self._to_json(self._obj_get(sample, "target")),
                "output": self._get_sample_output_text(sample),
                "scores": self._to_json(self._scores_to_dict(scores)),
                "events_count": len(events) if isinstance(events, list) else None,
                "total_time": self._obj_get(sample, "total_time"),
                "working_time": self._obj_get(sample, "working_time"),
                "error": self._to_json(self._obj_get(sample, "error")),
            }
            for key, value in usage_totals.items():
                sample_row[f"usage_{key}"] = value
            tables["samples"].append(sample_row)

            tables["sample_scores"].extend(
                self._extract_sample_score_rows(
                    eval_id=eval_id,
                    task_name=task_name,
                    sample_id=sample_id,
                    scores=scores,
                )
            )
            tables["messages"].extend(
                self._extract_message_rows(
                    eval_id=eval_id,
                    task_name=task_name,
                    sample_id=sample_id,
                    sample=sample,
                )
            )
            tables["events"].extend(
                self._extract_event_rows(
                    eval_id=eval_id,
                    task_name=task_name,
                    sample_id=sample_id,
                    sample=sample,
                )
            )
            tables["model_usage"].extend(
                self._extract_model_usage_rows(
                    eval_id=eval_id,
                    task_name=task_name,
                    sample_id=sample_id,
                    sample=sample,
                )
            )

        return tables

    def _extract_sample_score_rows(
        self,
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
            raw_value = self._obj_get(score, "value")
            if raw_value is None:
                raw_value = score
            explanation = self._obj_get(score, "explanation")
            rows.append(
                {
                    "task_name": task_name,
                    "eval_id": eval_id,
                    "sample_id": sample_id,
                    "scorer": str(scorer_name),
                    "raw_value": self._to_string(raw_value),
                    "numeric_value": score_to_numeric(raw_value),
                    "explanation": truncate(explanation, 500) if explanation else None,
                }
            )
        return rows

    def _extract_message_rows(
        self,
        *,
        eval_id: str,
        task_name: str,
        sample_id: Any,
        sample: Any,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        messages = self._obj_get(sample, "messages")
        if not isinstance(messages, list):
            return rows

        for idx, message in enumerate(messages):
            rows.append(
                {
                    "task_name": task_name,
                    "eval_id": eval_id,
                    "sample_id": sample_id,
                    "message_index": idx,
                    "role": self._obj_get(message, "role"),
                    "source": self._obj_get(message, "source"),
                    "content": self._to_json(self._obj_get(message, "content")),
                    "tool_calls": self._to_json(self._obj_get(message, "tool_calls")),
                    "tool_call_id": self._obj_get(message, "tool_call_id"),
                    "model": self._obj_get(message, "model"),
                    "stop_reason": self._obj_get(message, "stop_reason"),
                }
            )
        return rows

    def _extract_event_rows(
        self,
        *,
        eval_id: str,
        task_name: str,
        sample_id: Any,
        sample: Any,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        events = self._obj_get(sample, "events")
        if not isinstance(events, list):
            return rows

        for idx, event in enumerate(events):
            event_type = self._obj_get(event, "event") or self._obj_get(event, "type")
            row: dict[str, Any] = {
                "task_name": task_name,
                "eval_id": eval_id,
                "sample_id": sample_id,
                "event_index": idx,
                "event_type": self._to_string(event_type),
                "timestamp": self._to_json(self._obj_get(event, "timestamp")),
            }

            if event_type == "model":
                row["model"] = self._obj_get(event, "model")
                output = self._obj_get(event, "output")
                row["completion"] = self._to_json(self._obj_get(output, "completion"))
                usage = self._usage_to_dict(self._obj_get(output, "usage"))
                for key, value in usage.items():
                    row[f"usage_{key}"] = value
            elif event_type == "tool":
                row["tool_function"] = self._obj_get(event, "function")
                row["tool_arguments"] = self._to_json(self._obj_get(event, "arguments"))
                row["tool_result"] = self._to_json(self._obj_get(event, "result"))
                row["tool_error"] = self._to_json(self._obj_get(event, "error"))
            elif event_type == "error":
                row["error"] = self._to_json(self._obj_get(event, "error"))

            rows.append(row)
        return rows

    def _extract_model_usage_rows(
        self,
        *,
        eval_id: str,
        task_name: str,
        sample_id: Any,
        sample: Any,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        usage_map = self._obj_get(sample, "model_usage")

        if isinstance(usage_map, dict) and usage_map:
            for model_name, usage in usage_map.items():
                usage_dict = self._usage_to_dict(usage)
                if not usage_dict:
                    continue
                rows.append(
                    {
                        "task_name": task_name,
                        "eval_id": eval_id,
                        "sample_id": sample_id,
                        "model": self._to_string(model_name),
                        **usage_dict,
                    }
                )
            return rows

        usage_from_events = self._extract_usage_from_events(sample)
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

    def _extract_usage_from_events(self, sample: Any) -> dict[str, dict[str, int]]:
        events = self._obj_get(sample, "events")
        if not isinstance(events, list):
            return {}

        usage_by_model: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for event in events:
            event_type = self._obj_get(event, "event") or self._obj_get(event, "type")
            if event_type != "model":
                continue
            output = self._obj_get(event, "output")
            usage = self._usage_to_dict(self._obj_get(output, "usage"))
            if not usage:
                continue
            model_name = (
                self._to_string(self._obj_get(event, "model"))
                or self._to_string(self._obj_get(output, "model"))
                or "unknown"
            )
            model_usage = usage_by_model[model_name]
            for key, value in usage.items():
                model_usage[key] = int(model_usage.get(key, 0)) + int(value)

        return {
            model: {key: int(value) for key, value in usage.items()}
            for model, usage in usage_by_model.items()
        }

    def _sum_usage_map(self, usage_map: Any) -> dict[str, int]:
        if not isinstance(usage_map, dict):
            return {}

        totals: dict[str, int] = defaultdict(int)
        for usage in usage_map.values():
            usage_dict = self._usage_to_dict(usage)
            for key, value in usage_dict.items():
                totals[key] += int(value)
        return dict(totals)

    def _scores_to_dict(self, scores: Any) -> dict[str, Any]:
        if scores is None or not hasattr(scores, "items"):
            return {}

        out: dict[str, Any] = {}
        for scorer_name, score in scores.items():
            value = self._obj_get(score, "value")
            if value is None:
                value = score
            out[str(scorer_name)] = value
        return out

    def _get_sample_output_text(self, sample: Any) -> str | None:
        output = self._obj_get(sample, "output")
        if output is None:
            return None

        choices = self._obj_get(output, "choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            message = self._obj_get(first_choice, "message")
            message_text = self._obj_get(message, "text") or self._obj_get(message, "content")
            if message_text is not None:
                return truncate(message_text, 500)
            choice_text = self._obj_get(first_choice, "text") or self._obj_get(
                first_choice, "completion"
            )
            if choice_text is not None:
                return truncate(choice_text, 500)

        output_text = self._obj_get(output, "completion") or self._obj_get(output, "text")
        if output_text is not None:
            return truncate(output_text, 500)

        serialized = self._to_json(output)
        return truncate(serialized, 500) if serialized is not None else None

    @staticmethod
    def _usage_to_dict(usage: Any) -> dict[str, int]:
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

    @staticmethod
    def _rows_to_columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
        columns: dict[str, list[Any]] = {}
        for row in rows:
            for key in row:
                columns.setdefault(str(key), [])
        for row in rows:
            for key in columns:
                columns[key].append(row.get(key))
        return columns

    @staticmethod
    def _obj_get(obj: Any, key: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        with contextlib.suppress(Exception):
            return getattr(obj, key)
        return None

    @staticmethod
    def _to_string(value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _to_json(value: Any) -> str | int | float | bool | None:
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

    def _log_sample_table(self, run_id: str, log: Any) -> None:
        if not log.samples:
            return

        rows = []
        for sample in log.samples:
            row: dict[str, Any] = {
                "id": sample.id,
                "epoch": sample.epoch,
                "input": truncate(sample.input, 500),
                "target": truncate(sample.target, 300),
                "total_time": sample.total_time,
                "error": getattr(sample, "error", None),
            }
            if sample.output and sample.output.choices:
                first_choice = sample.output.choices[0]
                row["output"] = truncate(first_choice.message.text, 500)
            else:
                row["output"] = ""
            if sample.scores:
                for scorer_name, score in sample.scores.items():
                    row[f"score/{scorer_name}"] = score.value
                    if score.explanation:
                        row[f"explanation/{scorer_name}"] = truncate(score.explanation, 300)
            rows.append(row)

        eval_id = log.eval.eval_id if log.eval else "unknown"
        fd, path = tempfile.mkstemp(prefix=f"sample_results_{eval_id}_", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(rows, f, indent=2, default=str)
            self.client.log_artifact(run_id, path, artifact_path="sample_results")
        finally:
            os.unlink(path)

    def _log_eval_json(self, run_id: str, log: Any) -> None:
        eval_id = log.eval.eval_id if log.eval else "unknown"
        log_data = log.model_dump(mode="json", exclude={"samples"})

        fd, path = tempfile.mkstemp(prefix=f"eval_log_{eval_id}_", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(log_data, f, indent=2, default=str)
            self.client.log_artifact(run_id, path, artifact_path="eval_logs")
        finally:
            os.unlink(path)

    async def on_sample_end(self, data: SampleEnd) -> None:
        eval_id = data.eval_id
        task_run_id = self._task_run_ids.get(eval_id)
        if not task_run_id:
            return

        with self._lock:
            sample_idx = self._sample_counts.get(eval_id, 0)
            self._sample_counts[eval_id] = sample_idx + 1

        sample = data.sample

        if sample.scores:
            for scorer_name, score in sample.scores.items():
                numeric = score_to_numeric(score.value)
                if numeric is not None:
                    with contextlib.suppress(Exception):
                        self.client.log_metric(
                            task_run_id, f"sample/{scorer_name}", numeric, step=sample_idx
                        )

        if sample.total_time is not None:
            with contextlib.suppress(Exception):
                self.client.log_metric(
                    task_run_id, "sample/total_time", sample.total_time, step=sample_idx
                )

    async def on_sample_event(self, data: SampleEvent) -> None:
        eval_id = data.eval_id
        task_run_id = self._task_run_ids.get(eval_id)
        if not task_run_id:
            return

        with self._lock:
            if eval_id not in self._event_counts:
                self._event_counts[eval_id] = {"model_calls": 0, "tool_calls": 0}
            counters = self._event_counts[eval_id]

        event = data.event

        if isinstance(event, ModelEvent):
            with self._lock:
                step = counters["model_calls"]
                counters["model_calls"] += 1
            try:
                self.client.log_metric(task_run_id, "event/model_call", step, step=step)
                if event.output and event.output.usage:
                    usage = event.output.usage
                    self.client.log_metric(
                        task_run_id, "event/input_tokens", usage.input_tokens, step=step
                    )
                    self.client.log_metric(
                        task_run_id, "event/output_tokens", usage.output_tokens, step=step
                    )
                if event.working_time is not None:
                    self.client.log_metric(
                        task_run_id, "event/model_time", event.working_time, step=step
                    )
            except Exception:
                pass

        elif isinstance(event, ToolEvent):
            with self._lock:
                step = counters["tool_calls"]
                counters["tool_calls"] += 1
            try:
                self.client.log_metric(task_run_id, "event/tool_call", step, step=step)
                _safe_log_param(
                    self.client, task_run_id, f"tool_call.{step}.function", event.function[:500]
                )
                if event.error:
                    self.client.log_metric(task_run_id, "event/tool_error", 1, step=step)
                if event.working_time is not None:
                    self.client.log_metric(
                        task_run_id, "event/tool_time", event.working_time, step=step
                    )
            except Exception:
                pass

    async def on_model_usage(self, data: ModelUsageData) -> None:
        model = data.model_name
        with self._lock:
            if model not in self._model_usage:
                self._model_usage[model] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "total_duration": 0.0,
                }
            stats = self._model_usage[model]
            stats["calls"] += 1
            stats["input_tokens"] += data.usage.input_tokens
            stats["output_tokens"] += data.usage.output_tokens
            stats["total_tokens"] += data.usage.total_tokens
            stats["total_duration"] += data.call_duration
