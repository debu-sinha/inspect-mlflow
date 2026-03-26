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
import logging
import threading
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
from inspect_mlflow.artifacts.manager import ArtifactManager
from inspect_mlflow.artifacts.tables import (
    extract_event_rows,
    extract_inspect_table_rows,
    extract_message_rows,
    extract_model_usage_rows,
    extract_sample_score_rows,
    extract_usage_from_events,
    get_sample_output_text,
    obj_get,
    rows_to_columns,
    scores_to_dict,
    sum_usage_map,
    to_json,
    to_string,
    usage_to_dict,
)
from inspect_mlflow.config import MLflowSettings, load_settings
from inspect_mlflow.util import score_to_numeric

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
        self._artifact_manager: ArtifactManager | None = None
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

    @property
    def artifact_manager(self) -> ArtifactManager:
        if self._artifact_manager is None:
            self._artifact_manager = ArtifactManager(self.client, _logger)
        return self._artifact_manager

    def enabled(self) -> bool:
        return load_settings().tracking_uri is not None

    async def on_run_start(self, data: RunStart) -> None:
        self._settings = load_settings()
        self._client = MlflowClient()
        self._artifact_manager = ArtifactManager(self._client, _logger)
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
        self.artifact_manager.log_eval_artifacts(run_id, log)

    def _log_inspect_tables(self, run_id: str, log: Any) -> None:
        self.artifact_manager.log_inspect_tables(run_id, log)

    def _load_full_eval_log(self, log: Any) -> Any | None:
        return self.artifact_manager.load_full_eval_log(log)

    def _extract_inspect_table_rows(
        self, *, eval_id: str, task_name: str, log: Any
    ) -> dict[str, list[dict[str, Any]]]:
        return extract_inspect_table_rows(eval_id=eval_id, task_name=task_name, log=log)

    def _extract_sample_score_rows(
        self,
        *,
        eval_id: str,
        task_name: str,
        sample_id: Any,
        scores: Any,
    ) -> list[dict[str, Any]]:
        return extract_sample_score_rows(
            eval_id=eval_id,
            task_name=task_name,
            sample_id=sample_id,
            scores=scores,
        )

    def _extract_message_rows(
        self,
        *,
        eval_id: str,
        task_name: str,
        sample_id: Any,
        sample: Any,
    ) -> list[dict[str, Any]]:
        return extract_message_rows(
            eval_id=eval_id,
            task_name=task_name,
            sample_id=sample_id,
            sample=sample,
        )

    def _extract_event_rows(
        self,
        *,
        eval_id: str,
        task_name: str,
        sample_id: Any,
        sample: Any,
    ) -> list[dict[str, Any]]:
        return extract_event_rows(
            eval_id=eval_id,
            task_name=task_name,
            sample_id=sample_id,
            sample=sample,
        )

    def _extract_model_usage_rows(
        self,
        *,
        eval_id: str,
        task_name: str,
        sample_id: Any,
        sample: Any,
    ) -> list[dict[str, Any]]:
        return extract_model_usage_rows(
            eval_id=eval_id,
            task_name=task_name,
            sample_id=sample_id,
            sample=sample,
        )

    def _extract_usage_from_events(self, sample: Any) -> dict[str, dict[str, int]]:
        return extract_usage_from_events(sample)

    def _sum_usage_map(self, usage_map: Any) -> dict[str, int]:
        return sum_usage_map(usage_map)

    def _scores_to_dict(self, scores: Any) -> dict[str, Any]:
        return scores_to_dict(scores)

    def _get_sample_output_text(self, sample: Any) -> str | None:
        return get_sample_output_text(sample)

    @staticmethod
    def _usage_to_dict(usage: Any) -> dict[str, int]:
        return usage_to_dict(usage)

    @staticmethod
    def _rows_to_columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
        return rows_to_columns(rows)

    @staticmethod
    def _obj_get(obj: Any, key: str) -> Any:
        return obj_get(obj, key)

    @staticmethod
    def _to_string(value: Any) -> str | None:
        return to_string(value)

    @staticmethod
    def _to_json(value: Any) -> str | int | float | bool | None:
        return to_json(value)

    def _log_sample_table(self, run_id: str, log: Any) -> None:
        self.artifact_manager.log_sample_table(run_id, log)

    def _log_eval_json(self, run_id: str, log: Any) -> None:
        self.artifact_manager.log_eval_json(run_id, log)

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
