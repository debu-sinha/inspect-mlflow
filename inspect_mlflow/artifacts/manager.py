"""Artifact manager for MLflow tracking hook."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import posixpath
import tempfile
from typing import Any

from inspect_ai.log import read_eval_log
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_LOGGED_ARTIFACTS

from inspect_mlflow.artifacts.tables import extract_inspect_table_rows, obj_get, rows_to_columns
from inspect_mlflow.util import truncate


class ArtifactManager:
    """Handle Inspect artifact extraction and MLflow artifact logging."""

    def __init__(self, client: MlflowClient, logger: logging.Logger | None = None) -> None:
        self.client = client
        self.logger = logger or logging.getLogger(__name__)

    def log_eval_artifacts(self, run_id: str, log: Any) -> None:
        try:
            self.log_inspect_tables(run_id, log)
        except Exception:
            self.logger.debug("Failed to log inspect table artifacts", exc_info=True)

        try:
            self.log_sample_table(run_id, log)
        except Exception:
            self.logger.debug("Failed to log sample results artifact", exc_info=True)

        try:
            self.log_eval_json(run_id, log)
        except Exception:
            self.logger.debug("Failed to log eval log artifact", exc_info=True)

    def log_inspect_tables(self, run_id: str, log: Any) -> None:
        eval_id = obj_get(obj_get(log, "eval"), "eval_id") or "unknown"
        task_name = obj_get(obj_get(log, "eval"), "task") or "unknown"

        source_log = log
        tables = extract_inspect_table_rows(
            eval_id=str(eval_id),
            task_name=str(task_name),
            log=source_log,
        )
        if not tables["samples"]:
            full_log = self.load_full_eval_log(log)
            if full_log is not None:
                source_log = full_log
                tables = extract_inspect_table_rows(
                    eval_id=str(eval_id),
                    task_name=str(task_name),
                    log=source_log,
                )

        for name, rows in tables.items():
            if not rows:
                continue
            artifact_file = f"inspect/{name}.json"
            try:
                self.log_table(run_id, rows_to_columns(rows), artifact_file)
            except Exception:
                self.logger.debug("Failed to log table %s", artifact_file, exc_info=True)

    def log_table(self, run_id: str, columns: dict[str, list[Any]], artifact_file: str) -> None:
        """Log a column-oriented table the way ``MlflowClient.log_table`` does.

        ``MlflowClient.log_table`` imports pandas, which ``mlflow-skinny`` does not
        ship. Without pandas, write the same split-orient JSON directly and tag it
        as a table so the MLflow UI still renders it.
        """
        if _pandas_available():
            self.client.log_table(run_id=run_id, data=columns, artifact_file=artifact_file)
            return

        payload = {
            "columns": list(columns),
            "data": [list(row) for row in zip(*columns.values(), strict=True)],
        }
        artifact_dir, file_name = posixpath.split(artifact_file)
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = os.path.join(tmp_dir, file_name)
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, default=str)
            self.client.log_artifact(run_id, local_path, artifact_path=artifact_dir or None)

        logged = json.loads(
            self.client.get_run(run_id).data.tags.get(MLFLOW_LOGGED_ARTIFACTS, "[]")
        )
        entry = {"path": artifact_file, "type": "table"}
        if entry not in logged:
            logged.append(entry)
            self.client.set_tag(run_id, MLFLOW_LOGGED_ARTIFACTS, json.dumps(logged))

    def load_full_eval_log(self, log: Any) -> Any | None:
        location = obj_get(log, "location")
        if not isinstance(location, str) or not location:
            return None

        try:
            return read_eval_log(location)
        except Exception:
            self.logger.debug("Could not load full eval log from %s", location, exc_info=True)
            return None

    def log_sample_table(self, run_id: str, log: Any) -> None:
        source_log = log
        samples = obj_get(source_log, "samples")
        if not samples:
            full_log = self.load_full_eval_log(log)
            if full_log is not None:
                source_log = full_log
                samples = obj_get(source_log, "samples")

        if not samples:
            return

        rows = []
        for sample in samples:
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

        eval_spec = obj_get(source_log, "eval")
        eval_id = obj_get(eval_spec, "eval_id") or "unknown"
        fd, path = tempfile.mkstemp(prefix=f"sample_results_{eval_id}_", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(rows, f, indent=2, default=str)
            self.client.log_artifact(run_id, path, artifact_path="sample_results")
        finally:
            os.unlink(path)

    def log_eval_json(self, run_id: str, log: Any) -> None:
        eval_id = log.eval.eval_id if log.eval else "unknown"
        log_data = log.model_dump(mode="json", exclude={"samples"})

        fd, path = tempfile.mkstemp(prefix=f"eval_log_{eval_id}_", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(log_data, f, indent=2, default=str)
            self.client.log_artifact(run_id, path, artifact_path="eval_logs")
        finally:
            os.unlink(path)


def _pandas_available() -> bool:
    return importlib.util.find_spec("pandas") is not None
