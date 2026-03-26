"""Artifact manager for MLflow tracking hook."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from typing import Any

from inspect_ai.log import read_eval_log
from mlflow.tracking import MlflowClient

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
            with contextlib.suppress(Exception):
                self.client.log_table(
                    run_id=run_id,
                    data=rows_to_columns(rows),
                    artifact_file=f"inspect/{name}.json",
                )

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
