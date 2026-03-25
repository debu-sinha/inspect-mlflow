"""Tests for the MLflow tracking hook.

Uses real SQLite-backed MLflow tracking store (no mocks for MLflow API).
This matches MLflow's own testing best practices.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mlflow
import pytest
from inspect_ai.hooks._hooks import (
    RunEnd,
    RunStart,
    SampleEnd,
    TaskEnd,
    TaskStart,
)
from inspect_ai.log._log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalMetric,
    EvalResults,
    EvalSample,
    EvalScore,
    EvalSpec,
)
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.scorer._metric import Score

from inspect_mlflow.tracking import MlflowTrackingHooks
from inspect_mlflow.util import score_to_numeric


def _make_eval_spec(
    task: str = "test_task",
    model: str = "openai/gpt-4",
    eval_id: str = "eval-001",
    run_id: str = "run-001",
) -> EvalSpec:
    return EvalSpec(
        eval_id=eval_id,
        run_id=run_id,
        created="2026-03-20T00:00:00",
        task=task,
        task_version=1,
        task_file="test.py",
        task_id=f"{task}@test.py",
        model=model,
        dataset=EvalDataset(name="test_dataset", samples=10),
        solver="generate",
        config=EvalConfig(),
        model_generate_config=GenerateConfig(temperature=0.7, max_tokens=100),
    )


def _make_eval_log(
    eval_id: str = "eval-001",
    status: str = "success",
    scores: list[EvalScore] | None = None,
    samples: list[EvalSample] | None = None,
) -> EvalLog:
    results = None
    if scores:
        results = EvalResults(scores=scores, total_samples=3, completed_samples=3)
    return EvalLog(
        eval=EvalSpec(
            eval_id=eval_id,
            run_id="run-001",
            created="2026-03-20T00:00:00",
            task="test_task",
            task_version=1,
            task_file="test.py",
            task_id="test_task@test.py",
            model="openai/gpt-4",
            dataset=EvalDataset(name="test_dataset", samples=3),
            solver="generate",
            config=EvalConfig(),
            model_generate_config=GenerateConfig(temperature=0.7, max_tokens=100),
        ),
        status=status,
        results=results,
        samples=samples,
    )


# --- Enabled/Disabled ---


def test_enabled_requires_tracking_uri(monkeypatch):
    hook = MlflowTrackingHooks()

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert hook.enabled() is False

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    assert hook.enabled() is True


@pytest.mark.anyio
async def test_run_start_enables_autolog_when_enabled(tmp_tracking_uri, monkeypatch):
    monkeypatch.setenv("INSPECT_MLFLOW_AUTOLOG_ENABLED", "true")
    monkeypatch.setenv("INSPECT_MLFLOW_AUTOLOG_MODELS", "openai,anthropic")
    hook = MlflowTrackingHooks()

    with patch("inspect_mlflow.tracking.enable_autolog", return_value=True) as mock_enable:
        await hook.on_run_start(RunStart(eval_set_id=None, run_id="run-auto-1", task_names=["task"]))
        mock_enable.assert_called_once_with(["openai", "anthropic"])
        assert hook._autolog_enabled is True


@pytest.mark.anyio
async def test_run_start_skips_autolog_when_disabled(tmp_tracking_uri, monkeypatch):
    monkeypatch.setenv("INSPECT_MLFLOW_AUTOLOG_ENABLED", "false")
    hook = MlflowTrackingHooks()

    with patch("inspect_mlflow.tracking.enable_autolog", return_value=True) as mock_enable:
        await hook.on_run_start(RunStart(eval_set_id=None, run_id="run-auto-2", task_names=["task"]))
        mock_enable.assert_not_called()
        assert hook._autolog_enabled is False


@pytest.mark.anyio
async def test_run_end_disables_mlflow_autolog_when_enabled(tmp_tracking_uri, monkeypatch):
    monkeypatch.setenv("INSPECT_MLFLOW_AUTOLOG_ENABLED", "true")
    hook = MlflowTrackingHooks()

    with (
        patch("inspect_mlflow.tracking.enable_autolog", return_value=True),
        patch("inspect_mlflow.tracking.mlflow.autolog") as mock_mlflow_autolog,
    ):
        await hook.on_run_start(RunStart(eval_set_id=None, run_id="run-auto-3", task_names=["task"]))
        assert hook._autolog_enabled is True
        await hook.on_run_end(
            RunEnd(eval_set_id=None, run_id="run-auto-3", exception=None, logs=[])
        )
        mock_mlflow_autolog.assert_called_once_with(disable=True)
        assert hook._autolog_enabled is False


# --- Run lifecycle ---


@pytest.mark.anyio
async def test_run_lifecycle(tmp_tracking_uri):
    hook = MlflowTrackingHooks()

    await hook.on_run_start(
        RunStart(eval_set_id=None, run_id="run-abc123", task_names=["task_a", "task_b"])
    )

    assert hook._parent_run_id is not None
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(hook._parent_run_id)
    assert run.info.status == "RUNNING"
    assert run.data.tags["inspect.run_id"] == "run-abc123"
    assert "inspect-run-abc1" in run.info.run_name

    await hook.on_run_end(RunEnd(eval_set_id=None, run_id="run-abc123", exception=None, logs=[]))

    run = client.get_run(run.info.run_id)
    assert run.info.status == "FINISHED"
    assert hook._parent_run_id is None


@pytest.mark.anyio
async def test_run_end_with_exception(tmp_tracking_uri):
    hook = MlflowTrackingHooks()

    await hook.on_run_start(RunStart(eval_set_id=None, run_id="run-001", task_names=["t"]))
    parent_id = hook._parent_run_id

    await hook.on_run_end(
        RunEnd(eval_set_id=None, run_id="run-001", exception=RuntimeError("boom"), logs=[])
    )

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(parent_id)
    assert run.info.status == "FAILED"


# --- Task lifecycle ---


@pytest.mark.anyio
async def test_task_creates_nested_run(tmp_tracking_uri):
    hook = MlflowTrackingHooks()
    spec = _make_eval_spec()

    await hook.on_run_start(RunStart(eval_set_id=None, run_id="run-001", task_names=["test_task"]))
    await hook.on_task_start(
        TaskStart(eval_set_id=None, run_id="run-001", eval_id="eval-001", spec=spec)
    )

    assert "eval-001" in hook._task_run_ids
    task_run_id = hook._task_run_ids["eval-001"]

    client = mlflow.tracking.MlflowClient()
    task_run = client.get_run(task_run_id)
    assert task_run.data.tags["mlflow.parentRunId"] == hook._parent_run_id
    assert task_run.data.tags["inspect.model"] == "openai/gpt-4"
    assert task_run.data.tags["inspect.task"] == "test_task"


@pytest.mark.anyio
async def test_task_end_logs_metrics(tmp_tracking_uri):
    hook = MlflowTrackingHooks()
    spec = _make_eval_spec()

    await hook.on_run_start(RunStart(eval_set_id=None, run_id="run-001", task_names=["test_task"]))
    await hook.on_task_start(
        TaskStart(eval_set_id=None, run_id="run-001", eval_id="eval-001", spec=spec)
    )

    task_run_id = hook._task_run_ids["eval-001"]

    log = _make_eval_log(
        scores=[
            EvalScore(
                name="match",
                scorer="match",
                metrics={"accuracy": EvalMetric(name="accuracy", value=0.85)},
            )
        ]
    )
    await hook.on_task_end(TaskEnd(eval_set_id=None, run_id="run-001", eval_id="eval-001", log=log))

    client = mlflow.tracking.MlflowClient()
    task_run = client.get_run(task_run_id)
    assert task_run.data.metrics["match/accuracy"] == 0.85
    assert task_run.data.metrics["total_samples"] == 3
    assert task_run.info.status == "FINISHED"


# --- Sample metrics ---


@pytest.mark.anyio
async def test_sample_end_logs_step_metrics(tmp_tracking_uri):
    hook = MlflowTrackingHooks()
    spec = _make_eval_spec()

    await hook.on_run_start(RunStart(eval_set_id=None, run_id="run-001", task_names=["test_task"]))
    await hook.on_task_start(
        TaskStart(eval_set_id=None, run_id="run-001", eval_id="eval-001", spec=spec)
    )

    sample = MagicMock()
    sample.scores = {"match": Score(value="C", explanation="correct")}
    sample.total_time = 1.5

    await hook.on_sample_end(
        SampleEnd(
            eval_set_id=None,
            run_id="run-001",
            eval_id="eval-001",
            sample_id="s1",
            sample=sample,
        )
    )

    # Verify sample count was incremented (proves on_sample_end ran)
    assert hook._sample_counts["eval-001"] == 1


# --- Artifact logging disabled ---


@pytest.mark.anyio
async def test_artifact_logging_disabled(tmp_tracking_uri, monkeypatch):
    monkeypatch.setenv("MLFLOW_INSPECT_LOG_ARTIFACTS", "false")
    hook = MlflowTrackingHooks()
    spec = _make_eval_spec()

    await hook.on_run_start(RunStart(eval_set_id=None, run_id="run-001", task_names=["test_task"]))
    await hook.on_task_start(
        TaskStart(eval_set_id=None, run_id="run-001", eval_id="eval-001", spec=spec)
    )

    task_run_id = hook._task_run_ids["eval-001"]
    log = _make_eval_log()
    await hook.on_task_end(TaskEnd(eval_set_id=None, run_id="run-001", eval_id="eval-001", log=log))

    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(task_run_id)
    assert len(list(artifacts)) == 0


# --- Score conversion ---


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("C", 1.0),
        ("I", 0.0),
        ("P", 0.5),
        ("correct", 1.0),
        ("incorrect", 0.0),
        (True, 1.0),
        (False, 0.0),
        (0.75, 0.75),
        (42, 42.0),
        ("unknown", None),
        (None, None),
    ],
)
def test_score_to_numeric_conversion(value, expected):
    assert score_to_numeric(value) == expected


# --- Full lifecycle integration ---


@pytest.mark.anyio
async def test_full_lifecycle(tmp_tracking_uri):
    hook = MlflowTrackingHooks()
    spec = _make_eval_spec()

    # Start
    await hook.on_run_start(RunStart(eval_set_id=None, run_id="run-001", task_names=["test_task"]))
    await hook.on_task_start(
        TaskStart(eval_set_id=None, run_id="run-001", eval_id="eval-001", spec=spec)
    )

    # Sample
    sample = MagicMock()
    sample.scores = {"match": Score(value="C")}
    sample.total_time = 2.0
    await hook.on_sample_end(
        SampleEnd(
            eval_set_id=None,
            run_id="run-001",
            eval_id="eval-001",
            sample_id="s1",
            sample=sample,
        )
    )

    # End task
    log = _make_eval_log(
        scores=[
            EvalScore(
                name="match",
                scorer="match",
                metrics={"accuracy": EvalMetric(name="accuracy", value=1.0)},
            )
        ]
    )
    await hook.on_task_end(TaskEnd(eval_set_id=None, run_id="run-001", eval_id="eval-001", log=log))

    # End run
    await hook.on_run_end(RunEnd(eval_set_id=None, run_id="run-001", exception=None, logs=[]))

    # Verify everything via real MLflow API
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("test-experiment")
    runs = client.search_runs([exp.experiment_id])

    assert len(runs) == 2  # parent + task
    parent = next(r for r in runs if "mlflow.parentRunId" not in r.data.tags)
    child = next(r for r in runs if "mlflow.parentRunId" in r.data.tags)

    assert parent.info.status == "FINISHED"
    assert child.info.status == "FINISHED"
    assert child.data.tags["mlflow.parentRunId"] == parent.info.run_id
    assert child.data.metrics["match/accuracy"] == 1.0
    assert child.data.metrics["sample/match"] == 1.0
