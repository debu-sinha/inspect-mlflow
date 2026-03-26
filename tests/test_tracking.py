"""Tests for the MLflow tracking hook.

Uses real SQLite-backed MLflow tracking store (no mocks for MLflow API).
This matches MLflow's own testing best practices.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
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
from inspect_ai.model._chat_message import ChatMessageAssistant
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model_output import ChatCompletionChoice, ModelOutput, ModelUsage
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
        await hook.on_run_start(
            RunStart(eval_set_id=None, run_id="run-auto-1", task_names=["task"])
        )
        mock_enable.assert_called_once_with(["openai", "anthropic"])
        assert hook._autolog_enabled is True


@pytest.mark.anyio
async def test_run_start_skips_autolog_when_disabled(tmp_tracking_uri, monkeypatch):
    monkeypatch.setenv("INSPECT_MLFLOW_AUTOLOG_ENABLED", "false")
    hook = MlflowTrackingHooks()

    with patch("inspect_mlflow.tracking.enable_autolog", return_value=True) as mock_enable:
        await hook.on_run_start(
            RunStart(eval_set_id=None, run_id="run-auto-2", task_names=["task"])
        )
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
        await hook.on_run_start(
            RunStart(eval_set_id=None, run_id="run-auto-3", task_names=["task"])
        )
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


# --- Artifact table extraction helpers ---


def test_extract_inspect_table_rows_includes_core_tables():
    hook = MlflowTrackingHooks()

    sample = SimpleNamespace(
        id="sample-1",
        input="What is 2+2?",
        target="4",
        total_time=1.2,
        working_time=0.9,
        error=None,
        model_usage={
            "openai/gpt-4.1-mini": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            }
        },
    )

    output_choice = SimpleNamespace(message=SimpleNamespace(text="4"))
    sample.output = SimpleNamespace(choices=[output_choice])
    sample.scores = {"match": Score(value="C", explanation="correct")}

    sample.messages = [
        SimpleNamespace(
            role="assistant",
            source="generate",
            content="4",
            tool_calls=[],
            tool_call_id=None,
            model="openai/gpt-4.1-mini",
            stop_reason="stop",
        )
    ]

    sample.events = [
        {
            "event": "model",
            "timestamp": "2026-03-25T00:00:00Z",
            "model": "openai/gpt-4.1-mini",
            "output": {
                "completion": "4",
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        },
        {
            "event": "tool",
            "timestamp": "2026-03-25T00:00:01Z",
            "function": "calculator",
            "arguments": {"expression": "2+2"},
            "result": "4",
            "error": None,
        },
    ]

    log = SimpleNamespace(
        eval=_make_eval_spec(eval_id="eval-001"),
        samples=[sample],
    )
    tables = hook._extract_inspect_table_rows(eval_id="eval-001", task_name="test_task", log=log)

    assert set(tables.keys()) == {
        "tasks",
        "samples",
        "messages",
        "sample_scores",
        "events",
        "model_usage",
    }
    assert len(tables["tasks"]) == 1
    assert len(tables["samples"]) == 1
    assert len(tables["messages"]) == 1
    assert len(tables["sample_scores"]) == 1
    assert len(tables["events"]) == 2
    assert len(tables["model_usage"]) == 1

    sample_row = tables["samples"][0]
    assert sample_row["sample_id"] == "sample-1"
    assert sample_row["output"] == "4"
    assert sample_row["usage_total_tokens"] == 15

    score_row = tables["sample_scores"][0]
    assert score_row["scorer"] == "match"
    assert score_row["numeric_value"] == 1.0


def test_extract_model_usage_rows_falls_back_to_events():
    hook = MlflowTrackingHooks()

    sample = SimpleNamespace(
        model_usage=None,
        events=[
            {
                "event": "model",
                "model": "openai/gpt-4.1-mini",
                "output": {
                    "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
                },
            },
            {
                "event": "model",
                "model": "openai/gpt-4.1-mini",
                "output": {
                    "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                },
            },
        ],
    )

    rows = hook._extract_model_usage_rows(
        eval_id="eval-001",
        task_name="test_task",
        sample_id="sample-1",
        sample=sample,
    )

    assert len(rows) == 1
    assert rows[0]["model"] == "openai/gpt-4.1-mini"
    assert rows[0]["input_tokens"] == 9
    assert rows[0]["output_tokens"] == 4
    assert rows[0]["total_tokens"] == 13


def test_rows_to_columns_aligns_missing_keys():
    rows = [
        {"a": 1, "b": "x"},
        {"a": 2, "c": "y"},
    ]

    columns = MlflowTrackingHooks._rows_to_columns(rows)
    assert columns["a"] == [1, 2]
    assert columns["b"] == ["x", None]
    assert columns["c"] == [None, "y"]


def test_log_inspect_tables_falls_back_to_log_location(monkeypatch):
    hook = MlflowTrackingHooks()
    hook._client = MagicMock()

    sample = SimpleNamespace(
        id="sample-1",
        input="What is 2+2?",
        target="4",
        total_time=1.0,
        working_time=0.8,
        error=None,
        model_usage={
            "mockllm/model": {
                "input_tokens": 5,
                "output_tokens": 3,
                "total_tokens": 8,
            }
        },
        output=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(text="4"))],
        ),
        scores={"exact": Score(value="C", explanation="correct")},
        messages=[SimpleNamespace(role="assistant", source="generate", content="4")],
        events=[
            {
                "event": "model",
                "model": "mockllm/model",
                "output": {
                    "completion": "4",
                    "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                },
            }
        ],
    )
    full_log = SimpleNamespace(
        eval=_make_eval_spec(eval_id="eval-001"),
        samples=[sample],
    )
    partial_log = SimpleNamespace(
        eval=_make_eval_spec(eval_id="eval-001"),
        samples=[],
        location="logs/fake.eval",
    )

    monkeypatch.setattr("inspect_mlflow.artifacts.manager.read_eval_log", lambda _path: full_log)

    hook._log_inspect_tables(run_id="run-123", log=partial_log)

    artifact_files = [c.kwargs["artifact_file"] for c in hook._client.log_table.call_args_list]
    assert "inspect/tasks.json" in artifact_files
    assert "inspect/samples.json" in artifact_files
    assert "inspect/sample_scores.json" in artifact_files
    assert "inspect/messages.json" in artifact_files
    assert "inspect/events.json" in artifact_files
    assert "inspect/model_usage.json" in artifact_files


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


@pytest.mark.anyio
async def test_logs_inspect_table_artifacts(tmp_tracking_uri):
    hook = MlflowTrackingHooks()
    spec = _make_eval_spec()

    await hook.on_run_start(RunStart(eval_set_id=None, run_id="run-001", task_names=["test_task"]))
    await hook.on_task_start(
        TaskStart(eval_set_id=None, run_id="run-001", eval_id="eval-001", spec=spec)
    )

    task_run_id = hook._task_run_ids["eval-001"]

    sample = EvalSample(
        id="sample-1",
        epoch=1,
        input="What is 2+2?",
        target="4",
        messages=[
            ChatMessageAssistant(
                content="4",
                source="generate",
                model="openai/gpt-4.1-mini",
            )
        ],
        output=ModelOutput(
            model="openai/gpt-4.1-mini",
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content="4"),
                    stop_reason="stop",
                )
            ],
            usage=ModelUsage(input_tokens=7, output_tokens=3, total_tokens=10),
        ),
        model_usage={
            "openai/gpt-4.1-mini": ModelUsage(
                input_tokens=7,
                output_tokens=3,
                total_tokens=10,
            )
        },
        scores={"match": Score(value="C", explanation="correct")},
        total_time=1.5,
    )

    # Inject simple raw events for artifact-table extraction compatibility.
    object.__setattr__(
        sample,
        "events",
        [
            {
                "event": "model",
                "timestamp": "2026-03-25T00:00:00Z",
                "model": "openai/gpt-4.1-mini",
                "output": {
                    "completion": "4",
                    "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
                },
            },
            {
                "event": "tool",
                "timestamp": "2026-03-25T00:00:01Z",
                "function": "calculator",
                "arguments": {"expression": "2+2"},
                "result": "4",
                "error": None,
            },
        ],
    )

    log = _make_eval_log(
        scores=[
            EvalScore(
                name="match",
                scorer="match",
                metrics={"accuracy": EvalMetric(name="accuracy", value=1.0)},
            )
        ],
        samples=[sample],
    )
    await hook.on_task_end(TaskEnd(eval_set_id=None, run_id="run-001", eval_id="eval-001", log=log))

    client = mlflow.tracking.MlflowClient()
    root_artifacts = {a.path for a in client.list_artifacts(task_run_id)}
    assert "inspect" in root_artifacts

    inspect_artifacts = {a.path for a in client.list_artifacts(task_run_id, path="inspect")}
    expected = {
        "inspect/tasks.json",
        "inspect/samples.json",
        "inspect/messages.json",
        "inspect/sample_scores.json",
        "inspect/events.json",
        "inspect/model_usage.json",
    }
    assert expected.issubset(inspect_artifacts)

    samples_local = mlflow.artifacts.download_artifacts(
        run_id=task_run_id, artifact_path="inspect/samples.json"
    )
    samples_payload = json.loads(Path(samples_local).read_text())
    assert samples_payload["columns"]
    assert samples_payload["data"]

    events_local = mlflow.artifacts.download_artifacts(
        run_id=task_run_id, artifact_path="inspect/events.json"
    )
    events_payload = json.loads(Path(events_local).read_text())
    assert "event_type" in events_payload["columns"]
    assert len(events_payload["data"]) >= 2
