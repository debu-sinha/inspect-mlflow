"""Tests for the Scout import source."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from inspect_mlflow.scout import _trace_to_transcript, import_mlflow_traces


def _make_mock_span(
    span_type: str = "CHAIN",
    name: str = "test",
    parent_id: str | None = None,
    span_id: str = "span-1",
    inputs: dict | None = None,
    outputs: dict | None = None,
    attributes: dict | None = None,
    start_time_ns: int = 1000000000,
    end_time_ns: int = 2000000000,
):
    span = MagicMock()
    span.span_type = span_type
    span.name = name
    span.parent_id = parent_id
    span.span_id = span_id
    span.inputs = inputs or {}
    span.outputs = outputs or {}
    span.attributes = attributes or {}
    span.start_time_ns = start_time_ns
    span.end_time_ns = end_time_ns
    span.status = MagicMock()
    return span


def _make_mock_trace(spans=None, trace_id="tr-test123"):
    trace = MagicMock()
    trace.info.trace_id = trace_id
    trace.info.request_time = "2026-03-20T12:00:00Z"
    trace.data.spans = spans or []
    return trace


def test_trace_to_transcript_with_llm_spans():
    root = _make_mock_span(
        span_type="CHAIN",
        name="eval_run",
        span_id="root",
        attributes={"inspect.run_id": "run-123", "inspect.task_count": "1"},
    )
    llm = _make_mock_span(
        span_type="LLM",
        name="model:openai/gpt-4o-mini",
        parent_id="root",
        span_id="llm-1",
        inputs={"model": "openai/gpt-4o-mini", "messages": "What is 2+2?"},
        outputs={"response": "4"},
        attributes={
            "inspect.model": "openai/gpt-4o-mini",
            "input_tokens": 15,
            "output_tokens": 8,
            "total_tokens": 23,
            "working_time": 0.5,
        },
    )

    with patch("inspect_mlflow.scout.mlflow") as mock_mlflow:
        mock_mlflow.get_trace.return_value = _make_mock_trace(spans=[root, llm], trace_id="tr-abc")
        transcript = _trace_to_transcript("tr-abc")

    assert transcript is not None
    assert transcript.transcript_id == "tr-abc"
    assert transcript.source_type == "mlflow"
    assert transcript.model == "openai/gpt-4o-mini"
    assert transcript.total_tokens == 23
    assert transcript.total_time == 1.0  # (2e9 - 1e9) / 1e9
    assert len(transcript.messages) == 2  # user + assistant
    assert len(transcript.events) == 1  # ModelEvent
    assert transcript.metadata["run_id"] == "run-123"


def test_trace_to_transcript_with_tool_spans():
    root = _make_mock_span(span_type="CHAIN", span_id="root")
    tool = _make_mock_span(
        span_type="TOOL",
        name="tool:calculator",
        parent_id="root",
        span_id="tool-1",
        inputs={"function": "calculator", "arguments": {"expression": "47*89"}},
        outputs={"result": "4183"},
        attributes={"working_time": 0.1},
    )

    with patch("inspect_mlflow.scout.mlflow") as mock_mlflow:
        mock_mlflow.get_trace.return_value = _make_mock_trace(spans=[root, tool])
        transcript = _trace_to_transcript("tr-tool")

    assert transcript is not None
    assert len(transcript.events) == 1
    event = transcript.events[0]
    assert event.function == "calculator"
    assert event.arguments == {"expression": "47*89"}


def test_trace_to_transcript_with_score_spans():
    root = _make_mock_span(span_type="CHAIN", span_id="root")
    score = _make_mock_span(
        span_type="EVALUATOR",
        name="score",
        parent_id="root",
        span_id="score-1",
        inputs={"target": "4"},
        outputs={"value": "C", "explanation": "Correct"},
    )

    with patch("inspect_mlflow.scout.mlflow") as mock_mlflow:
        mock_mlflow.get_trace.return_value = _make_mock_trace(spans=[root, score])
        transcript = _trace_to_transcript("tr-score")

    assert transcript is not None
    assert transcript.score == "C"
    assert len(transcript.events) == 1


def test_trace_to_transcript_returns_none_for_empty_trace():
    with patch("inspect_mlflow.scout.mlflow") as mock_mlflow:
        mock_mlflow.get_trace.return_value = None
        assert _trace_to_transcript("tr-none") is None

    with patch("inspect_mlflow.scout.mlflow") as mock_mlflow:
        trace = MagicMock()
        trace.data = None
        mock_mlflow.get_trace.return_value = trace
        assert _trace_to_transcript("tr-nodata") is None


@pytest.mark.anyio
async def test_import_mlflow_traces_yields_transcripts():
    root = _make_mock_span(span_type="CHAIN", span_id="root")
    llm = _make_mock_span(
        span_type="LLM",
        span_id="llm-1",
        parent_id="root",
        attributes={"inspect.model": "gpt-4o", "total_tokens": 100},
        inputs={"messages": "test"},
        outputs={"response": "answer"},
    )

    mock_traces_df = MagicMock()
    mock_traces_df.__len__ = MagicMock(return_value=1)
    mock_traces_df.iloc.__getitem__ = MagicMock(return_value={"trace_id": "tr-1"})

    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "exp-1"

    with patch("inspect_mlflow.scout.mlflow") as mock_mlflow:
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        mock_mlflow.search_traces.return_value = mock_traces_df
        mock_mlflow.get_trace.return_value = _make_mock_trace(spans=[root, llm], trace_id="tr-1")

        results = []
        async for t in import_mlflow_traces(
            experiment_name="test", tracking_uri="http://localhost:5000"
        ):
            results.append(t)

    assert len(results) == 1
    assert results[0].transcript_id == "tr-1"
    assert results[0].source_type == "mlflow"


@pytest.mark.anyio
async def test_import_mlflow_traces_respects_limit():
    mock_traces_df = MagicMock()
    mock_traces_df.__len__ = MagicMock(return_value=5)
    mock_traces_df.iloc.__getitem__ = MagicMock(side_effect=lambda i: {"trace_id": f"tr-{i}"})

    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "exp-1"

    root = _make_mock_span(span_type="CHAIN", span_id="root")

    with patch("inspect_mlflow.scout.mlflow") as mock_mlflow:
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        mock_mlflow.search_traces.return_value = mock_traces_df
        mock_mlflow.get_trace.return_value = _make_mock_trace(spans=[root])

        results = []
        async for t in import_mlflow_traces(experiment_name="test", limit=2):
            results.append(t)

    assert len(results) == 2


@pytest.mark.anyio
async def test_import_mlflow_traces_no_experiment():
    with patch("inspect_mlflow.scout.mlflow") as mock_mlflow:
        mock_mlflow.get_experiment_by_name.return_value = None

        results = []
        async for t in import_mlflow_traces(experiment_name="nonexistent"):
            results.append(t)

    assert len(results) == 0
