"""Scout import source for MLflow traces.

Imports MLflow traces into an Inspect Scout transcript database,
enabling Scout scanners to analyze any MLflow-traced LLM application.

Usage:

    from inspect_mlflow.scout import import_mlflow_traces
    from inspect_scout import transcripts_db

    async with transcripts_db("./my-transcripts") as db:
        await db.insert(import_mlflow_traces(
            experiment_name="inspect-mlflow-demo",
            tracking_uri="http://localhost:5000",
        ))
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import mlflow
from inspect_ai.event._model import ModelEvent
from inspect_ai.event._score import ScoreEvent
from inspect_ai.event._tool import ToolEvent
from inspect_ai.model._chat_message import (
    ChatMessageAssistant,
    ChatMessageUser,
)
from inspect_ai.model._model_output import ModelUsage
from inspect_scout import Transcript

_logger = logging.getLogger(__name__)


async def import_mlflow_traces(
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
    limit: int | None = None,
) -> AsyncIterator[Transcript]:
    """Import MLflow traces as Scout transcripts.

    Args:
        experiment_name: MLflow experiment name to import from.
            Defaults to MLFLOW_EXPERIMENT_NAME env var or "inspect_ai".
        tracking_uri: MLflow tracking server URI.
            Defaults to MLFLOW_TRACKING_URI env var.
        limit: Maximum number of traces to import. None for all.

    Yields:
        Transcript objects ready for Scout database insertion.
    """
    import os

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    exp_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "inspect_ai")
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        _logger.warning("Experiment '%s' not found", exp_name)
        return

    traces = mlflow.search_traces(experiment_ids=[experiment.experiment_id])

    count = 0
    for idx in range(len(traces)):
        if limit is not None and count >= limit:
            break

        trace_id = traces.iloc[idx]["trace_id"]
        try:
            transcript = _trace_to_transcript(trace_id)
            if transcript is not None:
                yield transcript
                count += 1
        except Exception:
            _logger.debug("Failed to convert trace %s", trace_id, exc_info=True)


def _trace_to_transcript(trace_id: str) -> Transcript | None:
    """Convert a single MLflow trace to a Scout Transcript."""
    trace = mlflow.get_trace(trace_id)
    if trace is None or trace.data is None or not trace.data.spans:
        return None

    spans = trace.data.spans
    root_span = next((s for s in spans if s.parent_id is None), None)

    # Extract messages from LLM spans
    messages: list[Any] = []
    events: list[Any] = []
    total_tokens = 0
    model_name = None
    total_time = None

    for span in spans:
        if span.span_type == "LLM":
            model_name = model_name or _get_attr(span, "inspect.model")
            _extract_llm_messages(span, messages)
            tokens = _get_attr(span, "total_tokens")
            if isinstance(tokens, (int, float)):
                total_tokens += int(tokens)

            # Build a ModelEvent for Scout
            model_event = _span_to_model_event(span)
            if model_event is not None:
                events.append(model_event)

        elif span.span_type == "TOOL":
            tool_event = _span_to_tool_event(span)
            if tool_event is not None:
                events.append(tool_event)

        elif span.span_type == "EVALUATOR":
            score_event = _span_to_score_event(span)
            if score_event is not None:
                events.append(score_event)

    # Calculate total time from root span
    if root_span and root_span.start_time_ns and root_span.end_time_ns:
        total_time = (root_span.end_time_ns - root_span.start_time_ns) / 1e9

    # Extract score from EVALUATOR spans
    score = None
    for span in spans:
        if span.span_type == "EVALUATOR" and span.outputs:
            score = span.outputs.get("value")
            if score is not None:
                break

    # Build metadata from root span attributes
    metadata: dict[str, Any] = {}
    if root_span and root_span.attributes:
        for k, v in root_span.attributes.items():
            if k.startswith("inspect."):
                metadata[k.removeprefix("inspect.")] = v

    return Transcript(
        transcript_id=trace_id,
        source_type="mlflow",
        source_id=trace_id,
        source_uri=None,
        date=traces_request_time(trace),
        model=model_name,
        score=score,
        message_count=len(messages),
        total_time=total_time,
        total_tokens=total_tokens if total_tokens > 0 else None,
        metadata=metadata,
        messages=messages,
        events=events,
    )


def traces_request_time(trace: Any) -> str | None:
    """Extract request time from trace info."""
    if trace.info and hasattr(trace.info, "request_time"):
        rt = trace.info.request_time
        if rt:
            return str(rt)
    return None


def _get_attr(span: Any, key: str) -> Any:
    """Safely get a span attribute."""
    if span.attributes:
        return span.attributes.get(key)
    return None


def _extract_llm_messages(span: Any, messages: list[Any]) -> None:
    """Extract user/assistant messages from an LLM span's inputs/outputs."""
    if span.inputs:
        model_input = span.inputs.get("model") or span.inputs.get("messages")
        if isinstance(model_input, str) and model_input:
            messages.append(ChatMessageUser(content=model_input, role="user"))

    if span.outputs:
        response = span.outputs.get("response")
        if isinstance(response, str) and response:
            messages.append(ChatMessageAssistant(content=response, role="assistant"))


def _span_to_model_event(span: Any) -> ModelEvent | None:
    """Convert an LLM span to a ModelEvent for Scout."""
    try:
        model = _get_attr(span, "inspect.model") or "unknown"
        input_tokens = _get_attr(span, "input_tokens") or 0
        output_tokens = _get_attr(span, "output_tokens") or 0
        total = _get_attr(span, "total_tokens") or 0
        working_time = _get_attr(span, "working_time")

        from inspect_ai.model._generate_config import GenerateConfig
        from inspect_ai.model._model_output import ModelOutput

        return ModelEvent(
            model=model,
            input=[],
            tools=[],
            tool_choice="auto",
            config=GenerateConfig(),
            output=ModelOutput(
                usage=ModelUsage(
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    total_tokens=int(total),
                )
            ),
            working_time=float(working_time) if working_time is not None else None,
        )
    except Exception:
        _logger.debug("Failed to build ModelEvent from span", exc_info=True)
        return None


def _span_to_tool_event(span: Any) -> ToolEvent | None:
    """Convert a TOOL span to a ToolEvent for Scout."""
    try:
        inputs = span.inputs or {}
        function = inputs.get("function", "unknown")
        arguments = inputs.get("arguments", {})

        outputs = span.outputs or {}
        result = outputs.get("result", "")
        working_time = _get_attr(span, "working_time")

        return ToolEvent(
            id=span.span_id,
            function=str(function),
            arguments=arguments if isinstance(arguments, dict) else {},
            result=str(result) if result else None,
            working_time=float(working_time) if working_time is not None else None,
        )
    except Exception:
        _logger.debug("Failed to build ToolEvent from span", exc_info=True)
        return None


def _span_to_score_event(span: Any) -> ScoreEvent | None:
    """Convert an EVALUATOR span to a ScoreEvent for Scout."""
    try:
        from inspect_ai.scorer._metric import Score

        outputs = span.outputs or {}
        value = outputs.get("value")
        explanation = outputs.get("explanation")
        inputs = span.inputs or {}
        target = inputs.get("target")

        return ScoreEvent(
            score=Score(value=value, explanation=explanation),
            target=str(target) if target else None,
        )
    except Exception:
        _logger.debug("Failed to build ScoreEvent from span", exc_info=True)
        return None
