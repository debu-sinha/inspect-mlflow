"""Verify installed Inspect hooks against a local MLflow store without API credentials."""

import asyncio
import json
import os
from pathlib import Path

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput, get_model
from inspect_ai.scorer import match
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import ToolCall, tool
from mlflow.tracking import MlflowClient


@tool
def add():
    """Add two numbers."""

    async def execute(a: int, b: int) -> str:
        """Add two integers.

        Args:
            a: First integer.
            b: Second integer.
        """
        return str(a + b)

    return execute


def main():
    root = Path.cwd()
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{root / 'tracking.db'}"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "local-verification"
    os.environ["MLFLOW_INSPECT_TRACING"] = "true"
    os.environ["INSPECT_MLFLOW_AUTOLOG_ENABLED"] = "false"
    os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "false"
    client = MlflowClient()
    experiment_id = client.create_experiment(
        "local-verification", artifact_location=(root / "artifacts").as_uri()
    )
    outputs = []
    for index in range(2):
        outputs.extend(
            [
                ModelOutput(
                    model="mockllm/model",
                    choices=[
                        ChatCompletionChoice(
                            message=ChatMessageAssistant(
                                content="",
                                tool_calls=[
                                    ToolCall(
                                        id=f"add-{index}",
                                        function="add",
                                        arguments={"a": 2, "b": 2},
                                    )
                                ],
                            )
                        )
                    ],
                ),
                ModelOutput.from_content(model="mockllm/model", content="4"),
            ]
        )
    logs = eval(
        Task(
            name="local-arithmetic",
            dataset=[
                Sample(id="one", input="Use add to calculate 2 + 2.", target="4"),
                Sample(id="two", input="Use add to calculate 2 + 2.", target="4"),
            ],
            solver=[use_tools([add()]), generate()],
            scorer=match(),
        ),
        model=get_model("mockllm/model", custom_outputs=outputs),
        max_samples=1,
        log_dir=str(root / "logs"),
        display="none",
    )
    assert len(logs) == 1 and logs[0].status == "success", logs
    assert all(sample.scores["match"].value == "C" for sample in logs[0].samples)
    runs = client.search_runs([experiment_id])
    assert len(runs) == 2, [(r.info.run_id, r.info.status) for r in runs]
    child = next(r for r in runs if "mlflow.parentRunId" in r.data.tags)
    parent = next(r for r in runs if "mlflow.parentRunId" not in r.data.tags)
    assert child.data.tags["mlflow.parentRunId"] == parent.info.run_id
    assert all(r.info.status == "FINISHED" for r in runs)
    assert child.data.metrics["match/accuracy"] == 1.0
    assert child.data.metrics["completed_samples"] == 2
    assert child.data.metrics["total_model_calls"] == 4
    assert child.data.metrics["total_tool_calls"] == 2
    artifacts = {a.path for a in client.list_artifacts(child.info.run_id, "inspect")}
    assert {"inspect/tasks.json", "inspect/samples.json", "inspect/events.json"} <= artifacts
    traces = client.search_traces(experiment_ids=[experiment_id])
    assert len(traces) == 1, traces
    spans = traces[0].data.spans
    counts = {
        kind: sum(s.span_type == kind for s in spans) for kind in ["LLM", "TOOL", "EVALUATOR"]
    }
    assert counts["LLM"] == 4 and counts["TOOL"] == 2 and counts["EVALUATOR"] == 2, counts
    assert all(s.end_time_ns is not None for s in spans)

    async def check_scout():
        from inspect_mlflow.scout import import_mlflow_traces

        transcripts = [t async for t in import_mlflow_traces(experiment_name="local-verification")]
        assert len(transcripts) == 1
        assert len(transcripts[0].events) == 8
        return len(transcripts)

    imported = asyncio.run(check_scout())
    print(
        json.dumps(
            {
                "runs": len(runs),
                "traces": len(traces),
                "spans": counts,
                "artifacts": sorted(artifacts),
                "scout_transcripts": imported,
            }
        )
    )


if __name__ == "__main__":
    main()
