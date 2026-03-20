"""Run a complex eval with tools to demonstrate TOOL spans + full tracking."""

import os
import sys

os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5557"
os.environ["MLFLOW_INSPECT_TRACING"] = "true"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "inspect-mlflow-demo"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import tool

from inspect_mlflow.tracing import MlflowTracingHooks  # noqa: F401
from inspect_mlflow.tracking import MlflowTrackingHooks  # noqa: F401


@tool
def calculator():
    """Perform arithmetic calculations. Pass a math expression like '2+2' or '47*89'."""

    async def run(expression: str) -> str:
        """Evaluate a math expression and return the result.

        Args:
            expression: A math expression to evaluate, e.g. "47 * 89"
        """
        try:
            allowed = {"__builtins__": {}}
            result = eval(expression, allowed)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    return run


def main():
    dataset = [
        Sample(
            input="Use the calculator tool to compute 47 * 89, then tell me the answer.",
            target="4183",
        ),
        Sample(
            input="Use the calculator tool to compute 1024 / 16, then tell me the answer.",
            target="64",
        ),
        Sample(
            input="Use the calculator tool to compute 123 + 456 + 789, then tell me the answer.",
            target="1368",
        ),
        Sample(
            input="What is 7 * 8? Use the calculator tool.",
            target="56",
        ),
        Sample(
            input="Use the calculator to find 999 - 333.",
            target="666",
        ),
    ]

    task = Task(
        dataset=dataset,
        solver=[use_tools([calculator()]), generate()],
        scorer=match(),
    )

    print("Running complex eval with tools + tracking + tracing...")
    logs = eval(
        task,
        model="openai/gpt-4o-mini",
        log_dir="/tmp/inspect-mlflow-demo-logs",
    )

    log = logs[0]
    print(f"\nStatus: {log.status}")
    if log.results and log.results.scores:
        for score in log.results.scores:
            for m, v in score.metrics.items():
                print(f"  {score.name}/{m}: {v.value}")
    print("\nOpen http://127.0.0.1:5557 to see results with tool call spans.")


if __name__ == "__main__":
    main()
