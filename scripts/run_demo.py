"""Run a demo eval with both tracking and tracing hooks active.

Produces MLflow runs (metrics, params, artifacts) AND traces (span tree).
Used to generate README screenshots showing both hooks in action.
"""

import os
import sys

os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5557"
os.environ["MLFLOW_INSPECT_TRACING"] = "true"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "inspect-mlflow-demo"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlflow
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate

from inspect_mlflow.tracing import MlflowTracingHooks  # noqa: F401

# Import both hooks
from inspect_mlflow.tracking import MlflowTrackingHooks  # noqa: F401


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5557")
    mlflow.set_experiment("inspect-mlflow-demo")

    dataset = [
        Sample(input="What is 2 + 2?", target="4"),
        Sample(input="What is 3 * 5?", target="15"),
        Sample(input="What is 10 - 7?", target="3"),
        Sample(input="What is 8 / 2?", target="4"),
        Sample(input="What is 6 + 9?", target="15"),
    ]

    task = Task(
        dataset=dataset,
        solver=generate(),
        scorer=match(),
    )

    print("Running eval with tracking + tracing hooks...")
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

    # Verify both hooks produced data
    exp = mlflow.get_experiment_by_name("inspect-mlflow-demo")
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    traces = mlflow.search_traces(experiment_ids=[exp.experiment_id])
    print(f"\nMLflow runs: {len(runs)}")
    print(f"MLflow traces: {len(traces)}")
    print("\nOpen http://127.0.0.1:5557 to see results.")


if __name__ == "__main__":
    main()
