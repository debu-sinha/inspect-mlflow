"""Evaluation comparison and regression detection.

Compare results from two Inspect AI evaluation runs to detect score
regressions, compute statistical significance, and generate reports.

Example usage:

.. code-block:: python

    from inspect_mlflow.comparison import compare_evals

    result = compare_evals("logs/baseline.eval", "logs/candidate.eval")
    print(result.summary())

    for r in result.regressions:
        print(f"Sample {r.id} regressed: {r.baseline_score} -> {r.candidate_score}")
"""

from inspect_mlflow.comparison._compare import compare_evals
from inspect_mlflow.comparison._statistics import cohens_d
from inspect_mlflow.comparison._types import (
    ComparisonResult,
    MetricComparison,
    SampleComparison,
)

__all__ = [
    "ComparisonResult",
    "MetricComparison",
    "SampleComparison",
    "cohens_d",
    "compare_evals",
]
