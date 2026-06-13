"""Data types for evaluation comparison results."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MetricComparison:
    """Comparison of an aggregate metric between two evaluation runs."""

    name: str
    """Metric name (e.g., 'accuracy', 'mean')."""

    scorer: str
    """Scorer that produced this metric."""

    baseline_value: float
    """Metric value in the baseline run."""

    candidate_value: float
    """Metric value in the candidate run."""

    delta: float
    """Absolute difference (candidate - baseline)."""

    relative_delta: float | None
    """Relative change as a fraction (delta / baseline). None if baseline is zero."""

    significant: bool
    """Whether the difference is statistically significant."""

    p_value: float | None
    """P-value from significance test. None if not computed."""

    ci_lower: float | None
    """Lower bound of confidence interval for the difference."""

    ci_upper: float | None
    """Upper bound of confidence interval for the difference."""

    effect_size: float | None = None
    """Cohen's d effect size. None if not computed."""


@dataclass
class SampleComparison:
    """Comparison of a single sample's score between two runs."""

    id: int | str
    """Sample ID."""

    epoch: int
    """Epoch number."""

    scorer: str
    """Scorer that produced this score."""

    baseline_score: float | None
    """Score value in the baseline run. None if sample missing from baseline."""

    candidate_score: float | None
    """Score value in the candidate run. None if sample missing from candidate."""

    delta: float | None
    """Score difference (candidate - baseline). None if either score is missing."""

    direction: Literal["improved", "regressed", "unchanged", "new", "missing"]
    """Classification of the score change between runs."""


@dataclass
class ComparisonResult:
    """Complete comparison of two evaluation runs."""

    baseline_log: str
    """Path to baseline log file."""

    candidate_log: str
    """Path to candidate log file."""

    baseline_task: str
    """Task name from baseline."""

    candidate_task: str
    """Task name from candidate."""

    baseline_model: str
    """Model name from baseline."""

    candidate_model: str
    """Model name from candidate."""

    metrics: list[MetricComparison] = field(default_factory=list)
    """Aggregate metric comparisons."""

    samples: list[SampleComparison] = field(default_factory=list)
    """Per-sample score comparisons."""

    baseline_total_cost_usd: float | None = None
    """Total estimated USD cost of the baseline run, summed across all
    models. None if no provider in inspect_ai surfaced a cost estimate."""

    candidate_total_cost_usd: float | None = None
    """Total estimated USD cost of the candidate run, summed across all
    models. None if no provider in inspect_ai surfaced a cost estimate."""

    cost_delta_usd: float | None = None
    """Cost difference (candidate - baseline). Positive means the
    candidate run cost more. None if either run lacks cost data."""

    baseline_latency_p95_seconds: float | None = None
    """95th-percentile per-sample wall-clock latency for the baseline run.
    None if no per-sample timings are recorded."""

    candidate_latency_p95_seconds: float | None = None
    """95th-percentile per-sample wall-clock latency for the candidate run.
    None if no per-sample timings are recorded."""

    latency_p95_delta_seconds: float | None = None
    """Latency p95 difference (candidate - baseline). Positive means the
    candidate is slower at the tail. None if either run lacks timing data."""

    @functools.cached_property
    def _direction_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {
            "improved": 0,
            "regressed": 0,
            "unchanged": 0,
            "new": 0,
            "missing": 0,
        }
        for s in self.samples:
            counts[s.direction] += 1
        return counts

    @property
    def regressions(self) -> list[SampleComparison]:
        """Samples where the candidate scored lower than baseline."""
        return [s for s in self.samples if s.direction == "regressed"]

    @property
    def improvements(self) -> list[SampleComparison]:
        """Samples where the candidate scored higher than baseline."""
        return [s for s in self.samples if s.direction == "improved"]

    @property
    def unchanged(self) -> list[SampleComparison]:
        """Samples with identical scores in both runs."""
        return [s for s in self.samples if s.direction == "unchanged"]

    @property
    def aligned_count(self) -> int:
        """Number of samples present in both runs."""
        c = self._direction_counts
        return c["improved"] + c["regressed"] + c["unchanged"]

    @property
    def missing_count(self) -> int:
        """Samples in baseline but not in candidate."""
        return self._direction_counts["missing"]

    @property
    def new_count(self) -> int:
        """Samples in candidate but not in baseline."""
        return self._direction_counts["new"]

    @property
    def win_rate(self) -> float | None:
        """Fraction of aligned samples where candidate outperformed baseline."""
        aligned = self.aligned_count
        if aligned == 0:
            return None
        return self._direction_counts["improved"] / aligned

    def summary(self) -> str:
        """Generate a text summary of the comparison."""
        lines = [
            f"Baseline:  {self.baseline_model} ({self.baseline_task})",
            f"Candidate: {self.candidate_model} ({self.candidate_task})",
            f"Samples:   {self.aligned_count} aligned, "
            f"{self.missing_count} missing, {self.new_count} new",
            "",
        ]

        if self.metrics:
            metric_names = [f"{m.scorer}/{m.name}" for m in self.metrics]
            name_width = max(len(n) for n in metric_names) + 2

            header = (
                f"  {'Metric':<{name_width}}"
                f"{'Baseline':>10}"
                f"{'Candidate':>11}"
                f"{'Delta':>18}"
                f"{'Sig.':>12}"
            )
            lines.append(header)
            lines.append("  " + "-" * (len(header) - 2))

            for m, name in zip(self.metrics, metric_names, strict=True):
                delta_str = f"{m.delta:+.4f}"
                if m.relative_delta is not None:
                    delta_str += f" ({m.relative_delta:+.1%})"

                sig_str = ""
                if m.p_value is not None:
                    sig_str = f"p={m.p_value:.3f}"
                    if m.significant:
                        sig_str += "*"

                lines.append(
                    f"  {name:<{name_width}}"
                    f"{m.baseline_value:>10.4f}"
                    f"{m.candidate_value:>11.4f}"
                    f"{delta_str:>18}"
                    f"  {sig_str}"
                )
            for m, name in zip(self.metrics, metric_names, strict=True):
                if m.effect_size is not None:
                    d = abs(m.effect_size)
                    size = "small" if d < 0.5 else ("medium" if d < 0.8 else "large")
                    lines.append(
                        f"  Effect size ({name}): Cohen's d = {m.effect_size:+.2f} ({size} effect)"
                    )
            lines.append("")

        c = self._direction_counts
        lines.append(
            f"Regressions: {c['regressed']}, "
            f"Improvements: {c['improved']}, "
            f"Unchanged: {c['unchanged']}"
        )

        if self.win_rate is not None:
            lines.append(
                f"Candidate won on {self._direction_counts['improved']} "
                f"of {self.aligned_count} samples ({self.win_rate:.1%})"
            )

        if self.cost_delta_usd is not None and self.baseline_total_cost_usd is not None:
            sign = "+" if self.cost_delta_usd >= 0 else "-"
            rel = ""
            if self.baseline_total_cost_usd > 0:
                pct = self.cost_delta_usd / self.baseline_total_cost_usd
                pct_sign = "+" if pct >= 0 else "-"
                rel = f" ({pct_sign}{abs(pct):.1%})"
            lines.append(
                f"Cost:        ${self.baseline_total_cost_usd:.4f} -> "
                f"${self.candidate_total_cost_usd:.4f} "
                f"({sign}${abs(self.cost_delta_usd):.4f}{rel})"
            )

        if (
            self.latency_p95_delta_seconds is not None
            and self.baseline_latency_p95_seconds is not None
        ):
            sign = "+" if self.latency_p95_delta_seconds >= 0 else "-"
            rel = ""
            if self.baseline_latency_p95_seconds > 0:
                pct = self.latency_p95_delta_seconds / self.baseline_latency_p95_seconds
                pct_sign = "+" if pct >= 0 else "-"
                rel = f" ({pct_sign}{abs(pct):.1%})"
            lines.append(
                f"Latency p95: {self.baseline_latency_p95_seconds:.3f}s -> "
                f"{self.candidate_latency_p95_seconds:.3f}s "
                f"({sign}{abs(self.latency_p95_delta_seconds):.3f}s{rel})"
            )

        return "\n".join(lines)
