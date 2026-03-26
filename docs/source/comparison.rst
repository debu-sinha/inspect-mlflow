Evaluation Comparison
=====================

Compare results from two Inspect AI evaluation runs to detect score regressions,
compute statistical significance, and generate comparison reports.

Quick Start
-----------

.. code-block:: python

   from inspect_mlflow.comparison import compare_evals

   result = compare_evals("logs/baseline.eval", "logs/candidate.eval")
   print(result.summary())

   for r in result.regressions:
       print(f"Sample {r.id}: {r.baseline_score} -> {r.candidate_score}")

Output:

.. code-block:: text

   Baseline:  openai/gpt-4o-mini (math_task)
   Candidate: openai/gpt-4o-mini (math_task)
   Samples:   5 aligned, 0 missing, 0 new

     Metric            Baseline  Candidate             Delta        Sig.
     -------------------------------------------------------------------
     match/accuracy      0.6000     0.4000   -0.2000 (-33.3%)  p=0.048*
     Effect size (match/accuracy): Cohen's d = -0.73 (medium effect)

   Regressions: 2, Improvements: 1, Unchanged: 2
   Candidate won on 1 of 5 samples (20.0%)

Features
--------

- **Sample alignment** by ``(id, epoch)`` key with string/int ID normalization
- **Automatic test selection**: McNemar's test for binary scores (0/1), bootstrap CI for continuous
- **Effect size**: Cohen's d computed independently of sample size
- **Regression threshold**: filter noise with ``regression_threshold=0.05``
- **Sample filtering**: ``sample_filter=lambda s: s.id in subset``
- **Win rate** tracking across aligned samples
- **No scipy dependency**: all statistics implemented with NumPy only

Parameters
----------

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``baseline``
     - (required)
     - Path to baseline eval log or ``EvalLog`` object
   * - ``candidate``
     - (required)
     - Path to candidate eval log or ``EvalLog`` object
   * - ``scorers``
     - ``None``
     - Scorer names to compare. ``None`` compares all common scorers
   * - ``significance``
     - ``0.05``
     - P-value threshold for significance tests
   * - ``regression_threshold``
     - ``0.0``
     - Minimum delta to count as regression or improvement
   * - ``sample_filter``
     - ``None``
     - Function to filter samples before comparison

Statistical Tests
-----------------

The comparison module selects the appropriate test based on score distribution:

**Binary scores** (all values are 0.0 or 1.0): McNemar's test with continuity
correction. Tests whether discordant pairs (one run correct, other incorrect)
are asymmetrically distributed.

**Continuous scores**: Shifted bootstrap confidence interval with 10,000
resamples. Computes a two-sided p-value under the null hypothesis of no
difference.

**Effect size**: Cohen's d is always computed for primary metrics.
Values around 0.2 are small, 0.5 medium, and 0.8 large.

Result Objects
--------------

``ComparisonResult`` provides these properties:

- ``metrics``: aggregate metric comparisons with significance results
- ``samples``: per-sample score comparisons with direction classification
- ``regressions``: samples where candidate scored lower
- ``improvements``: samples where candidate scored higher
- ``unchanged``: samples with identical scores
- ``aligned_count``, ``missing_count``, ``new_count``: alignment counts
- ``win_rate``: fraction of aligned samples where candidate won
- ``summary()``: formatted text report

API Reference
-------------

.. autofunction:: inspect_mlflow.comparison.compare_evals

.. autofunction:: inspect_mlflow.comparison._statistics.cohens_d

.. autoclass:: inspect_mlflow.comparison.ComparisonResult
   :members:

.. autoclass:: inspect_mlflow.comparison.MetricComparison
   :members:

.. autoclass:: inspect_mlflow.comparison.SampleComparison
   :members:
