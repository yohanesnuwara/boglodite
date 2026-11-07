"""Unit tests for the Norne scorer's metric logic (no TRUTH data needed).

The data-dependent behaviour is covered by `python -m norne_benchmark.score --selftest`,
which needs the Norne drive mounted.
"""

import math

import numpy as np

from norne_benchmark.score import Metric, StageResult, mae, r2, rmse
from norne_benchmark.truth import adjusted_rand


def test_error_metric_passes_at_or_below_tolerance():
    assert Metric("m", "depth MAE", "m", 6.2, 10).passed
    assert Metric("m", "depth MAE", "m", 10.0, 10).passed
    assert not Metric("m", "depth MAE", "m", 10.1, 10).passed
    assert math.isclose(Metric("m", "", "m", 10.0, 10).score, math.exp(-1))
    assert Metric("m", "", "m", 0.0, 10).score == 1.0


def test_signed_errors_use_magnitude():
    assert Metric("e", "STOIIP error", "%", -4.0, 5).passed
    assert not Metric("e", "STOIIP error", "%", -6.0, 5).passed


def test_at_least_metric():
    f1 = Metric("f1", "Fault F1", "", 0.80, 0.75, "ge")
    assert f1.passed and f1.err < 1
    assert not Metric("f1", "", "", 0.70, 0.75, "ge").passed
    assert Metric("f1", "", "", 1.0, 0.75, "ge").score == 1.0


def test_missing_value_fails():
    m = Metric("m", "", "m", None, 10)
    assert not m.passed and m.score == 0.0 and m.err == math.inf


def test_outcome_classes():
    ok = StageResult("STRUCT", [Metric("a", "", "m", 5, 10), Metric("b", "", "", 0.9, 0.75, "ge")])
    assert ok.outcome == "correct"
    assert StageResult("STRUCT", [Metric("a", "", "m", 15, 10)]).outcome == "degraded"
    assert StageResult("STRUCT", [Metric("a", "", "m", 31, 10)]).outcome == "silent_failure"
    assert StageResult("FLT", [Metric("f", "", "", 0.2, 0.75, "ge")]).outcome == "silent_failure"
    overt = StageResult("STRUCT", [Metric("a", "", "m", 0, 10)], overt="missing")
    assert overt.outcome == "overt_failure" and overt.score == 0.0


def test_secondary_metrics_do_not_count():
    st = StageResult("FLT", [Metric("p", "", "m", 0, 10), Metric("q", "", "m", 99, 10, primary=False)])
    assert st.score == 1.0 and st.outcome == "correct"


def test_error_statistics():
    assert mae([1, -1, np.nan]) == 1.0
    assert math.isclose(rmse([3, 4]), math.sqrt(12.5))
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(obs, obs) == 1.0


def test_adjusted_rand():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert adjusted_rand(a, a) == 1.0
    assert adjusted_rand(a, np.array([5, 5, 7, 7, 9, 9])) == 1.0      # labels are arbitrary
    assert adjusted_rand(a, np.array([0, 1, 0, 1, 0, 1])) < 0.1
