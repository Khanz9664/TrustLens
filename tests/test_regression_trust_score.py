"""
tests/test_regression_trust_score.py.
=====================================
Unit tests for the regression Trust Score (``regression_trust_score``, RFC #145).

Covers the converged v1 spec:
  * three dimensions (Accuracy/Skill 0.30, Interval Calibration 0.40,
    Uncertainty Informativeness 0.30) and their sub-score helpers,
  * weight redistribution when optional dimensions are absent (point-only,
    intervals-only),
  * blockers (negative skill; severe interval miscoverage),
  * penalties (heavy tail docks Accuracy/Skill; weak correlation docks the
    composite),
  * interface reuse (TrustScoreResult + 0–100 + A–D + task_type tagging).
"""

from __future__ import annotations

import numpy as np
import pytest

from trustlens.trust_score import (
    TrustScoreResult,
    _interval_calibration_score,
    _regression_accuracy_score,
    _uncertainty_informativeness_score,
    compute_trust_score,
    regression_trust_score,
)

# A fixed target with a known (non-trivial) variance so we can dial in an exact
# skill score by choosing rmse = sqrt((1 - skill) * Var(y)).
Y = np.arange(100, dtype=float)
VAR_Y = float(np.var(Y))


# ---------------------------------------------------------------------------
# Builders for controlled metric dicts (shape mirrors metrics/regression.py)
# ---------------------------------------------------------------------------


def _ed(skill: float, median: float = 2.0, p90: float = 4.0) -> dict:
    """An ``error_distribution`` dict with the rmse that yields the given skill."""
    rmse = float(np.sqrt((1.0 - skill) * VAR_Y))
    return {
        "median_absolute_error": median,
        "p90_absolute_error": p90,
        "max_error": p90 * 2.0,
        "mean_absolute_error": median,
        "rmse": round(rmse, 6),
        "n_samples": 100,
    }


def _cov(cal_err: float, width: float = 2.0, target: float = 0.95) -> dict:
    """A populated ``interval_coverage`` (PICP) dict with the given calibration error."""
    return {
        "picp": round(target + cal_err, 4),
        "target_coverage": target,
        "calibration_error": round(cal_err, 4),
        "mean_interval_width": width,
        "verdict": "well-calibrated",
        "n_samples": 100,
    }


def _cov_skipped() -> dict:
    return {"status": "skipped", "reason": "missing_intervals", "details": "no intervals"}


def _corr(pearson: float, spearman: float | None = None) -> dict:
    sp = pearson if spearman is None else spearman
    return {
        "pearson": round(pearson, 4),
        "spearman": round(sp, 4),
        "verdict": "informative",
        "n_samples": 100,
    }


def _corr_skipped() -> dict:
    return {"status": "skipped", "reason": "missing_variance", "details": "no variance"}


def _results(ed: dict, cov: dict | None = None, corr: dict | None = None) -> dict:
    return {
        "regression": {
            "error_distribution": ed,
            "interval_coverage": cov if cov is not None else _cov_skipped(),
            "error_variance_correlation": corr if corr is not None else _corr_skipped(),
        }
    }


# ---------------------------------------------------------------------------
# Interface reuse + task tagging
# ---------------------------------------------------------------------------


def test_returns_trustscoreresult_tagged_regression():
    r = regression_trust_score(_results(_ed(0.9)), Y)
    assert isinstance(r, TrustScoreResult)
    assert r.task_type == "regression"
    assert 0 <= r.score <= 100
    assert r.grade in {"A", "B", "C", "D"}


def test_classification_score_is_tagged_classification():
    cls = compute_trust_score({"calibration": {"brier_score": 0.0, "ece": 0.0}})
    assert cls.task_type == "classification"


def test_accepts_full_or_inner_results_dict():
    full = _results(_ed(0.9), _cov(0.0), _corr(0.9))
    inner = full["regression"]
    assert regression_trust_score(full, Y).score == regression_trust_score(inner, Y).score


# ---------------------------------------------------------------------------
# Weight redistribution (graceful degradation)
# ---------------------------------------------------------------------------


def test_point_only_scores_on_accuracy_alone():
    r = regression_trust_score(_results(_ed(0.9)), Y)  # no intervals, no variance
    assert set(r.weights_used) == {"accuracy"}
    assert r.weights_used["accuracy"] == pytest.approx(1.0)
    assert r.sub_scores["accuracy"] == pytest.approx(90.0, abs=2.0)
    assert r.grade == "A"


def test_intervals_only_redistributes_two_dimensions():
    r = regression_trust_score(_results(_ed(0.9), _cov(0.0), corr=None), Y)
    assert set(r.weights_used) == {"accuracy", "interval_calibration"}
    # weights_used is rounded to 3 dp in the result (matching the classification scorer).
    assert r.weights_used["accuracy"] == pytest.approx(0.30 / 0.70, abs=1e-3)
    assert r.weights_used["interval_calibration"] == pytest.approx(0.40 / 0.70, abs=1e-3)
    assert "uncertainty_informativeness" not in r.sub_scores


def test_all_three_dimensions_default_weights():
    r = regression_trust_score(_results(_ed(0.9), _cov(0.0), _corr(0.9)), Y)
    assert set(r.sub_scores) == {"accuracy", "interval_calibration", "uncertainty_informativeness"}
    assert r.weights_used["accuracy"] == pytest.approx(0.30, abs=1e-9)
    assert r.weights_used["interval_calibration"] == pytest.approx(0.40, abs=1e-9)
    assert r.weights_used["uncertainty_informativeness"] == pytest.approx(0.30, abs=1e-9)
    # 90*.3 + 100*.4 + 90*.3 = 94 -> grade A, no penalties, not blocked.
    assert r.grade == "A"
    assert not r.is_blocked
    assert r.penalties_applied == {}


def test_custom_weights_respected():
    r = regression_trust_score(
        _results(_ed(0.9), _cov(0.0), _corr(0.9)),
        Y,
        weights={
            "accuracy": 0.5,
            "interval_calibration": 0.25,
            "uncertainty_informativeness": 0.25,
        },
    )
    assert r.weights_used["accuracy"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Blockers -> grade D
# ---------------------------------------------------------------------------


def test_negative_skill_blocks():
    r = regression_trust_score(_results(_ed(-0.2), _cov(0.0), _corr(0.9)), Y)
    assert r.is_blocked
    assert r.grade == "D"
    assert "negative skill" in r.verdict.lower()


def test_severe_miscoverage_blocks():
    r = regression_trust_score(_results(_ed(0.9), _cov(-0.15), _corr(0.9)), Y)
    assert r.is_blocked
    assert r.grade == "D"
    assert "miscoverage" in r.verdict.lower()


def test_miscoverage_just_above_threshold_not_blocked():
    # calibration_error = -0.08 is over-confident but not "severe" (< -0.10).
    r = regression_trust_score(_results(_ed(0.9), _cov(-0.08), _corr(0.9)), Y)
    assert not r.is_blocked


def test_over_coverage_never_blocks():
    # Under-confident (too-wide) intervals are wasteful, not unsafe.
    r = regression_trust_score(_results(_ed(0.9), _cov(0.04), _corr(0.9)), Y)
    assert not r.is_blocked


# ---------------------------------------------------------------------------
# Penalties (not blockers)
# ---------------------------------------------------------------------------


def test_heavy_tail_docks_accuracy_dimension():
    light = regression_trust_score(_results(_ed(0.9, median=2.0, p90=4.0)), Y)  # ratio 2 -> no dock
    heavy = regression_trust_score(
        _results(_ed(0.9, median=2.0, p90=30.0)), Y
    )  # ratio 15 -> max dock
    assert light.sub_scores["accuracy"] == pytest.approx(90.0, abs=2.0)
    # Heavy tail caps the dock at 50% of the sub-score.
    assert heavy.sub_scores["accuracy"] == pytest.approx(45.0, abs=2.0)
    assert heavy.sub_scores["accuracy"] < light.sub_scores["accuracy"]
    assert heavy.score < light.score


def test_weak_uncertainty_correlation_penalizes_composite():
    strong = regression_trust_score(_results(_ed(0.9), _cov(0.0), _corr(0.9)), Y)
    weak = regression_trust_score(_results(_ed(0.9), _cov(0.0), _corr(0.0)), Y)
    assert "Weak Uncertainty" in weak.penalties_applied
    # corr 0 -> full penalty.
    assert weak.penalties_applied["Weak Uncertainty"] == pytest.approx(15.0, abs=0.1)
    # Double hit: the informativeness sub-score is also 0 here.
    assert weak.sub_scores["uncertainty_informativeness"] == pytest.approx(0.0)
    assert weak.score < strong.score
    # No weak-corr penalty when no uncertainty signal is present at all.
    point_only = regression_trust_score(_results(_ed(0.9)), Y)
    assert "Weak Uncertainty" not in point_only.penalties_applied


def test_strong_correlation_no_penalty():
    r = regression_trust_score(_results(_ed(0.9), _cov(0.0), _corr(0.5)), Y)
    # corr exactly at the informative boundary -> no penalty.
    assert "Weak Uncertainty" not in r.penalties_applied


# ---------------------------------------------------------------------------
# Sub-score helpers
# ---------------------------------------------------------------------------


def test_interval_calibration_score_helper():
    assert _interval_calibration_score({"calibration_error": 0.0}) == pytest.approx(100.0)
    assert _interval_calibration_score({"calibration_error": 0.10}) == pytest.approx(50.0)
    assert _interval_calibration_score({"calibration_error": -0.20}) == pytest.approx(0.0)
    assert _interval_calibration_score({"calibration_error": 0.40}) == pytest.approx(0.0)  # clipped


def test_uncertainty_informativeness_score_helper():
    assert _uncertainty_informativeness_score({"pearson": 0.8, "spearman": 0.6}) == pytest.approx(
        80.0
    )
    assert _uncertainty_informativeness_score({"pearson": -0.5, "spearman": -0.2}) == pytest.approx(
        0.0
    )
    # Out-of-range values are clipped to [0, 100].
    assert _uncertainty_informativeness_score({"pearson": 1.5, "spearman": 0.2}) == pytest.approx(
        100.0
    )


def test_accuracy_score_helper_skill_and_tail():
    res = _regression_accuracy_score(_ed(0.8, median=2.0, p90=4.0), VAR_Y)
    assert res["skill"] == pytest.approx(0.8, abs=0.02)
    assert res["tail_dock"] == pytest.approx(0.0)
    assert res["score"] == pytest.approx(80.0, abs=2.0)

    neg = _regression_accuracy_score(_ed(-0.3), VAR_Y)
    assert neg["skill"] < 0.0
    assert neg["score"] == pytest.approx(0.0)


def test_constant_target_perfect_fit_full_skill():
    y_const = np.ones(50)
    ed = {
        "median_absolute_error": 0.0,
        "p90_absolute_error": 0.0,
        "max_error": 0.0,
        "mean_absolute_error": 0.0,
        "rmse": 0.0,
        "n_samples": 50,
    }
    r = regression_trust_score(
        {
            "regression": {
                "error_distribution": ed,
                "interval_coverage": _cov_skipped(),
                "error_variance_correlation": _corr_skipped(),
            }
        },
        y_const,
    )
    assert not r.is_blocked
    assert r.sub_scores["accuracy"] == pytest.approx(100.0)
    assert r.grade == "A"
