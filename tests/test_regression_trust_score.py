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

import warnings

import numpy as np
import pytest

from trustlens.trust_score import (
    TrustScoreResult,
    _informativeness_from_sharpness,
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


def _cov_ml(ice: float, sharpness_skill: float | None = 0.4, worst: float | None = None) -> dict:
    """A multi-level interval_coverage dict (ICE + sharpness proxy), as emitted by
    ``multilevel_interval_coverage`` (RFC #155)."""
    return {
        "ice": round(ice, 4),
        "sharpness_skill": sharpness_skill,
        "n_levels": 3,
        "n_calibrated_levels": 3 if sharpness_skill is not None else 0,
        "worst_calibration_error": round(worst if worst is not None else -ice, 4),
        "mean_interval_width": 2.0,
        "verdict": "well-calibrated" if ice <= 0.05 else "miscalibrated",
        "n_samples": 100,
    }


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


def test_constant_target_imperfect_fit_zero_skill_not_blocked():
    # Locks down the constant-target edge raised in review (#147): with Var(y)=0
    # a perfect fit earns full skill (covered above); an IMPERFECT fit collapses
    # to *zero* skill. Crucially skill is 0.0, not negative, so this is NOT the
    # negative-skill blocker — it degrades gracefully to a zero Accuracy/Skill
    # dimension, and point-only redistribution carries that onto the composite.
    assert _regression_accuracy_score({"rmse": 5.0}, 0.0)["skill"] == pytest.approx(0.0)

    y_const = np.ones(50)
    ed = {
        "median_absolute_error": 2.0,
        "p90_absolute_error": 4.0,  # tail_ratio 2.0 < threshold → no heavy-tail dock
        "max_error": 8.0,
        "mean_absolute_error": 2.0,
        "rmse": 5.0,  # imperfect fit against a constant target
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
    assert not r.is_blocked  # skill 0.0 is not < 0 → not a blocker
    assert r.sub_scores["accuracy"] == pytest.approx(0.0)
    assert r.grade == "D"


# ---------------------------------------------------------------------------
# Persisted target_variance — recompute a regression Trust Score from a stored
# report without the original y_true (issue #150)
# ---------------------------------------------------------------------------


def test_stored_variance_recompute_equals_y_true():
    """A score recomputed from a persisted target_variance is identical to the
    score computed from y_true."""
    res = _results(_ed(0.9), _cov(0.0), _corr(0.9))
    from_y_true = regression_trust_score(res, Y)
    res["regression"]["target_variance"] = VAR_Y  # what the pipeline persists
    from_stored = regression_trust_score(res)  # no y_true in memory
    assert from_stored.score == from_y_true.score
    assert from_stored.grade == from_y_true.grade
    assert from_stored.sub_scores == from_y_true.sub_scores
    assert from_stored.breakdown == from_y_true.breakdown


def test_missing_both_y_true_and_stored_variance_raises():
    """Neither y_true nor a persisted variance → explicit error, never a silent
    0.0 that would masquerade as a constant target."""
    res = _results(_ed(0.9))  # no target_variance persisted
    with pytest.raises(ValueError, match="target variance"):
        regression_trust_score(res)  # and no y_true


def test_stored_zero_variance_is_honoured_not_treated_as_missing():
    """A genuine constant target persists as target_variance == 0.0 and must be
    used (constant-target branch), not rejected as 'missing'."""
    ed = _ed(0.0)
    ed["rmse"] = 0.0  # perfect fit on a constant target
    res = {
        "regression": {
            "error_distribution": ed,
            "interval_coverage": _cov_skipped(),
            "error_variance_correlation": _corr_skipped(),
            "target_variance": 0.0,
        }
    }
    r = regression_trust_score(res)  # no y_true; 0.0 must be used, not rejected
    assert r.task_type == "regression"
    assert r.sub_scores["accuracy"] == pytest.approx(100.0)  # perfect constant-target fit


def test_y_true_wins_and_warns_on_mismatch_with_stored():
    """When both are present and disagree, y_true wins and a warning fires."""
    res = _results(_ed(0.9), _cov(0.0), _corr(0.9))
    res["regression"]["target_variance"] = VAR_Y * 4.0  # deliberately wrong
    with pytest.warns(UserWarning, match="disagrees with the persisted"):
        mismatched = regression_trust_score(res, Y)
    clean = regression_trust_score(_results(_ed(0.9), _cov(0.0), _corr(0.9)), Y)
    assert mismatched.score == clean.score  # y_true wins → stored value ignored


def test_matching_stored_variance_does_not_warn():
    """A persisted variance equal to Var(y_true) must not warn."""
    res = _results(_ed(0.9), _cov(0.0), _corr(0.9))
    res["regression"]["target_variance"] = VAR_Y
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a test failure
        regression_trust_score(res, Y)


# ---------------------------------------------------------------------------
# Multi-level interval calibration: ICE + sharpness proxy (RFC #155)
# ---------------------------------------------------------------------------


class TestMultilevelIntervalCalibrationRFC155:
    def test_ice_drives_interval_calibration_subscore(self):
        # ICE maps through the same tolerance as |calibration_error|.
        assert _interval_calibration_score({"ice": 0.0}) == 100.0
        assert _interval_calibration_score({"ice": 0.20}) == 0.0
        assert _interval_calibration_score({"ice": 0.10}) == pytest.approx(50.0)

    def test_ice_preferred_over_single_picp_when_both_present(self):
        # A dict carrying both ICE and a (contradictory) calibration_error scores
        # from ICE — the multi-level summary wins.
        score = _interval_calibration_score({"ice": 0.0, "calibration_error": -0.2})
        assert score == 100.0

    def test_sharpness_proxy_drives_informativeness(self):
        assert _informativeness_from_sharpness({"sharpness_skill": 0.6}) == pytest.approx(60.0)
        assert _informativeness_from_sharpness({"sharpness_skill": -0.3}) == 0.0
        assert _informativeness_from_sharpness({"sharpness_skill": None}) == 0.0

    def test_informativeness_uses_proxy_not_correlation_when_intervals_present(self):
        # sharpness proxy present + a (strong) correlation also present -> proxy wins.
        r = regression_trust_score(
            _results(_ed(0.9), _cov_ml(0.0, sharpness_skill=0.6), _corr(0.95)), Y
        )
        assert r.sub_scores["uncertainty_informativeness"] == pytest.approx(60.0, abs=0.5)

    def test_proxy_path_suppresses_weak_correlation_penalty(self):
        # On the proxy path the weak-correlation composite penalty must NOT fire,
        # even when a weak correlation is also supplied.
        r = regression_trust_score(
            _results(_ed(0.9), _cov_ml(0.0, sharpness_skill=0.9), _corr(0.0)), Y
        )
        assert "Weak Uncertainty" not in r.penalties_applied

    def test_falls_back_to_correlation_when_no_sharpness(self):
        # Multi-level intervals with no calibrated level (sharpness_skill=None) ->
        # Informativeness falls back to the error-variance correlation.
        r = regression_trust_score(
            _results(_ed(0.9), _cov_ml(0.0, sharpness_skill=None), _corr(0.8)), Y
        )
        assert r.sub_scores["uncertainty_informativeness"] == pytest.approx(80.0, abs=0.5)

    def test_severe_multilevel_miscoverage_blocks(self):
        # Worst per-level gap below the severe-miscoverage threshold -> grade D.
        r = regression_trust_score(
            _results(_ed(0.9), _cov_ml(0.15, sharpness_skill=None, worst=-0.15), _corr(0.9)), Y
        )
        assert r.is_blocked
        assert r.grade == "D"

    def test_multilevel_well_calibrated_not_blocked(self):
        r = regression_trust_score(
            _results(_ed(0.9), _cov_ml(0.02, sharpness_skill=0.5, worst=-0.02), _corr(0.9)), Y
        )
        assert not r.is_blocked
        assert "interval_calibration" in r.sub_scores
        assert "uncertainty_informativeness" in r.sub_scores

    # -- Zero-calibrated-levels handling (PR #161 review follow-up) -----------
    # Multi-level intervals supplied but NO level passes the calibration gate
    # (and no correlation fallback): score Informativeness a truthful 0.0
    # ("unusable uncertainty") instead of dropping the dimension.

    def test_zero_calibrated_multilevel_scores_informativeness_zero(self):
        # sharpness_skill=None (n_calibrated_levels=0) + corr absent -> the
        # supplied uncertainty was unusable, not absent: keep the dimension at 0.0.
        r = regression_trust_score(
            _results(_ed(0.9), _cov_ml(0.02, sharpness_skill=None, worst=-0.02), corr=None), Y
        )
        assert "uncertainty_informativeness" in r.sub_scores
        assert r.sub_scores["uncertainty_informativeness"] == pytest.approx(0.0)
        assert r.informativeness_status == "unusable_uncertainty"
        # Weight is KEPT (a real penalty), not redistributed onto the other dims.
        assert r.weights_used["uncertainty_informativeness"] == pytest.approx(0.30, abs=1e-9)

    def test_unusable_uncertainty_scores_below_usable(self):
        # Identical accuracy + interval calibration; only informativeness differs
        # (usable sharpness vs all-levels-failed). The unusable case must score
        # strictly lower, confirming 0.0 pulls the weighted score down.
        usable = regression_trust_score(
            _results(_ed(0.9), _cov_ml(0.02, sharpness_skill=0.5, worst=-0.02), corr=None), Y
        )
        unusable = regression_trust_score(
            _results(_ed(0.9), _cov_ml(0.02, sharpness_skill=None, worst=-0.02), corr=None), Y
        )
        assert unusable.score < usable.score
        assert usable.informativeness_status == "present"
        assert unusable.informativeness_status == "unusable_uncertainty"

    def test_zero_calibrated_multilevel_with_correlation_keeps_fallback(self):
        # Precedence guard: a usable error-variance correlation still wins. The
        # 0.0 rule only fires when nothing else can score the dimension.
        r = regression_trust_score(
            _results(_ed(0.9), _cov_ml(0.02, sharpness_skill=None, worst=-0.02), _corr(0.8)), Y
        )
        assert r.sub_scores["uncertainty_informativeness"] == pytest.approx(80.0, abs=0.5)
        assert r.informativeness_status == "present"

    def test_single_level_no_correlation_still_redistributes(self):
        # Scope guard: the 0.0 rule is multi-level only (n_levels >= 2). A
        # single-level PICP dict carries no such signal, so an absent correlation
        # keeps the old redistribute path (dimension dropped), not a 0.0.
        r = regression_trust_score(_results(_ed(0.9), _cov(0.0), corr=None), Y)
        assert "uncertainty_informativeness" not in r.sub_scores
        assert r.informativeness_status == "absent"

    def test_informativeness_status_present_on_sharpness_path(self):
        r = regression_trust_score(
            _results(_ed(0.9), _cov_ml(0.0, sharpness_skill=0.6), corr=None), Y
        )
        assert r.informativeness_status == "present"

    def test_informativeness_status_none_for_classification(self):
        cls = compute_trust_score({"calibration": {"brier_score": 0.0, "ece": 0.0}})
        assert cls.informativeness_status is None
