"""
trustlens.trust_score.
======================
The TrustLens Trust Score — a single 0–100 composite measure of model
trustworthiness.

Responsibilities
----------------
* Aggregate metrics from various modules (calibration, failure, bias, representation).
* Apply weightings and penalties to calculate a final 0-100 score.
* Determine the model's deployment verdict and letter grade.

Relationship to other components
--------------------------------
Used primarily by `TrustReport` to instantly summarize the complex `results`
dictionary into an actionable metric.

Why a single score?
-------------------
Practitioners face "metric overload": ECE, Brier Score, silhouette scores,
confidence gaps — great individually but hard to act on as a whole.

The Trust Score distils all TrustLens analysis into one instantly readable
number:

 * **< 40** — Serious issues. Do not deploy.
 * **40–60** — Moderate trust. Investigate flagged dimensions.
 * **60–80** — Good. Minor improvements recommended.
 * **80–100** — High trust. Model is production-ready.

Formula
-------
The Trust Score is a weighted sum of four normalized sub-scores (0–100 each):

 TrustScore = w_cal * CalibrationScore
       + w_fail * FailureScore
       + w_bias * BiasScore
       + w_rep * RepresentationScore

Default weights (tuned to reflect deployment risk):
 w_cal = 0.35  (calibration matters most — drives overconfidence risk)
 w_fail = 0.30  (failure patterns drive safety risk)
 w_bias = 0.25  (bias drives fairness/regulatory risk)
 w_rep = 0.10  (representation is a bonus signal; not always available)

If a dimension is unavailable (e.g., no embeddings → no representation score),
its weight is redistributed proportionally to the other available dimensions.

Sub-score Normalization
-----------------------
All sub-scores are normalized to [0, 100]:

 * CalibrationScore = 100 × (1 - clip(0.5×BS + 0.5×ECE, 0, 1))
   - Brier Score and ECE are both in [0, 1]; lower is better.
   - Perfect calibration → 100. Worst case (BS=1, ECE=1) → 0.

 * FailureScore = 100 × clip(confidence_gap, 0, 1)
   - Confidence gap in [0, 1] (clipped); higher is better.
   - A model that is highly confident *only* when correct → 100.

 * BiasScore = 100 × (1 - clip(bias_penalty, 0, 1))
   - bias_penalty = 0.5 × clip(imbalance_ratio / 20, 0, 1)
           + 0.5 × clip(subgroup_gap, 0, 1)
   - Perfectly balanced dataset, zero subgroup gap → 100.

 * RepresentationScore = 100 × clip(0.5 + 0.5 × silhouette, 0, 1)
   - Silhouette ∈ [-1, 1]; mapped to [0, 100].
   - Perfect separation → 100. Total overlap → 0.

References
----------
* Brier (1950), Guo et al. (2017) — calibration
* Hardt et al. (2016) — fairness
* Rousseeuw (1987) — silhouette
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from trustlens.visualization.style import BRAND_COLORS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
    "calibration": 0.35,
    "failure": 0.30,
    "bias": 0.25,
    "representation": 0.10,
}

_GRADE_THRESHOLDS = [
    (80, "A", "High Trust - production-ready"),
    (60, "B", "Good Trust - minor issues to address"),
    (40, "C", "Moderate Trust - investigate flagged dimensions"),
    (0, "D", "Low Trust - serious issues, do not deploy"),
]

_MAX_PENALTY_FAILURE = 20.0
_MAX_PENALTY_CALIBRATION = 15.0
_MAX_PENALTY_FAIRNESS = 15.0
_MAX_TOTAL_PENALTY = 35.0


# ---------------------------------------------------------------------------
# Sub-score computers
# ---------------------------------------------------------------------------


def _calibration_score(cal_data: dict) -> float:
    """
    Compute calibration sub-score (0–100).

    CalibScore = 100 × (1 − clip(BS + 1.5×ECE, 0, 1))
    """
    bs = float(cal_data.get("brier_score", 0.5))
    ece = float(cal_data.get("ece", 0.5))
    composite = bs + 1.5 * ece
    return 100.0 * (1.0 - float(np.clip(composite, 0.0, 1.0)))


def _failure_score(fail_data: dict) -> float:
    """
    Compute failure sub-score (0–100).

    FailScore = 100 × clip(confidence_gap, 0, 1)

    A large gap means the model is confident when right and uncertain when
    wrong — the ideal behaviour.
    """
    gap_data = fail_data.get("confidence_gap", {})
    gap = float(gap_data.get("gap", 0.0))

    # Also penalize high-confidence misclassifications
    misc = fail_data.get("misclassification_summary", {})
    overall = misc.get("__overall__", {})
    error_rate = float(overall.get("overall_error_rate", 0.5))

    # Combine: gap contribution (80%) + accuracy contribution (20%)
    gap_score = float(np.clip(gap, 0.0, 1.0))
    acc_score = 1.0 - float(np.clip(error_rate, 0.0, 1.0))
    score = 0.8 * gap_score + 0.2 * acc_score
    return 100.0 * float(np.clip(score, 0.0, 1.0))


def _bias_score(bias_data: dict) -> float:
    """
    Compute bias sub-score (0–100).

    BiasScore = 100 × (1 − clip(bias_penalty, 0, 1))
    bias_penalty = 0.5 × clip(imbalance_ratio/20, 0, 1)
           + 0.5 × max_subgroup_performance_gap
    """
    imbalance = bias_data.get("class_imbalance", {})
    ratio = float(imbalance.get("imbalance_ratio", 1.0))
    imbalance_penalty = float(np.clip((ratio - 1.0) / 19.0, 0.0, 1.0))

    # Subgroup performance gap (worst across all sensitive features)
    max_gap = 0.0
    subgroup = bias_data.get("subgroup_performance", {})
    for feat_data in subgroup.values():
        summary = feat_data.get("__summary__", {})
        gap = float(summary.get("performance_gap", 0.0))
        max_gap = max(max_gap, gap)

    subgroup_penalty = float(np.clip(max_gap, 0.0, 1.0))

    bias_penalty = 0.5 * imbalance_penalty + 0.5 * subgroup_penalty
    return 100.0 * (1.0 - float(np.clip(bias_penalty, 0.0, 1.0)))


def _representation_score(rep_data: dict) -> float:
    """
    Compute representation sub-score (0–100).

    RepScore = 100 × clip(0.5 + 0.5 × silhouette, 0, 1)
    """
    sep = rep_data.get("separability", {})
    sil = float(sep.get("silhouette_score", 0.0))
    if np.isnan(sil):
        sil = 0.0
    return 100.0 * float(np.clip(0.5 + 0.5 * sil, 0.0, 1.0))


# ---------------------------------------------------------------------------
# TrustScoreResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrustScoreResult:
    """
    Structured result from the Trust Score computation.

    Attributes
    ----------
    score : int
      Overall Trust Score in [0, 100].
    grade : str
      Letter grade: A / B / C / D.
    verdict : str
      Plain-English deployment recommendation.
    sub_scores : dict
      Per-dimension scores in [0, 100].
    weights_used : dict
      Actual weights used (after redistribution for missing dimensions).
    breakdown : dict
      Weighted contribution of each dimension to the final score.
    task_type : str
      The task the score was computed for — ``"classification"`` (default) or
      ``"regression"``. Both share this interface (0–100, A–D, verdicts), but a
      regression ``75`` and a classification ``75`` are **not** directly
      comparable: they aggregate different underlying dimensions.
    """

    score: int
    grade: str
    verdict: str
    sub_scores: dict[str, float] = field(default_factory=dict)
    weights_used: dict[str, float] = field(default_factory=dict)
    breakdown: dict[str, float] = field(default_factory=dict)
    penalties_applied: dict[str, float] = field(default_factory=dict)
    base_score: int = 0
    is_blocked: bool = False
    task_type: str = "classification"

    def __str__(self) -> str:
        lines = [
            f"Trust Score: {self.score}/100 [{self.grade}]",
            f"Assessment : {self.verdict}",
            "\nDimension Breakdown:",
        ]
        for dim, score in self.sub_scores.items():
            lines.append(f"  - {dim:<18} {score:5.1f}/100")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"TrustScoreResult(score={self.score}, grade={self.grade!r})"

    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks."""
        from trustlens.visualization.summary_plot import _color_for_grade, _color_for_score

        gc = _color_for_grade(self.grade)

        html = f"""
        <div style="font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                    max-width: 450px; padding: 20px; border-radius: 12px;
                    border: 1px solid {gc}40; background-color: #ffffff;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin: 10px 0;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 15px;">
                <div style="font-size: 14px; font-weight: 600; color: {BRAND_COLORS["gray"]}; text-transform: uppercase; letter-spacing: 0.5px;">
                    Trust Analysis Result
                </div>
                <div style="padding: 4px 12px; border-radius: 20px; background-color: {gc}; color: white;
                            font-size: 13px; font-weight: 700;">
                    GRADE {self.grade}
                </div>
            </div>

            <div style="display: flex; align-items: baseline; margin-bottom: 8px;">
                <span style="font-size: 48px; font-weight: 800; color: {gc}; line-height: 1;">{self.score}</span>
                <span style="font-size: 20px; font-weight: 600; color: {BRAND_COLORS["gray"]}; margin-left: 4px;">/100</span>
            </div>

            <div style="font-size: 16px; font-weight: 600; color: {BRAND_COLORS["dark"]}; margin-bottom: 20px;">
                {self.verdict}
            </div>

            <div style="border-top: 1px solid #f0f0f0; pt: 15px;">
                <div style="font-size: 12px; font-weight: 700; color: {BRAND_COLORS["gray"]}; margin: 12px 0 8px 0; text-transform: uppercase;">
                    Dimension Breakdown
                </div>
        """

        for dim, score in self.sub_scores.items():
            sc = _color_for_score(score)
            html += f"""
                <div style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px;">
                        <span style="color: {BRAND_COLORS["dark"]}; font-weight: 500;">{dim.capitalize()}</span>
                        <span style="color: {sc}; font-weight: 700;">{score:.1f}</span>
                    </div>
                    <div style="width: 100%; height: 6px; background-color: #f0f0f0; border-radius: 3px; overflow: hidden;">
                        <div style="width: {score}%; height: 100%; background-color: {sc}; border-radius: 3px;"></div>
                    </div>
                </div>
            """

        html += """
            </div>
        </div>
        """
        return html


def _score_bar(score: float, width: int = 12) -> str:
    """Return empty string (ASCII bars removed for professional output)."""
    return ""


# ---------------------------------------------------------------------------
# Main computation function
# ---------------------------------------------------------------------------


def compute_trust_score(
    results: dict,
    weights: dict[str, float] | None = None,
) -> TrustScoreResult:
    """
    Compute the overall Trust Score from a TrustReport's results dict.

    Parameters
    ----------
    results : dict
      The ``TrustReport.results`` dictionary.
    weights : dict, optional
      Custom dimension weights. Keys: ``"calibration"``, ``"failure"``,
      ``"bias"``, ``"representation"``. Values must sum to 1.0.
      If None, uses default weights.

    Returns
    -------
    TrustScoreResult
      Structured score result with per-dimension breakdown.

    Examples
    --------
    >>> from trustlens.trust_score import compute_trust_score
    >>> result = compute_trust_score(report.results)
    >>> print(result)
    >>> print(result.score)  # e.g. 74
    >>> print(result.grade)  # e.g. 'B'
    """
    w = dict(_DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    # ------------------------------------------------------------------
    # 1. Compute available sub-scores
    # ------------------------------------------------------------------
    sub_scores: dict[str, float] = {}

    if "calibration" in results:
        sub_scores["calibration"] = _calibration_score(results["calibration"])

    if "failure" in results:
        sub_scores["failure"] = _failure_score(results["failure"])

    if "bias" in results:
        sub_scores["bias"] = _bias_score(results["bias"])

    if "representation" in results:
        sub_scores["representation"] = _representation_score(results["representation"])

    # ------------------------------------------------------------------
    # 2. Redistribute weights for missing dimensions
    # ------------------------------------------------------------------
    active_dims = [d for d in w if d in sub_scores]
    total_active_weight = sum(w[d] for d in active_dims)

    weights_used: dict[str, float] = {}
    if total_active_weight > 0:
        for dim in active_dims:
            weights_used[dim] = w[dim] / total_active_weight
    else:
        # Fallback: equal weights
        for dim in active_dims:
            weights_used[dim] = 1.0 / len(active_dims) if active_dims else 0.0

    # ------------------------------------------------------------------
    # 3. Weighted sum and Weak-Dimension Penalties → final score
    # ------------------------------------------------------------------
    raw_score = sum(sub_scores[d] * weights_used[d] for d in active_dims)
    total_penalty = 0.0
    penalties_applied: dict[str, float] = {}

    # Scaled failure penalty (if under 60.0, apply linearly up to _MAX_PENALTY_FAILURE)
    failure_score = sub_scores.get("failure", 100.0)
    if failure_score < 60.0:
        penalty = _MAX_PENALTY_FAILURE * ((60.0 - failure_score) / 60.0)
        actual_p = float(np.clip(penalty, 0.0, _MAX_PENALTY_FAILURE))
        total_penalty += actual_p
        penalties_applied["Failure"] = round(actual_p, 1)

    # Scaled calibration penalty (if ece > 0.05, apply linearly)
    calibration_data = results.get("calibration", {})
    if "ece" in calibration_data and calibration_data["ece"] is not None:
        try:
            ece = float(calibration_data["ece"])
            if ece > 0.05:
                # ECE=0.15 gives max penalty
                penalty = _MAX_PENALTY_CALIBRATION * ((ece - 0.05) / 0.10)
                actual_p = float(np.clip(penalty, 0.0, _MAX_PENALTY_CALIBRATION))
                total_penalty += actual_p
                penalties_applied["Calibration"] = round(actual_p, 1)
        except (ValueError, TypeError):
            pass

    bias_has_severe_violation = False
    max_gap = 0.0
    bias_module = results.get("bias", {})

    # Consolidate subgroup and equalized_odds into a single fairness penalty
    for feat_data in bias_module.get("subgroup_performance", {}).values():
        if isinstance(feat_data, dict):
            gap = feat_data.get("__summary__", {}).get("performance_gap", 0.0)
            if gap is not None:
                try:
                    gap_val = float(gap)
                    max_gap = max(max_gap, gap_val)
                    if gap_val > 0.15:
                        bias_has_severe_violation = True
                except (ValueError, TypeError):
                    pass

    for val in bias_module.get("equalized_odds", {}).values():
        if not isinstance(val, dict):
            continue
        summary = val.get("__summary__", {})
        if summary.get("tpr_violation") == "severe" or summary.get("fpr_violation") == "severe":
            bias_has_severe_violation = True
            break

    if bias_has_severe_violation:
        actual_p = float(_MAX_PENALTY_FAIRNESS)
        total_penalty += actual_p
        penalties_applied["Fairness"] = round(actual_p, 1)
    elif max_gap > 0.05:
        # Scale penalty based on gap from 0.05 up to 0.15
        penalty = _MAX_PENALTY_FAIRNESS * ((max_gap - 0.05) / 0.10)
        actual_p = float(np.clip(penalty, 0.0, _MAX_PENALTY_FAIRNESS))
        total_penalty += actual_p
        penalties_applied["Fairness"] = round(actual_p, 1)

    # Cap total penalty to preserve general score variance
    if total_penalty > _MAX_TOTAL_PENALTY:
        scale = _MAX_TOTAL_PENALTY / total_penalty
        for k in penalties_applied:
            penalties_applied[k] = round(penalties_applied[k] * scale, 1)
        total_penalty = float(_MAX_TOTAL_PENALTY)

    base_score = int(round(float(np.clip(raw_score, 0.0, 100.0))))
    raw_score -= total_penalty

    final_score = int(round(float(np.clip(raw_score, 0.0, 100.0))))

    breakdown = {d: round(sub_scores[d] * weights_used[d], 2) for d in active_dims}

    # ------------------------------------------------------------------
    # 4. Assign grade & Check Blockers
    # ------------------------------------------------------------------
    conf_gap = results.get("failure", {}).get("confidence_gap", {}).get("gap", 0.0)
    ece_val = calibration_data.get("ece", 0.0) if isinstance(calibration_data, dict) else 0.0
    is_confidently_wrong = failure_score < 50.0 and ece_val > 0.15 and conf_gap < 0.05

    is_blocked = False
    block_reason = ""
    # Hierarchy: Failure > Fairness > Calibration
    if is_confidently_wrong:
        is_blocked = True
        block_reason = (
            "Blocked by 'confidently wrong' behavior (mismatched confidence-weighted errors)"
        )
    elif failure_score < 40.0:
        is_blocked = True
        block_reason = (
            "Blocked by high diagnostic risk (misaligned confidence-weighted error distribution)"
        )

    elif bias_has_severe_violation:
        is_blocked = True
        block_reason = "Blocked by severe fairness violations"
    elif ece_val > 0.1:
        is_blocked = True
        block_reason = "Blocked due to poor calibration (ECE > 0.1)"

    if is_blocked:
        grade = "D"
        verdict = f"Low Trust - {block_reason}"
    else:
        grade, verdict = "D", "Low Trust - serious issues"
        for threshold, g, v in _GRADE_THRESHOLDS:
            if final_score >= threshold:
                grade, verdict = g, v
                break

    return TrustScoreResult(
        score=final_score,
        grade=grade,
        verdict=verdict,
        sub_scores={d: round(sub_scores[d], 1) for d in active_dims},
        weights_used={d: round(weights_used[d], 3) for d in active_dims},
        breakdown=breakdown,
        penalties_applied=penalties_applied,
        base_score=base_score,
        is_blocked=is_blocked,
        task_type="classification",
    )


# ---------------------------------------------------------------------------
# Regression Trust Score
# ---------------------------------------------------------------------------
#
# A regression-specific scorer that reuses the classification interface
# (``TrustScoreResult``, the 0–100 scale, the A/B/C/D bands, the deployment
# verdicts and the weight-redistribution mechanism) but scores three
# regression-native dimensions instead of the classification four.
#
# Design (RFC #145, converged with the maintainer):
#   Accuracy / Skill            0.30   skill S = 1 − MSE/Var(y)  (= R² vs the
#                                      mean-predictor baseline), docked by a
#                                      p90/median heavy-tail penalty.
#   Interval Calibration        0.40   ICE (mean |emp(tau)-tau| across levels)
#                                      when multi-level intervals are supplied,
#                                      else single-level PICP |calibration_error|,
#                                      through a tolerance (regression analog of ECE).
#   Uncertainty Informativeness 0.30   calibration-conditioned sharpness proxy vs a
#                                      climatology reference (RFC #155) when multi-
#                                      level intervals are supplied, else max(pearson,
#                                      spearman) of predicted uncertainty vs error.
#
# Point-prediction-only reports score on Accuracy alone (the other two are
# redistributed away), exactly as a no-embeddings classification report drops
# Representation today.
#
# Blockers (→ grade D):  negative skill (S < 0); severe interval miscoverage
#                        (calibration_error < −0.10 — materially over-confident).
# Penalties (not blockers): a heavy tail docks the Accuracy/Skill dimension; weak
#                        uncertainty correlation docks the composite.

_REGRESSION_DEFAULT_WEIGHTS: dict[str, float] = {
    "accuracy": 0.30,
    "interval_calibration": 0.40,
    "uncertainty_informativeness": 0.30,
}

# Interval-calibration sub-score: the |calibration_error| (or ICE) at which the
# sub-score reaches 0. A ±0.10 miss → 50/100; ±0.20 → 0/100.
#
# NB: this SCORING tolerance (0.20) is deliberately distinct from — and looser
# than — the 0.05 calibration GATE in metrics/regression.py
# (``multilevel_interval_coverage``'s ``tolerance``). They answer different
# questions and should NOT be unified:
#   * 0.05 (metric layer) is a strict binary gate: which levels are calibrated
#     *enough* to be admitted into the sharpness proxy, so over-confident
#     intervals cannot be rewarded for looking "sharp". A hard pass/fail per level.
#   * 0.20 (here) is a smooth ramp mapping the continuous ICE / |calibration_error|
#     onto the 0–100 sub-score, so calibration quality degrades gracefully rather
#     than cliff-edging. A gradient, not a gate.
# Using 0.05 for the ramp would turn the sub-score into a cliff; using 0.20 for
# the gate would let materially miscalibrated levels inflate the sharpness proxy.
_REG_CALIBRATION_TOLERANCE = 0.20

# Heavy-tail penalty (docks the Accuracy/Skill dimension). The p90/median
# absolute-error ratio at/below the threshold incurs no dock; the dock then ramps
# linearly with the excess up to a capped fraction of the sub-score.
_REG_TAIL_RATIO_THRESHOLD = 3.0
_REG_TAIL_RATIO_SCALE = 7.0
_REG_MAX_TAIL_DOCK_FRACTION = 0.50

# Weak-uncertainty-correlation penalty (docks the composite). No penalty at/above
# the "informative" boundary; scales up as the correlation falls toward (and
# below) zero.
_REG_INFORMATIVE_CORR = 0.50
_REG_MAX_WEAK_CORR_PENALTY = 15.0

# Severe-miscoverage blocker: realised coverage this far below nominal means the
# intervals are materially over-confident (the regression "confidently wrong").
_REG_SEVERE_MISCOVERAGE = -0.10


def _regression_accuracy_score(error_dist: dict, target_variance: float) -> dict[str, float]:
    """
    Accuracy/Skill sub-score (0–100) plus the diagnostics needed downstream.

    The skill score ``S = 1 − MSE / Var(y)`` is the coefficient of determination
    against a predict-the-mean baseline (scale-free and comparable across
    datasets). The base sub-score is ``100 × clip(S, 0, 1)``, then docked by a
    heavy-tail penalty derived from the ``p90 / median`` absolute-error ratio so
    a handful of catastrophic errors hidden by the aggregate still lower the
    dimension.

    Returns a dict with ``score`` (the docked sub-score), ``skill`` (the raw
    ``S``, which may be negative → blocker), ``tail_ratio`` and ``tail_dock``
    (points removed by the heavy-tail penalty).
    """
    rmse = float(error_dist.get("rmse", 0.0))
    mse = rmse * rmse
    if target_variance > 0.0:
        skill = 1.0 - mse / target_variance
    else:
        # Constant target: a mean-predictor is already exact, so skill "over the
        # mean" is undefined. Treat a perfect fit as full skill, else none.
        # NB: a 0.0 here unambiguously means a genuine constant target — callers
        # resolve a *missing* variance to a ValueError upstream, never to 0.0
        # (issue #150), so this branch is never reached for absent ground truth.
        skill = 1.0 if mse <= 1e-12 else 0.0

    base = 100.0 * float(np.clip(skill, 0.0, 1.0))

    median = float(error_dist.get("median_absolute_error", 0.0))
    p90 = float(error_dist.get("p90_absolute_error", 0.0))
    if median > 1e-12:
        tail_ratio = p90 / median
    elif p90 > 1e-12:
        # Degenerate (median ~0 but a real tail) → treat as maximally heavy.
        tail_ratio = _REG_TAIL_RATIO_THRESHOLD + _REG_TAIL_RATIO_SCALE
    else:
        tail_ratio = 1.0  # all-zero errors → no tail

    excess = max(0.0, tail_ratio - _REG_TAIL_RATIO_THRESHOLD)
    dock_fraction = float(np.clip(excess / _REG_TAIL_RATIO_SCALE, 0.0, _REG_MAX_TAIL_DOCK_FRACTION))
    tail_dock = base * dock_fraction
    return {
        "score": base - tail_dock,
        "skill": skill,
        "tail_ratio": tail_ratio,
        "tail_dock": tail_dock,
    }


def _interval_calibration_score(coverage: dict) -> float:
    """
    Interval-calibration sub-score (0–100).

    Uses the multi-level ICE (``mean |emp(tau) - tau|``) when present —
    ``100 × (1 − clip(ICE / T, 0, 1))`` — otherwise the single-level PICP
    ``|calibration_error|`` through the same tolerance. Both are the regression
    analog of how :func:`_calibration_score` maps ECE for classification (RFC #155).
    """
    if coverage.get("ice") is not None:
        ice = float(coverage["ice"])
        return 100.0 * (1.0 - float(np.clip(ice / _REG_CALIBRATION_TOLERANCE, 0.0, 1.0)))
    cal_err = float(coverage.get("calibration_error", 0.0))
    return 100.0 * (1.0 - float(np.clip(abs(cal_err) / _REG_CALIBRATION_TOLERANCE, 0.0, 1.0)))


def _uncertainty_informativeness_score(corr: dict) -> float:
    """
    Uncertainty-informativeness sub-score (0–100).

    ``100 × clip(max(pearson, spearman), 0, 1)`` — rewards uncertainty that is
    larger exactly where the realised error is larger.
    """
    strongest = max(float(corr.get("pearson", 0.0)), float(corr.get("spearman", 0.0)))
    return 100.0 * float(np.clip(strongest, 0.0, 1.0))


def _informativeness_from_sharpness(coverage: dict) -> float:
    """
    Uncertainty-informativeness sub-score (0–100) from the calibration-conditioned
    sharpness proxy (RFC #155).

    ``100 × clip(sharpness_skill, 0, 1)`` — rewards intervals sharper than the
    climatology baseline *among the levels that pass calibration*, the
    CRPS-Resolution analog of the correlation-based score. Preferred over the
    error-variance correlation when multi-level intervals are available.
    """
    skill = float(coverage.get("sharpness_skill") or 0.0)
    return 100.0 * float(np.clip(skill, 0.0, 1.0))


def _reg_metric_present(metric: dict | None) -> bool:
    """True when an optional regression metric was actually computed (not skipped)."""
    return isinstance(metric, dict) and metric.get("status") != "skipped"


def regression_trust_score(
    results: dict,
    y_true: np.ndarray | None = None,
    weights: dict[str, float] | None = None,
) -> TrustScoreResult:
    """
    Compute the regression Trust Score from a regression report's results.

    Mirrors :func:`compute_trust_score` but scores three regression-native
    dimensions — Accuracy/Skill, Interval Calibration and Uncertainty
    Informativeness — while reusing the same :class:`TrustScoreResult`
    interface, the 0–100 scale, the A/B/C/D bands, the deployment verdicts and
    the weight-redistribution mechanism (see the module-level notes / RFC #145).

    Parameters
    ----------
    results : dict
      Either the full report results dict (with a ``"regression"`` key, as built
      by the regression pipeline) or the inner regression metrics dict directly.
      Expected keys: ``error_distribution`` (always present) plus the optional
      ``interval_coverage`` / ``error_variance_correlation`` (which may be
      ``status="skipped"`` dicts when their inputs were absent).
    y_true : np.ndarray, optional
      Ground-truth targets, used to compute ``Var(y)`` for the skill score. May
      be omitted **only** when ``results`` carries a persisted
      ``regression["target_variance"]`` (as emitted by the regression pipeline,
      issue #150), which is then used as the fallback so a regression Trust Score
      can be recomputed from a stored report alone. If both are supplied, the
      explicitly-passed ``y_true`` wins and a mismatch beyond tolerance warns. If
      neither is available a ``ValueError`` is raised.
    weights : dict, optional
      Custom dimension weights (keys: ``"accuracy"``, ``"interval_calibration"``,
      ``"uncertainty_informativeness"``). Defaults to ``0.30 / 0.40 / 0.30``.

    Returns
    -------
    TrustScoreResult
      With ``task_type="regression"``. A regression ``75`` and a classification
      ``75`` share this interface but are **not** directly comparable.

    Examples
    --------
    >>> from trustlens.trust_score import regression_trust_score
    >>> result = regression_trust_score(report.results, report.y_true)
    >>> print(result.score, result.grade)
    """
    reg = results.get("regression", results)
    error_dist = reg.get("error_distribution", {}) or {}
    coverage = reg.get("interval_coverage", {})
    corr = reg.get("error_variance_correlation", {})

    # Resolve Var(y) for the skill denominator. Priority (issue #150):
    #   1. explicit y_true → np.var (population, ddof=0), no behaviour change;
    #   2. else a persisted results["regression"]["target_variance"];
    #   3. else raise — never silently fall through to 0.0, which would route a
    #      missing variance through the genuine "constant target" branch and
    #      yield a misleading skill score.
    stored_variance = reg.get("target_variance")
    if y_true is not None:
        y_true_arr = np.asarray(y_true, dtype=float)
        target_variance = float(np.var(y_true_arr)) if y_true_arr.size else 0.0
        if stored_variance is not None and not np.isclose(
            target_variance, float(stored_variance), rtol=1e-6, atol=1e-9
        ):
            warnings.warn(
                "regression_trust_score: Var(y) from the supplied y_true "
                f"({target_variance:.6g}) disagrees with the persisted "
                f"target_variance ({float(stored_variance):.6g}); using y_true.",
                stacklevel=2,
            )
    elif stored_variance is not None:
        target_variance = float(stored_variance)
    else:
        raise ValueError(
            "regression_trust_score needs the target variance Var(y) to compute "
            "the skill score, but neither was available: pass y_true, or use a "
            "report whose regression results carry a persisted 'target_variance' "
            "(emitted by the regression pipeline; see issue #150)."
        )

    w = dict(_REGRESSION_DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    sub_scores: dict[str, float] = {}
    penalties_applied: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 1. Sub-scores (Accuracy always available; the other two are optional)
    # ------------------------------------------------------------------
    accuracy = _regression_accuracy_score(error_dist, target_variance)
    sub_scores["accuracy"] = accuracy["score"]
    skill = accuracy["skill"]

    coverage_present = _reg_metric_present(coverage)
    interval_present = coverage_present and ("ice" in coverage or "calibration_error" in coverage)
    calibration_error: float | None = None
    if interval_present:
        # Multi-level reports carry the worst per-level gap (most over-confident);
        # single-level PICP reports carry calibration_error. Either drives the
        # severe-miscoverage blocker below.
        calibration_error = float(
            coverage.get("worst_calibration_error", coverage.get("calibration_error", 0.0))
        )
        sub_scores["interval_calibration"] = _interval_calibration_score(coverage)

    # Uncertainty Informativeness: prefer the calibration-conditioned sharpness
    # proxy (multi-level intervals, RFC #155); fall back to the error-variance
    # correlation when only predicted variance is available.
    sharpness_skill = coverage.get("sharpness_skill") if coverage_present else None
    strongest_corr: float | None = None
    if sharpness_skill is not None:
        sub_scores["uncertainty_informativeness"] = _informativeness_from_sharpness(coverage)
    elif _reg_metric_present(corr) and ("pearson" in corr or "spearman" in corr):
        strongest_corr = max(float(corr.get("pearson", 0.0)), float(corr.get("spearman", 0.0)))
        sub_scores["uncertainty_informativeness"] = _uncertainty_informativeness_score(corr)
    informativeness_present = "uncertainty_informativeness" in sub_scores

    # ------------------------------------------------------------------
    # 2. Redistribute weights across the dimensions actually present
    #    (a point-only report collapses to Accuracy = 1.00)
    # ------------------------------------------------------------------
    active_dims = [d for d in w if d in sub_scores]
    total_active_weight = sum(w[d] for d in active_dims)
    weights_used: dict[str, float] = {}
    if total_active_weight > 0:
        for dim in active_dims:
            weights_used[dim] = w[dim] / total_active_weight
    else:
        for dim in active_dims:
            weights_used[dim] = 1.0 / len(active_dims) if active_dims else 0.0

    raw_score = sum(sub_scores[d] * weights_used[d] for d in active_dims)
    base_score = int(round(float(np.clip(raw_score, 0.0, 100.0))))

    # ------------------------------------------------------------------
    # 3. Composite penalty: weak uncertainty correlation (only when present).
    #    The heavy-tail penalty is NOT applied here — it already docked the
    #    Accuracy sub-score in step 1, so base_score − composite_penalties stays
    #    consistent with final_score.
    # ------------------------------------------------------------------
    if (
        informativeness_present
        and strongest_corr is not None
        and strongest_corr < _REG_INFORMATIVE_CORR
    ):
        frac = float(
            np.clip((_REG_INFORMATIVE_CORR - strongest_corr) / _REG_INFORMATIVE_CORR, 0.0, 1.0)
        )
        weak_corr_penalty = _REG_MAX_WEAK_CORR_PENALTY * frac
        raw_score -= weak_corr_penalty
        penalties_applied["Weak Uncertainty"] = round(weak_corr_penalty, 1)

    final_score = int(round(float(np.clip(raw_score, 0.0, 100.0))))
    breakdown = {d: round(sub_scores[d] * weights_used[d], 2) for d in active_dims}

    # ------------------------------------------------------------------
    # 4. Blockers → grade D (negative skill; severe interval miscoverage)
    # ------------------------------------------------------------------
    is_blocked = False
    block_reason = ""
    if skill < 0.0:
        is_blocked = True
        block_reason = "Blocked by negative skill (worse than predicting the mean; R^2 < 0)"
    elif (
        interval_present
        and calibration_error is not None
        and calibration_error < _REG_SEVERE_MISCOVERAGE
    ):
        is_blocked = True
        block_reason = (
            f"Blocked by severe interval miscoverage (coverage {calibration_error:+.2f} "
            "below nominal - over-confident intervals)"
        )

    if is_blocked:
        grade = "D"
        verdict = f"Low Trust - {block_reason}"
    else:
        grade, verdict = "D", "Low Trust - serious issues"
        for threshold, g, v in _GRADE_THRESHOLDS:
            if final_score >= threshold:
                grade, verdict = g, v
                break

    return TrustScoreResult(
        score=final_score,
        grade=grade,
        verdict=verdict,
        sub_scores={d: round(sub_scores[d], 1) for d in active_dims},
        weights_used={d: round(weights_used[d], 3) for d in active_dims},
        breakdown=breakdown,
        penalties_applied=penalties_applied,
        base_score=base_score,
        is_blocked=is_blocked,
        task_type="regression",
    )
