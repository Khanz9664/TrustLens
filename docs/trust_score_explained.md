# Trust Score Explained

This page is the canonical reference for how TrustLens turns diagnostics into a deployment recommendation.

## Why a Composite Score Exists

Raw metrics are useful, but teams still need one final decision signal for release gates and model comparison.
TrustLens computes a composite Trust Score from calibration, failure behavior, bias risk, and representation quality.

The score is not meant to replace detailed metrics. It is meant to summarize them for decision making.

## Inputs Used

Trust score logic consumes the module outputs produced by `analyze()`:

**For Classification:**
- `calibration`: Brier score, ECE
- `failure`: confidence gap, overall error context
- `bias`: class imbalance plus subgroup/equalized-odds information
- `representation`: silhouette-based separability (optional)

**For Regression:**
- `accuracy`: skill score (R²), RMSE, heavy-tail metrics
- `interval_calibration`: single-level PICP or multi-level ICE
- `uncertainty_informativeness`: sharpness proxy or error-variance correlation

## Scoring Workflow

The scoring process has four stages:

1. Compute sub-scores for each available dimension (0 to 100).
2. Apply dimension weights.
3. Redistribute weights if some dimensions are missing.
4. Apply risk penalties and deployment blockers.

This keeps the score usable across different datasets and metadata availability.

## Default Weights

**Classification Defaults:**
- Calibration: 0.35
- Failure: 0.30
- Bias: 0.25
- Representation: 0.10

If representation is missing, the other dimensions are re-normalized so total weight remains 1.0.

**Regression Defaults:**
- Interval Calibration: 0.40
- Accuracy (Skill): 0.30
- Uncertainty Informativeness: 0.30

If intervals or uncertainty estimates are missing, those dimensions are redistributed similarly.

## Penalties

TrustLens applies deductions when severe risk signals are detected. These penalties are intended to stop the final score from hiding critical issues behind good average performance.

Examples of penalty triggers:

**Classification:**
- Failure sub-score falls below a safety threshold
- ECE exceeds calibration tolerance
- Fairness violations exceed subgroup disparity limits

**Regression:**
- Heavy-tail penalty: the p90 absolute error is disproportionately larger than the median error
- Weak uncertainty correlation: prediction uncertainty has low correlation with actual errors
- Unusable uncertainty: prediction intervals are provided but completely fail the calibration gate (sub-score set to 0.0)

The final penalty burden is capped to preserve score interpretability.

## Deployment Blockers

Some conditions produce an explicit deployment block even when the numeric score is moderate.

Typical blockers include:

**Classification:**
- Strong confidently-wrong behavior
- Severe fairness violations
- Very poor calibration quality

**Regression:**
- Negative skill (model performs worse than simply predicting the mean)
- Severe interval miscoverage (intervals are materially over-confident)

When blocked, verdict messaging is forced into a do-not-deploy stance.

## Grade Interpretation

TrustLens maps final score to grade bands:

- A: high trust, production-ready profile
- B: good trust, minor issues to address
- C: moderate trust, investigate before deployment
- D: low trust, deployment should be blocked

Blockers can force a low-trust verdict regardless of grade boundaries.

## How to Use This in Practice

- Use the score for ranking and release gating.
- Use sub-scores to identify which dimension needs work.
- Use penalty details for remediation priorities.
- Use full metric pages for root-cause analysis.

## Related Reading

- [Features and Modules](features.md)
- [Known Limitations](known_limitations.md)
- [Metric: Calibration](metrics/calibration.md)
- [Metric: Failure](metrics/failure.md)
- [Metric: Bias](metrics/bias.md)
- [Metric: Regression](metrics/regression.md)
