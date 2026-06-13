# CI and Deployment Gate Workflow

This guide describes a practical way to use TrustLens outputs as a deployment gate in CI pipelines.

## Goal

Fail fast when a model has high diagnostic risk, even if raw validation accuracy looks good.

## Basic Approach

1. Run model evaluation job.
2. Generate TrustLens report artifacts.
3. Parse trust score output in CI.
4. Enforce policy thresholds (for example score floor and no blockers).
5. Publish artifacts for review.

## Example Policy

Use a staged policy at first, then tighten:

- Initial phase:
  - score >= 65
  - not blocked
- Mature phase:
  - score >= 75
  - not blocked
  - fairness penalty below defined domain threshold

## Minimal Pattern

```python
report = analyze(model, X_val, y_val, y_prob=model.predict_proba(X_val))
report.save("artifacts/trustlens")

if report.trust_score.is_blocked:
    raise SystemExit("Deployment blocked by TrustLens diagnostics")

if report.trust_score.score < 65:
    raise SystemExit("Trust score below CI threshold")
```

## Operational Recommendations

- Keep both machine-readable (`report.json`, `trust_score.json`) and human-readable (`report.txt`) outputs.
- Do not rely on one threshold alone; include blocker status and penalty context.
- Store historical scores to detect reliability drift over time.
