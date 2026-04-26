# Model Comparison Workflow

This guide shows how to compare candidate models using TrustLens and select a deployment candidate with explicit risk trade-offs.

## When to Use This

Use this workflow when two or more models have similar accuracy and you need a reliability-first decision.

## Workflow

1. Train candidate models on the same train split.
2. Run `analyze()` on each model using the same validation data.
3. Compare trust reports with `compare()`.
4. Review blocker status, penalties, and dimension-level differences.
5. Select the candidate with highest safe trust profile, not only highest score.

## Example

```python
from trustlens import analyze, compare

report_rf = analyze(model_rf, X_val, y_val, y_prob=model_rf.predict_proba(X_val))
report_lr = analyze(model_lr, X_val, y_val, y_prob=model_lr.predict_proba(X_val))

compare([report_rf, report_lr])
```

## Decision Checklist

- Are any candidates blocked from deployment?
- Which model has lower failure penalty burden?
- Are fairness penalties acceptable for your domain?
- Is calibration quality sufficient for confidence-based decisions?

## Recommended Output Artifacts

- `report.json` per candidate
- `trust_score.json` per candidate
- one short decision memo that captures rationale and rejected alternatives
