# Fairness Audit Workflow

This guide provides a practical fairness-audit flow using TrustLens subgroup and equalized-odds outputs.

## Goal

Detect and triage subgroup performance disparities before deployment decisions.

## Inputs Needed

- `y_true` and model predictions
- sensitive feature arrays aligned with validation rows
- binary target labels for equalized-odds checks

## Workflow

1. Define sensitive features (for example gender, age_group, region).
2. Run analysis with `sensitive_features`.
3. Visualize the disparities:
   ```python
   # Deep-dive into all diagnostic plots
   plots = report.plot_bias(mode="all", save_path="fairness_audit")
   ```
4. Decide whether to proceed, recalibrate, or retrain with mitigation.

## Example

```python
sensitive = {
    "gender": gender_val,
    "age_group": age_group_val,
}

report = analyze(
    model,
    X_val,
    y_val,
    y_prob=model.predict_proba(X_val),
    sensitive_features=sensitive,
)

report.show()
```

## Interpretation Checklist

- Which subgroup has the lowest performance?
- Is performance gap above your policy threshold?
- Are equalized-odds violations moderate or severe?
- Is this driven by class imbalance or model behavior?

## Remediation Options

- Data balancing and subgroup-aware sampling
- Threshold calibration and probability calibration
- Segment-specific error analysis and feature review
- Domain review before any production rollout
