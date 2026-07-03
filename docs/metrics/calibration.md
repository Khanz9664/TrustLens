# Calibration Metrics

Calibration metrics measure whether confidence values are trustworthy for decision-making.

## Why This Matters

If a model predicts 0.90 confidence, teams often act as if it is correct 90 percent of the time.
Calibration metrics test whether that assumption is true.

## When to Use

- when confidence values drive downstream thresholds
- when ranking or triage depends on probability quality
- when validating reliability before deployment

## Inputs and Assumptions

- `y_true`: ground-truth labels
- `y_prob`: predicted probabilities
- **Multiclass Support**: TrustLens v0.5.0 supports multiclass calibration using top-label ECE and Multiclass Brier Score.

## Output and Interpretation

Key outputs include:

- **Brier score**: lower values indicate better probabilistic accuracy
- **ECE**: lower values indicate confidence is better aligned with observed accuracy *on average*
- **MCE**: the worst-case per-bin confidence gap — a low ECE with a high MCE means one confidence region is severely miscalibrated even though the average looks fine (MCE >= ECE always)
- **Reliability curve data**: supports visual overconfidence or underconfidence inspection

## Limitations and Caveats

- calibration quality estimates are less stable on very small datasets
- poor probability estimation upstream can dominate all calibration outputs

## API Reference

```{eval-rst}
.. automodule:: trustlens.metrics.calibration
   :members:
   :show-inheritance:
```

## Related Pages

- [Features and Modules](../features.md)
- [Trust Score Explained](../trust_score_explained.md)
- [Known Limitations](../known_limitations.md)
