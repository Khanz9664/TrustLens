# Failure Analysis Metrics

Failure analysis metrics identify where errors are concentrated and whether error confidence creates operational risk.

## Why This Matters

Two models with similar error rates can have very different risk profiles.
A model that is highly confident when wrong is usually harder to monitor and safer to block early.

## When to Use

- when incident cost is high for false confidence
- when investigating model behavior beyond aggregate accuracy
- when selecting between candidates with close top-line metrics

## Inputs and Assumptions

- `y_true`: ground-truth labels
- `y_pred`: predicted labels
- `y_prob`: predicted probabilities

## Output and Interpretation

Key outputs include:

- **Misclassification summary**: class-wise error context
- **Confidence gap**: separation between confidence on correct versus incorrect predictions
- **High-confidence error patterns**: useful for targeted inspection

A narrow or negative confidence gap is a warning signal in most operational contexts.

## Limitations and Caveats

- confidence signals depend on quality of upstream probability calibration
- low error counts can make distribution-level interpretation noisy

## API Reference

```{eval-rst}
.. automodule:: trustlens.metrics.failure
   :members:
   :show-inheritance:
```

## Related Pages

- [Features and Modules](../features.md)
- [Trust Score Explained](../trust_score_explained.md)
- [Known Limitations](../known_limitations.md)
