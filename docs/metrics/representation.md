# Representation Metrics

Representation metrics evaluate embedding geometry to estimate whether class structure is separable in latent space.

## Why This Matters

Even when aggregate metrics look acceptable, weak latent separation can signal fragile generalization and higher failure risk.

## When to Use

- when embeddings are available from model internals
- when diagnosing class overlap or feature-space quality
- when comparing representation quality across model variants

## Inputs and Assumptions

- `embeddings`: latent vectors for each evaluated sample
- `y_true`: labels aligned with embeddings
- representation analysis is optional and only runs when embeddings are provided

## Output and Interpretation

Key outputs include:

- **Silhouette score**: higher values indicate better class separation
- **Within/between class distances**: additional separation signal
- **CKA utility**: representation similarity support for analysis workflows

Low separability should trigger deeper feature and model diagnostics, not immediate standalone conclusions.

## Limitations and Caveats

- representation quality depends on embedding extraction method
- silhouette estimates can be unstable with small or highly imbalanced samples
- representation outputs should be interpreted with calibration and failure diagnostics, not in isolation

## API Reference

```{eval-rst}
.. automodule:: trustlens.metrics.representation
   :members:
   :show-inheritance:
```

## Related Pages

- [Features and Modules](../features.md)
- [Known Limitations](../known_limitations.md)
- [Trust Score Explained](../trust_score_explained.md)
