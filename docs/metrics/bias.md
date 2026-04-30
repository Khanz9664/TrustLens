# Bias and Fairness Metrics

Bias and fairness metrics expose subgroup disparities so teams can evaluate equity risk before release.

## Why This Matters

A model can look strong in aggregate while underperforming for specific groups.
Fairness diagnostics make those gaps visible and actionable.

## When to Use

- when model decisions affect people across demographic or policy-sensitive segments
- when governance requires subgroup performance reporting
- when release policy includes fairness checks

## Inputs and Assumptions

- `y_true`: ground-truth labels
- `y_pred`: predicted labels
- `sensitive_features`: dictionary of aligned subgroup arrays
- equalized-odds analysis assumes binary target labels

## Output and Interpretation

Key outputs include:

- **Class imbalance report**: class distribution risk context
- **Subgroup performance**: per-group metrics and gap summaries
- **Equalized odds summary**: TPR/FPR disparity severity across groups

Large subgroup gaps or severe equalized-odds violations should be treated as release blockers in high-impact domains.

## Visualization

TrustLens provides comprehensive visualization modes via `report.plot_bias(mode="...")` to help interpret fairness diagnostics:

- **`"summary"` (Default)**: Combines key fairness signals into a single diagnostic view.
- **`"subgroup"`**: Detailed performance metric comparison (e.g., accuracy, precision) across all groups.
- **`"equalized_odds"`**: Visualizes TPR and FPR side-by-side to identify specific types of disparity.
- **`"gap"`**: High-level summary of the maximum demographic parity or opportunity gaps.
- **`"all"`**: Generates and returns all three diagnostic plots for a full audit.

These visualizations ensure that fairness gaps are not just calculated but are immediately visible and actionable.

```python
# Generate all diagnostic plots for a full audit
plots = report.plot_bias(mode="all")
```

## Limitations and Caveats

- fairness metrics are sensitive to subgroup sample size
- skipped equalized-odds checks are input constraints, not fairness clearance
- outputs are statistical diagnostics, not causal proof

## API Reference

```{eval-rst}
.. automodule:: trustlens.metrics.bias
   :members:
   :show-inheritance:
```

## Related Pages

- [Features and Modules](../features.md)
- [Known Limitations](../known_limitations.md)
- [Fairness Audit Workflow](../guides/fairness_audit_workflow.md)
