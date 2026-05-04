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

## Multi-Feature Fairness Visualization

When multiple sensitive features are provided, TrustLens generates per-feature plots for every visualization type — no feature is silently dropped.

### Usage

Pass multiple sensitive features to `analyze()`:

```python
from trustlens import analyze

report = analyze(
    model, X_test, y_test,
    sensitive_features={
        "gender": gender_array,
        "age_group": age_array,
        "income level": income_array,
    },
)
```

### Generating Per-Feature Plots

**Option 1 — Multi wrappers (direct):**

```python
from trustlens.visualization.fairness import (
    plot_subgroup_performance_multi,
    plot_equalized_odds_multi,
    plot_fairness_gap_multi,
)

bias_data = report.results["bias"]

figs = plot_subgroup_performance_multi(
    bias_data["subgroup_performance"], save_dir="plots/", show=False
)
# → plots/subgroup_performance_age_group.png
# → plots/subgroup_performance_gender.png
# → plots/subgroup_performance_income_level.png
```

**Option 2 — Orchestrated via `plot_module` (recommended for batch saving):**

```python
from trustlens.visualization import plot_module

plot_module("bias", report.results["bias"], save_dir="plots/")
```

When `class_imbalance` is present, this saves a single `bias_plot.png`.
When only fairness metrics are present, it saves per-feature files with standardized names:

```text
plots/
├── bias_subgroup_age_group.png
├── bias_subgroup_gender.png
├── bias_subgroup_income_level.png
├── bias_equalized_odds_age_group.png
├── bias_equalized_odds_gender.png
├── bias_equalized_odds_income_level.png
├── bias_fairness_gap_age_group.png
├── bias_fairness_gap_gender.png
└── bias_fairness_gap_income_level.png
```

### Filename Safety

Feature names with spaces or special characters are automatically sanitized:

| Feature Name | Filename Component |
|---|---|
| `gender` | `gender` |
| `age_group` | `age_group` |
| `income level` | `income_level` |
| `race/ethnicity` | `race_ethnicity` |

### Design Notes

- Features are processed in **sorted order** for deterministic output.
- `plot_module` is the sole owner of file saving — no duplicate writes.
- All returned figures are independent and can be saved or displayed individually.

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
