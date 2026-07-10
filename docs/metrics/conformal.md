# Conformal Prediction Diagnostics

Conformal diagnostics measure whether a set-valued predictor's coverage guarantee actually holds *where it matters*, not just on average.

## Why This Matters

Split conformal prediction turns any classifier into a set predictor `C(x)` with a **marginal** guarantee: the true label lands in the set at least `1 - α` of the time, averaged over the whole distribution. That average is weak. A predictor can hit 90% marginal coverage while systematically under-covering a hard class, or under-covering every large (ambiguous) prediction set, and marginal coverage alone will never show it. These diagnostics surface that hidden conditional failure from the prediction sets themselves.

## When to Use

- when you deploy conformal / set-valued predictions and need to trust the coverage claim per class, not just overall
- when comparing conformal methods (LAC, APS, RAPS, Mondrian, …) on the same footing
- when a "90% covered" headline needs an honesty check before it drives a decision

## Inputs and Assumptions

- `y_true`: ground-truth integer class labels, shape `(n_samples,)`
- `y_pred_sets`: the prediction sets, in either form —
  - an `(n, K)` binary / boolean membership matrix (`S[i, k]` is truthy iff class `k` is in the set for sample `i`), or
  - a ragged sequence of per-sample label lists (converted internally by `to_membership_matrix`).
  - *Disambiguation*: NumPy arrays are unambiguous (2D → matrix, 1D object array → label lists). A **native rectangular list whose entries are all `0`/`1`** is genuinely ambiguous — it could be either — so it **raises** unless you pass `n_classes` (a matrix has exactly `n_classes` columns) or a NumPy array.
- `n_classes` (optional): the class space `K`. Every public metric accepts it. It drives the same validation/disambiguation everywhere, and when given it is treated as a **contract**: a `y_true` label outside `[0, n_classes)` raises, rather than being counted as a coverage miss (the miss behavior is reserved for the inferred-`K` case, where an unseen class is a genuine "never predicted" diagnostic).
- `nominal_coverage` (optional): the target `1 - α` the sets were built for. Coverage and set-size numbers are reported without it; the gap/violation fields require it and are `None` (never a silent `0`) when it is absent.
- **Method-agnostic**: TrustLens *evaluates* prediction sets; it does not generate them and takes no dependency on any conformal library.

## Output and Interpretation

`conformal_diagnostics(...)` returns a single report. Key fields:

- **`marginal_coverage`** / **`coverage_gap`**: overall `P(y ∈ C(x))` and its signed gap to nominal (negative = under-covers).
- **`worst_class_coverage`** / **`worst_class_gap`**: the lowest per-class coverage and how far it sits below nominal — the class-conditional failure signal.
- **`ssc_violation`**: `max(0, nominal − worst size-stratum coverage)` (Angelopoulos & Bates size-stratified coverage). `0` means no size stratum is meaningfully under-covering; a positive value flags the singletons-under-cover / large-sets-over-cover failure that marginal coverage hides.
- **`avg_set_size`**, **`singleton_rate`**, **`size_efficiency`** (`1` = all singletons, `0` = all-`K` sets), **`empty_rate`**: informativeness — are the sets actually tight enough to be useful?

The individual functions (`marginal_coverage`, `class_conditional_coverage`, `size_stratified_coverage`, `set_size_summary`) return their full per-group tables with sample counts for deeper inspection.

## Using it through `analyze()`

The diagnostics are also available end-to-end from the top-level API. Pass your
prediction sets as `y_pred_sets` to `analyze()`; TrustLens emits the block as
`results["calibration"]["conformal"]` and renders a **Conformal Prediction**
panel under Calibration Analysis in `report.show()`. `confidence_level` is the
nominal coverage `1 - α` the sets claim.

```python
import numpy as np
from trustlens import analyze

# prediction sets from any conformal method (LAC/APS/RAPS/…), here a simple
# threshold on the model's probabilities → an (n, K) 0/1 membership matrix
y_pred_sets = (y_prob >= 0.30).astype(int)

report = analyze(
    model, X_val, y_val,
    y_prob=y_prob,            # ECE/Brier still computed from probabilities
    y_pred_sets=y_pred_sets,  # conformal coverage computed from the sets
    confidence_level=0.90,    # the nominal target the sets claim
)
report.show()

conf = report.results["calibration"]["conformal"]
print(conf["marginal_coverage"], conf["worst_class_gap"], conf["ssc_violation"])
```

Notes:

- **Coexists with probabilities.** `y_prob` drives ECE/Brier/MCE; `y_pred_sets`
  drives the coverage diagnostics. Either, both, or neither may be supplied — the
  conformal block is emitted whenever sets are present (even if probabilities are
  withheld), and simply omitted when they're absent.
- **Sign conventions are spelled out in the panel.** `coverage_gap` is
  `marginal − nominal` (positive = *over*-coverage) while `worst_class_gap` is
  `nominal − worst` (positive = *under*-coverage); the panel annotates each as
  "over/under by …" so the opposite signs can't be misread. The raw fields keep
  their documented conventions for downstream consumers.
- **Diagnostic-only.** These values are reported but do **not** influence the
  Trust Score (Phase-2 scoring integration is deferred by design).
- **Graceful degradation.** Malformed sets (wrong length, all-empty, ambiguous
  native `0/1` lists) surface as a visible skipped sub-block with the reason,
  rather than aborting the whole analysis.
- Classification only — `y_pred_sets` is ignored on the regression path.

## Limitations and Caveats

- These are **approximate** conditional diagnostics. Exact conditional coverage (coverage that holds for every fixed x, class, or set size) is impossible to guarantee distribution-free with finitely many samples (Vovk, 2012; Barber et al., 2021). Read the class- and size-conditional numbers together with their per-group counts.
- Size strata below `min_stratum` (default 20) samples are reported but excluded from the worst-stratum penalty, so a tiny stratum can't dominate the violation.
- Empty sets are counted (`empty_rate`) and treated as a coverage miss — an empty set cannot contain the label.

## API Reference

```{eval-rst}
.. automodule:: trustlens.metrics.conformal
   :members:
   :show-inheritance:
```

## Related Pages

- [Calibration Metrics](calibration.md)
- [Features and Modules](../features.md)
- [Trust Score Explained](../trust_score_explained.md)
- [Known Limitations](../known_limitations.md)
