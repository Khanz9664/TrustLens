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
