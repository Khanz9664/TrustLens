# Known Limitations

This page documents current limits of TrustLens so users can interpret outputs correctly.

## Scope

TrustLens currently targets classification reliability workflows. Regression support is not a first-class path in the core analysis pipeline.

## Probability Dependency

Calibration and several failure diagnostics require valid probability outputs.

- If your model has no `predict_proba`, you must provide `y_prob` manually.
- Low-quality probability estimates reduce the quality of trust conclusions.

## Dataset Size Effects

Small validation sets can make calibration and subgroup diagnostics unstable.

- Very small sample sizes may produce noisy ECE and subgroup gap values.
- Fairness metrics should be interpreted with caution when subgroup counts are low.

## Fairness Constraints

Current equalized-odds logic assumes a binary target and meaningful subgroup diversity.

- If conditions are not met, equalized-odds analysis is skipped.
- Skipped fairness outputs should not be treated as evidence of fairness.

## Representation Constraints

Representation analysis is optional and depends on embedding quality.

- No embeddings means no representation sub-score.
- Poorly aligned embeddings can mislead separability interpretation.

## Threshold and Penalty Design

Some trust-score thresholds and penalty boundaries are expert-designed heuristics.

- They are practical defaults, not universal constants.
- Domain-specific validation is recommended before using hard release gates.

## Not a Causal Fairness Auditor

TrustLens surfaces statistical disparities. It does not prove causality or policy compliance by itself.

- Human review and domain policy checks are still required.
- Regulatory and legal conclusions should include additional evidence.

## Recommended Mitigations

- Pair score-based gating with manual review for high-impact applications.
- Validate thresholds on your own datasets before strict automation.
- Track score behavior over time instead of relying on one run.
- Preserve full report artifacts for auditability.
