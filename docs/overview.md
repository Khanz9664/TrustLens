# Overview

TrustLens is a reliability-focused evaluation layer for classification models. It helps teams move from metric reporting to deployment decisions backed by evidence.

## Why This Matters

Accuracy alone is often insufficient for production decisions. A model can score high on accuracy while still being unsafe to deploy because of:

- overconfident errors
- subgroup performance disparity
- weak probability calibration

TrustLens addresses this by combining diagnostic modules and explicit decision logic.

## What TrustLens Evaluates

TrustLens evaluates models across four dimensions:

- **Calibration**: are predicted probabilities aligned with real outcomes?
- **Failure behavior**: are errors concentrated in high-confidence regions?
- **Bias and fairness**: do important subgroups see uneven performance?
- **Representation quality**: are embeddings well separated when provided?

These diagnostics are combined into a Trust Score, with penalties and blocker rules applied for high-risk conditions.

## Typical Workflow

1. Run `analyze(model, X_val, y_val, y_prob=...)`.
2. Inspect the returned `TrustReport`.
3. Review score, blockers, and dimension-level outputs.
4. Export artifacts for CI, governance, or comparison.

## What You Get

A TrustLens run produces:

- module-level metrics
- a composite Trust Score with grade and verdict
- narrative insights and detected risk patterns
- saveable artifacts for downstream workflows

## Related Pages

- [Getting Started](getting_started.md)
- [Features and Modules](features.md)
- [Trust Score Explained](trust_score_explained.md)
- [Known Limitations](known_limitations.md)
