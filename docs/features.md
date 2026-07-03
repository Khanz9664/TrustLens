# Features and Modules

TrustLens is organized as focused modules that diagnose model reliability from different angles. Together, they produce one deployment-oriented verdict while preserving detailed diagnostics.

## Calibration Module

Calibration checks whether predicted probabilities match real-world outcomes.

- **Expected Calibration Error (ECE)**: weighted gap between confidence and observed accuracy across bins.
- **Brier Score**: mean squared error of probabilistic predictions.
- **Reliability Curve Data**: confidence versus observed correctness for calibration plots.

Use calibration results when your downstream decision threshold depends on confidence quality rather than raw class labels.

> [!TIP]
> **Multiclass Support**: TrustLens v0.5.0 supports multiclass calibration using top-label Expected Calibration Error (ECE) and class-wise Mean Squared Error (Multiclass Brier Score).

## Failure Analysis Module

Failure analysis focuses on risk concentration, not only total error rate.

- **Misclassification Summary**: class-wise error rates and high-confidence mistakes.
- **Confidence Gap**: difference between confidence on correct predictions and confidence on incorrect predictions.
- **Failure Pattern Signals**: identifies behavior such as overconfident mistakes.

Use this module to prioritize error analysis where failure impact is highest.

## Bias and Fairness Module

Fairness diagnostics identify performance disparity between subgroups.

- **Class Imbalance Report**: distribution imbalance and majority/minority ratios.
- **Subgroup Performance**: group-wise accuracy and macro-F1 with gap summaries.
- **Equalized Odds Checks**: group-wise TPR/FPR disparity for binary classification.
- **Multi-Feature Fairness Visualization**: pass multiple sensitive features (e.g., gender, age, income) and TrustLens generates per-feature plots for every visualization type — no feature is silently dropped.
  - Features are processed in sorted order for deterministic, reproducible output.
  - Filenames are automatically sanitized for safety (e.g., `"income level"` → `income_level`).
  - `plot_module` serves as the sole orchestrator for saving and closing figures.
  ```python
  # High-level: generate all diagnostic plots
  report.plot_bias(mode="all")

  # Orchestrated batch save: one call, all per-feature files
  from trustlens.visualization import plot_module
  plot_module("bias", report.results["bias"], save_dir="plots/")
  ```

Use these outputs to detect whether your model systematically underperforms on particular segments.

## Representation Module

Representation diagnostics evaluate geometric quality of embeddings when latent vectors are available.

- **Embedding Separability**: silhouette-based estimate of class separation.
- **Within/Between Distance Statistics**: distance-based signal for overlap versus separation.
- **CKA Utility**: centered kernel alignment support for representation similarity studies.

Representation analysis is optional and only runs when embeddings are provided.

## Regression Module

The regression module evaluates the accuracy of continuous predictions and the reliability of their uncertainty bounds.

- **Accuracy and Skill**: Evaluates point predictions using RMSE, MSE, and R² (Skill Score against a predict-the-mean baseline). Includes heavy-tail penalty diagnostics to catch catastrophic misses.
- **Interval Calibration**: Checks if prediction intervals are honest. Evaluates single-level PICP and multi-level Interval Calibration Error (ICE).
- **Uncertainty Informativeness**: Evaluates whether uncertainty bounds are useful. Uses a calibration-conditioned sharpness proxy (to reject artificially narrow, over-confident intervals) or error-variance correlation.

Use these outputs when your continuous target predictions drive risk-sensitive decisions where both the predicted value and the confidence in that value matter.

## Trust Scoring Engine

The trust scoring engine combines module outputs into one decision support signal.

- **Composite Score (0-100)** with a grade and deployment verdict.
- **Weight Redistribution** when some modules are unavailable.
- **Risk Penalties** applied for severe calibration, failure, and fairness conditions.
- **Deployment Blockers** that force a do-not-deploy verdict despite high raw score.

For exact rules, see [Trust Score Explained](trust_score_explained.md).
