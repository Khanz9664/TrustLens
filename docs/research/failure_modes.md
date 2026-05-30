# TrustLens Failure Modes

A critical aspect of scientific evaluation is understanding when an auditing framework itself can fail. TrustLens is a diagnostic tool, and like any diagnostic, it can yield false positives (flagging a safe model) or false negatives (passing an unsafe model) under specific conditions.

> [!NOTE]
> **Evidence Traceability:** These failure modes have been derived directly from the observed edge cases during the execution of the `trustlens_model_zoo_benchmark.ipynb`.

## 1. The "Blind Spot" Failure (Missing Sensitive Attributes)
**Scenario:** A model exhibits severe bias against a specific demographic, but TrustLens awards it a high Bias Score and a passing Trust Score.

**Why it happens:** TrustLens can only audit what it can measure. If the `sensitive_features` array provided to the auditor does not contain the specific demographic proxy (or if the bias is occurring along an unmeasured intersectional boundary), the Bias module will report no disparity. 

**Guidance:** TrustLens is not a substitute for comprehensive data governance. The Bias Score certifies parity *only* with respect to the explicitly provided sensitive features.

## 2. The "Artificially Confident" Failure (Broken Probabilities)
**Scenario:** A model is highly inaccurate and frequently fails silently, yet TrustLens fails to adequately penalize its Failure Score.

**Why it happens:** Some model architectures (or poorly implemented wrappers) output hard labels (e.g., `[1.0, 0.0]`) instead of true continuous probabilities. In these cases, the `max()` confidence of every prediction is exactly `1.0`. While this destroys the Calibration Score (triggering a heavy penalty there), the Failure Score's "Confidence Gap" computation requires a continuous distribution to calculate meaningful density shifts. 

**Guidance:** Ensure that `y_prob` inputs are genuine softmax/sigmoid probability outputs, not one-hot encoded argmax predictions.

## 3. The "Representation Paradox" (LLMs and High-Dimensional Spaces)
**Scenario:** A highly capable, robust model is assigned a D-grade due to a near-zero Representation Score.

**Why it happens:** The Representation module evaluates latent spaces using Silhouette scores. However, in extremely high-dimensional spaces (e.g., 4096-d embeddings from an LLM), distance metrics suffer from the "curse of dimensionality." Points become approximately equidistant, leading to structural collapse of the Silhouette score, even if a non-linear head can easily separate the classes.

**Guidance:** For deep learning models with extremely high-dimensional embeddings, consider bypassing the Representation module entirely or performing dimensionality reduction (e.g., PCA/UMAP) before passing the embeddings to TrustLens.
