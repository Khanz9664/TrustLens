# Metric Limitations & Assumptions

Scientific transparency demands a rigorous account of what a framework *cannot* do. TrustLens relies on specific statistical methodologies that carry inherent limitations and edge cases.

> [!NOTE]
> **Evidence Traceability:** The limitations discussed on this page reflect the bounded scope of the `trustlens_model_zoo_benchmark.ipynb` and are derived from the empirical bounds of the framework's diagnostic capabilities.

## 1. Calibration Score Assumptions
The primary driver of the TrustLens Calibration module is the **Expected Calibration Error (ECE)**.
- **Limitation**: ECE is highly sensitive to the number of bins chosen and the binning strategy (uniform vs. quantile). TrustLens uses a standard 10-bin uniform approach.
- **Edge Case**: If a model outputs extremely clustered probabilities (e.g., heavily quantized outputs), the ECE might artificially inflate or deflate depending on where those clusters land relative to the bin edges.
- **Assumption**: The metric assumes that the target variable is stationary. If the base rate of the underlying population changes drastically post-deployment, the pre-computed ECE becomes invalid.

## 2. Failure Score & The Confidence Gap
The Failure Score penalizes models that make incorrect predictions with high confidence.
- **Limitation**: The metric assumes that the model's output probabilities are somewhat correlated with epistemic uncertainty.
- **Edge Case**: If a model is perfectly calibrated but inherently low-confidence (e.g., predicting 0.51 vs 0.49 for all samples), the Confidence Gap shrinks to zero. TrustLens will score this model highly on "Failure Stability" because it doesn't fail *confidently*, even though it fails frequently. (The Trust Score balances this by penalizing the baseline accuracy/calibration).

## 3. Bias Score Dependencies
The Bias module evaluates disparity across defined subgroups (e.g., via Equalized Odds or Demographic Parity).
- **Limitation**: TrustLens cannot detect bias against subgroups that are not explicitly defined in the `sensitive_features` array. It does not perform unsupervised intersectional bias discovery.
- **Edge Case**: In scenarios with extremely small minority subgroups (e.g., N < 10), the variance of the subgroup metric (like TPR/FPR) becomes so high that the resulting Bias penalty may be statistically noisy rather than structurally meaningful.

## 4. Representation Score (Silhouette)
The Representation module utilizes Silhouette scores to assess the clustering quality of the model's latent embeddings.
- **Limitation**: The Silhouette score inherently prefers convex, hyper-spherical clusters.
- **Edge Case**: Complex, non-linear manifold topologies (common in advanced vision models or LLMs) may receive artificially low Silhouette scores despite being highly separable by a downstream non-linear classifier. 
- **Assumption**: The module requires explicit access to internal model representations. For API-based black-box models, this module cannot be executed, and the framework gracefully degrades to omit the penalty.
