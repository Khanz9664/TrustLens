# Methodology & Threats to Validity

This document outlines the scientific methodology used to validate the TrustLens framework in our official benchmarks and explicitly details the limitations and threats to validity of our findings.

> [!NOTE]
> **Evidence Traceability:** All methodological parameters described below are sourced directly from the reproducible setup in `examples/trustlens_model_zoo_benchmark.ipynb`.

## Benchmark Methodology

### Experimental Setup
To validate the trustworthiness dimensions of TrustLens (Calibration, Failure, Bias, and Representation), we constructed a controlled experimental environment evaluating four standard machine learning architectures:
- **Logistic Regression** (Linear baseline)
- **Random Forest** (Ensemble baseline)
- **XGBoost** (Gradient boosting baseline)
- **MLP Classifier** (Deep learning baseline)

### Reproducibility
Scientific rigor requires reproducibility. The benchmark employs:
- **Statistical Aggregation**: All experiments are run across three distinct random seeds (`42`, `123`, `999`).
- **Fixed Sample Sizes**: Synthetic datasets are generated with a strict `N=1500` samples to maintain consistent variance across experimental runs.

### Evaluation Protocol & Corruption Assumptions
The framework is evaluated not just on clean data, but under progressive distribution shifts designed to simulate real-world failure modes:
1. **Baseline**: Unperturbed, clean synthetic data.
2. **Label Noise**: Gaussian noise injected into the targets (Low: 10%, Moderate: 20%, Severe: 30%) to test the framework's sensitivity to data quality degradation.
3. **Subgroup Imbalance**: Controlled class imbalance injected to evaluate the framework's ability to penalize models that sacrifice minority performance for aggregate accuracy.

## Threats to Validity

To maintain scientific honesty, we explicitly acknowledge the following threats to the validity of the TrustLens benchmark results. A high Trust Score in this benchmark does **not** guarantee real-world safety.

### 1. Benchmark Mostly Relies on Synthetic Datasets
Our fairness and robustness experiments rely heavily on controlled synthetic perturbations (e.g., explicit subgroup generation and Gaussian noise injection). Real-world data distributions are vastly more complex, containing intersecting, unobserved confounders that these synthetic benchmarks fail to capture.

### 2. Strict Binary Classification Focus
The current TrustLens metrics, particularly the Calibration and Failure modules, are strictly evaluated under binary classification assumptions in this benchmark. The degradation slopes and metric behaviors may behave unpredictably in multi-class or multi-label environments.

### 3. Sensitive Attribute Simulation and Fairness Assumptions
The Bias module assumes that all relevant sensitive attributes are explicitly known, perfectly measured, and provided during evaluation. TrustLens cannot detect bias against unmeasured groups or intersectional identities absent from the configuration.

### 4. Calibration Dependence on Probability Quality
The Failure and Calibration metrics inherently assume that the model provides meaningful probability outputs (e.g., via `predict_proba`). If a model outputs hard labels (or broken probabilities like `[1.0, 0.0]`), the framework's confidence gap calculations become invalid.

### 5. ECE Binning Instability
Our calibration penalty relies on Expected Calibration Error (ECE), which is notoriously sensitive to the number of bins and the binning strategy (uniform vs. quantile). Minor changes to binning hyperparameters could artificially inflate or deflate the calibration penalty.

### 6. Trust Score Weighting Assumptions
The final Trust Score is aggregated using heuristic weights (e.g., 35% Calibration, 25% Bias). These weights reflect our internal prioritization of risks and are not derived from fundamental mathematical laws. Different deployment contexts may require entirely different weightings.

### 7. Representation Metrics May Fail on Nonlinear Manifolds
The Representation module utilizes Silhouette scores to assess the clustering quality of latent embeddings. This inherently prefers convex, hyper-spherical clusters and may completely fail to accurately score the complex, nonlinear manifolds typically produced by modern deep learning models or LLMs.

### 8. Benchmark ≠ Real-World Guarantee
Ultimately, the TrustLens benchmark proves that the framework reacts mathematically to specific, controlled data degradations. It does **not** provide a guarantee of deployment safety. Results should always be interpreted alongside rigorous domain expertise and contextual risk assessment.
