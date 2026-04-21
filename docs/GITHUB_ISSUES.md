## 🚀 Published Issues
> These issues are currently live on the GitHub tracker.

### Issue #1: Add Maximum Calibration Error (MCE) metric
**Label:** `good first issue`, `metrics`
**Description:**
MCE is the maximum (worst-case) calibration gap across all confidence bins.
Unlike ECE which averages across bins, MCE highlights the single worst miscalibrated region.

**Expected output:**
```python
from trustlens.metrics.calibration import maximum_calibration_error
mce = maximum_calibration_error(y_true, y_prob) # float in [0, 1]
```
**Difficulty:** Beginner

---

### Issue #7: Add `SpearmanCorrelation` between confidence and correctness
**Label:** `good first issue`, `metrics`
**Description:**
Compute Spearman rank correlation between `max_confidence` and `is_correct`.
Higher correlation → model's confidence better tracks accuracy.
Add to `failure.py` as `confidence_correctness_correlation(y_true, y_pred, y_prob)`.
**Difficulty:** Beginner

---

### Issue #13: Add `TrustReport.to_dict()` serialization method
**Label:** `good first issue`, `api`
**Description:**
Add a `to_dict()` method to `TrustReport` that returns all results as a flat Python dictionary (JSON-compatible).
This enables users to log results to tools like MLflow or W&B.
**Difficulty:** Beginner

---

### Issue #19: Add HTML report export via `report.to_html()`
**Label:** `enhancement`, `reporting`
**Description:**
Add a `to_html()` method that generates a self-contained HTML report with:
- Embedded base64 PNG plots
- Metric tables with color-coded cells (green=good, red=bad)
- Collapsible sections per module
**Difficulty:** Intermediate

---

### Issue #20: Add `critical_failures()` method to `TrustReport`
**Label:** `enhancement`, `api`
**Description:**
Add a method that returns the top-N highest-confidence misclassifications across all classes.

```python
failures = report.critical_failures(n=20)
# Returns DataFrame: index, y_true, y_pred, confidence
```
**Difficulty:** Intermediate

---

### Issue #22: Add UMAP/t-SNE visualization for embeddings
**Label:** `enhancement`, `visualization`
**Description:**
In `trustlens/visualization/representation_plots.py`, add `plot_embedding_2d()`:
- Compute 2D projection using UMAP (preferred) or t-SNE (fallback)
- Color points by class label
- Add legend and silhouette score annotation
**Difficulty:** Intermediate

---

### Issue #2: Add per-class accuracy metric to failure analysis
**Label:** `good first issue`, `metrics`
**Description:**
Accuracy is often misleading in imbalanced datasets. We need to complement the `misclassification_summary` with a straightforward dictionary of per-class accuracy values.

**Task:**
Implement `per_class_accuracy(y_true, y_pred) -> dict` in `trustlens/metrics/failure.py`.

**Expected output:**
```python
acc = per_class_accuracy(y_true, y_pred)
# {0: 0.943, 1: 0.872}
```
**Difficulty:** Beginner

---

### Issue #3: Add shared pytest fixtures via conftest.py
**Label:** `good first issue`, `testing`
**Description:**
As the test suite grows, we are seeing significant boilerplate duplication in dataset creation. We should centralize common test assets using a root-level `conftest.py`.

**Task:**
Create `tests/conftest.py` and implement the following fixtures:
- `binary_dataset`: A tuple of small NumPy arrays for binary classification.
- `multiclass_dataset`: A 3-class dataset for verifying the upcoming multi-class support.
- `trained_rf`: A pre-fitted `RandomForestClassifier` to avoid training time in every test.

**Difficulty:** Beginner

---

### Issue #10: Fix visualization crash for single-class edge case
**Label:** `bug`, `visualization`
**Description:**
The `plot_class_distribution()` function assumes the presence of at least two classes. When provided with a single-class dataset (common in filtered data or specific edge cases), it raises a `ValueError` or crashes during plotting.

**Task:**
Add a safety guard in `trustlens/visualization/bias_plots.py` to handle 1-item class distributions gracefully. Instead of crashing, it should display a warning message or a simplified bar chart.

**Test case:**
```python
# This should no longer crash
report = class_imbalance_report(np.zeros(50, dtype=int))
plot_class_distribution(report)
```
**Difficulty:** Beginner

---

### Issue #28: Add tqdm progress bars to analyze()
**Label:** `enhancement`, `ux`
**Description:**
The `analyze()` pipeline can take several seconds when multiple compute-intensive modules (like representation or large-scale calibration) are active. Users currently get no feedback until the entire report is built.

**Task:**
Integrate `tqdm` into the main `analyze()` loop in `trustlens/api.py`.
- The progress bar should only show when `verbose=True`.
- `tqdm` must remain an **optional dependency**. The code should use a fallback (e.g., logging) if `tqdm` is not installed.

**Difficulty:** Intermediate

---

### Issue #34: Add MLflow integration for logging TrustReport
**Label:** `enhancement`, `integrations`
**Description:**
TrustLens results are often part of a larger experiment tracking workflow. We want to make it seamless to log a `TrustReport` directly to an active MLflow run.

**Task:**
Add a new integration module `trustlens/integrations/mlflow.py` with a `log_trust_report(report, run=None)` function.
Properties to log:
- Scalar metrics (Brier, ECE, individual trust scores) as MLflow metrics.
- High-level metadata.
- Plots (summary plot, reliability diagrams) as MLflow artifacts.

**Difficulty:** Intermediate

---

### Issue #35: Add Weights & Biases (W&B) integration for TrustReport logging
**Label:** `enhancement`, `integrations`
**Description:**
W&B is a primary experiment tracker. Users currently have to manually flatten reports or use `to_dict()` and `wandb.log()`. A first-class integration would handle metrics, metadata, and plots automatically.

**Task:**
Implement `trustlens.integrations.wandb.log_trust_report(report)`:
- Logs metrics via `wandb.log()`
- Logs reliability diagram via `wandb.Image()`
- Automatically initializes a run if one doesn't exist.

**Difficulty:** Intermediate

---

### Issue #25: Add `equalized_odds` metric to bias module — the fairness gold standard
**Label:** `enhancement`, `fairness`
**Description:**
Subgroup accuracy gap tells you *if* one group is worse off. Equalized Odds tells you *how* they're worse off — through predictive bias in TPR or FPR.

**Task:**
Add `equalized_odds(y_true, y_pred, sensitive_features) → dict` to `trustlens/metrics/bias.py`.
- Compute per-group TPR and FPR.
- Calculate the maximum disparity (gap) between groups.
- Label the violation as "severe" (>0.15), "moderate" (0.05-0.15), or "acceptable" (<0.05).

**Difficulty:** Intermediate

---

### Issue #51: Add a powerful CLI entry point: `trustlens analyze`
**Label:** `enhancement`, `ux`
**Description:**
Empower users to perform high-fidelity model audits directly from the terminal without writing a single line of Python.

**Task:**
Implement `trustlens analyze` with support for built-in datasets and local files.
- Command: `trustlens analyze --dataset breast_cancer`
- Command: `trustlens analyze --data path/to/data.csv --model path/to/model.pkl`
- Use `typer` for the CLI framework and register it as a console script.

**Difficulty:** Intermediate

---

### Issue #0: Add clearer error message for invalid input shapes
**Title:** Add clearer error message for invalid input shapes (great first PR)
**Label:** `good first issue`, `beginner-friendly`
**Problem**
When users pass incorrect input shapes (for example mismatched `y_true` and `y_prob`), the error messages are not always clear.
This can make debugging difficult, especially for beginners.
We should provide clear and helpful error messages.
Improve error messages for invalid inputs.

---

### Issue #5
**Title:** Improve docstring for `analyze()` with a full code example
**Label:** `good first issue`, `docs`
**Description:**
The `analyze()` docstring has a brief example.
Expand the `Examples` section with a 10-line runnable code snippet using `make_classification`.

**Difficulty:** Beginner

---

## Beginner Issues (0–15)

---

### Issue #0 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

### Issue #1 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

### Issue #2 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

### Issue #3 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

### Issue #4
**Title:** Add `__version__` test
**Label:** `good first issue`, `testing`
**Description:**
Add `tests/test_version.py` that asserts:
1. `trustlens.__version__` is importable
2. It matches a valid semantic version pattern (regex: `\d+\.\d+\.\d+`)

**Difficulty:** Beginner

---

### Issue #5 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

### Issue #6
**Title:** Add `n_samples` per bin to `reliability_curve()` output
**Label:** `good first issue`, `metrics`
**Description:**
The current `reliability_curve()` returns `(frac_pos, mean_pred, bin_counts)`.
The `bin_counts` computation is approximate. Refactor it to use exact binning logic from `expected_calibration_error`.

**Expected output:** Same tuple, but `bin_counts[i]` exactly equals samples in bin `i`.
**Difficulty:** Beginner

---

### Issue #7 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

### Issue #8
**Title:** Add Python 3.13 to CI matrix
**Label:** `good first issue`, `ci`
**Description:**
Add Python 3.13 to the GitHub Actions CI matrix in `.github/workflows/ci.yml`.
Ensure all tests pass or document known issues.

**Difficulty:** Beginner

---

### Issue #9
**Title:** Add `requirements-dev.txt` generated from `setup.cfg[dev]`
**Label:** `good first issue`, `packaging`
**Description:**
Add a `requirements-dev.txt` file pinning exact dev dependencies.
Generate with: `pip-compile setup.cfg --extra dev -o requirements-dev.txt`

**Difficulty:** Beginner

---

### Issue #10 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

### Issue #11
**Title:** Add `__all__` to remaining modules
**Label:** `good first issue`, `code quality`
**Description:**
Add `__all__` lists to:
- `trustlens/metrics/calibration.py`
- `trustlens/metrics/failure.py`
- `trustlens/metrics/bias.py`
- `trustlens/metrics/representation.py`

**Difficulty:** Beginner

---

### Issue #12
**Title:** Write `examples/bias_analysis_demo.py`
**Label:** `good first issue`, `examples`
**Description:**
Create a runnable example that:
1. Generates a dataset with gender + age_group sensitive features
2. Runs `subgroup_performance()`
3. Prints a formatted subgroup table to stdout

**Difficulty:** Beginner

---

### Issue #13 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

### Issue #14
**Title:** Add `SECURITY.md` file
**Label:** `good first issue`, `community`
**Description:**
Create `SECURITY.md` documenting how to report security vulnerabilities responsibly.
Follow GitHub's recommended security policy template.

**Difficulty:** Beginner

---

### Issue #15
**Title:** Add `CODE_OF_CONDUCT.md`
**Label:** `good first issue`, `community`
**Description:**
Add `CODE_OF_CONDUCT.md` using the Contributor Covenant v2.1.
Update README to link to it.

**Difficulty:** Beginner

---

## Intermediate Issues (16–35)

---

### Issue #16
**Title:** Implement Eigen-CAM for convolutional neural networks
**Label:** `enhancement`, `explainability`
**Description:**
Eigen-CAM uses the first eigenvector of the feature map matrix instead of gradients.
It is gradient-free, faster, and works with models that have non-differentiable components.

Implement `EigenCAM` in `trustlens/explainability/eigencam.py` with the same interface as `GradCAM`.

**Expected output:** Heatmap array (H, W) in [0, 1].
**Reference:** Muhammad & Yeasin (2020), Eigen-CAM. IJCNN.
**Difficulty:** Intermediate

---

### Issue #17
**Title:** Add multi-class ECE (label-wise decomposition)
**Label:** `enhancement`, `metrics`
**Description:**
The current ECE assumes binary classification.
Extend `expected_calibration_error` to support multi-class via one-versus-rest decomposition.
Return both per-class ECE and a macro-averaged ECE.

**Difficulty:** Intermediate

---

### Issue #18
**Title:** Add temperature scaling calibration
**Label:** `enhancement`, `calibration`
**Description:**
Implement `TemperatureScaler` — a post-hoc calibration method that learns a scalar temperature `T` to divide all logits.

```python
scaler = TemperatureScaler()
scaler.fit(logits_val, y_val)  # learns T
probs_cal = scaler.predict_proba(logits_test)
```

Add to `trustlens/calibrators/temperature.py`.
**Difficulty:** Intermediate

---

### Issue #19 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

### Issue #20 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

### Issue #21
**Title:** Implement Integrated Gradients for tabular models
**Label:** `enhancement`, `explainability`
**Description:**
Integrated Gradients attribute feature importance for any differentiable model.
Implement `IntegratedGradients` in `trustlens/explainability/integrated_gradients.py`.

**Input:** Tabular feature vector + baseline (e.g., zeros).
**Output:** Attribution vector per feature.
**Difficulty:** Intermediate

---

### Issue #22 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

### Issue #23
**Title:** Add precision-recall curve per class to failure module
**Label:** `enhancement`, `metrics`
**Description:**
Add `per_class_pr_curve(y_true, y_prob)` returning PR curve data and AUC-PR for each class.
Add corresponding visualization `plot_pr_curves()`.

**Difficulty:** Intermediate

---

### Issue #24
**Title:** Add Jupyter `_repr_html_` to TrustReport
**Label:** `enhancement`, `ux`
**Description:**
When `TrustReport` is returned in a Jupyter notebook cell, display a rich HTML card:
- Model name, sample count, modules run
- Key metrics (BS, ECE) as color-coded badges

**Difficulty:** Intermediate

---

### Issue #25 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

### Issue #26
**Title:** Add `demographic_parity` metric
**Label:** `enhancement`, `fairness`
**Description:**
Demographic Parity requires equal positive prediction rates across groups.
Add `demographic_parity(y_pred, sensitive_features)`.

**Difficulty:** Intermediate

---

### Issue #27
**Title:** Add subgroup ECE (calibration per demographic group)
**Label:** `enhancement`, `fairness`
**Description:**
Extend `subgroup_performance()` to also compute ECE per group.
A model can be well-calibrated overall but systematically miscalibrated for minority groups.

**Difficulty:** Intermediate

---

### Issue #28 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

### Issue #29
**Title:** Add SHAP wrapper integration
**Label:** `enhancement`, `explainability`
**Description:**
Add optional SHAP support:
```python
from trustlens.explainability.shap_wrapper import SHAPExplainer
exp = SHAPExplainer(model, X_background)
shap_values = exp.explain(X_val)
```
Depends on `shap` as an optional extra (`pip install "trustlens[full]"`).

**Difficulty:** Intermediate

---

### Issue #30
**Title:** Add overconfidence / underconfidence classifiers
**Label:** `enhancement`, `calibration`
**Description:**
Add `calibration_diagnosis(y_true, y_prob)` that categorizes a model as:
- `"well-calibrated"` — ECE < 0.05
- `"overconfident"` — mean predicted confidence > mean accuracy
- `"underconfident"` — mean predicted confidence < mean accuracy

Return a structured dict with diagnosis + supporting stats.
**Difficulty:** Intermediate

---

### Issue #31
**Title:** Add `plot_subgroup_comparison()` bar chart
**Label:** `enhancement`, `visualization`
**Description:**
Visualize per-group accuracy/F1 as a grouped bar chart, one group per sensitive feature.
Highlight the worst-performing group in red.

**Difficulty:** Intermediate

---

### Issue #32
**Title:** Add model comparison via `compare_reports()`
**Label:** `enhancement`, `api`
**Description:**
Add a top-level `compare_reports(reports: list[TrustReport])` function that generates a comparison table of key metrics across models.

**Output:**
```
       Model A  Model B  Model C
Brier Score  0.061   0.043   0.091
ECE      0.042   0.031   0.088
...
```
**Difficulty:** Intermediate

---

### Issue #33
**Title:** Add `prediction_flip_analysis()` to failure module
**Label:** `enhancement`, `metrics`
**Description:**
For each test sample, measure how much Gaussian noise is needed before the predicted class flips.
Higher = more robust prediction.
Add `prediction_flip_analysis(model, X, n_perturbations=50, sigma_range=(0.01, 1.0))`.

**Difficulty:** Intermediate

---

### Issue #34 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

### Issue #35 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

## Advanced Issues (36–50)

---

### Issue #36
**Title:** Implement Grad-CAM for Vision Transformers (ViT)
**Label:** `research`, `explainability`
**Description:**
Standard Grad-CAM relies on spatial feature maps from convolutional layers.
ViTs have attention layers instead.

Implement `ViTGradCAM` using:
1. Gradient w.r.t. the attention weights of the last transformer block
2. Project token-level gradients back to 2D image patches

**Reference:** Chefer et al. (2021), Transformer Interpretability Beyond Attention Visualization. CVPR.
**Difficulty:** Advanced

---

### Issue #37
**Title:** Implement Attention Rollout for ViT self-attention visualization
**Label:** `research`, `explainability`
**Description:**
Attention Rollout recursively multiplies attention matrices across all layers to propagate attention from output tokens to input patches.

Add `AttentionRollout` class in `trustlens/explainability/attention_rollout.py`.

**Reference:** Abnar & Zuidema (2020), Quantifying Attention Flow in Transformers.
**Difficulty:** Advanced

---

### Issue #38
**Title:** Implement layer-wise CKA heatmap
**Label:** `research`, `representation`
**Description:**
Compute CKA between all pairs of `n_layers` checkpoints and visualize as a symmetric heatmap.
This reveals when/where representations converge or diverge during training or across architectures.

**Input:** List of `n_layers` embedding matrices, each shape (n_samples, dim).
**Output:** (n_layers × n_layers) CKA matrix + heatmap figure.

**Difficulty:** Advanced

---

### Issue #39
**Title:** Add representation fragility score
**Label:** `research`, `representation`
**Description:**
Measure how much an input needs to be perturbed in input space before its embedding crosses a class boundary.
Compute per-class fragility scores.

**Difficulty:** Advanced

---

### Issue #40
**Title:** Implement intrinsic dimensionality estimation for embeddings
**Label:** `research`, `representation`
**Description:**
Estimate the intrinsic dimensionality of an embedding manifold using two-NN estimator.
High intrinsic dim → potentially redundant/overparameterized representation.

**Reference:** Facco et al. (2017), Estimating the Intrinsic Dimension of Datasets by a Minimal Neighborhood Information.
**Difficulty:** Advanced

---

### Issue #41
**Title:** Add token-level faithfulness test for NLP/BERT models
**Label:** `research`, `explainability`
**Description:**
Extend the pixel deletion framework to text:
- Replace top-k most salient tokens with [MASK]
- Track confidence drop curve
- Compute token AUPC

**Difficulty:** Advanced

---

### Issue #42
**Title:** Implement `benchmark()` on CIFAR-10 with standard models
**Label:** `research`, `benchmarking`
**Description:**
Create `trustlens.benchmark.vision.cifar10_benchmark()` that:
1. Downloads CIFAR-10 test set
2. Downloads pretrained torchvision ResNet50
3. Runs full TrustLens analysis
4. Saves results as standardized JSON for comparison

**Difficulty:** Advanced

---

### Issue #43
**Title:** Add Hugging Face `evaluate`-compatible metric modules
**Label:** `research`, `integrations`
**Description:**
Wrap `brier_score`, `expected_calibration_error`, and `embedding_separability` as HF `evaluate.Metric` objects.

**Expected output:**
```python
import evaluate
ece_metric = evaluate.load("trustlens/ece")
ece_metric.compute(predictions=probs, references=y_true)
```
**Difficulty:** Advanced

---

### Issue #44
**Title:** Design and implement the TrustLens web dashboard
**Label:** `research`, `platform`
**Description:**
Build a FastAPI + HTMX web dashboard launched via `trustlens serve`.
Must include:
- Reliability diagram (Plotly)
- Confidence gap histogram
- Subgroup comparison table
- JSON report download button

No React dependency — use HTMX for interactivity.
**Difficulty:** Advanced

---

### Issue #45
**Title:** Implement neuron activation statistics per class
**Label:** `research`, `representation`
**Description:**
For each neuron in a specified layer, compute:
- Mean activation per class
- Top-K classes each neuron responds to
- "Polysemanticity" score (how many classes trigger a neuron)

**Difficulty:** Advanced

---

### Issue #46
**Title:** Add consistency/agreement metric between explanation methods
**Label:** `research`, `explainability`
**Description:**
Given two saliency maps (e.g., Grad-CAM and SHAP), compute their rank-order correlation and spatial overlap.
Add `explanation_consistency(map_a, map_b)` returning Spearman ρ and IoU of top-k regions.

**Difficulty:** Advanced

---

### Issue #47
**Title:** Implement DINO-compatible self-attention visualization
**Label:** `research`, `explainability`
**Description:**
DINO-pretrained ViTs produce semantically meaningful self-attention maps without supervision.
Add `DINOAttentionMap` class that extracts the last-layer `[CLS]` attention.

**Reference:** Caron et al. (2021), Emerging Properties in Self-Supervised Vision Transformers. ICCV.
**Difficulty:** Advanced

---

### Issue #48
**Title:** Build community plugin submission process
**Label:** `research`, `community`
**Description:**
Design and implement a `trustlens-contrib` community plugin repository with:
- Plugin template (`cookiecutter` template)
- Automated CI checks for submitted plugins
- Plugin registry page on the docs site
- GitHub Actions bot for plugin validation

**Difficulty:** Advanced

---

### Issue #49
**Title:** Add prediction interval calibration for regression models
**Label:** `research`, `calibration`
**Description:**
Extend TrustLens to support regression models with prediction intervals.
Implement coverage probability and interval sharpness metrics.

```python
from trustlens.metrics.regression_calibration import interval_coverage
coverage = interval_coverage(y_true, y_pred_lower, y_pred_upper)
```

**Reference:** Angelopoulos & Bates (2022), Conformal Risk Control.
**Difficulty:** Advanced

---

### Issue #50
**Title:** Implement a public TrustLens leaderboard
**Label:** `research`, `platform`
**Description:**
Design a public leaderboard at `trustlens.dev/leaderboard` that:
- Accepts JSON report uploads from users
- Groups by dataset (CIFAR-10, ImageNet, custom)
- Ranks models by ECE, AUPC, silhouette score
- Displays anonymized or public model cards

Backend: FastAPI + SQLite (initial). Frontend: HTMX or minimal React.
**Difficulty:** Advanced
### Issue #51 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*
