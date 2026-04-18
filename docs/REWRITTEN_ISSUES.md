---

## 🚀 Published Issues
> These issues have been moved to the active GitHub tracker.

### Issue #1: Add Maximum Calibration Error (MCE) — the worst-case calibration metric
**Problem:** ECE tells you the average calibration gap. But what's the worst bin?
**Your task:** Add `maximum_calibration_error(y_true, y_prob, n_bins=10) → float` to `trustlens/metrics/calibration.py`.
**Difficulty:** Beginner

---

### Issue #7: Add confidence-correctness Spearman correlation to failure analysis
**Problem:** A model with high Spearman correlation between confidence and correctness is trustworthy by construction.
**Your task:** Add `confidence_correctness_correlation(y_true, y_pred, y_prob) → dict` to `trustlens/metrics/failure.py`.
**Difficulty:** Beginner

---

### Issue #13: Add `TrustReport.to_dict()` for JSON serialization
**Problem:** `TrustReport` is difficult to serialize for logging to experiment trackers (MLflow, W&B).
**Your task:** Implement a `to_dict()` method that returns a JSON-serializable dictionary (NumPy arrays to lists/floats).
**Difficulty:** Beginner

---

### Issue #19: Add HTML report export — `report.to_html()` with embedded charts
**Problem:** Stakeholders need a single-file shareable report — no Python required.
**Your task:** Create a self-contained `.html` file with embedded base64 PNGs and color-coded metrics.
**Difficulty:** Intermediate

---

### Issue #20: Add `report.critical_failures()` — ranked table of model mistakes
**Problem:** Users want a programmatic way to access high-confidence misclassifications.
**Your task:** Add `critical_failures(n=20) → list[dict]` to `TrustReport`.
**Difficulty:** Intermediate

---

### Issue #22: Add UMAP/t-SNE embedding visualization — see what your model learned
**Problem:** A silhouette score can't show class overlaps or ignore clusters visually.
**Your task:** Add `plot_embedding_2d(embeddings, y_true, method="umap")` to `trustlens/visualization/representation_plots.py`.
**Difficulty:** Intermediate

---

### Issue #2: Add per-class accuracy metric to failure analysis
**Problem:** Accuracy is often misleading in imbalanced datasets. We need to complement the `misclassification_summary` with a straightforward dictionary of per-class accuracy values.
**Your task:** Implement `per_class_accuracy(y_true, y_pred) -> dict` in `trustlens/metrics/failure.py`.
**Difficulty:** Beginner

---

### Issue #3: Add shared pytest fixtures via conftest.py
**Problem:** As the test suite grows, we are seeing significant boilerplate duplication in dataset creation. We should centralize common test assets using a root-level `conftest.py`.
**Your task:** Create `tests/conftest.py` with shared fixtures: `binary_dataset`, `multiclass_dataset`, and `trained_rf`.
**Difficulty:** Beginner

---

### Issue #10: Fix visualization crash for single-class edge case
**Problem:** The `plot_class_distribution()` function assumes the presence of at least two classes. When provided with a single-class dataset, it raises a `ValueError` or crashes during plotting.
**Your task:** Add a safety guard in `trustlens/visualization/bias_plots.py` to handle 1-item class distributions gracefully.
**Difficulty:** Beginner

---

### Issue #28: Add tqdm progress bars to analyze() — silence is the enemy of UX
**Problem:** `analyze()` is the heart of TrustLens. When modules like `representation` or `bias` run on large datasets, the terminal hangs without feedback.
**Your task:** Add `tqdm` progress tracking to the module execution loop in `trustlens/api.py`.
**Technical constraint:** `tqdm` must be an optional dependency (fallback to standard logging).
**Difficulty:** Intermediate

---

### Issue #34: Add MLflow integration for logging TrustReport — connect to the MLOps ecosystem
**Problem:** Most teams use MLflow to track experiments. Manually extracting metrics from a `TrustReport` and logging them individually is tedious and error-prone.
**Your task:** Implement `trustlens.integrations.mlflow.log_trust_report(report)` to log scalar scores and plot artifacts.
**Difficulty:** Intermediate

---

### Issue #35: Add Weights & Biases (W&B) integration — track Trust Scores across experiments
**Problem:** W&B is a primary experiment tracker. Users currently have to manually flatten reports or use `to_dict()` and `wandb.log()`. A first-class integration would handle metrics, metadata, and plots automatically.
**Your task:** Implement `trustlens.integrations.wandb.log_trust_report(report)` to log scalar scores and plot artifacts.
**Difficulty:** Intermediate

---

## Issue #1 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

## Issue #7 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

## Issue #2 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

## Issue #3 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

## Issue #10 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

## Issue #16 (Intermediate) — High-Value Explainability Feature

### Implement Eigen-CAM — gradient-free visual explanations for CNNs

**Problem:**
Grad-CAM requires a backward pass. For some model architectures (quantized models, models with non-differentiable ops), gradients are unavailable.

Eigen-CAM is a drop-in replacement — it uses the first principal component of the feature map to generate explanations. No gradients needed. Often more visually coherent.

**Your task:** Implement `EigenCAM` in `trustlens/explainability/eigencam.py`.

**Interface (must match GradCAM):**
```python
from trustlens.explainability import EigenCAM

cam = EigenCAM(model, target_layer=model.layer4[-1])
heatmap = cam.generate(image_tensor)     # shape (H, W), values in [0,1]
fig   = cam.overlay(image_np, heatmap)
```

**Algorithm:**
1. Forward pass → extract feature map `A` of shape `(C, H, W)` from target layer
2. Flatten to `(C, H*W)` → compute first right singular vector via SVD
3. Project: `cam = A.T @ v` → reshape to `(H, W)` → ReLU → normalize

**Reference:** Muhammad & Yeasin (2020), *EigenCAM: Class Activation Map using Principal Components*. IJCNN.

**Tests required:** `test_eigencam_output_shape`, `test_eigencam_values_in_range`, `test_eigencam_no_backward_needed`.

**Difficulty:** Intermediate · Est. time: 4–6 hours

---

## Issue #17 (Intermediate) — Extends Core Metric

### Extend ECE to multi-class classifiers (one-vs-rest label-wise decomposition)

**Problem:**
The current `expected_calibration_error()` only handles binary classifiers. But most real models are multi-class.

Multi-class ECE should compute per-class calibration and return a macro-averaged ECE alongside per-class breakdown.

**Your task:** Extend `expected_calibration_error()` to accept multi-dimensional `y_prob`.

**Expected output:**
```python
# Binary (current behavior, unchanged)
ece = expected_calibration_error(y_true, y_prob_binary) # float

# Multi-class (new behavior)
ece = expected_calibration_error(y_true, y_prob_multiclass)
# {
#  "macro_ece": 0.071,
#  "per_class": {0: 0.045, 1: 0.082, 2: 0.085}
# }
```

**Implementation strategy:** For each class `k`, create binary labels (`y_true == k`) and binary probs (`y_prob[:, k]`), compute ECE, average.

**Backward compatibility required:** If `y_prob` is 1-D, return a float (existing behavior).

**Difficulty:** Intermediate · Est. time: 3–4 hours

---

## Issue #19 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

## Issue #20 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

## Issue #22 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

## Issue #28 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

## Issue #34 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

## Issue #35 [PUBLISHED]
*This issue has been moved to the active GitHub tracker.*

---

## Issue #25 (Intermediate) — Fairness Research

### Add Equalized Odds — the fairness metric that catches systematic discrimination

**Problem:**
Subgroup accuracy gap tells you *if* one group is worse off.
Equalized Odds tells you *how* they're worse off — through predictive bias in True Positive Rate or False Positive Rate.

A model can have equal accuracy but a TPR gap of 30% between groups. That's systematic discrimination, invisible to accuracy-based metrics.

**Your task:** Add `equalized_odds(y_true, y_pred, sensitive_features) → dict` to `trustlens/metrics/bias.py`.

**Expected output:**
```python
result = equalized_odds(y_true, y_pred, {"gender": gender_arr})
# {
#  "gender": {
#   "M": {"tpr": 0.88, "fpr": 0.12},
#   "F": {"tpr": 0.61, "fpr": 0.08},
#   "__summary__": {
#    "tpr_gap": 0.27,
#    "fpr_gap": 0.04,
#    "violation": "severe"  # tpr_gap > 0.15
#   }
#  }
# }
```

**Violation thresholds to implement:**
- `> 0.15` → "severe"
- `0.05–0.15` → "moderate"
- `< 0.05` → "acceptable"

**Reference:** Hardt, Price, Srebo (2016), *Equality of Opportunity in Supervised Learning*. NeurIPS.

**Difficulty:** Intermediate · Est. time: 4–5 hours

---

## Issue #36 (Advanced) — The Feature That Gets Conference Talks

### Implement Grad-CAM for Vision Transformers (ViT) — the hardest explainability challenge

**Problem:**
Regular Grad-CAM requires convolutional feature maps. ViTs have attention heads instead.

Explaining ViT predictions is an open research problem. The best current approach uses gradients w.r.t. attention weights in the last transformer block, projected onto the 2D patch grid.

**Your task:** Implement `ViTGradCAM` in `trustlens/explainability/vit_gradcam.py`.

**Interface:**
```python
from trustlens.explainability import ViTGradCAM

cam = ViTGradCAM(vit_model, attention_layer_index=-1)
heatmap = cam.generate(image_tensor, class_idx=None)  # shape (H, W)
fig = cam.overlay(image_np, heatmap)
```

**Algorithm (Chefer et al., 2021):**
1. Run forward pass → extract attention weights from last block
2. Run backward pass → compute gradient of score w.r.t. attention weights
3. Weight attention maps by gradient magnitude
4. Average across heads → reshape from `(num_patches,)` to `(P, P)` patch grid
5. Upsample to input resolution

**Critical constraints:**
- Must handle both HuggingFace `ViTForImageClassification` and `timm` ViT models
- Must gracefully degrade to attention rollout if gradients are unavailable

**Reference:** Chefer, Gur, Wolf (2021), *Transformer Interpretability Beyond Attention Visualization*. CVPR.

**Difficulty:** Advanced · Est. time: 2–3 days

---

## Issue #44 (Advanced) — The Project That Unlocks Enterprise Users

### Build `trustlens serve` — the web dashboard that closes the loop with stakeholders

**Problem:**
TrustLens generates data. Stakeholders want a UI.

A product manager can't read `report.json`. A regulator wants a printable PDF.
A compliance team needs to compare Model v1 vs Model v2.

`trustlens serve report/` should launch a local web dashboard that any stakeholder can use.

**Your task:** Build the TrustLens web dashboard.

**Command:**
```bash
trustlens serve examples/output/ # auto-detects report.json + trust_score.json
# Serving at http://localhost:7788
```

**Tech stack (constraints):**
- **Backend:** FastAPI (already a Python project — keep it consistent)
- **Frontend:** HTMX + minimal vanilla JS — no React, no build step
- **Charts:** Plotly.js via CDN — interactive versions of matplotlib plots
- **No database needed** for single-session use

**Required panels:**
1. Trust Score gauge (interactive — click for dimension breakdown)
2. Reliability diagram (Plotly, interactive hover)
3. Confidence gap histogram
4. Critical failures table (sortable, filterable)
5. Model comparison (if multiple report dirs provided)
6. Export as PDF button (browser print)

**CLI integration:**
```python
# trustlens/cli.py
@app.command()
def serve(report_dir: str, port: int = 7788): ...
```

**Entry point in `setup.cfg`:**
```ini
[options.entry_points]
console_scripts =
  trustlens = trustlens.cli:app
```

**Why it matters:** This is the feature that takes TrustLens from "dev tool" to "team tool." It gets TrustLens into weekly model review meetings. It's the path to 1000+ stars.

**Difficulty:** Advanced · Est. time: 1–2 weeks
