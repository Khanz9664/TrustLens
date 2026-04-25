# TrustLens Roadmap

> Last updated: April 2026
> This roadmap reflects our current priorities. Community feedback shapes every phase.

---

## Current State (v0.2.0)

- Stable ML evaluation pipeline (calibration, failure, bias, representation)
- Experimental modules isolated (explainability, faithfulness)
- Strong contributor infrastructure in place

## Status Legend

- [x] Completed (production-ready, part of analyze pipeline)
- [~] In progress / Experimental (not part of core `analyze()` pipeline)
- [ ] Planned

---

## Active Work (Now)

These are high-priority items currently being developed or targeted for the next release.

- [~] **Equalized Odds** (Issue #25) — *[OPEN]*
- [~] **UMAP/t-SNE Visualization** (Issue #22) — *[OPEN]*
- [~] **HTML Report Export** (Issue #19) — *[OPEN]*

---

## Phase 1: MVP — *The Foundation*

**Target: v0.1.2**

The minimal set of features required to be genuinely useful to practitioners.

Note: As of v0.2.0, TrustLens focuses on classical ML evaluation.
Deep learning explainability features are under experimental development and not part of the core pipeline.

### Deliverables
- [x] Core `analyze()` API with module dispatch
- [x] `TrustReport` result container with `show()`, `plot()`, `save()`
- [x] **Calibration**: Brier Score, ECE, reliability curve + reliability diagram
- [x] **Failure Analysis**: misclassification summary, confidence gap histogram
- [x] **Bias Detection**: class imbalance report, subgroup accuracy/F1
- [x] **Representation Analysis**: silhouette separability, CKA metric
- [~] **Explainability**: Grad-CAM with PyTorch support — *[ASSIGNED: Maintainer]*
- [~] **Faithfulness**: pixel deletion + insertion tests (AUPC) — *[OPEN]*
- [x] Plugin system (BasePlugin + PluginRegistry)
- [x] Full test suite (>80% coverage)
- [x] Professional README (with logo), CONTRIBUTING, quickstart examples
- [x] PyPI package & GitHub Actions CI (lint/test/format)

---

## Phase 2: Core Expansion — *Going Deeper*

**Target: v0.2.x (ongoing)**

> **Focus:** High-impact ML features that integrate directly into the `analyze()` pipeline.

### High Priority
- [~] **Equalized Odds** (Issue #25) — [OPEN]
- [~] **UMAP/t-SNE Visualization** (Issue #22) — [OPEN]
- [~] **HTML Report Export** (Issue #19) — [OPEN]
- [ ] **Maximum Calibration Error (MCE)** (Issue #1)
- [ ] **Temperature Scaling** (Issue #18)
- [ ] **Jupyter Rich Display** (`_repr_html_`) (Issue #24)

### Nice to Have
- [ ] **Multi-class ECE** (label-wise decomposition)
- [ ] **Subgroup ECE** (calibration per demographic group)
- [ ] **Critical Failures** table for `TrustReport`
- [ ] **Per-class PR curves** and optimal threshold analysis
- [ ] **Prediction Flip Analysis** (robustness check)
- [ ] **Eigen-CAM** (gradient-free explainability)
- [ ] **Integrated Gradients (IG)** for tabular models
- [ ] **SHAP Wrapper** (optional dependency)
- [ ] **Intrinsic Dimensionality** estimation for embeddings
- [ ] **Linear Probing** accuracy per layer
- [ ] **Progress Bars** via `tqdm` (Issue #28)
- [ ] **Text-based Sensitive Feature Parsing**

---

## Phase 3: Research Features — *Frontier Methods*

**Target: v0.3.0**

Methods primarily of interest to ML researchers pushing state-of-the-art.

### Vision Transformers (ViTs)
- [ ] Generic attention rollout for ViT models
- [ ] DINO-compatible self-attention map visualization
- [ ] ViT Grad-CAM via register_hook on attention weights

### NLP & Sequence Models
- [ ] Attention-based saliency for BERT-style models
- [ ] Token-level faithfulness via masking tests
- [ ] Semantic similarity–based explanation consistency

### Advanced Representation
- [ ] Representation fragility score (adversarial perturbation in embedding space)
- [ ] Neuron activation statistics per class
- [ ] Layer-wise CKA heatmap (n_layers × n_layers)

### Benchmarking
- [ ] `benchmark()` function — run TrustLens on standard datasets (CIFAR-10, etc.)
- [ ] Baseline scores for common architectures
- [ ] Score comparison tables

---

## Phase 4: Community Growth — *Scaling Impact*

**Target: v0.4.0**

Making TrustLens a community standard.

### Contribution Infrastructure
- [ ] Contributor hall of fame in README
- [ ] Plugin submission process (community plugin registry)
- [ ] `trustlens-contrib` companion repository

### Integrations
- [ ] **Hugging Face**: `evaluate`-compatible metric modules
- [ ] **MLflow**: log TrustLens metrics as experiment artifacts
- [ ] **Weights & Biases**: log reliability diagrams as charts
- [ ] **DVC**: TrustLens report as a DVC stage output

### Documentation
- [ ] Full Read the Docs site with API reference
- [ ] Interactive Jupyter notebooks (Binder/Colab)
- [ ] Video walkthrough series

---

## Phase 5: Platform — *The Full Stack*

**Target: v1.0.0**

TrustLens as a complete model analysis platform.

### Web Dashboard
- [ ] Zero-config web UI (`trustlens serve`) using FastAPI + React
- [ ] Interactive reliability diagrams (Plotly)
- [ ] Model comparison views

### Leaderboard
- [ ] Public leaderboard for model calibration benchmarks
- [ ] Dataset-specific calibration baselines
- [ ] Community-submitted benchmark results

### Enterprise Features
- [ ] Scheduled monitoring reports (model drift detection)
- [ ] Slack/Teams alerting on metric regression
- [ ] Role-based access for team reports

---

## Feedback

Have thoughts on the roadmap?
[Open a discussion on GitHub](https://github.com/Khanz9664/TrustLens/discussions) — we read everything.
