# TrustLens Roadmap

> Last updated: May 2026
> This roadmap reflects our current priorities. Community feedback shapes every phase.

---

## Current State (v0.3.0)

- ✅ Stable ML evaluation pipeline (calibration, failure, bias, representation)
- ✅ Head-to-head model comparison API (`trustlens.compare`)
- ✅ Decision-ready Trust Score with penalty reasoning
- ✅ Multi-feature fairness and 2D embedding diagnostics
- ✅ Professional contributor infrastructure and documentation

---

## Status Legend

- [x] Completed (production-ready)
- [~] In progress / Experimental (limited availability)
- [ ] Planned

---

## Active Work (v0.3.x)

These are high-priority items currently being developed or targeted for the next release.

- [ ] **Policy Profiles** (Issue #56) — *Context-aware scoring (Strict/Lenient)*
- [ ] **TrustComparison** (Issue #57) — *Differential reliability audits*
- [ ] **XGBoost Support** — *Native prediction resolver for XGBClassifier*
- [~] **Deep Learning Backends** — *Experimental Keras & TensorFlow integration*
- [~] **HTML Report Export** (Issue #19) — *[OPEN]*
- [ ] **Maximum Calibration Error (MCE)** (Issue #1)

---

## Phase 1: MVP — *The Foundation*

**Status: COMPLETE**

The minimal set of features required to be genuinely useful to practitioners.

### Deliverables
- [x] Core `analyze()` API with module dispatch
- [x] `TrustReport` result container with `show()`, `plot()`, `save()`
- [x] **Calibration**: Brier Score, ECE, reliability curve + reliability diagram
- [x] **Failure Analysis**: misclassification summary, confidence gap histogram
- [x] **Bias Detection**: class imbalance report, subgroup accuracy/F1
- [x] **Representation Analysis**: silhouette separability, CKA metric
- [~] **Explainability**: Grad-CAM with PyTorch support — *[EXPERIMENTAL]*
- [~] **Faithfulness**: pixel deletion + insertion tests (AUPC) — *[EXPERIMENTAL]*
- [x] Plugin system (BasePlugin + PluginRegistry)
- [x] Full test suite (>230 tests)
- [x] Professional README, CONTRIBUTING, quickstart examples
- [x] PyPI package & GitHub Actions CI (lint/test/format)

---

## Phase 2: Core Expansion — *Going Deeper*

**Target: v0.3.x (ongoing)**

> **Focus:** High-impact ML features that integrate directly into the `analyze()` pipeline.

### High Priority
- [x] **Equalized Odds** (Issue #25)
- [x] **UMAP/t-SNE Visualization** (Issue #22)
- [x] **Jupyter Rich Display** (`_repr_html_`) (Issue #24)
- [x] **Progress Bars** via `tqdm` (Issue #28)
- [x] **Subgroup ECE** (calibration per demographic group)
- [~] **HTML Report Export** (Issue #19) — [OPEN]
- [ ] **Temperature Scaling** (Issue #18)
- [ ] **Maximum Calibration Error (MCE)** (Issue #1)

### Nice to Have
- [ ] **Multi-class ECE** (label-wise decomposition)
- [ ] **Critical Failures** table for `TrustReport`
- [ ] **Per-class PR curves** and optimal threshold analysis
- [ ] **Prediction Flip Analysis** (robustness check)
- [ ] **Eigen-CAM** (gradient-free explainability)
- [ ] **Integrated Gradients (IG)** for tabular models
- [ ] **SHAP Wrapper** (optional dependency)
- [ ] **Text-based Sensitive Feature Parsing**
- [ ] **Linear Probing** accuracy per layer
- [ ] **Intrinsic Dimensionality** estimation for embeddings

### Framework Support
- [ ] **XGBoost Integration** — Native `analyze()` support for `XGBClassifier`
- [~] **Keras Experimental** — Sequential and functional model support
- [~] **TensorFlow Experimental** — SavedModel loading and lazy import hygiene

---

## Phase 3: Comparative & Research — *Frontier Methods*

**Target: v0.3.x / v0.4.0**

Methods primarily of interest to ML researchers pushing state-of-the-art and production diffing.

### Comparative Audits
- [x] **Model recommendation engine** (`trustlens.compare`)
- [x] **Score comparison tables**
- [ ] **Policy-aware deltas** (Issue #57)

### Advanced Representation
- [ ] Layer-wise CKA heatmap (n_layers × n_layers)
- [ ] Neuron activation statistics per class
- [ ] Representation fragility score (adversarial perturbation)

### Benchmarking
- [ ] `benchmark()` function — run TrustLens on standard datasets (CIFAR-10, etc.)
- [ ] Baseline scores for common architectures

---

## Phase 4: Community Growth — *Scaling Impact*

**Target: v0.4.0**

Making TrustLens a community standard.

### Contribution & Documentation
- [x] Contributor hall of fame in README
- [x] Video walkthrough series
- [x] Interactive Jupyter notebooks (Colab-ready)
- [ ] Plugin submission process (community plugin registry)
- [ ] `trustlens-contrib` companion repository

### Integrations
- [ ] **MLflow**: log TrustLens metrics as experiment artifacts
- [ ] **Weights & Biases**: log reliability diagrams as charts
- [ ] **Hugging Face**: `evaluate`-compatible metric modules
- [ ] **DVC**: TrustLens report as a DVC stage output

---

## Phase 5: Platform — *The Full Stack*

**Target: v1.0.0**

TrustLens as a complete model analysis platform.

### Web Dashboard
- [ ] Zero-config web UI (`trustlens serve`) using FastAPI + React
- [ ] Interactive reliability diagrams (Plotly)
- [ ] Model comparison views

### Enterprise Features
- [ ] Scheduled monitoring reports (model drift detection)
- [ ] Slack/Teams alerting on metric regression
- [ ] Role-based access for team reports

---

## Feedback

Have thoughts on the roadmap?
[Open a discussion on GitHub](https://github.com/Khanz9664/TrustLens/discussions) — we read everything.
