<div align="center">

<img src="assets/banner.png" alt="TrustLens Banner" width="100%" />

<br/>

# TrustLens

### Your model has 92% accuracy. **That's not enough.**

**The open-source Python library that answers the questions accuracy never does.**
Calibration · Failure Analysis · Bias Detection · Explainability — in one function call.

<br/>

[![PyPI version](https://badge.fury.io/py/trustlens.svg)](https://pypi.org/project/trustlens)
[![CI](https://github.com/Khanz9664/TrustLens/actions/workflows/ci.yml/badge.svg)](https://github.com/Khanz9664/TrustLens/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Khanz9664/TrustLens?style=social)](https://github.com/Khanz9664/TrustLens/stargazers)
[![PyPI Downloads](https://img.shields.io/pypi/dm/trustlens)](https://pypi.org/project/trustlens)

<br/>

[**Get Started**](#quickstart) · [**Live Demo**](examples/trustlens_demo.ipynb) · [**PyPI**](https://pypi.org/project/trustlens) · [**Discussions**](https://github.com/Khanz9664/TrustLens/discussions)

</div>

---

## The Problem Nobody Ships Around

You trained a model. It hits **92% accuracy** on your validation set. You ship it.

Three months later:

- A minority-class user gets consistently wrong predictions
- The model is **90% confident on its worst mistakes**
- A regulator asks *"why did it make that decision?"* — and you have no answer

Sound familiar? You're not alone.

**Accuracy tells you how often your model is right.
It tells you nothing about *when* it fails, *why* it fails, or *who* it fails.**

TrustLens fixes that.

---

## Install

```bash
pip install trustlens
```

---

## Quickstart

### Zero-friction (no data needed)

```python
from trustlens import quick_analyze

# Loads breast cancer dataset, trains a baseline model,
# runs the full analysis — all in one call.
report = quick_analyze(dataset="breast_cancer")
```

### With your own model

```python
from trustlens import analyze

report = analyze(
    model,          # any sklearn-compatible model
    X_val,          # validation features
    y_val,          # ground truth labels
    y_prob=proba,   # predicted probabilities
)

print(report.trust_score)
report.show()
```

### Save Reports

```python
# Export to JSON (perfect for pipelines and logging)
report.save("report.json")

# Export to human-readable TXT
report.save("report.txt")

# Export full bundle (JSON + Plots) to a directory
report.save("my_trust_report")
```

JSON is optimized for machine-to-machine logging (NumPy-free, clean keys), while TXT provides a shareable, non-ANSI version of the console summary.

**Output:**

```
TrustLens Analysis Report

 Timestamp : 2026-04-16T15:43:02Z
 Model     : RandomForestClassifier
 Samples   : 2,500
 Classes   : 2


 TRUST SCORE: 61/100  [B]
 Assessment: Good Trust — minor issues to address


 Key Observations:
  * Calibration needs improvement (ECE > 0.1).
  * Model is overconfident on incorrect predictions (low confidence gap).


 Dimension breakdown:
  calibration      52.3 / 100
  failure          74.1 / 100
  bias             41.2 / 100    ← flagged
  representation   68.5 / 100
```

Your calibration is fine. Your bias score is not.
---

**TrustLens just saved you from a PR disaster.**

---

Early traction: receiving external contributions within hours of release

---

## The Trust Score

A single, actionable number: **0 to 100.**

Not a black box. Computed from four independently interpretable dimensions:

| Dimension | What it measures | Weight |
|---|---|---|
| **Calibration** | Do probabilities reflect reality? | 35% |
| **Failure** | Does confidence correlate with accuracy? | 30% |
| **Bias** | Are all groups treated equally? | 25% |
| **Representation** | Is the embedding space well-structured? | 10% |

| Score | Grade | Recommendation |
|---|---|---|
| 80–100 | A | Production-ready ✅ |
| 60–79 | B | Ship with a fix plan |
| 40–59 | C | Investigate before deployment |
| 0–39 | D | Do not deploy 🚫 |

---

## What You Can Do With It

### The Summary Dashboard

One line. One picture. Everything your team needs.

```python
report.summary_plot()
```

A presentation-ready **6-panel dashboard**:

- **Trust Score gauge** — overall trustworthiness at a glance
- **Reliability diagram** — is your model over or underconfident?
- **Confidence gap** — does high confidence actually mean high accuracy?
- **Error rate by class** — which classes are being failed?
- **Class distribution** — is your training data biased?
- **Sub-score breakdown** — which dimension needs the most work?

---

### Find Your Most Dangerous Mistakes

```python
report.show_failures(top_k=5)
```

```

 TOP 5 CRITICAL FAILURES
 GradientBoostingClassifier | 58 total errors / 700 samples (8.3%)

 #   Sample  True  Pred  Confidence  Danger
 -------------------------------------------------------
 1   412     1     0     97.4%       CRITICAL
 2   88      0     1     95.1%       CRITICAL
 3   301     1     0     91.8%       HIGH
 4   556     0     1     89.2%       HIGH

 Insights:
   Mean confidence on top failures: 93.4%
   These are high-confidence mistakes — the model is
   certain it is right, but it is wrong.
   Overconfidence detected — consider calibration.
```

High-confidence wrong predictions are the **most dangerous kind.** Now you know exactly where they are.

---

## Real-World Use Cases

**Medical AI**
A diagnostic model with 94% accuracy has an ECE of 0.18 — dangerously overconfident on edge cases. TrustLens surfaces it before it reaches a patient.

**Fraud Detection**
Your confidence gap is 0.04 — the model is nearly as confident on fraud it misses as fraud it catches. That's your false-negative problem, quantified.

**Hiring, Lending & Insurance**
Subgroup analysis reveals a 23% accuracy gap between applicant demographics. You have a fairness problem. You know before a regulator tells you.

**ML Research**
Use CKA to compare representation quality across architectures. Use faithfulness testing to benchmark explanation methods honestly.

---

## Why Not Just Use Standard Metrics?

Traditional metrics like accuracy, precision, and recall answer:

> "How often is the model correct?"

TrustLens answers a different question:

> "When is the model wrong, how confident is it, and who does it affect?"

It combines multiple reliability dimensions into one unified framework — something that individual metrics cannot provide.

---

## Architecture

```
trustlens/
├── metrics/
│   ├── calibration.py      # Brier Score, ECE
│   ├── failure.py          # Confidence gap, misclassification analysis
│   ├── bias.py             # Class imbalance, subgroup performance
│   └── representation.py  # Embedding separability, CKA
├── explainability/
│   ├── faithfulness.py
│   └── gradcam.py
├── visualization/
│   ├── summary_plot.py
│   ├── calibration_plots.py
│   ├── failure_plots.py
│   ├── bias_plots.py
│   └── representation_plots.py
├── plugins/
│   ├── base.py             # Extend TrustLens with custom metrics
│   └── registry.py
├── api.py                  # Public API: analyze(), quick_analyze()
├── report.py               # TrustReport object
└── trust_score.py          # Score computation
```

---

## Extend It: Plugin System

TrustLens is designed to grow. Adding a custom metric takes four steps:

```python
# 1. Write a pure function
def my_metric(y_true, y_pred) -> float:
    ...

# 2. Add it to the appropriate module (metrics/calibration.py, etc.)
# 3. Export it from metrics/__init__.py
# 4. Write a test in tests/test_<module>.py
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full guide including plugins and explainability methods.

---

## Quality & Reliability

TrustLens is built to production-grade standards — not a research prototype:

- ✅ **100+ unit tests**, 100% pass rate
- ✅ **Strict linting & type checking** — Ruff + MyPy
- ✅ **Continuous Integration** via GitHub Actions
- ✅ **Verified builds** on PyPI
- ✅ **67% test coverage**, actively growing toward 85%+

---

## Contributing

Open to contributors of all levels. Check the Issues tab for open tasks ready to be built.

> **Coverage policy:** All new contributions must maintain or improve the 67% baseline. Advanced modules (explainability, visualization) are the priority areas for new tests.

[**Read the contributing guide →**](CONTRIBUTING.md)

---

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for what's coming next.
Have a feature request? [Open a discussion →](https://github.com/Khanz9664/TrustLens/discussions)

---

## Citation

If you use TrustLens in research or production, please cite it:

```bibtex
@software{trustlens2026,
  author = {Shahid Ul Islam},
  title  = {TrustLens: Debug your ML models beyond accuracy},
  year   = {2026},
  url    = {https://github.com/Khanz9664/TrustLens},
}
```

---

## Author

**Shahid Ul Islam** — ML Engineer & Creator of TrustLens

[GitHub](https://github.com/Khanz9664) · [Portfolio](https://khanz9664.github.io/portfolio/) · [LinkedIn](https://www.linkedin.com/in/shahid-ul-islam-13650998/)

---

<div align="center">

**If TrustLens saved you from a bad deployment, give it a ⭐**
It helps other engineers find it before they make the same mistake.

[PyPI](https://pypi.org/project/trustlens) · [GitHub](https://github.com/Khanz9664/TrustLens) · [Discussions](https://github.com/Khanz9664/TrustLens/discussions)

</div>
