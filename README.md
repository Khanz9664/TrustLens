<div align="center">

<img src="assets/banner1.png" alt="TrustLens" width="100%" />

<br/>

# TrustLens

### Audit ML models beyond accuracy — calibration, fairness, latent health, and deployment verdicts.

<br/>

[![PyPI](https://badge.fury.io/py/trustlens.svg)](https://pypi.org/project/trustlens/)
[![Downloads](https://img.shields.io/pypi/dm/trustlens)](https://pypi.org/project/trustlens)
[![CI](https://github.com/Khanz9664/TrustLens/actions/workflows/ci.yml/badge.svg)](https://github.com/Khanz9664/TrustLens/actions)
[![Coverage](https://img.shields.io/badge/coverage-75%25-brightgreen)](https://github.com/Khanz9664/TrustLens)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-237%20passing-success)](https://github.com/Khanz9664/TrustLens/actions)

<br/>

[**Quickstart**](#quickstart) · [**How It Works**](#how-it-works) · [**Demo Video**](https://youtu.be/lZo7aWg6efI) · [**Docs**](docs/index.md) · [**Project Showcase**](https://khanz9664.github.io/portfolio/projects/trustlens.html)

</div>

---

> **Your model has 92% accuracy. It's still not safe for deployment.**
>
> Accuracy measures what went right. TrustLens measures what can go wrong — in production, on subgroups, and at high confidence.

---

## Why TrustLens

Standard evaluation stops at accuracy. Silent failures happen when:

- A model is **overconfident** — "90% sure" but right only 60% of the time
- Performance **collapses on subgroups** — gender, age, or region hidden inside a good aggregate score
- The model is **confidently wrong** — high-confidence errors that indicate systemic risk
- **Latent representations overlap** — classes bleed together where the model can't tell them apart

TrustLens surfaces all four with a single audit, and outputs a machine-readable deployment verdict.

---

## Quickstart

```bash
pip install trustlens
# Extended visualization support
pip install trustlens[full]
```

Run a one-line audit to see why 94% accuracy isn't the full story:

```python
from trustlens import quick_analyze

quick_analyze(dataset="breast_cancer")
```

```
TRUST SCORE: 68/100 [D]
Assessment : Low Trust — Blocked by high diagnostic risk

  Base Score        : 76
  Penalties Applied : -7.7 (Failure Risk)
  Final Score       : 68

→ Model shows high failure risk and is NOT ready for deployment.
```

---

## How It Works

TrustLens runs four diagnostic modules and combines them into a single **Trust Score (0–100)** with a CI/CD-ready deployment verdict.

| Module | What It Catches |
|---|---|
|  **Calibration** | Confidence vs. correctness mismatch, overconfidence, ECE |
|  **Fairness** | Subgroup performance gaps, equalized-odds violations |
|  **Representation** | Latent space health, class separation, overlap detection |
|  **Decision Engine** | Composite Trust Score + `Ready` / `Blocked` verdict |

---

## Full Audit

```python
from trustlens import analyze

report = analyze(
    model=clf,
    X=X_test,
    y_true=y_test,
    y_prob=clf.predict_proba(X_test),
    sensitive_features={"gender": gender_test, "region": region_test}
)

report.show()                  # human-readable summary
report.save("trust_report/")   # full artifact export
```

### Output artifacts

```
trust_report/
├── trust_score.json    ← deployment verdict & composite score
├── report.json         ← raw diagnostic metrics
├── metadata.json       ← environment, version, sample info
├── report.txt          ← human-readable summary
└── visuals/            ← per-module diagnostic plots (PNG)
```

### CI/CD gating

Gate model promotion on `trust_score.json` — no custom scripting needed:

```json
{
  "score": 68,
  "grade": "D",
  "verdict": "Low Trust — Blocked by high failure risk",
  "is_blocked": true
}
```

---

## Diagnostics in Practice

<div align="center">
<table width="100%">
  <tr>
    <td width="50%" align="center">
      <b>Calibration</b><br/>
      <img src="assets/calibration_plot.png" width="90%" /><br/>
      <sub>Does confidence align with correctness?</sub>
    </td>
    <td width="50%" align="center">
      <b>Fairness & Bias</b><br/>
      <img src="assets/fairness_gap_region.png" width="90%" /><br/>
      <sub>Are subgroups treated equally?</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <b>Latent Space Health</b><br/>
      <img src="assets/representation_embedding_2d.png" width="90%" /><br/>
      <sub>Is class separation clean?</sub>
    </td>
    <td width="50%" align="center">
      <b>Deployment Verdict</b><br/>
      <img src="assets/summary_dashboard.png" width="90%" /><br/>
      <sub>Is this model safe to ship?</sub>
    </td>
  </tr>
</table>
</div>

---

## Demo

[![Watch the demo](https://img.youtube.com/vi/lZo7aWg6efI/mqdefault.jpg)](https://youtu.be/lZo7aWg6efI)

15-minute walkthrough: diagnostics, trust scoring, fairness analysis, and visual dashboards.

Want a deeper look at the architecture and design decisions? → [**Interactive Project Showcase**](https://khanz9664.github.io/portfolio/projects/trustlens.html)

---

## Run the Full Demo

```bash
python demo.py
```

Generates multi-model comparisons, fairness deep-dives, latent space projections, JSON audits, and visual dashboards across all modules.

---

## Contributing

All contributions welcome — new metrics, diagnostic plugins, and visualizations.

→ [**Contributing Guide**](CONTRIBUTING.md) · [**Open an Issue**](https://github.com/Khanz9664/TrustLens/issues) · [**Docs**](docs/index.md)

<br/>

<a href="https://github.com/Khanz9664"><img src="https://github.com/Khanz9664.png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/jayssSmm"><img src="https://github.com/jayssSmm.png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/WeiGuang-2099"><img src="https://github.com/WeiGuang-2099.png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/CrepuscularIRIS"><img src="https://github.com/CrepuscularIRIS.png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/komoike-oss28-ui"><img src="https://github.com/komoike-oss28-ui.png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/sidharth-vijayan"><img src="https://github.com/sidharth-vijayan.png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/MustansirNisar"><img src="https://github.com/MustansirNisar.png" width="36" style="border-radius:50%" /></a>

---

## Citation

```bibtex
@software{trustlens2026,
  author = {Shahid Ul Islam},
  title  = {TrustLens: Audit ML models beyond accuracy},
  year   = {2026},
  url    = {https://github.com/Khanz9664/TrustLens}
}
```

<div align="center">
<br/>
Built by <a href="https://github.com/Khanz9664"><b>Shahid Ul Islam</b></a> &nbsp;·&nbsp;
<a href="https://khanz9664.github.io/portfolio/">Portfolio</a> &nbsp;·&nbsp;
<a href="https://www.linkedin.com/in/shahid-ul-islam-13650998/">LinkedIn</a>
</div>
