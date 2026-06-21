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

[**Why TrustLens**](#-why-traditional-evaluation-fails) · [**Visual Evidence**](#-visual-evidence-trustlens-in-action) · [**How It Works**](#-how-trustlens-works) · [**Architecture & Evolution**](#-community--project-evolution) · [**Quickstart**](#-quickstart) · [**Docs**](https://khanz9664.github.io/trustlensdocs/)

</div>

---

> **Your model has 92% accuracy. It's still not safe for deployment.**
>
> Standard evaluation stops at accuracy. Accuracy measures what went *right*. TrustLens measures what can go *wrong* — in production, on underrepresented subgroups, and at high confidence.

---

## Why Traditional Evaluation Fails

You train a model. The test set reports 92% Accuracy and a 0.95 ROC-AUC. By all traditional metrics, it is ready to ship. 

But behind those numbers, silent failures are lurking:
* **Overconfidence**: The model is "90% sure" about its predictions, but it's only right 60% of the time.
* **Subgroup Collapse**: The aggregate accuracy is 92%, but for a specific demographic, performance drops to 40%.
* **Latent Bleed**: In the embedding space, the model cannot distinguish between critical classes, leading to unpredictable edge-case behavior.
* **Confidently Wrong**: The model's most severe mistakes are made with >99% confidence, bypassing human-in-the-loop safety nets.

### TrustLens vs. Traditional Metrics

| Traditional Metrics | TrustLens Diagnostics | What It Tells You |
| :--- | :--- | :--- |
| **Accuracy, F1, Precision** | **Calibration (ECE, Brier)** | *Does the model know when it's guessing?* |
| **Aggregate ROC-AUC** | **Fairness & Bias** | *Are minority groups experiencing higher failure rates?* |
| **Loss Curve** | **Latent Space Health** | *Are the internal embeddings stable and separated?* |
| **Manual Error Analysis** | **Failure Diagnostics** | *Are the errors concentrated at high confidence?* |
| **"Looks good to me"** | **Deployment Verdict** | *Is this model mathematically safe to deploy?* |

TrustLens surfaces all these hidden risks with a single, statistically grounded audit, outputting a machine-readable deployment verdict.

---

## Visual Evidence: TrustLens in Action

TrustLens diagnostics are powered by visual evidence. We don't just give you a score; we show you exactly *why* a model is failing.

<div align="center">
<table width="100%">
  <tr>
    <td width="33%" align="center">
      <b>The Deployment Verdict</b><br/>
      <img src="assets/summary_dashboard.png" width="100%" /><br/>
      <div align="left">
      <sub><b>What it is:</b> The composite Trust Score.</sub><br/>
      <sub><b>Why it matters:</b> Gives a CI/CD-ready grade.</sub><br/>
      <sub><b>Risk:</b> Blocks shipping unsafe models.</sub>
      </div>
    </td>
    <td width="33%" align="center">
      <b>Calibration</b><br/>
      <img src="assets/calibration_plot.png" width="100%" /><br/>
      <div align="left">
      <sub><b>What it is:</b> Reliability diagram.</sub><br/>
      <sub><b>Why it matters:</b> Shows if the model is overconfident.</sub><br/>
      <sub><b>Risk:</b> High-confidence wrong answers.</sub>
      </div>
    </td>
    <td width="33%" align="center">
      <b>Subgroup Fairness Gaps</b><br/>
      <img src="assets/fairness_gap_region.png" width="100%" /><br/>
      <div align="left">
      <sub><b>What it is:</b> Error rates across demographics.</sub><br/>
      <sub><b>Why it matters:</b> Uncovers hidden biases.</sub><br/>
      <sub><b>Risk:</b> Regulatory failure and harm.</sub>
      </div>
    </td>
  </tr>
  <tr>
    <td width="33%" align="center">
      <b>Latent Space Health</b><br/>
      <img src="assets/representation_embedding_2d.png" width="100%" /><br/>
      <div align="left">
      <sub><b>What it is:</b> UMAP/t-SNE projection.</sub><br/>
      <sub><b>Why it matters:</b> Visualizes class separability.</sub><br/>
      <sub><b>Risk:</b> Feature collapse and instability.</sub>
      </div>
    </td>
    <td width="33%" align="center">
      <b>Equalized Odds Violations</b><br/>
      <img src="assets/equalized_odds_region.png" width="100%" /><br/>
      <div align="left">
      <sub><b>What it is:</b> True vs False Positive Rates.</sub><br/>
      <sub><b>Why it matters:</b> Ensures equitable outcomes.</sub><br/>
      <sub><b>Risk:</b> Systemic discrimination.</sub>
      </div>
    </td>
    <td width="33%" align="center">
      <b>Failure Analysis</b><br/>
      <img src="assets/failure_plot.png" width="100%" /><br/>
      <div align="left">
      <sub><b>What it is:</b> Confidence distribution of errors.</sub><br/>
      <sub><b>Why it matters:</b> Spots systemic failure modes.</sub><br/>
      <sub><b>Risk:</b> Unpredictable production behavior.</sub>
      </div>
    </td>
  </tr>
</table>
</div>

---

## How TrustLens Works

TrustLens evaluates your model through four distinct diagnostic modules, combining the findings into a **Trust Score (0–100)**.

1.  **Calibration Engine**: Computes Expected Calibration Error (ECE) and Brier Score to detect confidence mismatch.
2.  **Fairness Engine**: Evaluates Equalized Odds and Subgroup Performance gaps across sensitive features.
3.  **Representation Engine**: Analyzes latent embedding separability (Silhouette, CKA) to ensure stable decision boundaries.
4.  **Decision Engine**: Synthesizes the risks into a penalty-based Trust Score and a `Ready` / `Blocked` deployment verdict.

### The Prediction Resolver Architecture

You don't need to write boilerplate to extract probabilities. TrustLens features a **Prediction Resolver Architecture** that automatically detects your framework and standardizes the output.

We natively support:
*   **scikit-learn** (`ClassifierMixin` estimators)
*   **XGBoost** (`XGBClassifier`, `Booster`)
*   **LightGBM** (`LGBMClassifier`, `Booster`)
*   **CatBoost** (`CatBoostClassifier`)

---

## Scientific Validation

TrustLens is more than a visualization package—it is a statistically grounded diagnostic framework. We have systematically validated its behavior across 6 model architectures and multiple data corruption scenarios (noise, imbalance, bias).

**Key Finding**: TrustLens empirically decouples **Accuracy** from **Trust**, accurately flagging high-accuracy models that exhibit high reliability risks (the "Overconfidence Zone").

---

## Community & Project Evolution

TrustLens is an actively evolving framework driven by robust engineering discussions and RFCs (Request for Comments). We treat evaluation as a first-class architectural problem.

**Active Architectural Debates & Milestones:**
*   **[RFC #145: Regression Trust Score](https://github.com/Khanz9664/TrustLens/issues/145)** — Proposing the scoring framework for regression models.
*   **[PR #147: Implements RFC #145 (Regression Trust Score)](https://github.com/Khanz9664/TrustLens/pull/147)** — Core engine execution for regression contexts.
*   **[PR #102: Centralize plotting style](https://github.com/Khanz9664/TrustLens/pull/102)** — Unifying visual identity across the framework.
*   **[PR #68: Fairness multi-feature support](https://github.com/Khanz9664/TrustLens/pull/68)** — Scaling bias detection across complex datasets.

**The Evolution:**
*   **v0.1**: MVP — Core metrics and visualizations.
*   **v0.4 (Current)**: Framework-Agnostic Core — Native support for XGBoost, LightGBM, CatBoost.
*   **v0.5**: *In Progress* — Policy Profiles, Regression Support, TrustComparison.
*   **v1.0**: *Planned* — CI/CD enterprise integration and Web Dashboards.

---

## Quickstart

Install TrustLens (use `[full]` for extended plotting and framework support):

```bash
pip install trustlens
pip install trustlens[full]
```

Run a one-line audit on a built-in dataset to see why high accuracy isn't the full story:

```python
from trustlens import quick_analyze

quick_analyze(dataset="breast_cancer")
```

Or run a comprehensive audit on your own model:

```python
from trustlens import analyze
from xgboost import XGBClassifier

model = XGBClassifier().fit(X_train, y_train)

# TrustLens auto-detects the XGBoost model and extracts probabilities
report = analyze(
    model=model,
    X=X_test,
    y_true=y_test,
    sensitive_features={"gender": gender_test}
)

# Render the rich HTML dashboard or visual plots
report.show()

# Gate your CI/CD pipeline
report.save("trust_report/")
```

---

## Deep Dive Documentation

The README is just the tip of the iceberg. Explore the full TrustLens documentation site for methodology, API references, and architectural deep-dives:

*   🌐 **[Documentation Home](https://khanz9664.github.io/trustlensdocs/)**
*   🏛️ **[Architecture Guide](https://khanz9664.github.io/trustlensdocs/architecture.html)**
*   📖 **[API Reference](https://khanz9664.github.io/trustlensdocs/api_reference.html)**

---

## 🤝 Contributing

TrustLens is an open ecosystem. We welcome contributions—whether it's new diagnostic plugins, better visualizers, or core engine improvements. 

→ [**Contributing Guide**](CONTRIBUTING.md) · [**Open an Issue**](https://github.com/Khanz9664/TrustLens/issues)

**A massive thank you to our contributors:**

<a href="https://github.com/Khanz9664"><img src="https://github.com/Khanz9664.png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/komoike-oss28-ui"><img src="https://github.com/komoike-oss28-ui.png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/Whatsonyourmind"><img src="https://github.com/Whatsonyourmind.png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/WeiGuang-2099"><img src="https://github.com/WeiGuang-2099.png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/claude"><img src="https://github.com/claude.png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/jayssSmm"><img src="https://github.com/jayssSmm.png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/JavadTe"><img src="https://github.com/JavadTe.png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/gaoflow"><img src="https://github.com/gaoflow.png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/vaishnavidesai09"><img src="https://github.com/vaishnavidesai09.png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/nanookclaw"><img src="https://github.com/nanookclaw.png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/llamapreview[bot]"><img src="https://github.com/llamapreview[bot].png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/q404365631"><img src="https://github.com/q404365631.png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/dicnunz"><img src="https://github.com/dicnunz.png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/MustansirNisar"><img src="https://github.com/MustansirNisar.png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/sidharth-vijayan"><img src="https://github.com/sidharth-vijayan.png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/fuchan12040502-innu"><img src="https://github.com/fuchan12040502-innu.png" width="36" style="border-radius:50%" /></a>
<a href="https://github.com/CrepuscularIRIS"><img src="https://github.com/CrepuscularIRIS.png" width="36" style="border-radius:50%" /></a>

<br/>

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
Built by <a href="https://github.com/Khanz9664"><b>Shahid Ul Islam</b></a>
</div>
