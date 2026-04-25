<div align="center">
  <img src="assets/banner1.png" alt="TrustLens Banner" width="100%" />

<br/>

### Your model has 92% accuracy. **That's not enough.**

**The open-source Python library that transforms model metrics into actionable deployment decisions.**
TrustLens bridges the gap between model evaluation and deployment decision-making by evaluating reliability, robustness, and fairness in one function call.



<br/>

[![PyPI version](https://badge.fury.io/py/trustlens.svg)](https://pypi.org/project/trustlens/)
[![CI](https://github.com/Khanz9664/TrustLens/actions/workflows/ci.yml/badge.svg)](https://github.com/Khanz9664/TrustLens/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Khanz9664/TrustLens?style=social)](https://github.com/Khanz9664/TrustLens/stargazers)
[![PyPI Downloads](https://img.shields.io/pypi/dm/trustlens)](https://pypi.org/project/trustlens)
[![Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Contributor%20Covenant-blue.svg)](CODE_OF_CONDUCT.md)

<br/>

⭐ **Star the repo to support the project!**

🛠 **Actively looking for contributors - beginner-friendly issues available**

<br/>

[**Get Started**](#quickstart) · [**Docs**](docs/index.md) · [**Live Demo**](examples/trustlens_demo.ipynb) · [**PyPI**](https://pypi.org/project/trustlens) · [**Discussions**](https://github.com/Khanz9664/TrustLens/discussions)

</div>

---

## What TrustLens Does
TrustLens is a **decision-support system** designed to answer the hardest question in ML: *"Should I deploy this model?"*

Unlike standard metric libraries, TrustLens provides a cohesive diagnostic layer that evaluates:
- **Reliability (Calibration)**: Are the model's probabilities trust-worthy?
- **Robustness (Failure)**: Where and why does the model fail?
- **Fairness (Bias)**: Are errors disproportionately affecting certain groups?

The results are synthesized into a **Trust Score** and an actionable **deployment verdict**, enabling practitioners to move from simple metric evaluation to informed deployment decisions.



---

## One-Line Magic
**Full reliability analysis in one line.**

```python
from trustlens import quick_analyze
quick_analyze(dataset="breast_cancer").show()
```

> **Looking for the full documentation?**
> Head over to the [**TrustLens Documentation Hub**](docs/index.md) for deep dives into features, use cases, and architecture.

---

## Quickstart

### 1. Install
```bash
pip install trustlens
```

### 2. Analyze Your Model
```python
from trustlens import analyze

report = analyze(model, X_test, y_test, y_prob)
report.show()
```

**Real Output Snippet:**
```text
TRUST SCORE: 88/100 [B]
Assessment : High Trust - ready for controlled deployment

Score Explanation:
  - Dominant Issue  : Calibration (-12.0)
  - Secondary Issue : Fairness (-0.0)
```

### 3. Compare Models
```python
from trustlens import compare

# Compare multiple candidates and get a recommendation
compare([report_rf, report_logistic])
```

### 4. Save & Export

```python
report.save("report.json") # For CI/CD
report.save("report.txt")  # For humans
```

---

## Understanding the Output
TrustLens provides a multi-layered diagnostic report:
- **Trust Score**: A weighted combination (configurable, default ~40/40/20) of Calibration, Failure, and Bias. Includes **automatic penalties** for critical risks.
- **Failure Score**: Reflects **confidence-weighted errors**, not raw error rate. It identifies if errors are concentrated in high-certainty zones.
- **Calibration**: Measures probability reliability via Expected Calibration Error (ECE).
- **Fairness Margin**: Quantifies distance from the acceptable disparity threshold (0.10).
- **Pattern Detection**: Alerts you to behaviors like **Calibration Drift** (unreliable probabilities) or **Confidently Wrong** (high-certainty mistakes).



---

## Documentation

Detailed guides and architecture references are available in our [structured documentation](docs/index.md):

### Core Concepts
* [**The Problem**](docs/problem.md) — Why accuracy is a dangerous metric for production.
* [**Who This Is For**](docs/audience.md) — Targeted guidance for MLEs, Data Scientists, and Researchers.

### Deep Dive
* [**Features & Modules**](docs/features.md) — Deep dive into Calibration, Failure, Bias, and Representation.
* [**Real-World Use Cases**](docs/use_cases.md) — Medical AI, Fraud Detection, and Hiring.
* [**Architecture**](docs/architecture.md) — Modular design and internal logic.

### Development
* [**Contributing**](CONTRIBUTING.md) — How to set up your dev environment.
* [**Code of Conduct**](CODE_OF_CONDUCT.md) — Our community standards.
* [**Roadmap**](ROADMAP.md) — Future features and project vision.

<!-- Maintainers: Consider updating GitHub repository description to: "Debug your ML models beyond accuracy | Structured docs in /docs" -->

---

## Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Khanz9664">
        <img src="https://github.com/Khanz9664.png" width="100px;" style="border-radius:50%;"/>
        <br />
        <sub><b>Khanz9664</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/jayssSmm">
        <img src="https://github.com/jayssSmm.png" width="100px;" style="border-radius:50%;"/>
        <br />
        <sub><b>jayssSmm</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/WeiGuang-2099">
        <img src="https://github.com/WeiGuang-2099.png" width="100px;" style="border-radius:50%;"/>
        <br />
        <sub><b>WeiGuang-2099</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/CrepuscularIRIS">
        <img src="https://github.com/CrepuscularIRIS.png" width="100px;" style="border-radius:50%;"/>
        <br />
        <sub><b>CrepuscularIRIS</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/komoike-oss28-ui">
        <img src="https://github.com/komoike-oss28-ui.png" width="100px;" style="border-radius:50%;"/>
        <br />
        <sub><b>komoike-oss28-ui</b></sub>
      </a>
    </td>
  </tr>
</table>

Want to see your name here? Check out `good first issue` 👇

https://github.com/Khanz9664/TrustLens/issues

---

## Contributing

TrustLens is a production-grade tool, and we welcome developers of all levels. Whether it's fixing a bug, adding a metric, or improving our documentation, your help is appreciated.

[**Read the full Contributing Guide →**](CONTRIBUTING.md)

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

<p align="center">
  <strong>If TrustLens helped you understand your model better, give it a ⭐ — it helps others discover it.</strong><br><br>
  <a href="https://pypi.org/project/trustlens">PyPI</a> ·
  <a href="https://github.com/Khanz9664/TrustLens">GitHub</a> ·
  <a href="https://github.com/Khanz9664/TrustLens/discussions">Discussions</a>
</p>
