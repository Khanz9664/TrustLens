<div align="center">
  <img src="assets/banner1.png" alt="TrustLens Banner" width="100%" />

### Your model has 92% accuracy. That may still be unsafe.

TrustLens is an open-source Python library for evaluating model reliability beyond accuracy and producing deployment-ready decisions.

[![PyPI version](https://badge.fury.io/py/trustlens.svg)](https://pypi.org/project/trustlens/)
[![CI](https://github.com/Khanz9664/TrustLens/actions/workflows/ci.yml/badge.svg)](https://github.com/Khanz9664/TrustLens/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyPI Downloads](https://img.shields.io/pypi/dm/trustlens)](https://pypi.org/project/trustlens)
[![Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Contributor%20Covenant-blue.svg)](CODE_OF_CONDUCT.md)

[Get Started](#quickstart) · [Documentation](docs/index.md) · [Live Demo](examples/trustlens_demo.ipynb) · [PyPI](https://pypi.org/project/trustlens) · [Discussions](https://github.com/Khanz9664/TrustLens/discussions)
</div>

---

## Why TrustLens

Most model evaluations stop at accuracy, AUC, or F1. Deployment decisions require more:

- Can we trust model probabilities?
- Are failures concentrated in high-confidence regions?
- Is performance uneven across sensitive groups?
- Are we shipping a model with hidden reliability risk?

TrustLens answers these questions in one pipeline and produces:

- module-level diagnostics (calibration, failure, bias, representation)
- a composite Trust Score (0-100)
- penalty and blocker reasoning
- a deployment verdict suitable for review and CI gating

---

## Quickstart

### Install

```bash
pip install trustlens
```

### Analyze a Model

```python
from trustlens import analyze

report = analyze(model, X_test, y_test, y_prob=model.predict_proba(X_test))
report.show()
```

### Example Output

```text
TRUST SCORE: 88/100 [B]
Assessment : Good Trust - minor issues to address

Score Summary:
  Base Score        : 92
  Penalties Applied : -4.0 [Calibration (-4.0)]
  Final Score       : 88
```

### Compare Candidates

```python
from trustlens import compare

compare([report_model_a, report_model_b, report_model_c])
```

### Export Artifacts

```python
report.save("report.json")   # machine-readable
report.save("report.txt")    # human-readable
report.save("trust_report")  # full bundle with plots + metadata
```

---

## One-Line Demo

```python
from trustlens import quick_analyze
quick_analyze(dataset="breast_cancer")
```

---

## Core Capabilities

- **Calibration diagnostics**: Brier score, ECE, reliability curve data
- **Failure diagnostics**: misclassification analysis and confidence-gap risk signals
- **Bias and fairness diagnostics**: class imbalance, subgroup performance, equalized-odds checks
- **Representation diagnostics**: embedding separability when embeddings are provided
- **Decision engine**: weighted trust score with penalties and deployment blockers
- **Reporting**: console summary, plots, JSON/TXT export, model comparison utility

---

## How the Decision Works

TrustLens computes sub-scores across available dimensions, applies weights, then applies penalties and blockers for high-risk conditions.

Use this ordering when reading output:

1. Check `is_blocked` and verdict.
2. Check final Trust Score and grade.
3. Check penalties to identify dominant risk dimensions.
4. Inspect module-level diagnostics for remediation.

Deep dive: [Trust Score Explained](docs/trust_score_explained.md)

---

## Documentation Map

- Start: [Getting Started](docs/getting_started.md)
- Concepts: [Features and Modules](docs/features.md)
- Workflow guides: [Guides](docs/guides/model_comparison_workflow.md)
- Limits and caveats: [Known Limitations](docs/known_limitations.md)
- API docs: [API Reference](docs/api_reference.md)

---

## Use Cases

TrustLens is designed for practical ML workflows:

- release gating in CI
- candidate model selection
- fairness review before deployment
- post-deployment reliability audits

See [Use Cases](docs/use_cases.md) and [Guides](docs/guides/ci_deployment_gate.md).

---

## Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Khanz9664">
        <img src="https://github.com/Khanz9664.png" width="100px;" style="border-radius:50%;" alt="Khanz9664"/>
        <br />
        <sub><b>Khanz9664</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/jayssSmm">
        <img src="https://github.com/jayssSmm.png" width="100px;" style="border-radius:50%;" alt="jayssSmm"/>
        <br />
        <sub><b>jayssSmm</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/WeiGuang-2099">
        <img src="https://github.com/WeiGuang-2099.png" width="100px;" style="border-radius:50%;" alt="WeiGuang-2099"/>
        <br />
        <sub><b>WeiGuang-2099</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/CrepuscularIRIS">
        <img src="https://github.com/CrepuscularIRIS.png" width="100px;" style="border-radius:50%;" alt="CrepuscularIRIS"/>
        <br />
        <sub><b>CrepuscularIRIS</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/komoike-oss28-ui">
        <img src="https://github.com/komoike-oss28-ui.png" width="100px;" style="border-radius:50%;" alt="komoike-oss28-ui"/>
        <br />
        <sub><b>komoike-oss28-ui</b></sub>
      </a>
    </td>
  </tr>
</table>

Want to see your name here? Start with a [`good first issue`](https://github.com/Khanz9664/TrustLens/issues).

---

## Contributing

Contributions are welcome across metrics, docs, tests, and integrations.

- Read: [Contributing Guide](CONTRIBUTING.md)
- Browse issues: [Open Issues](https://github.com/Khanz9664/TrustLens/issues)
- Join discussions: [Discussions](https://github.com/Khanz9664/TrustLens/discussions)

---

## Citation

If you use TrustLens in research or production, cite:

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

Shahid Ul Islam
[GitHub](https://github.com/Khanz9664) · [Portfolio](https://khanz9664.github.io/portfolio/) · [LinkedIn](https://www.linkedin.com/in/shahid-ul-islam-13650998/)
