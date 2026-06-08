# Contributing to TrustLens

## First Contribution Guide

Welcome! We're excited to have you here. This guide will help you get started with your first contribution in minutes.

### 1. Find an Issue
Browse our [open issues](https://github.com/Khanz9664/TrustLens/issues). Look for the `good first issue` label for beginner-friendly tasks.

### 2. Set Up Your Environment
Run these commands to set up your local development environment:

```bash
# Fork the repo on GitHub, then clone your fork:
git clone https://github.com/YOUR_USERNAME/TrustLens.git
cd TrustLens

# Set up a virtual environment:
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# Install in editable mode with dev dependencies:
pip install -e ".[dev]"

# Install pre-commit hooks:
pre-commit install
```

### 3. Create a Branch
Always create a new branch for your work:

```bash
git checkout -b feat/my-new-feature
```

### 4. Make Changes & Verify
After making your changes, verify everything works:

```bash
# Run the test suite:
make test

# Run linting and formatting:
pre-commit run --all-files
```

### 5. Open a Pull Request
Push your branch and open a PR on GitHub. Don't worry about being perfect—we're here to help!

---

## How to Pick an Issue

We use labels to help you find the right task for your experience level:

- 🟢 **Beginner**: Small fixes (typos, docs, simple logging). Label: `good first issue`.
- 🟡 **Intermediate**: New metrics, CLI improvements, integrations. Label: `intermediate`.
- 🔵 **Advanced**: Core algorithm changes, performance optimization. Label: `advanced`.

---

## Need Help?

If you get stuck or have questions:
- **Comment on the issue**: We'll respond as soon as we can.
- **Join Discussions**: Ask general questions in our [GitHub Discussions](https://github.com/Khanz9664/TrustLens/discussions).
- **Draft PR**: Open a Draft PR if you want early feedback on your code.

---

## Quick PR Checklist

Before submitting your PR, ensure:
- [ ] Tests pass (`make test`)
- [ ] Pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] `CHANGELOG.md` is updated
- [ ] Linked issue is mentioned (e.g., `Fixes #123`)
- [ ] **Documentation is complete**: Public APIs have NumPy docstrings, new modules are documented, and usage examples are included when useful.

---

## Experimental Features

Some modules (e.g., `explainability/`) are under active development and are **not part of the stable API**. Before working on or importing these modules, please read the guidelines:

→ [**docs/EXPERIMENTAL.md**](docs/EXPERIMENTAL.md) — rules, promotion criteria, and what "experimental" means.

Roadmaps for **XGBoost**, **Keras**, and **TensorFlow** integrations live in [section 10](#10-framework-integration-plans).

---

## Code of Conduct

As a contributor, you are expected to respect and follow our code of conduct to ensure a welcoming environment for everyone.
→ [**CODE_OF_CONDUCT.md**](CODE_OF_CONDUCT.md)

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Adding a New Backend](#adding-a-new-backend)
4. [Adding a New Metric](#adding-a-new-metric)
5. [Adding a New Visualization](#adding-a-new-visualization)
6. [Writing a Plugin](#writing-a-plugin)
7. [Coding Standards](#coding-standards)
8. [Writing Tests](#writing-tests)
9. [Pull Request Guidelines](#pull-request-guidelines)
10. [Reporting Issues](#reporting-issues)
11. [Framework integration plans](#11-framework-integration-plans)

---

## 1. Development Setup

This is only needed if you're making code changes. For small documentation fixes, you can edit directly on GitHub.

### Prerequisites

- Python 3.9 or higher
- Git

### Fork and Clone

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/Khanz9664/trustlens.git
cd trustlens
```

### Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### Install in Editable Mode with Dev Dependencies

```bash
pip install -e ".[dev]"
```

### Install Pre-commit Hooks

```bash
pre-commit install
```

Pre-commit will automatically run linting and formatting checks on every `git commit`.

### Verify Installation

```bash
pytest --tb=short
python -c "from trustlens import analyze; print(' TrustLens ready')"
```

---

---

## 3. Project Structure & Architecture References

When modifying the codebase, refer to the following architectural layers and their corresponding documentation:

*   **API Layer**: The entry point for users. Resolves inputs and delegates to the core pipeline. (See `trustlens/api.py` and `trustlens/core/pipeline.py`)
*   **Backend Layer**: Resolves framework-specific models (sklearn, xgboost) into a standardized `PredictionBundle`. (See `trustlens/backends/`)
*   **Trust Score Layer**: Aggregates metric outputs into a single composite score and deployment verdict. (See `trustlens/trust_score.py`)
*   **Visualization Layer**: A centralized styling architecture that guarantees visual parity across all TrustLens plots. (See `trustlens/visualization/`)

```
trustlens/
 api.py          ← analyze() entry point
 report.py       ← TrustReport result container
 backends/       ← Framework Resolvers [NEW]
   registry.py   ← Framework detection & dispatch
   sklearn.py    ← Scikit-learn resolver
   xgboost.py    ← XGBoost resolver
 core/
   pipeline.py   ← Framework-agnostic execution engine
 metrics/        ← Agnostic diagnostic modules
 visualization/  ← Agnostic plotting modules
 plugins/        ← User-defined extensions
```

---

## 4. Adding a New Backend

TrustLens is framework-agnostic. To add support for a new library (e.g., CatBoost or PyTorch), follow these steps.

### Step 1 — Create the backend file
Create `trustlens/backends/myframework.py`.

### Step 2 — Implement the `resolve()` function
Your resolver must accept a model and features, and return a `PredictionBundle`. It is responsible for framework-specific prediction logic and normalization.

```python
# trustlens/backends/myframework.py
from trustlens.backends.types import PredictionBundle

def resolve(model, X, y_pred=None, y_prob=None):
    # 1. Resolve probabilities (framework-specific)
    if y_prob is None:
        y_prob = model.predict_proba(X)

    # 2. Normalize probabilities (Contract: always (n, classes))
    # For binary, convert (n,) -> (n, 2)

    # 3. Resolve class predictions
    if y_pred is None:
        y_pred = model.predict(X)

    # 4. Return the standard bundle
    return PredictionBundle(
        y_pred=y_pred,
        y_prob=y_prob,
        framework="myframework",
        metadata={
            "resolver": "myframework",
            "framework_version": "1.2.3"
        }
    )
```

### Step 3 — Register the framework
Update `trustlens/backends/registry.py`:

1. Add your framework to `FRAMEWORK_MAPPING` for auto-detection.
2. Add your framework name to `IMPLEMENTED_RESOLVERS`.
3. Update `get_resolver()` to return your `resolve` function.

### Step 4 — Add integration tests
Create `tests/test_backend_myframework.py`. Ensure it covers:
- Automatic framework detection.
- Prediction resolution (shapes and types).
- Gating/blocking unsupported tasks (e.g., regression).

---

## 5. Adding a New Metric

Here is the step-by-step workflow for adding a new metric.

**Example: Adding Maximum Calibration Error (MCE)**

### Step 1 — Write the metric function

Add your function to the appropriate module file (or create a new one):

```python
# trustlens/metrics/calibration.py

def maximum_calibration_error(
  y_true: np.ndarray,
  y_prob: np.ndarray,
  n_bins: int = 10,
) -> float:
  """
  Compute Maximum Calibration Error (MCE).

  MCE is the worst-case calibration gap across all bins.
  Lower is better; MCE=0.0 means perfect calibration.

  Parameters
  ----------
  y_true : np.ndarray
    Binary ground-truth labels (0 or 1).
  y_prob : np.ndarray
    Predicted probabilities for the positive class.
  n_bins : int
    Number of confidence bins.

  Returns
  -------
  float
    MCE in [0, 1].
  """
  y_true = np.asarray(y_true, dtype=float)
  y_prob = np.asarray(y_prob, dtype=float)

  bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
  max_gap = 0.0

  for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
    mask = (y_prob >= lo) & (y_prob < hi)
    if mask.sum() == 0:
      continue
    accuracy  = y_true[mask].mean()
    confidence = y_prob[mask].mean()
    max_gap = max(max_gap, abs(accuracy - confidence))

  return float(max_gap)
```

### Step 2 — Export from the metrics `__init__.py`

```python
# trustlens/metrics/__init__.py
from trustlens.metrics.calibration import maximum_calibration_error
```

### Step 3 — Integrate into `api.py` (optional)

If the metric should run automatically, add it to the calibration block in `api.py`:

```python
results["calibration"]["mce"] = maximum_calibration_error(y_true, y_prob_pos)
```

### Step 4 — Write tests

```python
# tests/test_calibration.py
def test_mce_perfect_is_zero():
  y_true = np.array([0, 1, 0, 1])
  y_prob = np.array([0.0, 1.0, 0.0, 1.0])
  assert maximum_calibration_error(y_true, y_prob) == pytest.approx(0.0)

def test_mce_geq_ece(binary_random):
  y_true, y_prob = binary_random
  ece = expected_calibration_error(y_true, y_prob)
  mce = maximum_calibration_error(y_true, y_prob)
  assert mce >= ece # MCE is always >= ECE
```

### Step 5 — Document

Add a docstring entry to `docs/api_reference.rst` and mention it in the changelog.

---

## 6. Adding a New Visualization

Visualization functions go in `trustlens/visualization/`.

**Interface contract:**
- Accept pre-computed data from TrustReport (never raw models/data).
- Return a `matplotlib.Figure` — do not call `plt.show()` internally.
- Accept an optional `save_path` parameter.
- Use `matplotlib.use("Agg")` at the top of each visualization file.

**Wire it up in `trustlens/visualization/__init__.py`:**
```python
from trustlens.visualization.my_plot_file import my_plot_function
```

**Wire it up in the dispatcher (`plot_module`):**
```python
def _plot_mycategory(data: dict):
  return my_plot_function(data)

dispatch["mycategory"] = _plot_mycategory
```

---

## 7. Writing a Plugin

Full plugin authoring guide: see `trustlens/plugins/__init__.py`.

Minimum required code:

```python
from trustlens.plugins.base import BasePlugin
from trustlens.plugins.registry import PluginRegistry

class MyCustomPlugin(BasePlugin):
  name = "my_metric"    # unique identifier
  description = "A description shown in the registry."

  def run(self, model, X, y_true, y_pred, y_prob, **kwargs):
    # Your logic here
    return {"result_key": 42.0} # must be JSON-serializable

PluginRegistry().register(MyCustomPlugin)
```

Then run it with:
```python
report = analyze(model, X, y, plugins=["my_metric"])
```

---

## 8. Coding Standards

- **Python 3.9+** — use type hints, `from __future__ import annotations`
- **Ruff** for linting (config in `pyproject.toml`)
- **Black** for formatting (line length = 100)
- **No bare `except:`** — always catch specific exceptions
- **No global state** — functions must be pure and stateless where possible
- **Imports** — standard library → third-party → trustlens internal

## 8.1. Documentation Standards

TrustLens strictly adheres to **NumPy-style docstrings**.

- **Format:** All docstrings must follow NumPy standards (Parameters, Returns, Raises, Examples).
- **Public API Expectations:** Every public function and class MUST have a complete docstring. Include a concise summary of "What it measures" and "Why it matters" for metrics where appropriate.
- **Module Documentation:** Every major module must contain a module-level docstring that details:
  - Responsibilities of the module
  - Relationship to other TrustLens components
  - High-level workflow

Run checks locally:
```bash
ruff check trustlens/
black --check trustlens/
mypy trustlens/
```

---

## 9. Writing Tests

- Place tests in `tests/` mirroring the source structure
- Use `pytest` fixtures for shared setup
- Target **100% branch coverage** for new functions
- Tests must be deterministic — seed all RNGs with a fixed value
- Name tests descriptively: `test_<function>_<scenario>`

```bash
pytest             # run all tests
pytest -k "calibration"    # run matching tests
pytest --cov=trustlens     # run with coverage
```

### Framework-Specific Tests
Some tests require optional dependencies. To run them locally, install the corresponding extra and use the pytest markers:

```bash
# Install the extra (example: xgboost)
pip install -e ".[xgboost]"

# Run only XGBoost tests
pytest -m requires_xgboost

# Run only TensorFlow tests
pytest -m requires_tensorflow
```

---

## 10. Pull Request Guidelines

1. **One PR per feature/fix** — keep changes focused
2. **Branch naming**: `feature/my-feature`, `fix/bug-description`, `docs/update-readme`
3. **Commit messages**: use [Conventional Commits](https://www.conventionalcommits.org/)
  - `feat: add MCE metric`
  - `fix: handle empty calibration bins`
  - `docs: expand plugin authoring guide`
4. **All tests must pass** before requesting review
5. **Write a clear PR description** — what changed, why, how to test it
6. **Link the related GitHub Issue**

---

## 11. Reporting Issues

Found a bug? Have a feature request?

Open an issue with:
- **Environment**: Python version, OS, TrustLens version
- **Minimal reproducible example** (MRE)
- **Expected vs. actual behaviour**

We aim to respond within 48 hours. ⏱

---

## 12. Framework integration plans

Use these when picking up or reviewing large integrations (optional backends, shared prediction resolver, CI). Each plan is **scoped to one library**:

| Plan | Scope |
|------|--------|
| [docs/plans/IMPLEMENTATION_PLAN_XGBoost.md](docs/plans/IMPLEMENTATION_PLAN_XGBoost.md) | **XGBoost [INTEGRATED]** — Native support added in v0.4.0. See `docs/internal/prediction_contract.md` for current backend standards. |
| [docs/plans/IMPLEMENTATION_PLAN_Keras.md](docs/plans/IMPLEMENTATION_PLAN_Keras.md) | **Keras** — `model.predict` semantics, shapes, `analyze_keras`, experimental API (see plan for `keras` vs `tf.keras` scope). |
| [docs/plans/IMPLEMENTATION_PLAN_TensorFlow.md](docs/plans/IMPLEMENTATION_PLAN_TensorFlow.md) | **TensorFlow** — optional `tensorflow` extra, lazy imports, CI, SavedModel/runtime notes; cross-links Keras plan for Keras API details. |

---

Thanks again for contributing to TrustLens.
You're helping make ML systems more honest, one metric at a time.
