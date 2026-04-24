# Experimental Modules

> This document is for **contributors and maintainers**. It explains which parts of TrustLens are experimental and what that means for development.

---

## What Does "Experimental" Mean?

An experimental module is code that:

- **Exists in the repository** and is functional
- **Is NOT used** by the core pipeline (`analyze()`, `quick_analyze()`, `TrustReport`)
- **Is NOT exposed** in the main `trustlens.__init__` docstring or marketing materials
- **May have heavy optional dependencies** (e.g., PyTorch)
- **Is still importable** for advanced users who explicitly need it

Experimental modules are **not deleted** — they are isolated until they meet the bar for promotion.

---

## Currently Experimental

| Module | Depends On | Why Experimental | Promotion Criteria |
|--------|-----------|------------------|-------------------|
| `trustlens/explainability/gradcam.py` | PyTorch | Requires deep learning framework; not part of the ML evaluation core | Full test coverage, CI-tested with PyTorch, documented API |
| `trustlens/explainability/faithfulness.py` | NumPy (+ SciPy optional) | Tied to explainability workflow | Same as above |
| `trustlens/metrics/faithfulness.py` | None | Wrapper over explainability results | Promoted when explainability is promoted |

### How to Use Experimental Modules

They are fully importable for advanced users:

```python
from trustlens.explainability import GradCAM
from trustlens.explainability import pixel_deletion_test, pixel_insertion_test
from trustlens.metrics.faithfulness import faithfulness_summary
```

### How to Identify Experimental Code

Every experimental file has this comment at the top:

```python
# NOTE:
# This module is under active development and is not part of the public API.
# Do not import into production pipelines until stabilized.
```

---

## Rules for Contributors

1. **Do NOT import experimental modules** into `api.py`, `report.py`, or `trust_score.py`.
2. **Do NOT add experimental modules** to `trustlens/__init__.py` or `metrics/__init__.py`.
3. **Do NOT reference experimental features** in user-facing documentation (README, PyPI description).
4. **You CAN** improve experimental modules — just keep them isolated.
5. **You CAN** add tests for experimental modules in `tests/` — this helps promotion.

---

## Promotion Process

To promote a module from experimental to stable:

1. Full test coverage with CI passing
2. No heavy required dependencies (must be optional via `try/except`)
3. Integration into the `analyze()` pipeline (optional module flag)
4. Documentation and examples updated
5. Maintainer approval via PR review

---

## Questions?

Open a [Discussion](https://github.com/Khanz9664/TrustLens/discussions) or comment on a related issue.
