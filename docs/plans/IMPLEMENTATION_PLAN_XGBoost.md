# Implementation plan: XGBoost

This document is for **contributors** and **maintainers** adding **first-class XGBoost** support to TrustLens. It covers **only** the XGBoost library and its interaction with the existing `analyze()` pipeline.

**Related plans (other libraries):** [Keras](IMPLEMENTATION_PLAN_Keras.md) · [TensorFlow](IMPLEMENTATION_PLAN_TensorFlow.md)

**Status target:** **Stable** — same public `analyze()` entry point as scikit-learn–compatible estimators, optional install extra, no `xgboost` import when users run `import trustlens`.

---

## Executive summary

| Item | Decision |
|------|----------|
| **Goal** | Users with `xgboost.XGBClassifier` call `analyze(model, X, y_true, ...)` without manually passing `y_prob` when `predict_proba` is available. |
| **Where it lives** | `trustlens.backends.predictions` (resolver branch) + `trustlens.api.analyze()`; optional `framework="xgboost"` or lazy `isinstance` after `import xgboost`. |
| **Dependencies** | Optional extra: `pip install trustlens[xgboost]` with `xgboost>=…` lower bound aligned with NumPy / scikit-learn. |
| **Import rule** | **Lazy** `import xgboost` only inside the resolver branch; never at package top level. |

TrustLens metrics already consume NumPy `y_true`, `y_pred`, `y_prob`. XGBoost work is **boundary normalization** only.

---

## Prerequisites (shared with other backends)

Before XGBoost-specific PRs, the codebase should have (or gain in **PR A**):

1. **Internal resolver** — `trustlens/backends/predictions.py` (or equivalent) with a function such as `resolve_predictions(model, X, y_prob=None, y_pred=None, framework=None) -> tuple[np.ndarray, np.ndarray]`.
2. **Sklearn path unchanged** — Current `analyze()` behavior remains the default; regression tests in `tests/test_api.py` stay green.
3. **Optional `y_pred=`** on `analyze()` — Recommended so users can bypass resolver edge cases without waiting for a fix.

Cross-team note: the same resolver hosts Keras and TensorFlow branches documented in their respective plans; merge order is maintainer choice, but **PR A (resolver + sklearn only)** should not require XGBoost installed.

---

## Objectives (XGBoost)

1. **Binary classification** — `XGBClassifier` with `objective` implying probabilities; `predict_proba` / `predict` return shapes compatible with `trustlens.api` binary calibration (`y_prob[:, 1]` when two columns).
2. **Multiclass** — `predict_proba` shape `(n_samples, n_classes)`; no crashes in calibration / failure / bias modules.
3. **Documentation** — `docs/getting_started.md` and `docs/api_reference.md`: minimal example with `XGBClassifier` + `analyze()`.
4. **Discoverability** — Optional extra documented in README / PyPI classifiers if appropriate.

---

## Non-goals (v1)

- `XGBRegressor` or non-classification tasks (unless calibration semantics are extended project-wide).
- Raw `xgboost.Booster` + `DMatrix` without an sklearn-style wrapper (document workaround: wrap or pass `y_prob` / `y_pred` manually).
- GPU-only CI as a merge blocker.
- XGBoost **training** utilities inside TrustLens (users train outside; TrustLens analyzes).

---

## Technical design

### 1. Detection

Pick **one** primary strategy and document it:

- **Explicit:** `analyze(..., framework="xgboost")` forces the XGBoost branch.
- **Implicit:** After lazy `import xgboost`, `isinstance(model, xgboost.XGBClassifier)`.

Avoid duck-typing that could mis-classify other estimators.

### 2. Prediction path

```text
y_prob = np.asarray(model.predict_proba(X), dtype=np.float64)
y_pred = np.asarray(model.predict(X))
```

**Binary edge case:** If `predict_proba` ever returns shape `(n_samples,)`, reshape to `(n_samples, 2)` using `[1 - p, p]` or match the convention already used in `trustlens/api.py` for positive-class scores. Add a unit test that mirrors XGBoost’s actual output for a toy fixture.

### 3. Sparse features

If `X` is `scipy.sparse` and XGBoost accepts it, either document “dense NumPy only for v1” or thread sparse through **only** where `analyze()` already allows it; do not silently densify huge matrices without documentation.

### 4. Metadata (optional v1)

Record in saved report metadata, e.g. `framework: xgboost`, `xgboost_version`, booster params summary string — helpful for audits.

---

## Files to add or change (checklist)

| Path | Action |
|------|--------|
| `pyproject.toml` | `[project.optional-dependencies]` → `xgboost = ["xgboost>=…"]`. |
| `trustlens/backends/predictions.py` | `_predict_xgboost(...)` or branch inside `resolve_predictions`. |
| `trustlens/api.py` | Call resolver from `analyze()`; optional `framework` / `y_pred` kwargs. |
| `docs/getting_started.md` | Short XGBoost example. |
| `docs/api_reference.md` | `framework`, shapes, manual overrides. |
| `tests/test_predictions_resolver.py` | Mocks + shape normalization without `xgboost` installed. |
| `tests/test_api_xgboost.py` | `pytest.importorskip("xgboost")` or `@pytest.mark.requires_xgboost`. |
| `CHANGELOG.md` | `feat:` entry. |

---

## Testing strategy

- **Default CI:** Either skip XGBoost tests when extra not installed, or add a **single** optional job (Linux, one Python) `pip install -e ".[dev,xgboost]"` and run marked tests — pick one and document in `CONTRIBUTING.md`.
- **Markers:** Register `requires_xgboost` in `[tool.pytest.ini_options]` markers.
- **Import smoke:** Subprocess or `sys.modules` assertion: `import trustlens` does not load `xgboost`.

---

## Acceptance criteria (XGBoost complete)

- [ ] `analyze(xgb_clf, X_val, y_val, verbose=False)` works on a binary `make_classification` fixture without `y_prob`.
- [ ] Three-class fixture completes without error.
- [ ] No `xgboost` import at `import trustlens` time.
- [ ] Ruff + mypy clean for new code.
- [ ] Docs and changelog updated.

---

## Suggested PR breakdown

1. **PR A — Resolver + sklearn parity**  
   Introduce `resolve_predictions`; wire `analyze()`; no XGBoost yet; tests prove identical behavior to today.

2. **PR B — XGBoost**  
   Optional extra, branch, tests, docs — isolated revert surface.

---

## Maintainer sign-off

| Gate | ☐ |
|------|---|
| API / docs reviewed | ☐ |
| Tests (default + optional) | ☐ |
| No optional dep on core import | ☐ |
| CHANGELOG entry | ☐ |

---

## FAQ

**Q: LightGBM / CatBoost?**  
**A:** Often work today via `predict` / `predict_proba`. Add library-specific tests in follow-up PRs; open a dedicated plan only if a quirk requires a new branch.

**Q: Order relative to Keras / TensorFlow?**  
**A:** Independent once PR A exists. XGBoost is lighter and is a good first optional backend.

---

## Document history

- **Scope:** XGBoost only; Keras and TensorFlow have separate plans in this directory.
