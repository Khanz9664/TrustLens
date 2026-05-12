# Implementation plan: TensorFlow

This document is for **contributors** and **maintainers** integrating the **TensorFlow** Python package (`tensorflow`) with TrustLens: dependency management, **lazy imports**, CI, and TensorFlow-specific loading paths. It does **not** replace the [Keras plan](IMPLEMENTATION_PLAN_Keras.md), which owns **`model.predict` semantics** and shape normalization for Keras-style models.

**Related plans:** [XGBoost](IMPLEMENTATION_PLAN_XGBoost.md) · [Keras](IMPLEMENTATION_PLAN_Keras.md)

**Status target:** **Experimental** for any code path that executes `import tensorflow` during a typical `import trustlens`. Follow [`EXPERIMENTAL.md`](../EXPERIMENTAL.md) until promotion.

---

## Executive summary

| Item | Decision |
|------|----------|
| **Goal** | Allow TrustLens to run against models and artifacts in the TensorFlow ecosystem **without** adding TensorFlow to core dependencies or to the import chain of `import trustlens`. |
| **Primary surfaces (v1)** | Optional `pip install trustlens[tensorflow]` (or equivalent extra name); experimental modules that call `import tensorflow as tf` **inside functions**. |
| **Keras API** | Most users use **`tf.keras`**. Prediction shape rules and `analyze_keras` behavior are specified in [IMPLEMENTATION_PLAN_Keras.md](IMPLEMENTATION_PLAN_Keras.md); this document covers TF packaging, versions, SavedModel, and CI. |

---

## Prerequisites

1. **Shared resolver / pipeline** — Same as [XGBoost plan](IMPLEMENTATION_PLAN_XGBoost.md): internal prediction resolution and `_run_analysis_pipeline` extracted from `analyze()` so TensorFlow integration does not fork metric code.
2. **Keras semantics** — Implementers should read [Keras plan](IMPLEMENTATION_PLAN_Keras.md) before merging TF-specific PRs that call `model.predict`.

---

## Objectives (TensorFlow)

1. **Optional dependency** — `pyproject.toml` includes e.g. `tensorflow = ["tensorflow>=X.Y"]` with a conservative **minimum** version; revisited when security advisories apply (align with existing `pip-audit` CI).

2. **Lazy import discipline** — No `import tensorflow` at top level in `trustlens/__init__.py`, `trustlens/api.py`, or any module imported by default trustlens entry points.

3. **Experimental entry points** — Code that needs TF lives under e.g. `trustlens/experimental/` (or clearly named submodule) and is not re-exported from `trustlens.__init__` in v1.

4. **CI policy** — Document whether PR CI installs TensorFlow (usually **no** due to size) and whether **weekly / manual** workflows run TF tests.

5. **Version metadata** — When saving reports from TF-backed runs, optional `tensorflow_version` field in metadata JSON.

---

## Non-goals (v1)

- TPU / `MirroredStrategy` / multi-worker training integration.
- TensorFlow Extended (TFX) pipelines.
- Full **SavedModel** serving parity (may be v2; see below for optional stretch).

---

## Technical design

### 1. Lazy import pattern

```python
def _require_tf():
    import tensorflow as tf
    return tf

def some_entry(...):
    tf = _require_tf()
    ...
```

No module-level `tensorflow` import in files that are transitively imported by `from trustlens import analyze`.

### 2. `tf.keras` as the default supported model class

- Type checks: `isinstance(model, tf.keras.Model)` after lazy import.
- For prediction output shapes, follow [Keras plan](IMPLEMENTATION_PLAN_Keras.md).

### 3. SavedModel (optional stretch / v2)

- Document `loaded = tf.saved_model.load(path)` and how (if) a concrete function maps to `(y_pred, y_prob)` for TrustLens.
- If v1 does not support SavedModel, state **explicitly** in `EXPERIMENTAL.md` to avoid user confusion.

### 4. Tensors vs NumPy

- **v1:** Convert model outputs to NumPy at the boundary before metrics (consistent with rest of TrustLens).
- Document memory implications for large validation sets.

### 5. GPU / CPU in CI

- Prefer **CPU-only** TensorFlow wheels in CI to reduce flakiness unless maintainers provision GPU runners.

---

## Files to add or change (checklist)

| Path | Action |
|------|--------|
| `pyproject.toml` | Optional `[tensorflow]` extra; version lower bound. |
| `trustlens/experimental/*.py` | Any TF-specific helpers; lazy imports only. |
| `.github/workflows/ci.yml` | Optional scheduled / `workflow_dispatch` job: `pip install -e ".[dev,tensorflow]"` + `pytest -m requires_tensorflow`. |
| `docs/EXPERIMENTAL.md` | TensorFlow subsection: install, lazy import guarantee, CI, limitations. |
| `CONTRIBUTING.md` | How to run TF-marked tests locally. |
| `tests/test_tensorflow_import_hygiene.py` | Subprocess: `import trustlens` → assert `'tensorflow' not in sys.modules`. |
| `CHANGELOG.md` | Experimental TF support entry. |

---

## Testing strategy

1. **Import hygiene tests** — Always run in default CI; no TensorFlow install required.
2. **Marked integration tests** — `@pytest.mark.requires_tensorflow` or `pytest.importorskip("tensorflow")`; run in optional job.
3. **Version matrix** — At most **one** TF version + **one** Python version in optional CI unless the team expands capacity.

---

## Acceptance criteria (TensorFlow track “ready”)

- [ ] Optional extra documented; core install unchanged.
- [ ] `import trustlens` does not import TensorFlow (automated test).
- [ ] Optional CI job **or** documented maintainer-only local run for TF tests.
- [ ] `docs/EXPERIMENTAL.md` describes scope and points to Keras plan for `predict` shapes.
- [ ] `CHANGELOG.md` updated.

---

## Suggested PR breakdown

1. **Extras + import hygiene tests only** — No functional TF analysis yet; proves packaging.
2. **Experimental TF-backed analyze path** — Wires lazy TF import to `analyze_keras` or equivalent implemented per Keras plan.

---

## Security and maintenance

- Pin minimum TensorFlow versions; monitor CVEs (project already runs `pip-audit`).
- Do not execute arbitrary user graphs without documentation of trust boundaries.

---

## Maintainer sign-off

| Gate | ☐ |
|------|---|
| Optional extra / no core bloat | ☐ |
| Lazy import verified | ☐ |
| CI / docs / changelog | ☐ |

---

## FAQ

**Q: Why a separate doc from Keras?**  
**A:** TensorFlow covers **package install, import policy, CI, SavedModel, runtime**; Keras covers **model.predict contract** for Keras API models (including `tf.keras`).

**Q: JAX?**  
**A:** Out of scope for this TensorFlow plan.

---

## Document history

- **Scope:** TensorFlow package only; Keras API semantics live in `IMPLEMENTATION_PLAN_Keras.md`.
