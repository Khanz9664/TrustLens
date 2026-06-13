# Implementation plan: Keras

This document is for **contributors** and **maintainers** integrating the **Keras API** (`keras.Model`, `Sequential`, `predict`) with TrustLens so classification models produce NumPy `y_pred` and `y_prob` for the existing analysis pipeline.

**Related plans:** [XGBoost](IMPLEMENTATION_PLAN_XGBoost.md) (tabular backend) · [TensorFlow](IMPLEMENTATION_PLAN_TensorFlow.md) (TensorFlow package, `tf.keras`, SavedModel, CI)

**Status target:** **Experimental** until [`EXPERIMENTAL.md`](../EXPERIMENTAL.md) promotion criteria are met. Do **not** require Keras (or TensorFlow) for `pip install trustlens`.

---

## Scope: “Keras” vs “TensorFlow” in this repo

| Topic | This document (Keras) | [TensorFlow plan](IMPLEMENTATION_PLAN_TensorFlow.md) |
|--------|-------------------------|------------------------------------------------------|
| `model.predict` output shapes, binary vs multiclass | **Primary** | References this plan |
| `keras` PyPI package, `keras.Model` type checks | **Primary** | N/A |
| `tensorflow` PyPI extra, lazy `import tensorflow` | Cross-reference | **Primary** |
| `tf.keras` as deployment vehicle | Note: same API patterns | **Primary** for import/version/CI |
| SavedModel, serving, GPU runtime | Out of scope (v1) | **Primary** |

Many users only use **`tf.keras`**. Implementation may ship **`analyze_keras`** in `trustlens.experimental.keras` using `tf.keras.Model` first; this Keras plan still defines **API semantics** and tests that apply to any Keras-compatible `predict` output.

---

## Executive summary

| Item | Decision |
|------|----------|
| **Goal** | Resolve `y_pred`, `y_prob` from a trained **classification** Keras model and run `_run_analysis_pipeline(...)` (shared with `analyze()`). |
| **Public API (until promotion)** | e.g. `from trustlens.experimental.keras import analyze_keras` — avoid silent heavy imports from `import trustlens`. |
| **Dependencies** | Optional extra, e.g. `keras = ["keras>=3"]` **and/or** overlap with TensorFlow extra when implementation uses `tf.keras` — document the **single supported install path for v1** in `pyproject.toml` and here once chosen. |
| **Import rule** | Lazy `import keras` or lazy `import tensorflow as tf` **only** inside experimental modules (see TensorFlow plan for TF-specific hygiene). |

---

## Prerequisites

1. **`_run_analysis_pipeline`** extracted from `trustlens.api.analyze()` so Keras code does not duplicate calibration / failure / bias / representation logic. See [XGBoost plan](IMPLEMENTATION_PLAN_XGBoost.md) PR A for resolver context; pipeline extraction can land in the same or adjacent PR.
2. Optional **`y_pred=`** / **`y_prob=`** on `analyze()` for power users who bypass Keras resolution.

---

## Objectives (Keras)

1. **Binary classification**
   - Sigmoid output `(n, 1)`: normalize to TrustLens binary convention (two-column `y_prob` or documented single-column path consistent with `api.py`).
   - Softmax output `(n, 2)`: use as-is; `y_pred = argmax(axis=1)`.

2. **Multiclass**
   Softmax `(n, C)`, `C > 2`: `y_prob` as-is; `y_pred = argmax(axis=1)`.

3. **Input**
   **v1:** NumPy `X` (and optional `embeddings`) only; call `model.predict(X, verbose=0)` then `np.asarray(..., dtype=np.float64)`.

4. **Examples**
   `examples/keras_audit.py`: small `Sequential` model on synthetic data, `analyze_keras`, optional `report.save(...)`.

5. **Documentation**
   `docs/EXPERIMENTAL.md`: Keras subsection — install, API, limitations, promotion checklist.

---

## Non-goals (v1)

- Multi-label, regression, object detection, dict / ragged inputs.
- Custom training loops, callbacks, or layer freezing logic.
- **Native** multi-backend Keras 3 on JAX/Torch for CI unless maintainers explicitly add jobs (numpy/TF path is acceptable for v1).
- Monkey-patching `predict_proba` onto Keras models.

---

## Technical design

### 1. Output shape matrix (canonical)

| Head | `predict` shape | `y_prob` (TrustLens) | `y_pred` |
|------|-----------------|----------------------|----------|
| Binary sigmoid | `(n, 1)` | `[1-p, p]` shape `(n, 2)` *recommended* | `argmax` or `(p > 0.5).astype(int)` — **pick one and test** |
| Binary softmax | `(n, 2)` | unchanged | `argmax(axis=1)` |
| Multiclass softmax | `(n, C)` | unchanged | `argmax(axis=1)` |

Centralize normalization in **`resolve_keras_predictions(model, X) -> tuple[y_pred, y_prob]`** in `trustlens/experimental/keras.py` (or submodule).

### 2. Model typing

- Prefer **`isinstance(model, keras.Model)`** when using the standalone `keras` package.
- If v1 targets **`tf.keras.Model`** only, state that explicitly in module docstring and still follow the shape matrix above (types live in TensorFlow plan).

### 3. Integration

```text
analyze_keras(model, X, y_true, *, embeddings=None, ...)
  -> resolve_keras_predictions(model, X)
  -> _run_analysis_pipeline(y_true, y_pred, y_prob, ...)
  -> TrustReport
```

### 4. Embeddings

Same as sklearn path: user-supplied `embeddings` NumPy array. Optional later: helper to attach a Keras intermediate layer (separate feature / not v1).

---

## Files to add or change (checklist)

| Path | Action |
|------|--------|
| `trustlens/experimental/keras.py` | `resolve_keras_predictions`, `analyze_keras`. |
| `trustlens/experimental/__init__.py` | Docstring / limited exports; no heavy imports at import time. |
| `trustlens/api.py` | Export `_run_analysis_pipeline` for experimental use **or** move pipeline to `trustlens/backends/pipeline.py` to avoid circular imports. |
| `pyproject.toml` | Optional `keras` and/or document use of `tensorflow` extra — one clear story. |
| `docs/EXPERIMENTAL.md` | Keras integration subsection. |
| `docs/getting_started.md` | One-line pointer to experimental Keras. |
| `examples/keras_audit.py` | End-to-end demo. |
| `tests/test_keras_experimental.py` | Shape unit tests (pure NumPy) + optional integration tests. |
| `pyproject.toml` markers | e.g. `requires_keras` / reuse `requires_tensorflow` if tests use tf.keras only. |

---

## Testing strategy

1. **Pure NumPy tests** — Feed `resolve_keras_predictions`–level helpers with fixed arrays mimicking `(n,1)`, `(n,2)`, `(n,C)` outputs; run in default CI without Keras installed.
2. **Integration tests** — Behind `pytest.importorskip("keras")` or `tensorflow` depending on v1 choice; binary + 3-class models.
3. **Import hygiene** — `import trustlens` must not import `keras` or `tensorflow` (subprocess test recommended).

---

## CI recommendations

- Default PR CI: **no** heavy Keras/TF install unless job time is acceptable.
- Optional: weekly / `workflow_dispatch` job installing the chosen extra and running marked tests.

Coordinate exact markers with [TensorFlow plan](IMPLEMENTATION_PLAN_TensorFlow.md) to avoid duplicate jobs.

---

## Acceptance criteria (Keras experimental “ready”)

- [ ] `analyze_keras` returns `TrustReport` with calibration, failure, bias, representation populated for binary and 3-class toy models.
- [ ] Binary sigmoid and softmax paths both tested.
- [ ] `import trustlens` does not load Keras/TF (per implementation choice).
- [ ] `docs/EXPERIMENTAL.md` updated.
- [ ] `CHANGELOG.md` entry (Experimental).

---

## Suggested PR breakdown

1. **Pipeline extraction** — `_run_analysis_pipeline` only; no Keras.
2. **Experimental Keras** — `resolve_keras_predictions` + `analyze_keras` + tests + example + docs.

---

## Promotion to stable API

When promoted per `EXPERIMENTAL.md`:

- Document `framework="keras"` on main `analyze()` **or** keep `analyze_keras` as the supported entry point.
- Expand CI if Keras becomes a first-class optional extra.

---

## FAQ

**Q: Keras 3 multi-backend vs tf.keras only?**
**A:** v1 should pick **one** supported install to reduce support burden; document the other as “community tested” or future work.

**Q: Overlap with TensorFlow plan?**
**A:** Keras plan owns **shapes and API**; TensorFlow plan owns **TF package**, versions, SavedModel, and lazy-import policy.

---

## Document history

- **Scope:** Keras API only; XGBoost and TensorFlow have separate plans.
