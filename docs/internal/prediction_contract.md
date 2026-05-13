# Internal RFC: Prediction Semantics Contract

This document defines the strict contract for how predictions must be normalized and structured within TrustLens backends. All `resolve()` functions must adhere to these rules.

## 1. Classification Semantics

### Binary Classification
TrustLens metrics expect binary probabilities as a 2D array.
- **Allowed Input Shapes**: `(n,)`, `(n, 1)`, `(n, 2)`.
- **Normalization Rule**: All binary outputs must be normalized to `(n, 2)` where index 1 represents the positive class.
- **Example**: If a model returns a 1D array of positive probabilities, the backend must expand it to `[1-p, p]`.

### Multiclass Classification
- **Allowed Input Shape**: `(n, C)`.
- **Normalization Rule**: Probabilities must sum to 1.0 per sample. No logits allowed in the `PredictionBundle`.

---

## 2. Regression Semantics
TrustLens does **NOT** currently support regression tasks.
- **Contract**: Backends should raise `NotImplementedError` if regression output is detected.

---

### Logits Policy
Raw logits are strictly forbidden inside the `PredictionBundle`.

Backends are responsible for converting logits to calibrated probabilities (e.g., via Softmax for multiclass or Sigmoid for binary tasks) before constructing the bundle. This ensures that downstream trust metrics, calibration plots, and uncertainty estimations receive standardized probability scores.

---

## 3. Label Consistency
- **Class Labels**: The `class_labels` field in `PredictionBundle` should be populated whenever possible to ensure that metric reports use the correct semantic labels (e.g., "Malignant" instead of `1`).

---

## 4. Metadata Hooks
Every `PredictionBundle` should include detection metadata for debugging:
```python
metadata = {
    "resolver": "sklearn",
    "detection_method": "module_name", # or "explicit", "capability"
    "framework_version": "1.3.0"
}
```

---

## 5. Non-Standard Frameworks
- **Ranking / Anomaly Detection**: If a framework does not provide probabilities, the backend must provide a valid `y_pred` and set `y_prob` to a placeholder or zeros (while documenting the limitation in `metadata`).
