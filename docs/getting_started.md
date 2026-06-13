# Getting Started

TrustLens is designed to be **zero-friction**. This guide will get you from zero to a production-grade model audit in less than two minutes.

## Framework Support

TrustLens works out-of-the-box with common ML frameworks:

* **scikit-learn**: All classifiers inheriting from `ClassifierMixin`.
* **XGBoost**: Both `XGBClassifier` and raw `Booster` objects.

The library **automatically detects** your framework. If you use a different framework or external inference system, you can still use TrustLens by providing predictions manually:

```python
# Pass model=None if you only have results, not the model object
report = analyze(None, X, y, y_pred=my_preds, y_prob=my_probs)
```

> [!NOTE]
> **Degraded Mode**: If you provide `y_pred` but not `y_prob`, TrustLens enters "Degraded Mode". It will skip confidence-based metrics (like calibration) while still auditing accuracy and fairness.

## Installation

```bash
pip install trustlens
```

## Minimal Working Example

The primary entry point is `trustlens.analyze()`. It orchestrates the entire evaluation pipeline and returns a `TrustReport`.

```python
from trustlens import analyze
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 1. Prepare your model and data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier().fit(X_train, y_train)

# 2. Run the decision-support audit
report = analyze(model, X_test, y_test)

# 3. Inspect the results
report.show()  # Prints a professional summary to the console
```

## What happened?

When you call `analyze()`, TrustLens performs a deep diagnostic sweep:

1. **Calibration Check**: It measures if your model's 90% confidence actually means 90% accuracy.
2. **Failure Modes**: It identifies high-confidence mistakes (the "Confidently Wrong" pattern).
3. **Trust Scoring**: It aggregates all signals into a single Trust Score (0-100) and provides a deployment **Verdict**.

## Next Steps

* View the [Features](features.md) to understand the metrics.
* Check the [Use Cases](use_cases.md) for domain-specific examples.
* Explore the [API Reference](api_reference.md) for advanced configuration.
