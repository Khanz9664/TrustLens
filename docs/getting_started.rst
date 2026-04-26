Getting Started
===============

TrustLens is designed to be **zero-friction**. This guide will get you from zero to a production-grade model audit in less than two minutes.

Prerequisites
-------------

TrustLens works with any model that exposes standard scikit-learn style methods:

* ``.predict(X)``: Returns class labels.
* ``.predict_proba(X)``: Returns class probabilities (required for calibration analysis).

If your model does not expose ``predict_proba``, you must provide the probabilities manually to the ``analyze()`` function.

Installation
------------

.. code-block:: bash

    pip install trustlens

Minimal Working Example
-----------------------

The primary entry point is ``trustlens.analyze()``. It orchestrates the entire evaluation pipeline and returns a ``TrustReport``.

.. code-block:: python

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

What happened?
--------------

When you call ``analyze()``, TrustLens performs a deep diagnostic sweep:

1. **Calibration Check**: It measures if your model's 90% confidence actually means 90% accuracy.
2. **Failure Modes**: It identifies high-confidence mistakes (the "Confidently Wrong" pattern).
3. **Trust Scoring**: It aggregates all signals into a single Trust Score (0-100) and provides a deployment **Verdict**.

Next Steps
----------

* View the :doc:`features` to understand the metrics.
* Check the :doc:`use_cases` for domain-specific examples.
* Explore the :doc:`api_reference` for advanced configuration.
