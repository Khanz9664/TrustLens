"""
examples/calibration_deep_dive.py.
===================================
Deep-dive into calibration analysis with TrustLens.

Compares three models at different calibration levels:
* A well-calibrated model (Platt-scaled logistic regression)
* A slightly miscalibrated SVM
* A severely overconfident neural network (simulated)

Generates side-by-side reliability diagrams for visual comparison.

Run with:
  python examples/calibration_deep_dive.py
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from trustlens.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_curve,
)
from trustlens.visualization.calibration_plots import plot_reliability_diagram


def main():
    # -----------------------------
    # 1. Load data
    # -----------------------------
    X, y = make_classification(
        n_samples=2_000,
        n_features=15,
        n_informative=6,
        random_state=0,
        weights=[0.6, 0.4],
    )
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

    # -----------------------------
    # 2. Train model
    # -----------------------------
    # 1. Well-calibrated logistic regression
    lr = LogisticRegression(max_iter=500, random_state=0)
    lr.fit(X_train, y_train)
    prob_lr = lr.predict_proba(X_val)[:, 1]

    # 2. Uncalibrated SVM (raw decision function passed through sigmoid)
    svm_cal = CalibratedClassifierCV(
        SVC(kernel="rbf", probability=False, random_state=0), cv=5, method="sigmoid"
    )
    svm_cal.fit(X_train, y_train)
    prob_svm = svm_cal.predict_proba(X_val)[:, 1]

    # 3. Simulated overconfident model (push probs toward extremes)
    prob_overconfident = np.clip(prob_lr**0.4, 0.01, 0.99)

    models = [
        ("Logistic Regression (calibrated)", prob_lr),
        ("SVM + Platt Scaling", prob_svm),
        ("Simulated Overconfident Model", prob_overconfident),
    ]

    # -----------------------------
    # 3. Analyze
    # -----------------------------
    # (Metrics are computed inside the visualization loop below)

    # -----------------------------
    # 4. Visualize
    # -----------------------------
    os.makedirs("examples/output", exist_ok=True)

    for name, probs in models:
        frac_pos, mean_pred, _ = reliability_curve(y_val, probs)
        bs = brier_score(y_val, probs)
        ece = expected_calibration_error(y_val, probs)

        # Save a detailed, professional diagram for each model
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        save_path = f"examples/output/calibration_{safe_name}.png"

        plot_reliability_diagram(
            frac_pos,
            mean_pred,
            ece=ece,
            brier_score=bs,
            title=name,
            save_path=save_path,
            show=False,
        )
        print(f"    ✔ Saved detailed diagram: {save_path}")

    # -----------------------------
    # 5. Output
    # -----------------------------
    print("\nCalibration deep-dive completed.")


if __name__ == "__main__":
    print("Running TrustLens calibration deep-dive...\n")
    main()
