"""
examples/quickstart.py.
=======================
TrustLens Quickstart — complete end-to-end example.

This script demonstrates the full TrustLens workflow:
1. Train a RandomForest on a synthetic binary classification dataset.
2. Run the full TrustLens analysis pipeline.
3. Print the report to console.
4. Save plots and JSON metrics to 'output/'.

Run with:
  python examples/quickstart.py
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import trustlens
from trustlens import analyze

print(f"TrustLens v{trustlens.__version__}")
print("=" * 60)


def main():
    # -----------------------------
    # 1. Load data
    # -----------------------------
    print("\n[1/4] Generating dataset …")

    X, y = make_classification(
        n_samples=1_000,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        n_classes=2,
        weights=[0.75, 0.25],  # 75/25 imbalance
        flip_y=0.05,  # 5% label noise
        random_state=42,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} samples | Val: {len(X_val)} samples")
    print(f"  Class counts (val): {dict(zip(*np.unique(y_val, return_counts=True)))}")

    # -----------------------------
    # 2. Train model
    # -----------------------------
    print("\n[2/4] Training RandomForestClassifier …")

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_val)
    accuracy = (clf.predict(X_val) == y_val).mean()
    print(f"  Validation accuracy: {accuracy:.2%}")

    # -----------------------------
    # 3. Analyze
    # -----------------------------
    rng = np.random.default_rng(7)
    gender = rng.choice(["M", "F"], size=len(y_val))
    age_grp = rng.choice(["18-35", "36-55", "55+"], size=len(y_val))

    # Simulate embeddings (in real use, these come from a model's penultimate layer)
    embeddings = rng.standard_normal((len(y_val), 32))
    # Make them slightly class-informative for interesting separability results
    embeddings += y_val.reshape(-1, 1) * 0.8

    print("\n[3/4] Running TrustLens analysis …")

    report = analyze(
        clf,
        X_val,
        y_val,
        y_prob=y_prob,
        embeddings=embeddings,
        sensitive_features={
            "gender": gender,
            "age_group": age_grp,
        },
        verbose=True,
    )

    # -----------------------------
    # 4. Visualize
    # -----------------------------
    print("\n[4/4] Report Summary:")
    report.show()

    print("\n" + "=" * 40)
    print("DEPLOYMENT RECOMMENDATION EXPLANATION")
    print("=" * 40)
    print(report.deployment_summary)
    print("=" * 40 + "\n")

    # Visualize fairness deep-dive
    print("\n[*] Generating fairness summary plot...")
    report.plot_bias(mode="summary", save_path="examples/output/quickstart_bias.png")

    # -----------------------------
    # 5. Output
    # -----------------------------
    out_dir = report.save("examples/output")
    print(f"\nReport artifacts saved to: {out_dir}")
    print("  - Metrics: report.json, metadata.json")
    print("  - Plots: calibration_plot.png, failure_plot.png, quickstart_bias.png")


if __name__ == "__main__":
    print("Running TrustLens quickstart demo...\n")
    main()
