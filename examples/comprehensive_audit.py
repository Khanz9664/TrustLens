"""
Full TrustLens Audit (End-to-End Demo)

If you're new:
→ Start with: examples/quickstart.py
→ Then come back here for full capabilities

This script demonstrates:
- Calibration, failure, bias, and representation analysis
- Fairness visualization via plot_bias(mode="all")
- Exporting reports and artifacts

Expected outputs:
- bias_subgroup.png
- bias_equalized_odds.png
- bias_gap.png
- report.json
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from trustlens import analyze
from trustlens.plugins.base import BasePlugin
from trustlens.plugins.registry import PluginRegistry


class AccuracyByConfidencePlugin(BasePlugin):
    """Computes accuracy specifically for high-confidence predictions."""

    name = "high_conf_acc"
    description = "Accuracy for predictions with confidence > 0.9"

    def run(self, model, X, y_true, y_pred, y_prob, **kwargs):
        high_conf_mask = np.max(y_prob, axis=1) > 0.9
        if not np.any(high_conf_mask):
            return {"accuracy": None, "sample_count": 0}

        acc = (y_true[high_conf_mask] == y_pred[high_conf_mask]).mean()
        return {"accuracy": round(float(acc), 4), "sample_count": int(np.sum(high_conf_mask))}


def main():
    # -----------------------------
    # 1. Load data
    # -----------------------------
    print("[*] Generating synthetic dataset with embeddings and sensitive features...")
    n_samples = 1200
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.6, 0.4],
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Generate sensitive features
    rng = np.random.default_rng(42)
    region = rng.choice(["North", "South", "East", "West"], size=len(y_test))
    user_type = rng.choice(["Free", "Premium", "Enterprise"], size=len(y_test))

    # Simulate model embeddings (penultimate layer)
    # We make embeddings somewhat informative of the class
    embeddings = rng.standard_normal((len(y_test), 64))
    embeddings += y_test.reshape(-1, 1) * 1.2

    # -----------------------------
    # 2. Train model
    # -----------------------------
    print("[*] Training RandomForest model...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)

    # -----------------------------
    # 3. Analyze
    # -----------------------------
    # Register the plugin
    registry = PluginRegistry()
    registry.register(AccuracyByConfidencePlugin)

    print("\n[*] Running Comprehensive TrustLens Audit...")
    report = analyze(
        clf,
        X_test,
        y_test,
        y_prob=y_prob,
        embeddings=embeddings,
        sensitive_features={"region": region, "user_type": user_type},
        plugins=["high_conf_acc"],
        verbose=True,
    )

    # -----------------------------
    # 4. Visualize
    # -----------------------------
    print("\n[*] Displaying Audit Summary...")
    report.show()

    print("\n[*] Generating Multi-Mode Fairness Visualizations...")
    # Generate subgroup, equalized_odds, and gap plots in one call
    audit_plots = report.plot_bias(mode="all", save_path="comprehensive_audit_plots")
    print(f"    ✔ Batch plots generated: {list(audit_plots.keys())}")

    # -----------------------------
    # 5. Output
    # -----------------------------
    print("\n[*] Exporting Results...")
    # 1. Save as directory bundle (JSON metrics + Plots)
    bundle_path = report.save("examples/trustlens_audit_report")
    print(f"    ✔ Audit bundle saved to: {bundle_path}")

    # 2. Save as single JSON
    json_path = "examples/audit_results.json"
    report.save(json_path)
    print(f"    ✔ Raw metrics saved to: {json_path}")

    print("\n========== Audit Completed Successfully ==========")


if __name__ == "__main__":
    print("Running full TrustLens audit...\n")
    main()
