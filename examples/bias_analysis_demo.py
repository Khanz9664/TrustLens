"""
Bias Analysis Demo — TrustLens
Shows how to detect subgroup performance gaps using sensitive attributes.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from trustlens import analyze


def main():
    # -----------------------------
    # 1. Load data
    # -----------------------------
    np.random.seed(42)
    n_samples = 500

    X, y = make_classification(n_samples=n_samples, n_features=5, random_state=42)

    # Add sensitive attributes (not used in training, only for bias analysis)
    gender = np.random.choice(["male", "female"], size=n_samples)
    age_group = np.random.choice(["young", "middle", "senior"], size=n_samples)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Split sensitive attributes the same way
    indices = np.arange(n_samples)
    _, test_idx = train_test_split(indices, test_size=0.3, random_state=42)

    gender_test = gender[test_idx]
    age_group_test = age_group[test_idx]

    # -----------------------------
    # 2. Train model
    # -----------------------------
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Get predicted probabilities
    y_prob = model.predict_proba(X_test)

    # -----------------------------
    # 3. Analyze
    # -----------------------------
    print("[*] Running TrustLens analysis with subgroup diagnostics...\n")

    report = analyze(
        model,
        X_test,
        y_test,
        y_prob=y_prob,
        sensitive_features={"gender": gender_test, "age_group": age_group_test},
    )

    # -----------------------------
    # 4. Visualize
    # -----------------------------
    print(f"Trust Score: {report.trust_score.score:.1f} / 100")
    print("(Bias accounts for 25% of this score — subgroup gaps lower it)\n")

    # Show the text report
    report.show()

    print("\n[*] Generating advanced fairness plots...")

    # Mode "all" generates subgroup performance, equalized odds, and fairness gap plots
    # It returns a dictionary of Figures and saves them with suffixes if save_path is provided
    plots = report.plot_bias(mode="all", save_path="examples/output/bias_audit")

    print(f"    ✔ Generated plots: {list(plots.keys())}")
    print("    ✔ Saved to examples/output/bias_audit_subgroup.png, etc.")

    # -----------------------------
    # 5. Output
    # -----------------------------
    # (Results are already displayed/saved above)
    print("\nBias analysis demo completed.")


if __name__ == "__main__":
    print("Running TrustLens bias analysis demo...\n")
    main()
