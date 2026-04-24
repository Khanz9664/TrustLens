"""
examples/model_comparison.py.
==============================
TrustLens Showdown: Gradient Boosting vs Logistic Regression

This script demonstrates the killer insight behind TrustLens:
Sometimes the more accurate model is the less trustworthy one.

Run with:
  python examples/model_comparison.py

Expected runtime: ~15 seconds
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import trustlens
from trustlens import analyze

# ---------------------------------------------------------------------------
print(f"\n{'-' * 66}")
print(f" TrustLens v{trustlens.__version__} - Model Trustworthiness Showdown")
print(f"{'-' * 66}\n")
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
print("[1/5] Generating dataset...")

X, y = make_classification(
    n_samples=2_000,
    n_features=20,
    n_informative=10,
    n_redundant=4,
    n_classes=2,
    weights=[0.70, 0.30],
    flip_y=0.06,
    class_sep=0.9,
    random_state=42,
)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.35, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

rng = np.random.default_rng(7)
gender_val = rng.choice(["M", "F"], size=len(y_val))
age_grp_val = rng.choice(["18-35", "36-55", "55+"], size=len(y_val))

print(
    f"  Train: {len(X_train):,} | Val: {len(X_val):,} | "
    f"Imbalance: {(y_val == 0).sum()}/{(y_val == 1).sum()} (neg/pos)"
)

# ---------------------------------------------------------------------------
# 2. Train Models
# ---------------------------------------------------------------------------
print("\n[2/5] Training models...")

clf_a = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.12,
    random_state=42,
)
clf_a.fit(X_train, y_train)
prob_a = clf_a.predict_proba(X_val)
acc_a = (clf_a.predict(X_val) == y_val).mean()

clf_b = LogisticRegression(
    C=0.8,
    max_iter=2000,
    random_state=42,
    class_weight="balanced",
)
clf_b.fit(X_train, y_train)
prob_b = clf_b.predict_proba(X_val)
acc_b = (clf_b.predict(X_val) == y_val).mean()

print("\nModel Details:")
print(f"Model A (GradientBoosting)\n Accuracy: {acc_a:.2%}")
print(f"\nModel B (LogisticRegression)\n Accuracy: {acc_b:.2%}")

# ---------------------------------------------------------------------------
# 3. TrustLens Analysis
# ---------------------------------------------------------------------------
print("\n[3/5] Running TrustLens analysis...\n")

report_a = analyze(
    clf_a,
    X_val,
    y_val,
    y_prob=prob_a,
    sensitive_features={"gender": gender_val, "age_group": age_grp_val},
    verbose=False,
)

report_b = analyze(
    clf_b,
    X_val,
    y_val,
    y_prob=prob_b,
    sensitive_features={"gender": gender_val, "age_group": age_grp_val},
    verbose=False,
)

# ---------------------------------------------------------------------------
# 4. Results
# ---------------------------------------------------------------------------
print("[4/5] Results - TrustLens Showdown\n")

_div = "=" * 66
_bar = "-" * 66

print(_div)
print(f" {'METRIC':<34} {'MODEL A':^13} {'MODEL B':^13}")
print(_bar)

print(f" {' Accuracy':<34} {acc_a:^13.2%} {acc_b:^13.2%}")

ts_a = report_a.trust_score
ts_b = report_b.trust_score
print(f" {' Trust Score':<34} {ts_a.score:^13}/100 {ts_b.score:^13}/100")
print(f" {' Grade':<34} {ts_a.grade:^13} {ts_b.grade:^13}")

print(_bar)

dims = sorted(set(ts_a.sub_scores) | set(ts_b.sub_scores))
for dim in dims:
    sa = ts_a.sub_scores.get(dim, float("nan"))
    sb = ts_b.sub_scores.get(dim, float("nan"))
    print(f" {' ' + dim.capitalize() + ' Score':<34} {sa:^13.1f} {sb:^13.1f}")

print(_bar)

bs_a = report_a.results["calibration"]["brier_score"]
ece_a = report_a.results["calibration"]["ece"]
bs_b = report_b.results["calibration"]["brier_score"]
ece_b = report_b.results["calibration"]["ece"]
print(f" {' Brier Score':<34} {bs_a:^13.4f} {bs_b:^13.4f}")
print(f" {' ECE':<34} {ece_a:^13.4f} {ece_b:^13.4f}")

gap_a = report_a.results["failure"]["confidence_gap"]["gap"]
gap_b = report_b.results["failure"]["confidence_gap"]["gap"]
print(f" {' Confidence Gap':<34} {gap_a:^13.4f} {gap_b:^13.4f}")

ratio_a = report_a.results["bias"]["class_imbalance"]["imbalance_ratio"]
ratio_b = report_b.results["bias"]["class_imbalance"]["imbalance_ratio"]
print(f" {' Imbalance Ratio':<34} {ratio_a:^13.2f} {ratio_b:^13.2f}")


def _sg_gap(report, feat="gender"):
    try:
        sg = report.results["bias"]["subgroup_performance"][feat]
        return sg["__summary__"]["performance_gap"]
    except Exception:
        return float("nan")


sg_a = _sg_gap(report_a, "gender")
sg_b = _sg_gap(report_b, "gender")
print(f" {' Gender Perf. Gap':<34} {sg_a:^13.4f} {sg_b:^13.4f}")

print(_div)

# ---------------------------------------------------------------------------
# 5. Top Failures
# ---------------------------------------------------------------------------
print("\n[5/5] Critical Failures (top 3)\n")

print(f"--- MODEL A: {clf_a.__class__.__name__} ---")
report_a.show_failures(top_k=3)

print(f"--- MODEL B: {clf_b.__class__.__name__} ---")
report_b.show_failures(top_k=3)

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------
print(_div)
print(" VERDICT")
print(_bar)
print("Model A:")
print(f" Accuracy: {acc_a:.0%}")
print(f" Trust Score: {ts_a.score}")
print("\nModel B:")
print(f" Accuracy: {acc_b:.0%}")
print(f" Trust Score: {ts_b.score}")
print("\nConclusion:")
if ts_b.score > ts_a.score:
    print(
        "Model B is more reliable despite lower accuracy due to better calibration and lower confidence errors."
    )
elif ts_a.score > ts_b.score:
    print("Model A is superior across both accuracy and overall trustworthiness.")
else:
    print("Both models possess matching Trust Scores. Evaluate by specific use-case needs.")

print(_div)

# ---------------------------------------------------------------------------
# Save summary plots
# ---------------------------------------------------------------------------
import os

os.makedirs("examples/output", exist_ok=True)

report_a.summary_plot(
    save_path="examples/output/model_a_summary.png",
    show=False,
)
report_b.summary_plot(
    save_path="examples/output/model_b_summary.png",
    show=False,
)
print("\n Summary dashboards saved:")
print("  examples/output/model_a_summary.png")
print("  examples/output/model_b_summary.png")
print()
