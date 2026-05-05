import os
import shutil
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# TrustLens Imports
import trustlens
from trustlens import analyze, quick_analyze, compare
from trustlens.plugins.base import BasePlugin
from trustlens.plugins.registry import PluginRegistry

# IMPORTANT: Importing specific visualization functions to demonstrate "everything" 
# even when the high-level API focuses on summaries.
from trustlens.visualization import (
    plot_subgroup_performance_multi,
    plot_equalized_odds_multi,
    plot_fairness_gap_multi,
    plot_class_distribution
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# =============================================================================
# 1. SETUP & OUTPUT ORGANIZATION
# =============================================================================

BASE_OUTPUT = "trustlens_demo_output"

def setup_output_dirs():
    """Ensures a clean single-folder output structure."""
    if os.path.exists(BASE_OUTPUT):
        shutil.rmtree(BASE_OUTPUT)
    # Folders for model assets
    for m in ["model_a", "model_b", "model_c"]:
        os.makedirs(os.path.join(BASE_OUTPUT, f"visuals/{m}/bias_deep_dive"), exist_ok=True)
    os.makedirs(os.path.join(BASE_OUTPUT, "reports"), exist_ok=True)

# =============================================================================
# 2. DATA PREPARATION (Engineered for Bias and Failure)
# =============================================================================

def generate_demo_dataset(n_samples=2000):
    """Generates a rich dataset with multi-feature bias and imbalance."""
    print(f"Generating synthetic dataset with {n_samples} samples and multiple sensitive features...")
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=15,
        n_informative=10,
        weights=[0.7, 0.3],  # Class Imbalance
        flip_y=0.08,           # Calibration noise
        random_state=42
    )
    
    rng = np.random.default_rng(42)
    # Multiple Sensitive Features
    gender = rng.choice(["Male", "Female"], size=n_samples)
    region = rng.choice(["North", "South", "East", "West"], size=n_samples)
    education = rng.choice(["High School", "Bachelor", "Master", "PhD"], size=n_samples)
    
    # Inject synthetic bias: PhDs in the West have higher flipped labels (Model Challenge)
    bias_mask = (education == "PhD") & (region == "West")
    y[bias_mask] = 1 - y[bias_mask]
    
    # Embeddings for representation analysis
    embeddings = np.random.randn(n_samples, 16)
    embeddings[y == 1] += 0.5
    
    sensitive_features = {
        "gender": gender,
        "region": region,
        "education": education
    }
    
    return X, y, sensitive_features, embeddings

# =============================================================================
# 3. CUSTOM PLUGIN: "ConfidenceStabilityPlugin"
# =============================================================================

class ConfidenceStabilityPlugin(BasePlugin):
    """Custom metric to check if the model is overconfident on wrong predictions."""
    name = "conf_stability"
    description = "Measures average confidence on incorrect predictions."
    version = "1.0.0"

    def run(self, model, X, y_true, y_pred, y_prob, **kwargs):
        incorrect = y_true != y_pred
        if not any(incorrect):
            return {"avg_err_conf": 0.0, "status": "stable"}
        
        err_conf = np.max(y_prob[incorrect], axis=1).mean()
        return {
            "avg_err_conf": round(float(err_conf), 4),
            "status": "Warning" if err_conf > 0.8 else "Safe"
        }

# =============================================================================
# 4. MAIN DEMO ENGINE
# =============================================================================

def run_comprehensive_demo():
    setup_output_dirs()
    
    print("\n" + "="*80)
    print(f" TRUSTLENS COMPLETE FEATURE SHOWCASE (v{trustlens.__version__})")
    print("="*80 + "\n")
    
    # --- Step 1: Data Preparation ---
    X, y, sens, embeddings = generate_demo_dataset()
    X_train, X_test, y_train, y_test, emb_train, emb_test = train_test_split(
        X, y, embeddings, test_size=0.4, random_state=42
    )
    
    # Prep sensitive features for test set
    sens_test = {k: v[len(y_train):] for k, v in sens.items()}
    
    # --- Step 2: Model Training (3 Models for comparison) ---
    print("\nTraining models with different reliability profiles...")
    clf_a = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf_a.fit(X_train, y_train)
    
    clf_b = LogisticRegression(class_weight="balanced", random_state=42)
    clf_b.fit(X_train, y_train)
    
    clf_c = GradientBoostingClassifier(n_estimators=50, random_state=42)
    clf_c.fit(X_train, y_train)

    # --- Step 3: Register Plugin ---
    PluginRegistry().register(ConfidenceStabilityPlugin)

    # --- Step 4: Run TrustLens Analysis ---
    print("\nRunning Diagnostics...")
    reports = []
    configs = [(clf_a, "Model A", "model_a"), (clf_b, "Model B", "model_b"), (clf_c, "Model C", "model_c")]
    
    for model, label, folder in configs:
        print(f"  > Analyzing {label}...")
        report = analyze(
            model=model,
            X=X_test,
            y_true=y_test,
            embeddings=emb_test,
            sensitive_features=sens_test,
            plugins=["conf_stability"],
            verbose=False
        )
        report.metadata["model_class"] = label
        reports.append(report)
        
        # --- [FEATURE: EXPOSING EVERY BIAS PLOT] ---
        # Manually calling multi-feature plotters to demonstrate "EVERYTHING" 
        # as requested, bypassing any default tool-level summary logic.
        bias_results = report.results["bias"]
        deep_dive_path = os.path.join(BASE_OUTPUT, f"visuals/{folder}/bias_deep_dive")
        
        print(f"    - Generating Multi-Feature Subgroup Performance...")
        plot_subgroup_performance_multi(bias_results["subgroup_performance"], save_dir=deep_dive_path, show=False)
        
        if "equalized_odds" in bias_results:
            print(f"    - Generating Multi-Feature Equalized Odds...")
            plot_equalized_odds_multi(bias_results["equalized_odds"], save_dir=deep_dive_path, show=False)
            print(f"    - Generating Multi-Feature Fairness Gaps...")
            plot_fairness_gap_multi(bias_results["equalized_odds"], save_dir=deep_dive_path, show=False)
        
        # Save the report bundle
        report.save(f"{BASE_OUTPUT}/reports/{folder}_report/")
        # Save the master dashboard
        report.summary_plot(save_path=f"{BASE_OUTPUT}/visuals/{folder}/summary_dashboard.png", show=False)

    # --- Step 5: Side-by-Side Showdown Table ---
    print("\n" + "="*80)
    print(f" {'METRIC':<28} {'MODEL A':^15} {'MODEL B':^15} {'MODEL C':^15}")
    print("-" * 80)
    
    def get_acc(m): return (m.predict(X_test) == y_test).mean()
    print(f" {'Accuracy':<28} {get_acc(clf_a):^15.2%} {get_acc(clf_b):^15.2%} {get_acc(clf_c):^15.2%}")
    print(f" {'Trust Score':<28} {reports[0].trust_score.score:^15} {reports[1].trust_score.score:^15} {reports[2].trust_score.score:^15}")
    print(f" {'Trust Grade':<28} {reports[0].trust_score.grade:^15} {reports[1].trust_score.grade:^15} {reports[2].trust_score.grade:^15}")
    print("-" * 80)
    
    # Module-level breakdowns
    for dim in ["calibration", "failure", "bias"]:
        s_a = reports[0].trust_score.sub_scores.get(dim, 0)
        s_b = reports[1].trust_score.sub_scores.get(dim, 0)
        s_c = reports[2].trust_score.sub_scores.get(dim, 0)
        print(f" {dim.capitalize() + ' Score':<28} {s_a:^15.1f} {s_b:^15.1f} {s_c:^15.1f}")
    
    print("="*80)

    # --- Step 6: TrustLens Decision Support ---
    print("\n[ FEATURE: compare() ]")
    compare(reports)

    # --- Step 7: Failure Analysis ---
    print("\n[ FEATURE: show_failures() ]")
    print("Worst offenders in Model A:")
    reports[0].show_failures(top_k=3)

    # --- Step 8: Quick Analyze ---
    print("\n" + "-"*80)
    print("INSTANT AUDIT: quick_analyze()")
    print("-" * 80)
    quick_analyze(dataset="breast_cancer")

    print(f"\n✅ All features showcased. Results consolidated in: ./{BASE_OUTPUT}/")
    print(f"   Note: Check ./{BASE_OUTPUT}/visuals/model_a/bias_deep_dive/ for every single fairness plot.")

if __name__ == "__main__":
    run_comprehensive_demo()
