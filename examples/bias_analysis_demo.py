"""
Bias Analysis Demo — TrustLens
Shows how to detect subgroup performance gaps using sensitive attributes.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from trustlens import analyze

# --- 1. Generate a synthetic dataset ---
np.random.seed(42)
n_samples = 500

X, y = make_classification(n_samples=n_samples, n_features=5, random_state=42)

# Add sensitive attributes (not used in training, only for bias analysis)
gender = np.random.choice(["male", "female"], size=n_samples)
age_group = np.random.choice(["young", "middle", "senior"], size=n_samples)

# --- 2. Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Split sensitive attributes the same way
indices = np.arange(n_samples)
_, test_idx = train_test_split(indices, test_size=0.3, random_state=42)

gender_test = gender[test_idx]
age_group_test = age_group[test_idx]

# --- 3. Train a simple model ---
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Get predicted probabilities for the positive class
y_prob = model.predict_proba(X_test)[:, 1]

# --- 4. Run TrustLens bias analysis ---
# sensitive_features tells TrustLens which groups to compare
print("Running TrustLens analysis with subgroup diagnostics...\n")

report = analyze(
    model,
    X_test,
    y_test,
    y_prob=y_prob,
    sensitive_features={"gender": gender_test, "age_group": age_group_test},
)

# --- 5. Display results ---
# Trust Score: 0-100, higher = more trustworthy
print(f"Trust Score: {report.trust_score:.1f} / 100")
print("(Bias accounts for 25% of this score — subgroup gaps lower it)\n")

# Show the full report including bias breakdown per group
report.show()
