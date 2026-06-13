import json
import os

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from trustlens import analyze


def compare_dicts(actual, expected, path=""):
    """Recursively compare dictionaries with tolerance for floats."""
    if type(actual) is not type(expected):
        pytest.fail(f"Type mismatch at {path}: actual={type(actual)}, expected={type(expected)}")

    if isinstance(actual, dict):
        for k in expected:
            if k not in actual:
                pytest.fail(f"Missing key at {path}: {k}")
            compare_dicts(actual[k], expected[k], path=f"{path}.{k}" if path else k)
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            pytest.fail(
                f"List length mismatch at {path}: actual={len(actual)}, expected={len(expected)}"
            )
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare_dicts(a, e, path=f"{path}[{i}]")
    elif isinstance(actual, float):
        assert actual == pytest.approx(expected, abs=1e-5), (
            f"Value mismatch at {path}: actual={actual}, expected={expected}"
        )
    else:
        assert actual == expected, f"Value mismatch at {path}: actual={actual}, expected={expected}"


@pytest.fixture
def baselines():
    baseline_path = os.path.join(os.path.dirname(__file__), "baselines.json")
    if not os.path.exists(baseline_path):
        pytest.skip("Baselines not found. Run generate_baselines.py first.")
    with open(baseline_path) as f:
        return json.load(f)


@pytest.mark.parametrize("model_name", ["rf", "lr"])
def test_analyze_parity(baselines, model_name):
    expected = baselines[model_name]

    # Reproduce exactly the same data as generate_baselines.py
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sensitive features (re-seeding for reproducibility)
    np.random.seed(42)
    sensitive_features = {
        "gender": np.random.choice(["Male", "Female"], size=len(y_test)),
        "age": np.random.choice(["Young", "Old"], size=len(y_test)),
    }

    # Embeddings
    np.random.seed(42)
    embeddings = np.random.randn(len(y_test), 8)

    if model_name == "rf":
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
    else:
        clf = LogisticRegression(random_state=42)

    clf.fit(X_train, y_train)

    report = analyze(
        model=clf,
        X=X_test,
        y_true=y_test,
        embeddings=embeddings,
        sensitive_features=sensitive_features,
        verbose=False,
    )

    actual = {
        "results": report.results,
        "trust_score": {
            "score": report.trust_score.score,
            "grade": report.trust_score.grade,
            "sub_scores": report.trust_score.sub_scores,
            "penalties": report.trust_score.penalties_applied,
        },
        "metadata": {
            "n_samples": report.metadata["n_samples"],
            "n_classes": report.metadata["n_classes"],
        },
    }

    # Convert actual to JSON-compatible format for comparison
    # (Handling numpy arrays in results)
    import json

    from tests.characterization.generate_baselines import NpEncoder

    actual_json = json.loads(json.dumps(actual, cls=NpEncoder))

    compare_dicts(actual_json, expected)
