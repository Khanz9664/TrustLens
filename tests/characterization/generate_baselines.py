import json

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from trustlens import analyze


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def generate_baselines():
    print("Generating characterization baselines...")

    # Setup data
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sensitive features
    np.random.seed(42)
    sensitive_features = {
        "gender": np.random.choice(["Male", "Female"], size=len(y_test)),
        "age": np.random.choice(["Young", "Old"], size=len(y_test)),
    }

    # Embeddings
    np.random.seed(42)
    embeddings = np.random.randn(len(y_test), 8)

    models = {
        "rf": RandomForestClassifier(n_estimators=10, random_state=42),
        "lr": LogisticRegression(random_state=42),
    }

    baselines = {}

    for name, clf in models.items():
        print(f"  Analyzing {name}...")
        clf.fit(X_train, y_train)

        report = analyze(
            model=clf,
            X=X_test,
            y_true=y_test,
            embeddings=embeddings,
            sensitive_features=sensitive_features,
            verbose=False,
        )

        # Capture relevant parts of the report
        baselines[name] = {
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

    # Save to file
    output_path = "tests/characterization/baselines.json"
    with open(output_path, "w") as f:
        json.dump(baselines, f, indent=2, cls=NpEncoder)

    print(f"Baselines saved to {output_path}")


if __name__ == "__main__":
    generate_baselines()
