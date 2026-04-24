"""
examples/custom_plugin_demo.py.
================================
Demonstrate how to write and register a custom TrustLens plugin.

This plugin computes the "Negative Predictive Value Gap" — NVP_gap —
which measures how much worse a model's NPV is for the minority class.

Run with:
  python examples/custom_plugin_demo.py
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from trustlens import analyze
from trustlens.plugins.base import BasePlugin
from trustlens.plugins.registry import PluginRegistry

# ---------------------------------------------------------------------------
# Define the custom plugin
# ---------------------------------------------------------------------------


class NPVGapPlugin(BasePlugin):
    """
    Negative Predictive Value (NPV) gap between majority and minority class.

    NPV = TN / (TN + FN)

    A large gap indicates the model under-predicts the minority class,
    which is common in imbalanced datasets.
    """

    name = "npv_gap"
    description = "Computes NPV per class and reports the inter-class gap."
    version = "0.2.0"

    def run(self, model, X, y_true, y_pred, y_prob, **kwargs):

        classes = np.unique(y_true)
        npv_per_class = {}

        for cls in classes:
            # Binary: cls = positive, everything else = negative
            binary_true = (y_true == cls).astype(int)
            binary_pred = (y_pred == cls).astype(int)

            tn = int(((binary_pred == 0) & (binary_true == 0)).sum())
            fn = int(((binary_pred == 0) & (binary_true == 1)).sum())
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            npv_per_class[int(cls)] = round(npv, 4)

        if len(classes) >= 2:
            npv_values = list(npv_per_class.values())
            gap = round(max(npv_values) - min(npv_values), 4)
        else:
            gap = 0.0

        return {
            "npv_per_class": npv_per_class,
            "npv_gap": gap,
        }


# ---------------------------------------------------------------------------
# Register and run
# ---------------------------------------------------------------------------

registry = PluginRegistry()
registry.register(NPVGapPlugin)

print("Registered plugins:", registry.list_plugins())

# Train a model
X, y = make_classification(n_samples=800, random_state=7, weights=[0.8, 0.2])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=7)

clf = GradientBoostingClassifier(n_estimators=50, random_state=7)
clf.fit(X_train, y_train)

# Run with plugin
report = analyze(
    clf,
    X_val,
    y_val,
    plugins=["npv_gap"],
    verbose=True,
)

report.show()
print("\nPlugin result:", report.results["plugin_npv_gap"])
