"""
TrustLens — Debug your ML models beyond accuracy.

TrustLens is a modular Python library for analyzing the reliability and
trustworthiness of machine learning models beyond standard metrics.

Core capabilities include:
- Calibration analysis (Brier Score, ECE, reliability diagrams)
- Failure analysis (misclassifications, confidence gaps)
- Bias detection (subgroup performance, class imbalance)
- Representation analysis (embedding geometry, CKA)
- Trust Score (0–100 composite reliability metric)
"""

__version__ = "0.2.0"

from .api import analyze, quick_analyze
from .comparison import compare
from .report import TrustReport
from .trust_score import TrustScoreResult, compute_trust_score

__all__ = [
    "analyze",
    "quick_analyze",
    "TrustReport",
    "compute_trust_score",
    "TrustScoreResult",
    "__version__",
    "compare",
]
