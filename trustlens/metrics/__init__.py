"""
trustlens.metrics.
=================
Metric sub-package. Each module is independently importable.
"""

from trustlens.metrics.bias import (
    class_imbalance_report,
    subgroup_performance,
)
from trustlens.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_curve,
)
from trustlens.metrics.failure import (
    confidence_gap,
    misclassification_summary,
)
from trustlens.metrics.regression import (
    error_distribution,
    error_variance_correlation,
    prediction_interval_coverage,
)
from trustlens.metrics.representation import (
    centered_kernel_alignment,
    embedding_separability,
)

__all__ = [
    "brier_score",
    "expected_calibration_error",
    "reliability_curve",
    "misclassification_summary",
    "confidence_gap",
    "error_distribution",
    "prediction_interval_coverage",
    "error_variance_correlation",
    "class_imbalance_report",
    "subgroup_performance",
    "embedding_separability",
    "centered_kernel_alignment",
]
