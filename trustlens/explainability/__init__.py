# NOTE:
# This module is under active development and is not part of the public API.
# Do not import into production pipelines until stabilized.
# It is NOT used by analyze(), quick_analyze(), or TrustReport.

"""
trustlens.explainability.
========================
Explainability sub-package (experimental).

Provides gradient-based visual attribution methods for deep models,
and faithfulness evaluation tests for any explanation method.

These features require PyTorch and are intended for advanced users
working with deep learning models. They are not part of the core
ML evaluation pipeline.

Modules
-------
* ``gradcam``   — Gradient-weighted Class Activation Maps (Grad-CAM)
* ``faithfulness`` — Pixel deletion/insertion tests for faithfulness
"""

from trustlens.explainability.faithfulness import (
    pixel_deletion_test,
    pixel_insertion_test,
)
from trustlens.explainability.gradcam import GradCAM

__all__ = [
    "GradCAM",
    "pixel_deletion_test",
    "pixel_insertion_test",
]
