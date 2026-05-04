"""
Global test configuration for TrustLens.

Sets matplotlib to the non-interactive Agg backend before any test
imports visualization modules. This prevents GUI windows from opening
during CI and headless environments.
"""

import matplotlib

matplotlib.use("Agg")
