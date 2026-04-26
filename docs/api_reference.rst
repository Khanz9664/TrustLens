API Reference
=============

This section provides detailed documentation for the TrustLens API, categorized by the layer of the decision-support pipeline.

Top-level API
-------------

The primary entry points for orchestrating model audits.

.. autofunction:: trustlens.api.analyze

.. autofunction:: trustlens.api.quick_analyze

Reports and Decision Results
----------------------------

Data containers for analysis results and decision logic.

.. autoclass:: trustlens.report.TrustReport
   :members:
   :show-inheritance:

.. autoclass:: trustlens.trust_score.TrustScoreResult
   :members:
   :show-inheritance:

Metrics Modules
---------------

Specialized modules for computing diagnostic signals.

.. toctree::
   :maxdepth: 1

   metrics/calibration
   metrics/failure
   metrics/bias
   metrics/representation

Comparative Engine
------------------

.. autofunction:: trustlens.comparison.compare
