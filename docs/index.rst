TrustLens: The ML Decision Support System
==========================================

TrustLens is a production-grade **decision-support system** designed to transform raw model outputs into actionable deployment decisions.

Beyond standard accuracy, TrustLens provides a deep, multi-dimensional audit of your model's reliability, safety, and fairness. It bridges the gap between raw metrics and the technical evidence needed to approve (or block) a model for production.

Why TrustLens?
--------------

Traditional metrics like Accuracy or F1 hide critical failure modes. TrustLens makes them visible through three core pillars of trust:

* **Calibration**: Do your model's probabilities reflect reality? (ECE, Brier Score)
* **Failure Analysis**: Does confidence correlate with correctness? (Confidence Gap)
* **Bias Detection**: Are performance gaps hidden within subgroups? (Equalized Odds)

The result is a **Trust Score**—a single, explainable metric that provides a definitive deployment **Verdict**.

--------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   overview
   architecture
   features
   use_cases
   api_reference

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide:

   EXPERIMENTAL

--------------------------------------------------------------------------------

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
