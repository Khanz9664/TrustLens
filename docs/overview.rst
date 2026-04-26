Why Accuracy Isn’t Enough
=========================

You trained a model. It hits **92% accuracy** on your validation set. You ship it.

Three months later:

* A minority-class user gets consistently wrong predictions
* The model is **90% confident on its worst mistakes**
* A regulator asks *"why did it make that decision?"* — and you have no answer

Sound familiar? You're not alone.

--------------------------------------------------------------------------------

**Accuracy tells you how often your model is right.**

**It tells you nothing about *when* it fails, *why* it fails, or *who* it fails.**

Traditional metrics hide the very risks that matter most in production.

--------------------------------------------------------------------------------

Why standard metrics fall short
-------------------------------

Most ML pipelines rely on Accuracy, F1, or RMSE. While useful, these metrics are aggregate scores that mask critical failure modes:

* **Miscalibration**
  A model says "I'm 99% confident" but is only correct 60% of the time.

* **Silent Bias**
  High overall accuracy hides poor performance on minority groups.

* **Unstable Decision Boundaries**
  Small changes in input lead to inconsistent or unreliable predictions.

--------------------------------------------------------------------------------

**Traditional metrics tell you how the model performs.**

**They don’t tell you if the model is safe to deploy.**

--------------------------------------------------------------------------------

The TrustLens Approach
----------------------

TrustLens makes these hidden risks visible — before they reach production.

It goes beyond aggregate metrics to analyze:

* How confident your model is when it fails.
* How errors are distributed across subgroups.
* Whether predictions can be trusted in high-stakes scenarios.

TrustLens transforms raw metrics into **actionable deployment decisions**, giving you the evidence needed to approve — or block — a model for production.

--------------------------------------------------------------------------------

Learn how these issues are measured in :doc:`features`.
