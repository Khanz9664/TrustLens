# The Problem Nobody Ships Around

You trained a model. It hits **92% accuracy** on your validation set. You ship it.

Three months later:

* A minority-class user gets consistently wrong predictions.
* The model is **90% confident on its worst mistakes**.
* A regulator asks *"why did it make that decision?"* — and you have no answer.

Sound familiar? You're not alone.

**Accuracy tells you how often your model is right.**

**It tells you nothing about *when* it fails, *why* it fails, or *who* it fails.**

TrustLens makes those failures visible — before they reach production. Beyond standard metrics, machine learning practitioners need to understand the "certainty of failure" and the distribution of errors across subgroups.

### Why standard metrics fall short
Most ML pipelines rely on Accuracy, F1, or RMSE. While useful, these metrics are aggregate scores that hide systematic flaws:
- **Miscalibration**: A model saying "I'm 99% sure" when it's only right 60% of the time.
- **Silent Bias**: High overall accuracy that masks significant performance drops for minority classes.
- **Representation Fragility**: Latent spaces where classes are so closely packed that slight noise causes classification flips.

**Traditional metrics tell you how the model performs, but they don't tell you if the model is safe to deploy.**

TrustLens bridges the gap between **raw metrics and deployment decisions**. It transforms aggregate diagnostics into explainable narratives, providing machine learning practitioners with the evidence needed to approve (or block) a model for production.

Learn how these issues are measured in [Features & Modules](features.md).
