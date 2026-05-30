# Why TrustLens?

A common question from ML engineers is: *"Why should I use TrustLens instead of standard scikit-learn metrics like Accuracy, ROC-AUC, or Brier Score?"*

The short answer: **Accuracy measures if a model is mathematically correct; TrustLens is designed to support the assessment of whether that model is safe to deploy.**

> [!NOTE]
> **Evidence Traceability:** The examples on this page are illustrative, but directly mirror the empirical failure patterns discovered in the `trustlens_model_zoo_benchmark.ipynb` when evaluating Random Forests under severe data imbalance.

## The Measurement Gap

Traditional metrics optimize for aggregate predictive power. They are mathematically blind to how errors are distributed across confidence thresholds, subgroups, and latent spaces.

| Question | Traditional Metric (e.g. `sklearn`) | TrustLens |
| :--- | :---: | :---: |
| **Accuracy** | ✔ | ✔ |
| **Calibration (Prob. correctness)** | ✖ | ✔ |
| **Fairness & Subgroup Parity** | ✖ | ✔ |
| **Overconfidence Detection** | ✖ | ✔ |
| **Representation Quality** | ✖ | ✔ |
| **Deployment Readiness Verdict** | ✖ | ✔ |

*Note: While you can manually compute calibration or fairness using disparate libraries, TrustLens unifies them into a single, penalized gating metric.*

## A Tangible Example: The 97% Accuracy Trap

Consider a concrete scenario observed during the TrustLens Model Zoo Benchmark (Random Forest on Severely Imbalanced Data).

**Model A Performance:**
- **Accuracy:** 97.2%
- **Traditional Verdict:** Deploy immediately.

**Model A TrustLens Evaluation:**
- **Trust Score:** 50.0
- **Grade:** D
- **TrustLens Verdict:** DO NOT DEPLOY.

### Why did TrustLens block a 97% accurate model?
When we examine the TrustLens sub-scores, the hidden risks become apparent:
1. **Calibration Collapse:** The model was highly overconfident on the minority class. (Triggering a heavy calibration penalty).
2. **Confident Failures:** When the model made an error on the minority class, it predicted the wrong class with near 1.0 probability. (Triggering a failure penalty).
3. **Subgroup Disparity:** The 97% accuracy was achieved by correctly classifying 100% of the majority class while silently misclassifying 45% of the minority class.

## The Verdict

If you use Accuracy or ROC-AUC as your deployment gate, you will deploy Model A, resulting in silent failures and demographic bias in production.

If you use TrustLens, Model A receives a failing verdict, providing the engineering team with diagnostic signals to address the underlying data imbalance before release.
