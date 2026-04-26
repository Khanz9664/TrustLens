# Features & Modules

TrustLens provides a suite of specialized modules, each targeting a specific dimension of model reliability.

---

## 1. Calibration Module
*Ensure probabilities mean what they say.*

Calibration measures the reliability of a model's predicted probabilities. A well-calibrated model that predicts 70% confidence for a set of samples should be correct for approximately 70% of them.

- **ECE (Expected Calibration Error)**: Quantifies the average gap between confidence and accuracy across confidence bins.
- **Brier Score**: Measures the mean squared difference between predicted probabilities and actual outcomes.
- **Reliability Curves**: Visual diagnostics to identify overconfidence or underconfidence.

---

## 2. Failure Analysis Module
*Identify the "Confidently Wrong" patterns.*

High-accuracy models often hide dangerous failure modes. This module drills down into the errors.

- **Confidence Gap**: The difference between the model's confidence on correct vs. incorrect predictions. A narrow gap signals high diagnostic risk.
- **High-Confidence Failures**: Automatic flagging of samples where the model was >90% sure but wrong.
- **Severity Ranking**: Categorizes errors to prioritize manual review.

---

## 3. Bias & Fairness Module
*Detect performance disparities across subgroups.*

Bias detection reveals performance gaps across sensitive subgroups, helping ensure models are equitable before deployment.

- **Subgroup Disparity**: Compares Accuracy, Precision, and Recall across protected attributes (e.g., gender, age).
- **Equalized Odds**: Checks for parity in False Positive Rates (FPR) and True Positive Rates (TPR) across groups.
- **Fairness Margin**: A distance-to-limit metric that triggers penalties if disparities exceed acceptable thresholds.

---

## 4. Representation Module
*Audit data integrity and out-of-distribution risks.*

Leverages embeddings to understand if the model is operating on "familiar" ground.

- **Silhouette Scores**: Measures how well-defined the model's feature space is.
- **OOD Detection**: (Experimental) Identifies samples that fall far from the training representation cluster.

---

## 5. Trust Scoring Engine
*The unified decision-support signal.*

Aggregates all modules into a single **Trust Score (0-100)** and a **Grade (A-F)**.

- **Weights**: Configurable importance for each module.
- **Penalties**: Dynamic deductions based on risk severity.
- **Blockers**: Automatic "Blocked" verdict if critical thresholds are violated (e.g., severe bias or extreme overconfidence).
