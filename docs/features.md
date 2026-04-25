# Features & Output

TrustLens goes beyond pass/fail — it explains why your model should or shouldn't be trusted. It provides a deep dive into the four dimensions of model trust.

## The Trust Score

A single, actionable number: **0 to 100.** Computed from a **weighted combination (configurable, default ~40/40/20)** of three core dimensions with automatic **diagnostic penalties** for critical risks:

| Dimension | What it measures |
|---|---|
| **Calibration** | Do probabilities reflect reality? |
| **Failure** | Does confidence correlate with accuracy? |
| **Bias** | Are all groups treated equally? |


The final score is: `(Base Score) - (Critical Penalties)`.


---

## Core Modules

*Learn how these modules are structured internally in the [Architecture Guide](architecture.md).*

### 1. Calibration Analysis
Uncover if your model is overconfident or underconfident.
* **Metrics**: Brier Score, Expected Calibration Error (ECE).
* **Pattern Detection**: Detects **Calibration Drift** — when reliability of predicted probabilities is low.

### 2. Failure Analysis
Identify "confidence-weighted errors" — mistakes made when the model is certain.
* **Interpretation**: Automatically distinguishes between "Safe Failures" and "Confidently Wrong" patterns.
* **Insights**: Provides concrete analysis of confidence ranges where errors are concentrated.

### 3. Bias Detection
Surface performance gaps across subgroups using **Bias Margins**.
* **Metrics**: Explicitly calculates distance from the parity threshold (0.10 limit).
* **Integration**: Powered by `equalized_odds` for rigorous fairness auditing.

### 4. Model Comparison
The **Comparison Engine** recommend candidates for deployment.
* **Advantage**: Automatically ties penalty differences back to root causes (e.g., "higher due to poor calibration").
* **API**: `trustlens.comparison.compare(reports)` for head-to-head evaluation.


---

## Explainability Layer

TrustLens is designed to justify its verdicts:
- **Ranked Explanation**: Reports the top contributors to score deductions (Dominant → Secondary).
- **Verdicts & Blocks**: Decisive assessment strings (e.g., "Blocked by high diagnostic risk") mapped to score tiers.
- **Pattern Flags**: High-level semantic signals used for diagnostic interpretation:
  - **Calibration Drift**: Probabilities are unreliable (high ECE).
  - **Confidently Wrong**: High-confidence incorrect predictions.


## Output Formats
- **Interactive**: Rich Jupyter representations via `_repr_html_`.
- **Narrative**: High-density humans reports with structured methodology notes.
- **Structured**: JSON export for automated CI/CD gating (`report.save("report.json")`).
