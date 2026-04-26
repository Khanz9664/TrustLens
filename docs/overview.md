# Overview

TrustLens is a unified suite for auditing machine learning models, focusing on the gap between "standard accuracy" and "production safety."

## The Trust Gap

Accuracy is often a poor proxy for real-world reliability. A model can have 95% accuracy while still being:
- **Unreliable**: Predicting 99% confidence on its mistakes (Overconfidence).
- **Biased**: Discriminating against sensitive subgroups.
- **Fragile**: Failing catastrophically on edge cases that it should handle.

TrustLens bridges this "Trust Gap" by providing technical evidence to support a **Deployment Verdict**.

## Core Pillars

TrustLens evaluates models across four integrated dimensions:

### 1. Calibration (Reliability)
Does a 0.8 probability mean 80% accuracy? We use **Expected Calibration Error (ECE)** and **Brier Scores** to ensure your model's confidence reflects reality.

### 2. Failure Analysis (Diagnostic Risk)
Where does your model fail? We identify high-confidence errors and calculate the **Confidence Gap** to quantify how "confidently wrong" your model is.

### 3. Bias & Fairness (Equity)
Does the model perform equally for everyone? We support **Subgroup Analysis** and **Equalized Odds** to detect and flag hidden disparities.

### 4. Representation (Data Integrity)
Is the model seeing "familiar" data? We use **Silhouette Scores** and representation analysis to detect if the training distribution is still relevant to the current inputs.

## High-Level Workflow

1. **Audit**: Run `analyze(model, X, y)`.
2. **Score**: Review the **Trust Score** (aggregating results and penalties).
3. **Decide**: Use the **Verdict** (Deploy, Partial, or Blocked) to inform your CI/CD or governance gates.
