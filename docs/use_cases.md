# Real-World Use Cases

TrustLens is applied across industries to ensure model safety and governance.

---

## 1. Safety-Critical Selection (Medical AI)
**Scenario**: Choosing between a 94% accurate "Black Box" model and a 92% accurate well-calibrated model.
- **TrustLens Output**: Identifies that the 94% model has high Expected Calibration Error (ECE), making its probability scores unreliable for triage.
- **Verdict**: Recommend the 92% model due to superior calibration and reliability.

## 2. Model Governance & Compliance
**Scenario**: Auditing financial models for bias before submission to regulators.
- **TrustLens Output**: Automatically identifies performance gaps in `Equalized Odds` for sensitive protected attributes.
- **Verdict**: Flag model as "Blocked" due to fairness violations, providing clear evidence for retraining.

## 3. High-Throughput Maintenance
**Scenario**: Monitoring if a deployed model is starting to "drift" or become overconfident on new data.
- **TrustLens Output**: Detects a shrinking `Confidence Gap` over time, indicating the model is becoming less reliable at separating correct from incorrect predictions.
- **Verdict**: Trigger a manual review of the last 1000 samples.

## 4. Head-to-Head Comparison
**Scenario**: Testing three different candidate models for a production vacancy.
- **TrustLens Output**: A ranked comparison dashboard showing which model has the cleanest aggregate Trust Score.
- **Verdict**: Select the model with the lowest penalty burden, even if its raw accuracy is marginally lower than competitors.
