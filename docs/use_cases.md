# Real-World Use Cases

TrustLens is being applied across domains to ensure ML reliability where it matters most.

### Medical AI
Identify overconfidence in edge cases before a diagnostic model reaches a patient. TrustLens flags high ECE (>0.15) early, suggesting where a human-in-the-loop is most needed.

### Fraud Detection
Quantify your false-negative problem. If your confidence gap is low, your model is equally confident on the fraud it catches and the fraud it misses — a clear signal that the decision threshold needs tuning.

### Hiring & Lending
Automated subgroup analysis reveals performance gaps across demographics (gender, age, ethnicity) before they become regulatory liabilities or ethical failures.

### Manufacturing & Quality Control
Monitor representation drift. By analyzing CKA (Centered Kernel Alignment) between training and production embeddings, teams can detect when a model's understanding of "defective" is shifting.

### Enterprise MLOps
Connect TrustLens to your existing stack:
- **MLflow / W&B**: Log the Trust Score as a primary metric to track reliability decay over time.
- **CI/CD**: Block deployments if the Bias Score drops below a certain threshold.
