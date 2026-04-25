# Features & Output

TrustLens goes beyond pass/fail — it explains why your model should or shouldn't be trusted. It provides a deep dive into the four dimensions of model trust.

## The Trust Score

A single, actionable number: **0 to 100.** Computed from four independently interpretable dimensions:

| Dimension | What it measures | Weight |
|---|---|---|
| **Calibration** | Do probabilities reflect reality? | 35% |
| **Failure** | Does confidence correlate with accuracy? | 30% |
| **Bias** | Are all groups treated equally? | 25% |
| **Representation** | Is the embedding space well-structured? | 10% |

Weights are empirically chosen based on common production failure modes and will be configurable in future releases.

---

## Core Modules

*Learn how these modules are structured internally in the [Architecture Guide](architecture.md).*

### 1. Calibration Analysis
Uncover if your model is overconfident or underconfident.
* **Metrics**: Brier Score, Expected Calibration Error (ECE).
* **Visuals**: Reliability Diagrams, Calibration Histograms.

### 2. Failure Analysis
Identify the "silent killers" of production ML — high-confidence mistakes.
* **Method**: `report.show_failures(top_k=5)`
* **Output**: A ranked list of samples where the model was certain it was right, but was actually wrong.

### 3. Bias Detection
Surface performance gaps across subgroups before they become liabilities.
* **Metrics**: Class imbalance reports, subgroup accuracy/F1 breakdown.
* **Input**: Simply pass a dictionary of `sensitive_features`.

### 4. Representation Analysis
Monitor how your model "sees" the data in its latent space.
* **Metrics**: Silhouette separability, Centered Kernel Alignment (CKA).
* **Visuals**: Embedding plots (UMAP/t-SNE support in roadmap).

---

## Output Formats
- **Interactive**: Rich Jupyter representations via `_repr_html_`.
* **Visual**: Presentation-ready dashboard via `report.summary_plot()`.
* **Structured**: JSON export for automated systems (`report.save("report.json")`).
* **Narrative**: Human-readable text summaries (`report.save("report.txt")`).
