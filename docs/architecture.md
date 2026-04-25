# Architecture

TrustLens is built as a modular, extensible framework designed for zero-friction integration into existing ML workflows.

## Directory Structure

```
trustlens/
├── metrics/           # Math & logic for Brier, ECE, Confidence Gap, Subgroup Bias
├── visualization/     # Matplotlib-based dashboarding, Reliability Curves, Embedding Plots
├── explainability/    # [Experimental] Grad-CAM, Faithfulness tests (requires PyTorch)
├── plugins/           # Extensible plugin system for custom metrics
├── api.py             # Primary entry points (analyze, quick_analyze)
├── report.py          # Result container, serialisation, and exports
└── trust_score.py     # Weighted trust consensus algorithm
```

## Key Components

### 1. The Dispatcher (`api.py`)
Centrally manages the execution of analysis modules. It handles data validation, probability resolution, and triggers the required metrics based on the input data (e.g., only running representation analysis if embeddings are provided).

### 2. The Result Container (`report.py`)
The `TrustReport` object is the "brain" of the project. It stores all results, handles the logic for `show()` and `plot()`, and provides serialization methods like `to_dict()` and `save()`.

### 3. The Trust Scorer (`trust_score.py`)
Computes a weighted consensus of model reliability. It translates raw metrics (like 0.04 ECE) into human-understandable dimension scores (0-100) and finally a letter grade (A-F).

### 4. Plugin System (`plugins/`)
Developers can extend TrustLens by registering custom plugins. Any class inheriting from `BasePlugin` can be injected into the `analyze()` pipeline, allowing for domain-specific reliability checks.

---

> Modules marked `[Experimental]` are functional but not part of the core pipeline due to heavy dependencies (e.g., PyTorch) or ongoing research. See [`docs/EXPERIMENTAL.md`](EXPERIMENTAL.md) for details.
