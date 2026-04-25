# Architecture

TrustLens is built as a modular, extensible framework designed for zero-friction integration into existing ML workflows.

## Directory Structure

```
trustlens/
├── metrics/           # Math & logic for Brier, ECE, Confidence Gap, Subgroup Bias
├── visualization/     # Matplotlib-based dashboarding, Reliability Curves, Embedding Plots
├── comparison/        # Head-to-head model evaluation and recommendation logic
├── api.py             # Primary orchestration entry points (analyze, quick_analyze)
├── report.py          # Interpretation layer, human-readable summaries, serialization
├── trust_score.py     # Core scoring logic, penalty algorithms, and grading
└── comparison.py      # Exposed API for multi-report comparison
```


## Key Components

### 1. Orchestration (`api.py`)
Centrally manages the execution of analysis modules. It handles data validation, probability resolution, and triggers the required metrics based on the input data.

### 2. Interpretation Layer (`report.py`)
The `TrustReport` object is the system's "narrative brain." It transforms raw numeric results into human-readable text summaries, ranked explanations, and actionable insights.

### 3. Scoring & Diagnostics (`trust_score.py`)
Computes the final **Trust Score**. The logic is encapsulated in `TrustScoreResult`, which now tracks:
- `base_score`: The weighted performance across dimensions.
- `penalties_applied`: Deductions for critical diagnostic failures.
- `is_blocked`: Binary flag triggered by extreme risks (e.g., severe bias).

### 4. Comparison Engine (`comparison.py`)
Provides cross-report analysis. It evaluates multiple `TrustReport` objects to recommend the safest candidate, grounding its decisions in the "penalty burden" and diagnostic blocks of each model.


---

> Modules marked `[Experimental]` are functional but not part of the core pipeline due to heavy dependencies (e.g., PyTorch) or ongoing research. See [`docs/EXPERIMENTAL.md`](EXPERIMENTAL.md) for details.
