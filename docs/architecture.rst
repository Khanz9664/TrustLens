System Architecture
===================

TrustLens is a **decision-support framework** designed to transform raw model outputs into actionable deployment decisions. It operates through a decoupled pipeline that separates data orchestration from performance metrics, composite scoring, and narrative interpretation.

--------------------------------------------------------------------------------

1. High-Level Perspective
-------------------------

TrustLens is structured as a series of specialized layers, each governed by strict data contracts.

System-Level Data Flow
~~~~~~~~~~~~~~~~~~~~~~

This diagram illustrates the end-to-end pipeline from raw model inputs to the final deployment recommendation.

.. code-block:: text

    graph TD
        Input[Input: model, X, y_true, y_prob?, sensitive_features?] --> Validate[Validation & Prob Resolution]
        Validate --> Orchestrator[api.analyze Orchestration]
        
        subgraph Metrics Engine
            Orchestrator --> Calib[Calibration: Brier, ECE]
            Orchestrator --> Fail[Failure: Confidence Gap]
            Orchestrator --> Bias[Bias: Subgroup, Equalized Odds]
            Orchestrator --> Rep[Representation: Silhouette]
        end
        
        Calib --> Results[Results Dict]
        Fail --> Results
        Bias --> Results
        Rep --> Results
        
        Results --> Scoring[trust_score.py: base_score vs Penalties]
        Scoring --> Report[TrustReport: Narrative & Interpretation]
        Report --> Output[Console / Plots / Saved Files]

*This diagram shows how raw inputs are transformed into structured metrics, then aggregated into a trust score and final deployment decision.*

**Technical Note**: The orchestrator automatically handles ``predict_proba`` resolution if ``y_prob`` is not provided, and conditionally executes the **Bias** and **Representation** modules only when the required metadata (``sensitive_features`` or ``embeddings``) is present.

--------------------------------------------------------------------------------

2. Component Interactions
-------------------------

The system relies on structured data flow between internal components.

.. code-block:: text

    graph LR
        API[api.py: Orchestrator] -- "dict: results" --> Metrics[metrics/: Compute Nodes]
        Metrics -- "dict: raw metrics" --> API
        API -- "dict: final results" --> Report[report.py: TrustReport]
        Report -- "dict: results" --> Score[trust_score.py: Scoring Engine]
        Score -- "TrustScoreResult" --> Report
        Report -- "Insights/Patterns" --> User[User/Developer]

*This interaction map defines the flow of the internal results dictionary as it is enriched by the metrics and scoring engines before final interpretation.*

Data Contracts
~~~~~~~~~~~~~~

1. **API → Metrics**: Passes sanitized numpy arrays (``y_true``, ``y_prob``, ``y_pred``).
2. **Metrics → API**: Returns standard dictionaries containing scalars (e.g., ``ece``) and plotting vectors (e.g., ``reliability_curve_points``).
3. **API → TrustReport**: Assembles the ``results`` payload and provides a model reference for metadata extraction.
4. **TrustReport → TrustScore**: Consumes the ``results`` dict and evaluates it against predefined threshold constants.

--------------------------------------------------------------------------------

3. Execution Sequence
---------------------

The following sequence diagram tracks a single call to ``api.analyze()``, highlighting the synchronous handoffs and conditional execution paths.

.. code-block:: text

    sequenceDiagram
        participant User
        participant API as api.analyze
        participant Metrics as metrics/
        participant Score as trust_score.py
        participant Report as TrustReport

        User->>API: model, X, y_true
        API->>API: Resolve y_prob (predict_proba)
        Note over API, Metrics: Conditional: Bias runs only if sensitive_features provided
        Note over API, Metrics: Conditional: Rep runs only if embeddings provided
        API->>Metrics: Dispatch Module Jobs
        Metrics-->>API: Sub-metric Payloads
        API->>Report: Initialize(results, data)
        Report->>Score: compute_trust_score(results)
        Score->>Score: Apply Penalties / Blockers
        Score-->>Report: TrustScoreResult
        Report->>Report: Generate Insights & Patterns
        Report-->>User: TrustReport Object

*The sequence trace identifies where high-overhead or data-dependent modules are conditionally activated by the orchestrator.*

--------------------------------------------------------------------------------

4. Layer Responsibilities
-------------------------

Orchestration Layer (``api.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Centrally manages the execution of analysis modules.

* **Validation**: Ensures input consistency (e.g., matching shapes for ``X`` and ``y_true``).
* **Resolution**: Implements fallback logic for models without ``predict_proba``.
* **Dispatch**: Triggers modular units (``calibration``, ``failure``, and conditionally ``bias``/``representation``) and manages progress tracking via ``tqdm``.

Metrics Engine (``metrics/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dedicated compute nodes for specialized diagnostics.

* **Calibration**: Computes **Brier Score** and **Expected Calibration Error (ECE)** to measure probability reliability.
* **Failure Analysis**: Calculates the **Confidence Gap** (difference between correct and incorrect prediction confidence).
* **Bias Detection**: (*Optional*) Evaluates **Subgroup Performance Gap** and **Equalized Odds** (TPR/FPR parity).
* **Representation**: (*Optional / Advanced*) Representation analysis using **Silhouette Scores**.

Decision Layer (``trust_score.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The system's judging engine. It uses a hybrid scoring methodology:

1. **Base Score**: A weighted combination (configurable; default ≈ 35 / 30 / 25 / 10).
2. **Penalty Burden**: Linear deductions for specific failures (e.g., ECE > 0.05).
3. **Deployment Blockers**: Boolean flags (``is_blocked``) triggered by severe risks. **Critical Note**: Deployment blockers override numeric scores to prevent unsafe model approval regardless of performance in other dimensions.

Narrative & Interpretation Layer (``report.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Converts raw numbers into human-readable signals (often referred to as the "Narrative Brain").

* **Pattern Detection**: Identifies higher-order behaviors like *"Confidently Wrong"* or *"Calibration Drift"*.
* **Insight Generation**: Ranks observations by severity (Critical, Warning, Info).
* **Narrative Serialization**: Generates the final console summaries and the Plotly/Matplotlib dashboards.

Comparative Engine (``comparison.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Provides causal reasoning for multi-model selection. It ranks candidates by score and identifies the "Comparative Advantage".

* **Example Output**: *"Model A recommended due to lower failure penalty and no active fairness violations."*

--------------------------------------------------------------------------------

5. Design Principles
--------------------

* **Modular Architecture**: Modules are isolated; failure in one (e.g., bias) does not crash the entire pipeline.
* **Separation of Concerns**: Each layer operates independently with well-defined data contracts.
* **Explainability-First**: Every metric is tied to a human-readable "Insight" in the report.
* **Decision-Support**: TrustLens doesn't just report numbers; it provides a **Verdict** (Deploy vs. Investigate).
* **Zero-Friction**: Primary entry points (``quick_analyze``) require minimal configuration for demoing.

--------------------------------------------------------------------------------

6. Extensibility Architecture
-----------------------------

TrustLens is designed to be extended by researchers and engineers.

Example: Adding a New Metric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **Metric Implementation**: Create a function in ``trustlens/metrics/new_metric.py``.
2. **API Integration**: Add the module to the ``_ALL_MODULES`` registry in ``trustlens/api.py``.
3. **Dispatch Logic**: Add a dispatch block in ``api.analyze()`` to call your new metric and save results to the dictionary.
4. **Scoring Integration**: (Optional) Add a weight/sub-score computer in ``trust_score.py``.

--------------------------------------------------------------------------------

7. Limitations & Constraints
----------------------------

* **Model Compatibility**: Optimized for Classifiers (binary/multi-class). Regression support is currently experimental.
* **Data Density**: Calibration metrics (ECE) require a minimum of 30 samples for statistical validity.
* **Linear Penalties**: Penalty deduction is current linear; non-linear risk modeling is planned for future releases.
* **Input Requirements**: Interpretability depends on availability of probabilities and optional metadata (e.g., sensitive features).

--------------------------------------------------------------------------------

8. Experimental Ecosystem
-------------------------

Modules marked ``[Experimental]`` (e.g., **Explainability/Grad-CAM**) are kept isolated from the core production pipeline to:

1. Reduce required dependency bloat (e.g., avoiding mandatory PyTorch).
2. Allow for research-stage iteration without affecting system stability.
3. Provide a clear path for promotion based on usage and stability testing (see :doc:`EXPERIMENTAL`).
