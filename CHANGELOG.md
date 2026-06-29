# Changelog

All notable changes to TrustLens are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Radar Comparison Visualization**: Added `plot_radar_comparison()` for visually comparing multiple models across TrustLens dimensions (e.g., Calibration, Failure, Bias, Representation) using a publication-quality radar (spider) chart. Built on the centralized visualization styling system with themed colors, scoped styling, input validation, and support for saving figures. (closes #121, implemented in #156) Thanks @devanprigent

### Fixed

## Documentation

## Changed

### Improvements

---

## [v0.5.0] - 2026-06-27

### Added
- **Regression Metrics**: Added `trustlens/metrics/regression.py` with `error_distribution` (MedAE, 90th-percentile error, max, MAE, RMSE + histogram data), `prediction_interval_coverage` (PICP vs. nominal confidence, with graceful skip when intervals are absent), and `error_variance_correlation` (Pearson/Spearman between predicted uncertainty and actual error). A metrics-only first step toward regression support; `analyze()` auto-dispatch and visualization to follow. (refs #82) Thanks @Whatsonyourmind
- **Regression Analysis Pipeline & TrustReport Integration:** Added automatic regression-task detection and dispatch within `trustlens.analyze()`. Continuous targets are now routed through a dedicated regression evaluation pipeline with support for regression report rendering, serialization, export, and integration with `TrustReport`. (#142) (refs #82) Thanks @Whatsonyourmind
- **Regression Visualizations:** Added residual-analysis and error-distribution visualizations for regression workloads, enabling residual inspection, heteroscedasticity detection, and error-pattern analysis directly from regression reports. (#143) (refs #82) Thanks @Whatsonyourmind
- **Regression Trust Score:** Added a regression-specific Trust Score framework based on three trust dimensions:
  - Accuracy / Skill
  - Interval Calibration (PICP)
  - Uncertainty Informativeness
Includes A–D grading, deployment verdicts, blocker conditions for negative skill and severe interval miscoverage, weight redistribution when uncertainty signals are unavailable, regression/classification task separation via `task_type`, and full `TrustReport` integration. (#147) (refs #82) Thanks @Whatsonyourmind
- **Regression Trust Score Reproducibility**: Persisted `target_variance` (`Var(y)`) in regression analysis results, allowing `regression_trust_score()` to be recomputed from stored reports without requiring the original `y_true` array. Added backward-compatible fallback logic, mismatch warnings when persisted variance disagrees with supplied targets, and improved portability of serialized regression artifacts. (closes #150, implemented in #151) Thanks @Whatsonyourmind
- **Model Zoo Benchmark**: Introduced a comprehensive scientific validation notebook (`examples/trustlens_model_zoo_benchmark.ipynb`) that systematically evaluates TrustLens across 6 model architectures and multiple data corruption scenarios with statistical aggregation.
- **Centralized Visualization Styling**: Introduced an internal `trustlens/visualization/style.py` as the single source of truth for color palettes, semantic colors (severity, deployment verdict, grade, direction), typography, grid, and figure defaults. Added an `apply_style()` context manager that scopes `matplotlib.rcParams` mutations to a `with` block, preventing global state leakage when TrustLens is used inside notebooks or larger ML pipelines. Existing plotting modules are being migrated to the centralized system without changing visual output (so far: `calibration_plots.py`, `failure_plots.py`, `bias_plots.py`, `representation_plots.py`, `fairness.py`, `summary_plot.py`). (refs #57) Thanks @komoike-oss28-ui
- **Deployment Recommendation Explanations**: Added `TrustReport.deployment_explanation` and `TrustReport.deployment_summary` to provide structured deployment verdict explanations, identify primary risks, and surface actionable recommendations based on Trust Score penalties and sub-scores.
- **Deployment Recommendation UX**: Deployment recommendations are now surfaced directly in `TrustReport.show()`, text exports, and HTML report views. Users now receive deployment verdicts, primary risk identification, and actionable recommendations without needing to access `deployment_summary` manually.
- **Native LightGBM & CatBoost Backends**: Added automatic backend detection and prediction resolution for the LightGBM (LGBMClassifier, Booster) and CatBoost (CatBoostClassifier) models, including probability extraction, classification validation, and integration with the standard PredictionBundle pipeline. Thanks @vaishnavidesai09

### Fixed
- Fixed `reliability_curve(strategy="quantile")` for collapsed quantile bin edges, returning a valid single-bin curve for zero-variance probability distributions instead of raising `ValueError`. (fixes #144)
- **XGBoost Booster String Label Mapping**: Fixed an issue where raw `xgboost.Booster` models in multiclass classification returned ordinal prediction indices instead of semantic class labels when `y_true` contained string labels. TrustLens now correctly maps probability-column indices back to user-provided class labels, preventing downstream metric distortion and label-type mismatches. Added validation to ensure `class_labels` match the probability matrix shape and introduced regression tests covering raw Booster multiclass workflows. (fixes #117) Thanks @nanookclaw
- Fixed incorrect `top_mistake_indices` in `misclassification_summary()` to return **global dataset indices** instead of local filtered subset positions, improving downstream EDA and debugging workflows for high-confidence model errors. (PR #104) Thanks @dicnunz 🙌
- Fixed `Security Audit` CI failures caused by newly published upstream dependency vulnerabilities by updating `pip-audit` handling and ignore rules for unresolved ecosystem CVEs/PYSEC advisories. (PR #105)
- Fixed the visual narrative of the **Accuracy vs Trust (“Decoupling”)** analysis to more clearly communicate the relationship between predictive performance and trustworthiness in the benchmark notebook. (PR #100)

### Documentation
- **Core Architecture Documentation**: Improved module-level and class-level docstrings for `api.py`, `report.py`, `trust_score.py`, and `pipeline.py`.
- **Backend Architecture Documentation**: Documented the internal backend architecture in `trustlens/backends/`, detailing the `PredictionBundle` lifecycle, resolver architecture, probability extraction, and label mapping strategies.
- **Metrics Documentation**: Enhanced public docstrings for major metrics (`brier_score`, `expected_calibration_error`, `confidence_gap`, `equalized_odds`, `embedding_separability`) with clear explanations of what they measure, why they matter, their limitations, and how to interpret them.
- **Visualization Architecture Documentation**: Documented the centralized visualization architecture in `trustlens/visualization/style.py`, providing rules for maintaining visual parity and using semantic colors.
- **Developer Experience**: Updated `CONTRIBUTING.md` to formally outline documentation expectations, including a mandate for NumPy-style docstrings and a high-level architecture reference for new contributors.
- **Research & Validation Layer**: Added a comprehensive, research-grade documentation section (`docs/research/`) featuring empirical benchmark results, scientific trust score validation, robustness under distribution shift, metric limitations, and explicitly outlined failure modes.
- **Methodology & Threats to Validity**: Introduced a brutally honest `methodology.md` page detailing benchmark experimental setup and transparently acknowledging limitations such as reliance on synthetic datasets and binary classification constraints.
- **Why TrustLens**: Added a `why_trustlens.md` page to directly compare TrustLens against traditional metrics (like Accuracy and ROC-AUC) using tangible failure case studies.
- Generated publication-quality (300 DPI) visual assets demonstrating TrustLens's behavior under noise, calibration degradation, and severe class imbalance, inheriting the project's centralized visual styling.
- Added a complete, copy-paste runnable example to the analyze() docstring that demonstrates: Dataset creation using make_classification, Train/test split, Training a RandomForestClassifier, Predicting probabilities, Running analyze(), Displaying results with report.show(). Thanks @q404365631
- Added hosted TrustLens documentation website integration across the repository, including README links, package metadata (`pyproject.toml`), and documentation navigation improvements. (PR #101)

### Improvements
- Improved validation feedback in `brier_score()` with clearer and more beginner-friendly error messages for invalid input shape mismatches, making debugging easier for users. (PR #106) Thanks @JavadTe 🙌
- Improved macOS CI reliability by resolving `xgboost.core.XGBoostError` related to missing `libomp.dylib` discovery during GitHub Actions execution. (PR #96)
- Improved cross-platform CI stability and macOS workflow reliability. (PR #97)

### Changed
- Added explicit `__all__` exports to metrics modules (`calibration`, `failure`, `bias`, `representation`) to improve API clarity and consistency. Thanks @JavadTe 🙌

### Maintenance
- Added a temporary CI trigger workflow to validate and debug macOS GitHub Actions behavior during infrastructure stabilization. (PR #98, later superseded and closed)

---

## [v0.4.0] - 2026-05-15

### Major Architectural Milestone: Framework-Agnostic Core
This release marks the transition of TrustLens from a scikit-learn-specific library to a framework-agnostic trustworthiness platform.

### Added
- **Prediction Resolver Architecture**: A new plugin-based backend system for resolving predictions across different ML frameworks.
- **XGBoost Support**: Native support for `XGBClassifier` and raw `Booster` objects (including DMatrix conversion and objective-based task blocking).
- **Manual Override Mode**: Full support for `model=None` workflows where users provide `y_pred` and `y_prob` manually.
- **Degraded Mode Transparency**: Explicit metadata tracking for missing components (`degraded_mode`, `missing_components`) when probabilistic data is unavailable.
- **Hardened Prediction Contract**: Strict validation for non-finite values (NaN/Inf) and probability range enforcement with EPS tolerance.
- **Unified JSON Artifact Export**: `TrustReport.save("report.json")` now produces a single, self-contained JSON artifact containing results, metadata, and trust scores.

### Improved
- **Architectural Decoupling**: Fully refactored `trustlens/api.py` and `trustlens/core/pipeline.py` to be framework-agnostic.
- **Lazy Loading**: Framework-specific dependencies (like XGBoost) are now loaded lazily, ensuring a minimal footprint for scikit-learn users.
- **CI/CD Pipeline**: Added Python 3.13 support, security auditing (`pip-audit`), and automated build validation.
- **Documentation**: Fully synchronized Sphinx documentation, including new internal RFCs for backend developers.

### Fixed
- **Multiclass Brier Score**: Fixed a core metric bug where calibration analysis assumed binary probabilities for all models. TrustLens now correctly computes the Multiclass Brier Score (Mean Squared Error across all classes) for N-class problems.
- Fixed numerical instability in calibration metrics via automatic probability clipping.
- Improved classifier detection to support custom mock objects and non-BaseEstimator wrappers.
- Fixed Sphinx build warnings related to non-consecutive header levels and broken links.
- Corrected relative links in `docs/index.md` and `docs/EXPERIMENTAL.md`.
- Fixed CI failures by updating mypy configuration for Python 3.9 EOL compliance.
- Resolved type-shadowing and incompatible assignment errors in `GradCAM`.
- Hardened prediction resolver registry with explicit type hints and improved error handling.
- Fully propagated `Optional[y_prob]` support through the core pipeline, plugin architecture, and visualization dashboard, ensuring robust handling of non-probabilistic models.

### Compatibility / Migration
- **No breaking changes** for existing scikit-learn users.
- `analyze()` remains backward compatible; existing workflows will continue to work unchanged while benefiting from improved internal validation.

## [0.3.0] — 2026-05-06

### Added
- 2D embedding visualization (`plot_embedding_2d`) with automatic UMAP → t-SNE → PCA fallback, class-colored scatter plot, silhouette score annotation, and configurable subsampling (`n_max`). Integrated into `report.plot()` auto-dispatch. Thanks @WeiGuang-2099
- `embedding_separability` metric computing silhouette score, within/between-class distances, and separability ratio. Thanks @WeiGuang-2099
- 14 tests covering representation metrics, CKA, and 2D embedding visualization. Thanks @WeiGuang-2099
- Model comparison API (`trustlens.compare`) for head-to-head multi-model evaluation and recommendation.
- Pattern detection system (e.g., "Calibration Drift", "Confidently Wrong") to surface high-level semantic risks.
- Initial `equalized_odds()` fairness metric with per-group TPR/FPR analysis (closes #17). Thanks @komoike-oss28-ui
- Ranked score explanation layer to justify Trust Score deductions.
- `equalized_odds()`: added input validation, configurable violation thresholds (`severe_threshold`, `moderate_threshold`), and concrete docstring examples (closes #41) Thanks @komoike-oss28-ui
- Fairness visualization module (`trustlens/visualization/fairness.py`) with `plot_subgroup_performance()`, `plot_equalized_odds()`, and `plot_fairness_gap()` (closes #52) Thanks @komoike-oss28-ui
- Upgraded `TrustReport.plot_bias()` with multi-mode diagnostic support:
  - New `mode` parameter: `"summary"` (default), `"all"`, `"subgroup"`, `"equalized_odds"`, and `"gap"`.
  - Added deterministic return contracts (Returns `Figure` or `dict[str, Figure | None]`).
  - Implemented backend-safe `plt.show()` and automated `save_path` suffixing for batch plotting.
  - Hardened validation for bias data structures and added memory hygiene documentation.
- Added bias analysis demo with subgroup diagnostics (`examples/bias_analysis_demo.py`). Thanks @sidharth-vijayan
- Added SECURITY.md. Thanks @MustansirNisar
- Added unit tests for multi-feature fairness visualizations covering all-features-processed guarantee, output key matching, and figure smoke tests (`tests/test_fairness_visualization_multi.py`). Thanks @komoike-oss28-ui
- `_plot_multi_helper()` — internal helper that eliminates duplication across `*_multi` wrappers and enforces deterministic (sorted) feature iteration.
- `_safe_name()` — filename sanitizer for feature names containing spaces or special characters (e.g., `"income level"` → `income_level`).
- `_BIAS_PLOT_TYPES` — internal registry for deterministic plot-type dispatch ordering in `_plot_bias()`.
- `tests/conftest.py` — centralized Agg backend configuration for the test suite.
- `tests/test_plot_module_multi_feature.py` — 23 integration tests covering nested figure outputs, filename sanitization, orchestrated saving, and edge cases.
- `TrustReport.plot_bias()` now accepts an opt-in `multi_feature: bool = False` parameter for per-feature visualization output. With `multi_feature=True`, single modes (`"subgroup"`, `"equalized_odds"`, `"gap"`) return `dict[str, Figure]` keyed by feature name, and `mode="all"` returns a nested `dict[str, dict[str, Figure]]` keyed by mode then feature. The structure is fixed by the `(mode, multi_feature)` combination, missing components are represented by empty dicts (never `None`), and feature ordering is deterministic (`sorted(feature_names)`). Default behavior (`multi_feature=False`) is unchanged. `tests/test_plot_bias_multi_feature.py` adds 18 tests covering the four return-shape cells, partial-data handling, deterministic ordering, and invalid-mode interaction (closes #74). Thanks @komoike-oss28-ui

### Improved
- Final Trust Score logic now includes a base score, penalty breakdown, and decisive deployment verdicts.
- Standardized canonical terminology to "confidence-weighted errors".
- Enhanced failure diagnostics with confidence concentration insights (range analysis).
- Bias reporting now includes explicit margin calculations relative to the 0.10 threshold.
- Comparison engine includes causal reasoning (e.g., linking selection to lower penalty burdens).
- Integrated fairness metrics into the main `analyze()` pipeline with safe fallback handling and margin reporting.
- Unified validation error message format in `equalized_odds()` for consistency. Thanks @komoike-oss28-ui
- Enhanced `_violation_level()` docstring with parameter descriptions and threshold details. Thanks @komoike-oss28-ui
- Fairness visualization now supports multiple sensitive features via `plot_subgroup_performance_multi()`, `plot_equalized_odds_multi()`, and `plot_fairness_gap_multi()`, which return per-feature figures as `{feature_name: Figure}`. Fixed `_plot_bias()` to no longer silently drop features after the first (closes #56). Thanks @komoike-oss28-ui
- Enhanced bias module usability with visual diagnostics for easier interpretation.
- Refactored `_plot_bias()` into a pure figure-generation function (no file I/O or side effects). All saving and figure closing is now centralized in `plot_module()`.
- `plot_module()` now handles nested `dict[str, dict[str, Figure]]` outputs for multi-feature bias data, with standardized filenames (`bias_<type>_<feature>.png`).
- Updated `docs/metrics/bias.md`, `docs/features.md`, and `README.md` with multi-feature visualization documentation, usage examples, and file output reference.

### Fixed
- Removed all `matplotlib.use("Agg")` calls from library modules (6 visualization files, `report.py`, `gradcam.py`). This was silently overriding the user's matplotlib backend at import time, breaking interactive use in Jupyter and GUI environments.

### Stability
- Maintained full backward compatibility with the `analyze()` API.
- All 219 tests passing.


---

## [0.2.0] — 2026-04-24

### Added
- Extended CI test matrix to include Python 3.13 (closes #29). Thanks @CrepuscularIRIS
- Standardized GitHub contribution infrastructure:
  - Pull Request template with integrated checklists.
  - Structured YAML Issue templates for Bug Reports and Feature Requests.
  - Dedicated `good-first-issue` template and `config.yml` for triage.
- Overhauled `CONTRIBUTING.md` with a command-driven "First Contribution Guide" and difficulty labeling system.
- Comprehensive test suite in `tests/test_utils.py` covering edge cases for all utility functions.
- `report.save()` now supports direct export to single `.json` and `.txt` files.
- Human-readable text report generation without ANSI colors.
- `docs/EXPERIMENTAL.md` — contributor-facing guide for experimental module governance.


### Improved
- Enhanced `utils.py` with robust input validation and NumPy-aware numeric type checking.
- Added progress messages in `analyze()` for better runtime visibility. Thanks @jayssSmm
- Codebase stabilization: isolated experimental modules (`explainability/`, `metrics/faithfulness.py`) from the production pipeline with clear `# NOTE:` headers and documentation.
- Cleaned public API surface — `__init__.py` docstring now reflects only production-ready capabilities.
- Updated README architecture tree to distinguish stable vs experimental modules.
- Replaced misleading `pyproject.toml` keyword `"explainability"` with `"model trust"`.
- Renamed `examples/cnn_vs_vit_trustlens.py` → `examples/model_comparison.py` to match actual content (sklearn models, not deep learning).
- Added actionable Pipeline Module Registry guard in `api.py` to prevent accidental re-exposure of experimental code.

### Fixed
- Prevented crashes in `describe_array` for empty inputs.
- Corrected bin count computation in `reliability_curve()` to use exact binning logic. Thanks @WeiGuang-2099

---

## [0.1.2] — 2026-04-16

### Fixed
- Stabilized Matplotlib plotting backends for headless environments
- Resolved NumPy division-by-zero warnings in histograms
- Fixed trailing whitespace and end-of-file linting violations

### Improved
- Standardized `pyproject.toml` and documentation
- Enhanced small-dataset reliability warnings
- Robust CI/CD pipeline integration across Python versions

---

## [0.1.1] — 2026-04-16

### Fixed
- Resolved NumPy runtime warnings in histogram normalization
- Fixed Matplotlib non-interactive backend warning (`FigureCanvasAgg` warning suppressed via backend-aware `plt.show()` guard)
- Improved plotting stability with controlled rendering and `plt.close()` cleanup

### Improved
- Cleaner console output in headless and CI environments
- Small dataset warning added for `n < 30` samples
- `show: bool = True` parameter added to all visualization functions for optional interactive display

---

## [0.1.0] — 2026-04-16

- `trustlens.quick_analyze()` — zero-friction, branded entry point with auto-loading demo data
- `trustlens.analyze()` — primary analysis API with module dispatch
- `TrustReport` result container with rich `_repr_html_` for Jupyter, plus `show()`, `plot()`, `save()`
- **Calibration module**: `brier_score`, `expected_calibration_error`, `reliability_curve`
- **Failure module**: `misclassification_summary`, `confidence_gap`
- **Bias module**: `class_imbalance_report`, `subgroup_performance`
- **Representation module**: `embedding_separability`, `centered_kernel_alignment`
- **Explainability**: `GradCAM` class with hook-based PyTorch implementation
- **Faithfulness**: `pixel_deletion_test`, `pixel_insertion_test` with AUPC metric
- **Visualization**: Professional base64-rendered Jupyter dashboards and premium Matplotlib visualizations
- **UX**: `tqdm` progress tracking for long-running batch analysis
- **Plugin system**: `BasePlugin` ABC + `PluginRegistry` singleton
- Full test suite: `test_calibration`, `test_failure`, `test_bias`, `test_representation`, `test_api`, `test_plugins`
- Examples: `trustlens_demo.ipynb` (Colab-ready), `quickstart.py`, `calibration_deep_dive.py`
- GitHub Actions CI workflow (linting, testing, and formatting)
- Complete documentation: README (with logo), CONTRIBUTING, ROADMAP, this CHANGELOG

[Unreleased]: https://github.com/Khanz9664/TrustLens/compare/v0.5.0...HEAD
[v0.5.0]: https://github.com/Khanz9664/TrustLens/compare/v0.4.0...v0.5.0
[v0.4.0]: https://github.com/Khanz9664/TrustLens/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Khanz9664/TrustLens/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Khanz9664/TrustLens/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/Khanz9664/TrustLens/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Khanz9664/TrustLens/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Khanz9664/TrustLens/releases/tag/v0.1.0
