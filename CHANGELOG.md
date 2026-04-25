# Changelog

All notable changes to TrustLens are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Model comparison API (`trustlens.compare`) for head-to-head multi-model evaluation and recommendation.
- Pattern detection system (e.g., "Calibration Drift", "Confidently Wrong") to surface high-level semantic risks.
- Initial `equalized_odds()` fairness metric with per-group TPR/FPR analysis (closes #17). Thanks @komoike-oss28-ui
- Ranked score explanation layer to justify Trust Score deductions.

### Improved
- Final Trust Score logic now includes a base score, penalty breakdown, and decisive deployment verdicts.
- Standardized canonical terminology to "confidence-weighted errors".
- Enhanced failure diagnostics with confidence concentration insights (range analysis).
- Bias reporting now includes explicit margin calculations relative to the 0.10 threshold.
- Comparison engine includes causal reasoning (e.g., linking selection to lower penalty burdens).
- Integrated fairness metrics into the main `analyze()` pipeline with safe fallback handling and margin reporting.


### Stability
- Maintained full backward compatibility with the `analyze()` API.
- All 33 core tests passing.


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

[Unreleased]: https://github.com/Khanz9664/TrustLens/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Khanz9664/TrustLens/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/Khanz9664/TrustLens/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Khanz9664/TrustLens/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Khanz9664/TrustLens/releases/tag/v0.1.0
