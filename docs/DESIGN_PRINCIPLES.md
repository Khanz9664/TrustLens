# TrustLens Design Principles

> These principles guide every technical decision in TrustLens.
> New contributors should read this before writing any code.

---

## 1. Simplicity > Complexity

**A correct library that is never used has zero impact.**

TrustLens competes with the user's time. If the API is confusing, they will write a 10-line `sklearn` script instead. Every function must be usable without reading the docs.

**In practice:**
- The primary API is a single function: `analyze()`. No configuration classes, no session objects, no builders.
- Default parameters should produce a useful result for 80% of users.
- Errors must be actionable: "y_prob is required when model does not expose predict_proba()" not "ValueError: array mismatch."
- We prefer explicit over implicit — pass arrays in, get numbers out.

**What this rules out:**
Overengineered abstractions, "framework" patterns that require users to learn TrustLens-specific concepts before getting results.

---

## 2. Modular by Design

**Every feature is independently importable.**

Users who only want Brier Score should not pay the import cost of Grad-CAM. Researchers who want only CKA should not need `scikit-learn` installed.

**In practice:**
- Each analysis area (calibration, failure, bias, explainability, representation) lives in its own module.
- Modules have no circular imports.
- Deep learning dependencies (`torch`, `shap`, `captum`) are optional extras.
- The plugin system ensures new capabilities extend — not couple — the core.

**What this rules out:**
Monolithic files, cross-module imports that create hidden dependencies, mandatory heavy dependencies for lightweight features.

---

## 3. Visual-First Outputs

**Numbers without context mislead. Visuals reveal structure.**

An ECE of 0.042 means nothing until you see the reliability diagram and notice the model is overconfident specifically for high-confidence predictions.

**In practice:**
- Every metric has a corresponding visualization in `trustlens.visualization`.
- Plots are annotated with metric values so they are self-contained.
- Each visualization is designed to answer a specific question (e.g., "Is my model overconfident?"), not just display data.
- Plots can be saved as publication-quality PNGs (150 DPI minimum).
- Dark mode support and accessible color palettes are roadmap items.

**What this rules out:**
Raw number dumps with no visual companion, non-informative plots that don't annotate the metric they're meant to convey.

---

## 4. Research + Practical Balance

**A library used only in papers or only in production is half a library.**

TrustLens sits at the intersection: rigorous enough for researchers (proper citations, correct math, edge-case handling), accessible enough for practitioners (sklearn API, sensible defaults, fast runtimes).

**In practice:**
- Every metric links to its original paper in the module docstring.
- Mathematical formulas are included in NumPy-style docstrings using LaTeX notation.
- Metrics handle edge cases (empty bins, single-class inputs, NaN propagation) without crashing.
- Performance matters: large datasets use subsampling with documented behavior.

**What this rules out:**
Metrics that are implemented incorrectly for the sake of simplicity, or metrics so theoretically correct they're unusably slow.

---

## 5. Extensibility Without Fragility

**New capabilities should not break existing ones.**

As TrustLens grows, new metrics, visualizations, and integrations must not require touching core files.

**In practice:**
- The plugin system is the primary extension mechanism.
- `analyze()` dispatches to modules by name string — adding a new module never changes the function signature.
- Visualization functions accept dicts (not TrustReport objects) — they are decoupled from the report class.
- Backward compatibility is maintained within a major version.

**What this rules out:**
Tight coupling between modules, hardcoded module lists that must be updated on every addition, breaking API changes without a deprecation cycle.

---

## 6. Test Everything

**Code without tests is a liability, not an asset.**

TrustLens is a trust tool — it must itself be trustworthy.

**In practice:**
- Minimum 80% branch coverage for all modules.
- All metrics must have at least one test for the "perfect predictor" case and one for "random predictor."
- Edge cases (empty input, single class, NaN, infinite values) are explicitly tested.
- Integration tests verify the full `analyze()` → `TrustReport` → `save()` pipeline.

**What this rules out:**
Merging untested code, skipping tests for "obviously correct" utility functions.

---

## 7. Documentation as a Feature

**If it isn't documented, it doesn't exist.**

**In practice:**
- Every public function has a complete NumPy-style docstring.
- Every module has a module-level docstring explaining purpose, metrics implemented, and references.
- The examples directory contains runnable, realistic scripts (no toy `np.random` examples as primary usage).
- The README is updated with every public API change.

---

## 8. Performance is Not Optional

**A tool that takes 10 minutes to run won't be run.**

**In practice:**
- Silhouette score uses subsampling for `n > 5000`.
- Faithfulness tests support configurable `n_steps` to trade resolution for speed.
- Plots use `matplotlib`'s Agg backend (no display required) for CI/server compatibility.
- Expensive operations are lazy (only run when the corresponding module is requested).
