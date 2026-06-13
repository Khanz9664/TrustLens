# TrustLens v0.4.0 Release Checklist

This checklist tracks the final release engineering steps for the v0.4.0 "Framework-Agnostic" milestone.

## Quality Assurance
- [x] **Core Tests**: 263 tests passing (verified via `pytest tests/`)
- [x] **Coverage**: ~75% coverage maintained (verified via Codecov)
- [x] **Static Analysis**: `mypy trustlens/` passing with zero errors
- [x] **Linting**: `ruff check .` passing
- [x] **Format**: `ruff format .` check passing
- [x] **Pre-commit**: `pre-commit run --all-files` passing

## Architecture & Backends
- [x] **Sklearn Parity**: Zero behavior change for legacy sklearn workflows
- [x] **XGBoost Backend**: Verified support for `XGBClassifier` and raw `Booster`
- [x] **Degraded Mode**: Transparency flags and missing component tracking verified
- [x] **Lazy Imports**: `xgboost` is NOT imported when running sklearn-only pipelines
- [x] **Numerical Stability**: EPS tolerance and auto-clipping implemented in `PredictionBundle`

## Documentation
- [x] **README**: Updated with v0.4.0 architecture and manual override examples
- [x] **Architecture Docs**: Mermaid diagrams and resolver registry documented
- [x] **Internal RFCs**: `prediction_contract.md` updated with manual/degraded mode rules
- [x] **Sphinx Build**: Clean build with zero warnings/errors (`make html`)

## Release Engineering
- [x] **Version Bump**: Updated `trustlens/__init__.py` and `report.py` to `0.4.0`
- [x] **Changelog**: Finalized `CHANGELOG.md` with v0.4.0 entry
- [x] **Tag Release**: `git tag -a v0.4.0 -m "v0.4.0: Framework-Agnostic Architecture"` (READY)
- [x] **GitHub Release**: Drafted release notes in `RELEASE_NOTES_DRAFT.md` (READY)
- [x] **PyPI Publish**: `python -m build && twine upload dist/*` (VERIFIED LOCALLY)
