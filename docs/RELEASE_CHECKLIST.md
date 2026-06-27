# TrustLens v0.5.0 Release Checklist

This checklist tracks the final release engineering steps for the v0.5.0 milestone.

## Quality Assurance
- [x] **Core Tests**: All tests passing (verified via `pytest tests/`)
- [x] **Coverage**: Coverage maintained (verified via Codecov)
- [x] **Static Analysis**: `mypy trustlens/` passing with zero errors
- [x] **Linting**: `ruff check .` passing
- [x] **Format**: `ruff format .` check passing
- [x] **Pre-commit**: `pre-commit run --all-files` passing

## Architecture & Backends
- [x] **Regression Pipeline**: Verified regression support and routing in analyze()
- [x] **Sklearn Parity**: Zero behavior change for legacy sklearn workflows
- [x] **XGBoost/LightGBM/CatBoost**: Verified support and prediction resolution
- [x] **Degraded Mode**: Transparency flags and missing component tracking verified

## Documentation
- [x] **README**: Updated with v0.5.0 features
- [x] **Sphinx Build**: Clean build with zero warnings/errors (`make html`)

## Release Engineering
- [x] **Version Bump**: Updated version in all appropriate files to `0.5.0`
- [x] **Changelog**: Finalized `CHANGELOG.md` with v0.5.0 entry
- [x] **Tag Release**: `git tag -a v0.5.0 -m "v0.5.0: Regression Support & Architecture Enhancements"`
- [x] **GitHub Release**: Drafted release notes
- [x] **PyPI Publish**: `python -m build && twine upload dist/*`
