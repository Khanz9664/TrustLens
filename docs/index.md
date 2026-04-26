# TrustLens Documentation

TrustLens helps ML teams answer a practical question with evidence: should this model be deployed, delayed, or blocked?

This documentation is organized around real workflows used by ML engineers, not around internal package structure.

## Start Here

- New to TrustLens: read [Getting Started](getting_started.md)
- Need context first: read [Overview](overview.md) and [The Problem](problem.md)
- Need API details: open [API Reference](api_reference.md)

## I Want To...

- Evaluate one model quickly: [Getting Started](getting_started.md)
- Understand score logic and verdict rules: [Trust Score Explained](trust_score_explained.md)
- Investigate reliability and risk dimensions: [Features and Modules](features.md)
- Use TrustLens in production workflows: [Guides](guides/model_comparison_workflow.md)
- Understand what is and is not supported: [Known Limitations](known_limitations.md)
- Extend the system safely: [Architecture](architecture.md), [Contributing](../CONTRIBUTING.md)

## Documentation Map

### Foundations

- [Overview](overview.md)
- [The Problem](problem.md)
- [Audience](audience.md)

### Concepts

- [Features and Modules](features.md)
- [Trust Score Explained](trust_score_explained.md)
- [Known Limitations](known_limitations.md)

### Practical Guides

- [Model Comparison Workflow](guides/model_comparison_workflow.md)
- [CI and Deployment Gate Workflow](guides/ci_deployment_gate.md)
- [Fairness Audit Workflow](guides/fairness_audit_workflow.md)

### Reference

- [API Reference](api_reference.md)
- [Metric: Calibration](metrics/calibration.md)
- [Metric: Failure](metrics/failure.md)
- [Metric: Bias](metrics/bias.md)
- [Metric: Representation](metrics/representation.md)

### Project and Contribution

- [Architecture](architecture.md)
- [Design Principles](DESIGN_PRINCIPLES.md)
- [Experimental Modules](EXPERIMENTAL.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [Roadmap](../ROADMAP.md)
- [Code of Conduct](../CODE_OF_CONDUCT.md)

```{toctree}
:maxdepth: 2
:caption: Documentation

getting_started
overview
problem
audience
features
trust_score_explained
known_limitations
guides/model_comparison_workflow
guides/ci_deployment_gate
guides/fairness_audit_workflow
api_reference
architecture
DESIGN_PRINCIPLES
EXPERIMENTAL
```

## Indices and Tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
