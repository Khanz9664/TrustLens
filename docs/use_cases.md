# Real-World Use Cases

This page shows how TrustLens is used in practical decision points where accuracy alone is not enough.

## How to Read These Examples

Each use case has three parts:

- **Scenario**: the production decision context
- **Diagnostic signal**: what TrustLens surfaces
- **Decision impact**: what action follows

## Safety-Critical Model Selection

**Scenario**
You must choose between a higher-accuracy model and a slightly lower-accuracy but better-calibrated model.

**Diagnostic signal**
Calibration outputs indicate that the higher-accuracy model has materially higher ECE and less reliable confidence.

**Decision impact**
Prefer the model with safer confidence behavior for triage-heavy workflows.

## Governance and Fairness Review

**Scenario**
A model must pass internal fairness review before release.

**Diagnostic signal**
Subgroup gap and equalized-odds outputs show severe disparity for one sensitive feature.

**Decision impact**
Treat release as blocked until disparity is investigated and mitigated.

## Post-Deployment Reliability Monitoring

**Scenario**
A deployed model still meets top-line accuracy targets, but support teams report suspicious errors.

**Diagnostic signal**
Confidence gap trends shrink over time and high-confidence mistakes increase.

**Decision impact**
Trigger focused error review and retraining or recalibration cycle.

## Candidate Ranking for Release

**Scenario**
Several candidates pass baseline accuracy and latency targets.

**Diagnostic signal**
`compare()` shows one candidate has lower penalty burden and no blockers.

**Decision impact**
Select the safer candidate, even if raw accuracy is slightly lower.

## Related Pages

- [Model Comparison Workflow](guides/model_comparison_workflow.md)
- [CI and Deployment Gate Workflow](guides/ci_deployment_gate.md)
- [Fairness Audit Workflow](guides/fairness_audit_workflow.md)
