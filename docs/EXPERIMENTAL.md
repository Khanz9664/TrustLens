# Experimental Modules

TrustLens maintains an experimental ecosystem for cutting-edge research modules that are not yet part of the core production pipeline.

## Current Experimental Modules

### Explainability / Grad-CAM (`trustlens.explainability.gradcam`)
* **Purpose**: Visual attribution maps for convolutional neural networks.
* **Why Experimental**: Requires heavy external dependencies (PyTorch, Torchvision) and is currently limited to vision models.
* **Usage**:
  ```python
  from trustlens.explainability import GradCAM
  explainer = GradCAM(model, target_layer)
  heatmap = explainer.compute(input_tensor)
  ```

---

## Experimental Promotion Criteria

A module is promoted from `Experimental` to `Core` when it meets the following standards:

1. **Zero-Crash Stability**: Passes internal stress tests without unhandled exceptions.
2. **Standardized API**: Implements the standard `compute()` and `get_results_dict()` interfaces.
3. **Internal Documentation**: Complete docstrings for all public methods.
4. **Unit Test Coverage**: Minimum 80% coverage in isolation.
5. **Dependency Management**: Dependencies must be optional (`extras_require`) to keep the core install lightweight.

---

## Contributing to Experimental
We welcome contributions to experimental modules! See [Contributing](https://github.com/Khanz9664/TrustLens/blob/main/CONTRIBUTING.md) for how to get involved.
