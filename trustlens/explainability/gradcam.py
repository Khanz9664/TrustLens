# NOTE:
# This module is under active development and is not part of the public API.
# Do not import into production pipelines until stabilized.

"""
trustlens.explainability.gradcam.
=================================
Grad-CAM: Gradient-weighted Class Activation Maps.

Grad-CAM visualizes which spatial regions of an image most strongly
activate a convolutional neural network's prediction for a given class.
It requires only a single forward + backward pass and no model modification.

Usage
-----
>>> from trustlens.explainability import GradCAM
>>> cam = GradCAM(model, target_layer=model.layer4[-1])
>>> heatmap = cam.generate(image_tensor, class_idx=283)
>>> cam.overlay(image_np, heatmap, save_path="heatmap.png")

References
----------
* Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep
 networks via gradient-based localization. ICCV.
"""

from typing import Optional, cast

import numpy as np

try:
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class GradCAM:
    """
    Grad-CAM implementation for PyTorch CNN models.

    Parameters
    ----------
    model : torch.nn.Module
      Trained PyTorch model in evaluation mode.
    target_layer : torch.nn.Module
      The convolutional layer whose activations are used.
      Typically the last conv layer of the backbone
      (e.g., ``model.layer4[-1]`` for ResNet).

    Raises
    ------
    ImportError
      If PyTorch is not installed.
    """

    def __init__(self, model, target_layer) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for GradCAM. Install it with: pip install torch torchvision"
            )
        self.model = model
        self.target_layer = target_layer
        self._gradients: Optional[object] = None
        self._activations: Optional[object] = None
        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""

        def _forward_hook(module, inputs, output):
            self._activations = output.detach()

        def _backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(_forward_hook)
        self.target_layer.register_full_backward_hook(_backward_hook)

    # ------------------------------------------------------------------
    # Heatmap generation
    # ------------------------------------------------------------------

    def generate(
        self,
        image_tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for a single image.

        Parameters
        ----------
        image_tensor : Union[np.ndarray, torch.Tensor]
          Preprocessed image tensor of shape (1, C, H, W).
        class_idx : int, optional
          Target class index. If None, uses the predicted class.

        Returns
        -------
        np.ndarray
          Heatmap array of shape (H, W) in [0, 1], where H and W are
          the spatial dimensions of the target layer's feature maps,
          upsampled back to the input image resolution.
        """
        self.model.eval()
        import torch

        if not isinstance(image_tensor, torch.Tensor):
            image_tensor = torch.from_numpy(image_tensor)
        image_tensor = image_tensor.requires_grad_(False)

        # Forward pass
        logits = self.model(image_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Backward pass for the target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Pool gradients across spatial dimensions → channel weights
        if not isinstance(self._gradients, torch.Tensor) or not isinstance(
            self._activations, torch.Tensor
        ):
            raise RuntimeError(
                "Gradients or activations not captured. Ensure hooks are registered correctly."
            )
        gradients = self._gradients  # (1, C, H', W')
        activations = self._activations  # (1, C, H', W')

        weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')

        # ReLU + normalize
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cast(np.ndarray, cam.astype(np.float32))

    # ------------------------------------------------------------------
    # Overlay utility
    # ------------------------------------------------------------------

    def overlay(
        self,
        image_np: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.55,
        colormap: str = "jet",
        save_path: Optional[str] = None,
    ):
        """
        Overlay a Grad-CAM heatmap on the original image.

        Parameters
        ----------
        image_np : np.ndarray
          Original RGB image, shape (H, W, 3), dtype uint8 or float in [0,1].
        heatmap : np.ndarray
          Grad-CAM heatmap from ``generate()``, shape (H, W).
        alpha : float
          Heatmap blending strength. Default 0.55.
        colormap : str
          Matplotlib colormap name for the heatmap. Default ``"jet"``.
        save_path : str, optional
          If provided, saves the overlay figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

        # Original
        axes[0].imshow(image_np)
        axes[0].set_title("Original Image", fontsize=12)
        axes[0].axis("off")

        # Heatmap
        axes[1].imshow(heatmap, cmap=colormap, vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
        axes[1].axis("off")

        # Overlay
        if image_np.max() > 1:
            img_float = image_np.astype(float) / 255.0
        else:
            img_float = image_np.astype(float)

        cmap_fn = cm.get_cmap(colormap)
        heat_rgb = cmap_fn(heatmap)[..., :3]
        overlay = (1 - alpha) * img_float + alpha * heat_rgb
        overlay = np.clip(overlay, 0, 1)

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=12)
        axes[2].axis("off")

        plt.suptitle("Grad-CAM Explanation", fontsize=14, fontweight="bold")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
