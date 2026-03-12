"""
Base Model Abstract Class
==========================

Defines the interface contract that all models must implement for compatibility
with the training pipeline. Provides common utilities and enforces consistency.

Author: Ductho Le (ductho.le@outlook.com)
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


# =============================================================================
# TYPE ALIASES
# =============================================================================

# Spatial shape type aliases for model input dimensions
SpatialShape1D = tuple[int]
SpatialShape2D = tuple[int, int]
SpatialShape3D = tuple[int, int, int]
SpatialShape = SpatialShape1D | SpatialShape2D | SpatialShape3D


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def compute_num_groups(num_channels: int, preferred_groups: int = 32) -> int:
    """
    Compute valid num_groups for GroupNorm that divides num_channels evenly.

    GroupNorm requires num_channels to be divisible by num_groups. This function
    finds the largest valid divisor up to preferred_groups.

    Args:
        num_channels: Number of channels to normalize (must be positive)
        preferred_groups: Preferred number of groups (default: 32)

    Returns:
        Valid num_groups that satisfies num_channels % num_groups == 0

    Example:
        >>> compute_num_groups(64)  # Returns 32
        >>> compute_num_groups(48)  # Returns 16 (48 % 32 != 0)
        >>> compute_num_groups(7)  # Returns 1 (prime number)
    """
    # Try preferred groups first, then common divisors
    for groups in [preferred_groups, 16, 8, 4, 2, 1]:
        if groups <= num_channels and num_channels % groups == 0:
            return groups

    # Fallback: find any valid divisor (always returns at least 1)
    for groups in range(min(32, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups

    return 1  # Always valid


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all regression models.

    All models in this framework must inherit from BaseModel and implement
    the required abstract methods. This ensures compatibility with the
    training pipeline and provides a consistent interface.

    Supports any input dimensionality:
        - 1D: in_shape = (L,) for signals/waveforms
        - 2D: in_shape = (H, W) for images/spectrograms
        - 3D: in_shape = (D, H, W) for volumes

    Attributes:
        in_shape: Input spatial dimensions (varies by dimensionality)
        out_size: Number of output targets

    Example:
        from wavedl.models.base import BaseModel
        from wavedl.models.registry import register_model

        @register_model("my_model")
        class MyModel(BaseModel):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__(in_shape, out_size)
                # Build layers...

            def forward(self, x):
                # Forward pass...
                return output
    """

    @abstractmethod
    def __init__(
        self,
        in_shape: tuple[int] | tuple[int, int] | tuple[int, int, int],
        out_size: int,
        **kwargs,
    ):
        """
        Initialize the model.

        Args:
            in_shape: Input spatial dimensions, excluding batch and channel dims:
                      - 1D: (L,) for signal length
                      - 2D: (H, W) for image dimensions
                      - 3D: (D, H, W) for volume dimensions
            out_size: Number of regression output targets
            **kwargs: Model-specific hyperparameters
        """
        super().__init__()
        self.in_shape = in_shape
        self.out_size = out_size

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (B, C, *spatial_dims)
               - 1D: (B, C, L)
               - 2D: (B, C, H, W)
               - 3D: (B, C, D, H, W)

        Returns:
            Output tensor of shape (B, out_size)
        """
        pass

    def validate_input_shape(self, x: torch.Tensor) -> None:
        """
        Validate input tensor shape against model's expected shape.

        Call this at the start of forward() for explicit shape contract enforcement.
        Provides clear, actionable error messages instead of cryptic Conv layer errors.

        Args:
            x: Input tensor to validate

        Raises:
            ValueError: If shape doesn't match expected dimensions

        Example:
            def forward(self, x):
                self.validate_input_shape(x)  # Optional but recommended
                return self.model(x)
        """
        expected_ndim = len(self.in_shape) + 2  # +2 for (batch, channel)

        if x.ndim != expected_ndim:
            dim_names = {
                3: "1D (B, C, L)",
                4: "2D (B, C, H, W)",
                5: "3D (B, C, D, H, W)",
            }
            expected_name = dim_names.get(expected_ndim, f"{expected_ndim}D")
            actual_name = dim_names.get(x.ndim, f"{x.ndim}D")
            raise ValueError(
                f"Input shape mismatch: model expects {expected_name} input, "
                f"got {actual_name} with shape {tuple(x.shape)}.\n"
                f"Expected in_shape: {self.in_shape} -> input should be (B, C, {', '.join(map(str, self.in_shape))})\n"
                f"Hint: Check your data preprocessing - you may need to add/remove dimensions."
            )

        # Validate spatial dimensions match
        spatial_dims = tuple(x.shape[2:])  # Skip batch and channel
        if spatial_dims != tuple(self.in_shape):
            raise ValueError(
                f"Spatial dimension mismatch: model expects {self.in_shape}, "
                f"got {spatial_dims}.\n"
                f"Full input shape: {tuple(x.shape)} (B={x.shape[0]}, C={x.shape[1]})\n"
                f"Hint: Ensure your data dimensions match the model's in_shape."
            )

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count the number of parameters in the model.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def parameter_summary(self) -> dict[str, Any]:
        """
        Generate a summary of model parameters.

        Returns:
            Dictionary with parameter statistics
        """
        total = self.count_parameters(trainable_only=False)
        trainable = self.count_parameters(trainable_only=True)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "total_mb": total * 4 / (1024 * 1024),  # Assuming float32
        }

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """
        Return default configuration for this model.
        Override in subclasses to provide model-specific defaults.

        Returns:
            Dictionary of default hyperparameters
        """
        return {}

    def get_optimizer_groups(self, base_lr: float, weight_decay: float = 1e-4) -> list:
        """
        Get parameter groups for optimizer with optional layer-wise learning rates.
        Override in subclasses for custom parameter grouping (e.g., no decay on biases).

        Args:
            base_lr: Base learning rate
            weight_decay: Weight decay coefficient

        Returns:
            List of parameter group dictionaries
        """
        # Default: no weight decay on bias and normalization layers
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Skip weight decay for bias, normalization, and per-channel
            # scaling parameters (LayerScale gamma, GRN gamma/beta, etc.)
            if (
                "bias" in name
                or "norm" in name
                or "bn" in name
                or "gamma" in name
                or "beta" in name
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # Handle empty parameter lists gracefully
        groups = []
        if decay_params:
            groups.append(
                {"params": decay_params, "lr": base_lr, "weight_decay": weight_decay}
            )
        if no_decay_params:
            groups.append(
                {"params": no_decay_params, "lr": base_lr, "weight_decay": 0.0}
            )

        return (
            groups
            if groups
            else [
                {
                    "params": self.parameters(),
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                }
            ]
        )
