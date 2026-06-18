"""
CaFormer: MetaFormer with Convolution and Attention
====================================================

CaFormer implements the MetaFormer architecture using depthwise separable
convolutions in early stages and vanilla self-attention in later stages.

**Key Features**:
    - MetaFormer principle: architecture > token mixer
    - Hybrid: Conv (early) + Attention (late)
    - StarReLU activation for efficiency
    - State-of-the-art on ImageNet (85.5%)

**Variants**:
    - caformer_s18: 26M params
    - caformer_s36: 39M params
    - caformer_m36: 56M params

**Related Models**:
    - PoolFormer: Uses pooling instead of attention
    - ConvFormer: Uses only convolutions

**Requirements**:
    - timm >= 0.9.0 (for CaFormer models)

Reference:
    Yu, W., et al. (2023). MetaFormer Baselines for Vision.
    TPAMI 2023. https://arxiv.org/abs/2210.13452

Author: Ductho Le (ductho.le@outlook.com)
"""

import torch
import torch.nn as nn

from wavedl.models._pretrained_utils import build_regression_head
from wavedl.models.base import BaseModel
from wavedl.models.registry import register_model


__all__ = [
    "CaFormerBase",
    "CaFormerM36",
    "CaFormerS18",
    "CaFormerS36",
    "PoolFormerS12",
]


# =============================================================================
# CAFORMER BASE CLASS
# =============================================================================


class CaFormerBase(BaseModel):
    """
    CaFormer base class wrapping timm implementation.

    MetaFormer with conv (early) + attention (late) token mixing.
    2D only.
    """

    def __init__(
        self,
        in_shape: tuple[int, int],
        out_size: int,
        model_name: str = "caformer_s18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)

        if len(in_shape) != 2:
            raise ValueError(f"CaFormer requires 2D input (H, W), got {len(in_shape)}D")

        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.model_name = model_name

        # Try to load from timm
        try:
            import timm

            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
            )

            # Get feature dimension (eval mode to preserve pretrained BN stats)
            with torch.no_grad():
                self.backbone.eval()
                dummy = torch.zeros(1, 3, *in_shape)
                features = self.backbone(dummy)
                in_features = features.shape[-1]
                self.backbone.train()

        except ImportError:
            raise ImportError(
                "timm >= 0.9.0 is required for CaFormer. "
                "Install with: pip install timm>=0.9.0"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load CaFormer model '{model_name}': {e}"
            ) from e

        # Adapt input channels (3 -> 1)
        self._adapt_input_channels()

        # Regression head
        self.head = build_regression_head(in_features, out_size, dropout_rate)

        if freeze_backbone:
            self._freeze_backbone()

    def _adapt_input_channels(self):
        """Adapt first conv layer for single-channel input."""
        # CaFormer uses stem for first layer
        if not hasattr(self.backbone, "stem"):
            import warnings

            warnings.warn(
                "CaFormer backbone has no 'stem' attribute. "
                "Input channel adaptation skipped — model may fail "
                "with single-channel input.",
                UserWarning,
                stacklevel=2,
            )
            return

        first_conv = None
        # Find first conv in stem
        for name, module in self.backbone.stem.named_modules():
            if isinstance(module, nn.Conv2d):
                first_conv = (name, module)
                break

        if first_conv is not None:
            name, old_conv = first_conv
            new_conv = self._make_new_conv(old_conv)
            # Set the new conv (handle nested structure)
            self._set_module(self.backbone.stem, name, new_conv)
        else:
            import warnings

            warnings.warn(
                "No Conv2d found in CaFormer stem. "
                "Input channel adaptation skipped — model may fail "
                "with single-channel input.",
                UserWarning,
                stacklevel=2,
            )

    def _make_new_conv(self, old_conv: nn.Conv2d) -> nn.Conv2d:
        """Create new conv layer with 1 input channel."""
        new_conv = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if self.pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
        return new_conv

    def _set_module(self, parent: nn.Module, name: str, module: nn.Module):
        """Set a nested module by name, supporting integer-indexed children."""
        parts = name.split(".")
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        final = parts[-1]
        if final.isdigit():
            parent[int(final)] = module
        else:
            setattr(parent, final, module)

    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


# =============================================================================
# REGISTERED VARIANTS
# =============================================================================


@register_model("caformer_s18")
class CaFormerS18(CaFormerBase):
    """
    CaFormer-S18: ~23.2M backbone parameters.

    MetaFormer with conv + attention.
    2D only.

    Example:
        >>> model = CaFormerS18(in_shape=(224, 224), out_size=3)
        >>> x = torch.randn(4, 1, 224, 224)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="caformer_s18",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"CaFormer_S18(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


@register_model("caformer_s36")
class CaFormerS36(CaFormerBase):
    """
    CaFormer-S36: ~36.2M backbone parameters.

    Deeper MetaFormer variant.
    2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="caformer_s36",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"CaFormer_S36(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


@register_model("caformer_m36")
class CaFormerM36(CaFormerBase):
    """
    CaFormer-M36: ~52.6M backbone parameters.

    Medium-size MetaFormer variant.
    2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="caformer_m36",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"CaFormer_M36(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


@register_model("poolformer_s12")
class PoolFormerS12(CaFormerBase):
    """
    PoolFormer-S12: ~11.4M backbone parameters.

    MetaFormer with simple pooling token mixer.
    Proves that architecture matters more than complex attention.
    2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="poolformer_s12",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"PoolFormer_S12(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )
