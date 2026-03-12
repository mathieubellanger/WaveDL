"""
Optimizers for Deep Learning Training
=====================================

Provides a comprehensive set of optimizers with a factory function
for easy selection via CLI arguments.

Supported Optimizers:
    - adamw: AdamW (default, best for most cases)
    - adam: Adam (legacy)
    - sgd: SGD with momentum
    - nadam: NAdam (Adam + Nesterov momentum)
    - radam: RAdam (variance-adaptive Adam)
    - rmsprop: RMSprop (good for RNNs)

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

from collections.abc import Iterator

import torch
import torch.optim as optim
from torch.nn import Parameter


# ==============================================================================
# OPTIMIZER REGISTRY
# ==============================================================================
def list_optimizers() -> list[str]:
    """
    Return list of available optimizer names.

    Returns:
        List of registered optimizer names
    """
    return ["adamw", "adam", "sgd", "nadam", "radam", "rmsprop"]


def get_optimizer(
    name: str,
    params: Iterator[Parameter] | list[dict],
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    # SGD-specific
    momentum: float = 0.9,
    nesterov: bool = False,
    dampening: float = 0.0,
    # Adam/AdamW-specific
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    amsgrad: bool = False,
    # RMSprop-specific
    alpha: float = 0.99,
    centered: bool = False,
    **kwargs,
) -> optim.Optimizer:
    """
    Factory function to create optimizer by name.

    Args:
        name: Optimizer name (see list_optimizers())
        params: Model parameters or parameter groups
        lr: Learning rate
        weight_decay: Weight decay (L2 penalty)
        momentum: Momentum factor (SGD, RMSprop)
        nesterov: Enable Nesterov momentum (SGD)
        dampening: Dampening for momentum (SGD)
        betas: Coefficients for computing running averages (Adam variants)
        eps: Term for numerical stability (Adam variants, RMSprop)
        amsgrad: Use AMSGrad variant (Adam variants)
        alpha: Smoothing constant (RMSprop)
        centered: Compute centered RMSprop
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Instantiated optimizer

    Raises:
        ValueError: If optimizer name is not recognized

    Example:
        >>> optimizer = get_optimizer("adamw", model.parameters(), lr=1e-3)
        >>> optimizer = get_optimizer("sgd", model.parameters(), lr=1e-2, momentum=0.9)
    """
    name_lower = name.lower()

    if name_lower == "adamw":
        return optim.AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            **kwargs,
        )

    elif name_lower == "adam":
        return optim.Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            **kwargs,
        )

    elif name_lower == "sgd":
        return optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            **kwargs,
        )

    elif name_lower == "nadam":
        return optim.NAdam(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, **kwargs
        )

    elif name_lower == "radam":
        return optim.RAdam(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, **kwargs
        )

    elif name_lower == "rmsprop":
        return optim.RMSprop(
            params,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
            **kwargs,
        )

    else:
        available = ", ".join(list_optimizers())
        raise ValueError(f"Unknown optimizer: '{name}'. Available options: {available}")


def get_optimizer_with_param_groups(
    name: str,
    model: torch.nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    no_decay_keywords: list[str] = None,
    **kwargs,
) -> optim.Optimizer:
    """
    Create optimizer with automatic parameter grouping.

    Separates parameters into decay and no-decay groups based on
    parameter names. By default, bias and normalization layer parameters
    are excluded from weight decay.

    Args:
        name: Optimizer name
        model: PyTorch model
        lr: Learning rate
        weight_decay: Weight decay for applicable parameters
        no_decay_keywords: Keywords to identify no-decay parameters
                          Default: ['bias', 'norm', 'bn', 'ln', 'gamma', 'beta']
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer with parameter groups configured

    Example:
        >>> optimizer = get_optimizer_with_param_groups(
        ...     "adamw", model, lr=1e-3, weight_decay=1e-4
        ... )
    """
    if no_decay_keywords is None:
        no_decay_keywords = ["bias", "norm", "bn", "ln", "gamma", "beta"]

    decay_params = []
    no_decay_params = []

    for name_param, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if any no-decay keyword is in parameter name
        if any(kw in name_param.lower() for kw in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append(
            {
                "params": decay_params,
                "weight_decay": weight_decay,
            }
        )
    if no_decay_params:
        param_groups.append(
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            }
        )

    # Fall back to all parameters if grouping fails
    if not param_groups:
        param_groups = [{"params": model.parameters()}]

    return get_optimizer(name, param_groups, lr=lr, weight_decay=0.0, **kwargs)
