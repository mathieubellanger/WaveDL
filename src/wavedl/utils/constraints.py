"""
Physical Constraints for Training
=================================

Soft constraint enforcement via penalty-based loss terms.

Usage:
    # Expression constraints
    wavedl-train --constraint "y0 > 0" --constraint_weight 0.1

    # Complex constraints via Python file
    wavedl-train --constraint_file my_constraint.py

Author: Ductho Le (ductho.le@outlook.com)
Version: 2.0.0
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F


if TYPE_CHECKING:
    from collections.abc import Callable


# ==============================================================================
# SAFE EXPRESSION PARSING
# ==============================================================================
SAFE_FUNCTIONS: dict[str, Callable] = {
    "sin": torch.sin,
    "cos": torch.cos,
    "tan": torch.tan,
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "abs": torch.abs,
    "relu": F.relu,
    "sigmoid": torch.sigmoid,
    "softplus": F.softplus,
    "tanh": torch.tanh,
    "min": torch.minimum,
    "max": torch.maximum,
    "pow": torch.pow,
    "clamp": torch.clamp,
}

INPUT_AGGREGATES: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "x_mean": lambda x: x.mean(dim=tuple(range(1, x.ndim))),
    "x_sum": lambda x: x.sum(dim=tuple(range(1, x.ndim))),
    "x_max": lambda x: x.amax(dim=tuple(range(1, x.ndim))),
    "x_min": lambda x: x.amin(dim=tuple(range(1, x.ndim))),
    "x_std": lambda x: x.std(dim=tuple(range(1, x.ndim))),
    "x_energy": lambda x: (x**2).sum(dim=tuple(range(1, x.ndim))),
}


# ==============================================================================
# SOFT CONSTRAINTS
# ==============================================================================
class ExpressionConstraint(nn.Module):
    """
    Soft constraint via string expression.

    Parses mathematical expressions using Python's AST for safe evaluation.
    Supports output variables (y0, y1, ...), input aggregates (x_mean, ...),
    and whitelisted math functions.

    Example:
        >>> constraint = ExpressionConstraint("y0 - y1 * y2")
        >>> penalty = constraint(predictions, inputs)

        >>> constraint = ExpressionConstraint("sin(y0) + cos(y1)")
        >>> penalty = constraint(predictions, inputs)
    """

    def __init__(self, expression: str, reduction: str = "mse"):
        """
        Args:
            expression: Mathematical expression to evaluate (should equal 0)
            reduction: How to reduce violations - 'mse' or 'mae'
        """
        super().__init__()
        self.expression = expression
        self.reduction = reduction
        self._tree = ast.parse(expression, mode="eval")
        self._validate(self._tree)

    def _validate(self, tree: ast.Expression) -> None:
        """Validate that expression only uses safe functions."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id not in SAFE_FUNCTIONS:
                    raise ValueError(
                        f"Unsafe function '{node.func.id}' in constraint. "
                        f"Allowed: {list(SAFE_FUNCTIONS.keys())}"
                    )

    def _eval(
        self, node: ast.AST, pred: torch.Tensor, inputs: torch.Tensor | None
    ) -> torch.Tensor:
        """Recursively evaluate AST node."""
        if isinstance(node, ast.Constant):
            return torch.tensor(node.value, device=pred.device, dtype=pred.dtype)
        elif isinstance(node, ast.Name):
            name = node.id
            # Output variable: y0, y1, ...
            if name.startswith("y") and name[1:].isdigit():
                idx = int(name[1:])
                if idx >= pred.shape[1]:
                    raise ValueError(
                        f"Output index {idx} out of range. "
                        f"Model has {pred.shape[1]} outputs."
                    )
                return pred[:, idx]
            # Input aggregate: x_mean, x_sum, ...
            elif name in INPUT_AGGREGATES:
                if inputs is None:
                    raise ValueError(
                        f"Constraint uses '{name}' but inputs not provided."
                    )
                return INPUT_AGGREGATES[name](inputs)
            else:
                raise ValueError(
                    f"Unknown variable '{name}'. "
                    f"Use y0, y1, ... for outputs or {list(INPUT_AGGREGATES.keys())} for inputs."
                )
        elif isinstance(node, ast.BinOp):
            left = self._eval(node.left, pred, inputs)
            right = self._eval(node.right, pred, inputs)
            ops = {
                ast.Add: torch.add,
                ast.Sub: torch.sub,
                ast.Mult: torch.mul,
                ast.Div: torch.div,
                ast.Pow: torch.pow,
                ast.Mod: torch.remainder,
            }
            if type(node.op) not in ops:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return ops[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand, pred, inputs)
            if isinstance(node.op, ast.USub):
                return -operand
            elif isinstance(node.op, ast.UAdd):
                return operand
            else:
                raise ValueError(
                    f"Unsupported unary operator: {type(node.op).__name__}"
                )
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls supported (e.g., sin(x))")
            func_name = node.func.id
            if func_name not in SAFE_FUNCTIONS:
                raise ValueError(f"Unsafe function: {func_name}")
            args = [self._eval(arg, pred, inputs) for arg in node.args]
            return SAFE_FUNCTIONS[func_name](*args)
        elif isinstance(node, ast.Compare):
            # Comparison operators: y0 > 0, y0 < 1, y0 >= 0, y0 <= 1
            # Returns penalty (violation amount) when constraint is not satisfied
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ValueError(
                    "Only single comparisons supported (e.g., 'y0 > 0', not 'y0 > 0 > y1')"
                )
            left = self._eval(node.left, pred, inputs)
            right = self._eval(node.comparators[0], pred, inputs)
            op = node.ops[0]

            # Return violation amount (0 if satisfied, positive if violated)
            if isinstance(
                op, (ast.Gt, ast.GtE)
            ):  # y0 > right → penalize if y0 <= right
                return F.relu(right - left)
            elif isinstance(
                op, (ast.Lt, ast.LtE)
            ):  # y0 < right → penalize if y0 >= right
                return F.relu(left - right)
            elif isinstance(op, ast.Eq):  # y0 == right → penalize difference
                return torch.abs(left - right)
            elif isinstance(op, ast.NotEq):  # y0 != right → not useful as constraint
                raise ValueError(
                    "'!=' is not a valid constraint. Use '==' for equality constraints."
                )
            else:
                raise ValueError(
                    f"Unsupported comparison operator: {type(op).__name__}"
                )
        elif isinstance(node, ast.Subscript):
            # Input indexing: x[0], x[0,5], x[0,5,10]
            if not isinstance(node.value, ast.Name) or node.value.id != "x":
                raise ValueError(
                    "Subscript indexing only supported for 'x' (inputs). "
                    "Use x[i], x[i,j], or x[i,j,k]."
                )
            if inputs is None:
                raise ValueError("Constraint uses 'x[...]' but inputs not provided.")

            # Parse indices from the slice
            indices = self._parse_subscript_indices(node.slice)

            # Auto-squeeze channel dimension for single-channel inputs
            # This allows x[i,j] syntax for (B, 1, H, W) inputs instead of x[c,i,j]
            inputs_for_indexing = inputs
            if inputs.ndim >= 3 and inputs.shape[1] == 1:
                inputs_for_indexing = inputs.squeeze(1)  # (B, 1, H, W) → (B, H, W)

            # Validate dimensions match
            # inputs shape: (batch, dim1) or (batch, dim1, dim2) or (batch, dim1, dim2, dim3)
            input_ndim = inputs_for_indexing.ndim - 1  # Exclude batch dimension
            if len(indices) != input_ndim:
                raise ValueError(
                    f"Input has {input_ndim}D shape (after channel squeeze), but got {len(indices)} indices. "
                    f"Use x[i] for 1D, x[i,j] for 2D, x[i,j,k] for 3D inputs."
                )

            # Extract the value at the specified indices (for entire batch)
            if len(indices) == 1:
                return inputs_for_indexing[:, indices[0]]
            elif len(indices) == 2:
                return inputs_for_indexing[:, indices[0], indices[1]]
            elif len(indices) == 3:
                return inputs_for_indexing[:, indices[0], indices[1], indices[2]]
            else:
                raise ValueError("Only 1D, 2D, or 3D input indexing supported.")
        elif isinstance(node, ast.Expression):
            return self._eval(node.body, pred, inputs)
        else:
            raise ValueError(f"Unsupported AST node type: {type(node).__name__}")

    def _parse_subscript_indices(self, slice_node: ast.AST) -> list[int]:
        """Parse subscript indices from AST slice node."""
        if isinstance(slice_node, ast.Constant):
            # Single index: x[0]
            return [int(slice_node.value)]
        elif isinstance(slice_node, ast.Tuple):
            # Multiple indices: x[0,5] or x[0,5,10]
            indices = []
            for elt in slice_node.elts:
                if not isinstance(elt, ast.Constant):
                    raise ValueError(
                        "Only constant indices supported in x[...]. "
                        "Use x[0,5] not x[i,j]."
                    )
                indices.append(int(elt.value))
            return indices
        else:
            raise ValueError(
                f"Unsupported subscript type: {type(slice_node).__name__}. "
                "Use x[0], x[0,5], or x[0,5,10]."
            )

    def forward(
        self, pred: torch.Tensor, inputs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute constraint violation penalty.

        Args:
            pred: Model predictions of shape (N, num_outputs)
            inputs: Model inputs of shape (N, ...) for input-dependent constraints

        Returns:
            Scalar penalty value
        """
        violation = self._eval(self._tree, pred, inputs)
        if self.reduction == "mse":
            return (violation**2).mean()
        else:  # mae
            return violation.abs().mean()

    def __repr__(self) -> str:
        return (
            f"ExpressionConstraint('{self.expression}', reduction='{self.reduction}')"
        )


class FileConstraint(nn.Module):
    """
    Load constraint function from Python file.

    The file must define a function `constraint(pred, inputs=None)` that
    returns per-sample violation values.

    Example file (my_constraint.py):
        import torch

        def constraint(pred, inputs=None):
            # Monotonicity: y0 < y1 < y2
            diffs = pred[:, 1:] - pred[:, :-1]
            return torch.relu(-diffs).sum(dim=1)

    Usage:
        >>> constraint = FileConstraint("my_constraint.py")
        >>> penalty = constraint(predictions, inputs)
    """

    def __init__(self, file_path: str, reduction: str = "mse"):
        """
        Args:
            file_path: Path to Python file containing constraint function
            reduction: How to reduce violations - 'mse' or 'mae'
        """
        super().__init__()
        self.file_path = file_path
        self.reduction = reduction

        # Load module from file
        module_name = f"constraint_module_{id(self)}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load constraint file: {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "constraint"):
            raise ValueError(
                f"Constraint file must define 'constraint(pred, inputs)' function: {file_path}"
            )

        self._constraint_fn = module.constraint

    def forward(
        self, pred: torch.Tensor, inputs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Evaluate constraint from loaded function."""
        violation = self._constraint_fn(pred, inputs)
        if self.reduction == "mse":
            return (violation**2).mean()
        else:
            return violation.abs().mean()

    def __repr__(self) -> str:
        return f"FileConstraint('{self.file_path}')"


# ==============================================================================
# COMBINED LOSS WRAPPER
# ==============================================================================
class PhysicsConstrainedLoss(nn.Module):
    """
    Combine base loss with constraint penalties.

    Total Loss = Base Loss + Σ(weight_i × constraint_i)

    Constraints are evaluated in **physical space** (denormalized) while
    the base loss is computed in normalized space for stable training.

    Example:
        >>> base_loss = nn.MSELoss()
        >>> constraints = [ExpressionConstraint("y0 - y1*y2")]
        >>> criterion = PhysicsConstrainedLoss(
        ...     base_loss,
        ...     constraints,
        ...     weights=[0.1],
        ...     output_mean=[10, 5, 50],
        ...     output_std=[2, 1, 10],
        ... )
        >>> loss = criterion(pred, target, inputs)
    """

    def __init__(
        self,
        base_loss: nn.Module,
        constraints: list[nn.Module] | None = None,
        weights: list[float] | None = None,
        output_mean: torch.Tensor | list[float] | None = None,
        output_std: torch.Tensor | list[float] | None = None,
    ):
        """
        Args:
            base_loss: Base loss function (e.g., MSELoss)
            constraints: List of constraint modules
            weights: Weight for each constraint. If shorter than constraints,
                     last weight is repeated.
            output_mean: Mean of each output (for denormalization). Shape: (num_outputs,)
            output_std: Std of each output (for denormalization). Shape: (num_outputs,)
        """
        super().__init__()
        self.base_loss = base_loss
        self.constraints = nn.ModuleList(constraints or [])
        self.weights = weights or [0.1]

        # Store scaler as buffers (moves to correct device automatically)
        if output_mean is not None:
            if not isinstance(output_mean, torch.Tensor):
                output_mean = torch.tensor(output_mean, dtype=torch.float32)
            self.register_buffer("output_mean", output_mean)
        else:
            self.register_buffer("output_mean", None)

        if output_std is not None:
            if not isinstance(output_std, torch.Tensor):
                output_std = torch.tensor(output_std, dtype=torch.float32)
            self.register_buffer("output_std", output_std)
        else:
            self.register_buffer("output_std", None)

    def _denormalize(self, pred: torch.Tensor) -> torch.Tensor:
        """Convert normalized predictions to physical values."""
        if self.output_mean is None or self.output_std is None:
            return pred
        return pred * self.output_std + self.output_mean

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            pred: Model predictions (normalized)
            target: Ground truth targets (normalized)
            inputs: Model inputs (for input-dependent constraints)

        Returns:
            Combined loss value
        """
        # Base loss in normalized space (stable gradients)
        loss = self.base_loss(pred, target)

        # Denormalize for constraint evaluation (physical units)
        pred_physical = self._denormalize(pred)

        for i, constraint in enumerate(self.constraints):
            weight = self.weights[i] if i < len(self.weights) else self.weights[-1]
            penalty = constraint(pred_physical, inputs)
            loss = loss + weight * penalty

        return loss

    def __repr__(self) -> str:
        has_scaler = self.output_mean is not None
        return f"PhysicsConstrainedLoss(base={self.base_loss}, constraints={len(self.constraints)}, denormalize={has_scaler})"


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================
def build_constraints(
    expressions: list[str] | None = None,
    file_path: str | None = None,
    reduction: str = "mse",
) -> list[nn.Module]:
    """
    Build soft constraint modules from CLI arguments.

    Args:
        expressions: Expression constraints (e.g., ["y0 - y1*y2", "y0 > 0"])
        file_path: Path to Python constraint file
        reduction: Reduction mode for penalties

    Returns:
        List of constraint modules
    """
    constraints: list[nn.Module] = []

    for expr in expressions or []:
        constraints.append(ExpressionConstraint(expr, reduction))

    if file_path:
        constraints.append(FileConstraint(file_path, reduction))

    return constraints
