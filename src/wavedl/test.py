"""
WaveDL - Testing & Inference Script
====================================
Target Environment: NVIDIA HPC GPUs (Single/Multi-GPU) | PyTorch 2.x | Python 3.11+

Production-grade inference script for evaluating trained WaveDL models:
  1. Model-Agnostic: Works with any registered architecture
  2. Flexible Data Loading: Supports NPZ/HDF5/MAT formats with auto-detection
  3. Comprehensive Metrics: R², Pearson, per-parameter MAE with physical units
  4. Publication Plots: 10 diagnostic plots with LaTeX styling and multi-format export
  5. Batch Inference: Efficient GPU utilization for large test sets
  6. Model Export: ONNX format for deployment in production systems

Usage:
    # Basic inference
    wavedl-test --checkpoint ./best_checkpoint --data_path test_data.npz

    # With visualization and detailed output
    wavedl-test --checkpoint ./best_checkpoint --data_path test_data.npz \\
        --plot --plot_format png pdf --output_dir ./test_results --save_predictions

    # Export model to ONNX for deployment
    wavedl-test --checkpoint ./best_checkpoint --data_path test_data.npz \\
        --export onnx --export_path model.onnx

Author: Ductho Le (ductho.le@outlook.com)
"""

# ==============================================================================
# ENVIRONMENT CONFIGURATION (must be before matplotlib import)
# ==============================================================================
# Auto-configure writable cache directories when home is not writable.
# Uses current working directory as fallback - works on HPC and local machines.
import os

# Import and call HPC cache setup before any library imports
from wavedl.utils import setup_hpc_cache_dirs


setup_hpc_cache_dirs()

import argparse  # noqa: E402
import logging  # noqa: E402
import pickle  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

# Force Agg backend before pyplot import (headless HPC nodes have no display)
import matplotlib  # noqa: E402


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

# Local imports
from wavedl.models import build_model, list_models  # noqa: E402
from wavedl.utils import (  # noqa: E402
    FIGURE_DPI,
    calc_pearson,
    load_test_data,
    plot_bland_altman,
    plot_correlation_heatmap,
    plot_error_boxplot,
    plot_error_cdf,
    plot_error_histogram,
    plot_prediction_vs_index,
    plot_qq,
    plot_relative_error,
    plot_residuals,
    plot_scientific_scatter,
)


# Optional dependencies for sparse matrix handling (now handled in utils/data.py)
try:
    from scipy.sparse import issparse

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from safetensors.torch import load_file as load_safetensors

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for testing."""
    parser = argparse.ArgumentParser(
        description="WaveDL Testing & Inference Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., ./cnn_test/best_checkpoint)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to test data (NPZ or MAT format)",
    )

    # Model specification
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model architecture (auto-detect if not specified). Available: {list_models()}",
    )

    # Data format
    parser.add_argument(
        "--format",
        type=str,
        default="auto",
        choices=["auto", "npz", "mat", "hdf5"],
        help="Data format (auto-detect by extension)",
    )
    parser.add_argument(
        "--input_key",
        type=str,
        default=None,
        help="Custom key name for input data in MAT/HDF5/NPZ files (e.g., 'X', 'waveforms')",
    )
    parser.add_argument(
        "--output_key",
        type=str,
        default=None,
        help="Custom key name for output data in MAT/HDF5/NPZ files (e.g., 'Y', 'labels')",
    )
    parser.add_argument(
        "--param_names",
        type=str,
        nargs="+",
        default=None,
        help="Parameter names for output (e.g., 'h' 'v11' 'v12')",
    )
    parser.add_argument(
        "--input_channels",
        type=int,
        default=None,
        help="Explicit number of input channels. Bypasses auto-detection heuristics "
        "for ambiguous 4D shapes (e.g., 3D volumes with small depth).",
    )

    # Inference options
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for inference"
    )
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers")

    # Output options
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Directory for saving results"
    )
    parser.add_argument(
        "--save_predictions", action="store_true", help="Save predictions to CSV"
    )
    parser.add_argument("--plot", action="store_true", help="Generate diagnostic plots")
    parser.add_argument(
        "--plot_format",
        type=str,
        nargs="+",
        default=["png"],
        help="Output format(s) for plots: png, pdf, svg, eps, tiff (default: png)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-sample predictions"
    )

    # Export options
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        choices=["onnx"],
        help="Export model format (onnx)",
    )
    parser.add_argument(
        "--export_path",
        type=str,
        default=None,
        help="Output path for exported model (default: {model_name}.onnx)",
    )
    parser.add_argument(
        "--export_opset",
        type=int,
        default=17,
        help="ONNX opset version (11-17, higher = newer ops)",
    )
    parser.add_argument(
        "--no_denorm",
        action="store_true",
        help="Disable de-normalization in ONNX (output normalized values instead)",
    )
    parser.add_argument(
        "--out_size",
        type=int,
        default=None,
        help="Override output size (inferred from scaler.pkl if targets missing)",
    )

    return parser.parse_args()


# ==============================================================================
# DATA LOADING (delegated to utils.data.load_test_data)
# ==============================================================================
def load_data_for_inference(
    file_path: str,
    format: str = "auto",
    input_key: str | None = None,
    output_key: str | None = None,
    input_channels: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Load test data for inference using the unified data loading pipeline.

    This is a thin wrapper around utils.data.load_test_data that adds logging.

    Args:
        file_path: Path to data file (NPZ, HDF5, or MAT v7.3)
        format: Format hint ('auto', 'npz', 'mat', 'hdf5')
        input_key: Custom key for input data (overrides auto-detection)
        output_key: Custom key for output data (overrides auto-detection)

    Returns:
        Tuple of:
            - X: Input tensor with channel dimension (N, 1, *spatial_dims)
            - y: Target tensor (N, T) or None if targets not present
    """
    from pathlib import Path

    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Determine format for logging
    if format == "auto":
        suffix = file_path_obj.suffix.lower()
        format_name = {
            ".npz": "npz",
            ".mat": "mat",
            ".h5": "hdf5",
            ".hdf5": "hdf5",
        }.get(suffix, "npz")
    else:
        format_name = format

    logging.info(f"Loading test data from: {file_path} (format: {format_name})")
    if input_key:
        logging.info(f"   Using custom input key: '{input_key}'")
    if output_key:
        logging.info(f"   Using custom output key: '{output_key}'")

    # Use the unified loader from utils.data
    X, y = load_test_data(
        file_path,
        format=format,
        input_key=input_key,
        output_key=output_key,
        input_channels=input_channels,
    )

    # Log results
    if y is not None:
        logging.info(
            f"   ✔ Loaded {len(X)} samples | Input: {X.shape} | Target: {y.shape}"
        )
    else:
        logging.info(
            f"   ✔ Loaded {len(X)} samples | Input: {X.shape} | Target: None (predictions only)"
        )

    return X, y


# ==============================================================================
# MODEL LOADING
# ==============================================================================
def load_checkpoint(
    checkpoint_dir: str,
    in_shape: tuple[int, ...],
    out_size: int,
    model_name: str | None = None,
) -> tuple[nn.Module, Any]:
    """
    Load model and scaler from Accelerate checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory
        in_shape: Input spatial shape - (L,) for 1D, (H, W) for 2D, or (D, H, W) for 3D
        out_size: Number of output parameters
        model_name: Model architecture name (auto-detect if None)

    Returns:
        model: Loaded model in eval mode
        scaler: StandardScaler for inverse transform
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Load training metadata
    meta_path = checkpoint_dir / "training_meta.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        best_val = meta.get("best_val_loss")
        val_str = f"{best_val:.6f}" if isinstance(best_val, (int, float)) else "N/A"
        logging.info(
            f"   Checkpoint from epoch {meta.get('epoch', 'unknown')}, "
            f"val_loss: {val_str}"
        )

    # Auto-detect model architecture if not specified
    if model_name is None:
        # First, try to read from training_meta.pkl (most reliable)
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            if meta.get("model_name"):
                model_name = meta["model_name"]
                logging.info(f"   Auto-detected model from checkpoint: {model_name}")

        # Fallback: try to detect from parent directory name (e.g., 'cnn_test' -> 'cnn')
        if model_name is None:
            parent_dir = checkpoint_dir.parent.name
            detected_name = (
                parent_dir.split("_")[0]
                if "_" in parent_dir
                else parent_dir.split("-")[0]
            )

            if detected_name in list_models():
                model_name = detected_name
                logging.info(f"   Auto-detected model from folder: {model_name}")
            else:
                raise ValueError(
                    f"Could not auto-detect model architecture.\\n"
                    f"Checkpoint missing 'model_name' in training_meta.pkl and folder '{parent_dir}' "
                    f"doesn't start with a known model name.\\n"
                    f"Please specify --model explicitly. Available models: {list_models()}"
                )

    logging.info(f"   Building model: {model_name}")
    # Use pretrained=False: checkpoint weights will overwrite any pretrained weights,
    # so downloading ImageNet weights is wasteful and breaks offline/HPC inference.
    model = build_model(
        model_name, in_shape=in_shape, out_size=out_size, pretrained=False
    )

    # Load weights (check multiple formats in order of preference)
    weight_path = None
    for fname in ["model.safetensors", "model.bin", "pytorch_model.bin"]:
        candidate = checkpoint_dir / fname
        if candidate.exists():
            weight_path = candidate
            break

    if weight_path is None:
        raise FileNotFoundError(
            f"No model weights found in {checkpoint_dir}. "
            f"Expected one of: model.safetensors, model.bin, pytorch_model.bin"
        )

    if HAS_SAFETENSORS and weight_path.suffix == ".safetensors":
        state_dict = load_safetensors(str(weight_path))
    elif weight_path.suffix == ".safetensors":
        # Safetensors file exists but library not installed
        raise ImportError(
            f"Checkpoint uses safetensors format ({weight_path.name}) but "
            f"'safetensors' package is not installed. Install it with:\n"
            f"    pip install safetensors\n"
            f"Or convert the checkpoint to PyTorch format (model.bin)."
        )
    else:
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)

    # Remove wrapper prefixes from checkpoints:
    # - 'module.' from DDP (DistributedDataParallel)
    # - '_orig_mod.' from torch.compile()
    cleaned_dict = {}
    for k, v in state_dict.items():
        key = k
        if key.startswith("module."):
            key = key[7:]  # Remove 'module.' (7 chars)
        if key.startswith("_orig_mod."):
            key = key[10:]  # Remove '_orig_mod.' (10 chars)
        cleaned_dict[key] = v
    state_dict = cleaned_dict

    model.load_state_dict(state_dict)
    model.eval()

    logging.info(f"   ✔ Loaded weights from: {weight_path.name}")

    # Load scaler
    scaler_path = checkpoint_dir.parent / "scaler.pkl"
    if not scaler_path.exists():
        # Try in checkpoint dir itself
        scaler_path = checkpoint_dir / "scaler.pkl"

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found. Expected at: {checkpoint_dir.parent}/scaler.pkl"
        )

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    logging.info(f"   ✔ Loaded scaler from: {scaler_path.name}")

    param_info = model.parameter_summary()
    logging.info(
        f"   Model: {param_info['trainable_parameters']:,} parameters ({param_info['total_mb']:.2f} MB)"
    )

    return model, scaler


# ==============================================================================
# INFERENCE
# ==============================================================================
@torch.inference_mode()
def run_inference(
    model: nn.Module,
    X: torch.Tensor,
    batch_size: int = 128,
    device: torch.device = None,
    num_workers: int = 0,
) -> np.ndarray:
    """
    Run batch inference on test data.

    Args:
        model: Trained model in eval mode
        X: Input tensor (N, C, *spatial_dims)
        batch_size: Batch size for inference
        device: Target device (auto-detect if None)
        num_workers: DataLoader workers (0 for single-threaded)

    Returns:
        predictions: Numpy array (N, out_size) - still in normalized space
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    dataset = TensorDataset(X)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type in ("cuda", "mps"),
    )

    predictions = []

    for (batch_x,) in tqdm(loader, desc="Inference", leave=False):
        batch_x = batch_x.to(device, non_blocking=True)
        preds = model(batch_x).cpu().numpy()
        predictions.append(preds)

    if not predictions:
        return np.empty((0, 0))
    return np.vstack(predictions)


# ==============================================================================
# ONNX EXPORT
# ==============================================================================
class ModelWithDenormalization(nn.Module):
    """
    Wrapper that combines a trained model with scaler inverse transform.

    This embeds the StandardScaler's mean_ and scale_ as model buffers,
    allowing the ONNX export to include de-normalization directly.
    Users get original-scale predictions without manual post-processing.

    Args:
        model: Trained PyTorch model
        scaler_mean: Mean values from StandardScaler (shape: [out_size])
        scaler_scale: Scale (std) values from StandardScaler (shape: [out_size])
    """

    def __init__(
        self, model: nn.Module, scaler_mean: np.ndarray, scaler_scale: np.ndarray
    ):
        super().__init__()
        self.model = model
        # Register as buffers (not trainable parameters, but saved with model)
        self.register_buffer(
            "scaler_mean", torch.tensor(scaler_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "scaler_scale", torch.tensor(scaler_scale, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with integrated de-normalization."""
        # Get normalized predictions from base model
        normalized_preds = self.model(x)
        # Apply inverse transform: original = normalized * scale + mean
        return normalized_preds * self.scaler_scale + self.scaler_mean


def export_to_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: str,
    opset_version: int = 17,
    validate: bool = True,
    model_name: str = "WaveDL_Model",
    scaler: object | None = None,
    include_denorm: bool = False,
) -> bool:
    """
    Export PyTorch model to ONNX format for production deployment.

    Features:
        - Dynamic batch size support
        - Comprehensive validation with numerical comparison
        - Embedded metadata (input/output names, descriptions)
        - Compatibility testing with ONNX runtime

    Args:
        model: Trained PyTorch model in eval mode
        sample_input: Sample input tensor for tracing (N, C, *spatial_dims)
        output_path: Path to save the ONNX model
        opset_version: ONNX opset version (11-17, default 17)
        validate: Whether to validate the exported model
        model_name: Model name embedded in ONNX metadata

    Returns:
        True if export and validation successful, False otherwise

    Example:
        >>> success = export_to_onnx(model, X_test[:1], "model.onnx")
        >>> if success:
        ...     print("Model exported successfully!")

    Note:
        For deployment in MATLAB/LabVIEW/C++, use the exported .onnx file
        with the appropriate ONNX runtime for your target platform.
    """
    import warnings

    # Wrap model with de-normalization if requested
    if include_denorm:
        if scaler is None:
            raise ValueError("scaler must be provided when include_denorm=True")
        logging.info("   Wrapping model with de-normalization layer (scaler embedded)")
        model = ModelWithDenormalization(
            model=model, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_
        )

    # Ensure model is in eval mode on CPU for consistent export
    model = model.cpu()
    model.eval()
    sample_input = sample_input.cpu()

    # Determine input/output names based on dimensions
    input_dims = sample_input.ndim - 2  # Exclude batch and channel
    if input_dims == 1:
        spatial_desc = "1D signal (length)"
    elif input_dims == 2:
        spatial_desc = "2D image (height, width)"
    else:
        spatial_desc = f"{input_dims}D volume"

    # Build dynamic axes for variable batch size
    dynamic_axes = {"input": {0: "batch_size"}, "predictions": {0: "batch_size"}}

    logging.info(f"📦 Exporting model to ONNX (opset {opset_version})...")
    logging.info(f"   Input shape: {tuple(sample_input.shape)} ({spatial_desc})")

    try:
        # Export with comprehensive settings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            torch.onnx.export(
                model,
                sample_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,  # Optimize for inference
                input_names=["input"],
                output_names=["predictions"],
                dynamic_axes=dynamic_axes,
                verbose=False,
            )

        logging.info(f"   ✔ Export completed: {output_path}")

        # Validate exported model
        if validate:
            return _validate_onnx_export(model, sample_input, output_path)

        return True

    except Exception as e:
        logging.error(f"   ✘ ONNX export failed: {e}")
        return False


def _validate_onnx_export(
    pytorch_model: nn.Module,
    sample_input: torch.Tensor,
    onnx_path: str,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> bool:
    """
    Validate ONNX model by comparing outputs with PyTorch model.

    Args:
        pytorch_model: Original PyTorch model
        sample_input: Input tensor for comparison
        onnx_path: Path to exported ONNX model
        rtol: Relative tolerance for numerical comparison
        atol: Absolute tolerance for numerical comparison

    Returns:
        True if validation passes, False otherwise
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        logging.warning(
            "   ⚠ ONNX validation skipped (install: pip install onnx onnxruntime)"
        )
        return True

    logging.info("   Validating ONNX model...")

    try:
        # 1. Check ONNX model structure
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logging.info("   ✔ ONNX model structure valid")

        # 2. Compare numerical outputs
        # PyTorch inference
        pytorch_model.eval()
        with torch.inference_mode():
            pytorch_output = pytorch_model(sample_input.cpu()).numpy()

        # ONNX Runtime inference
        ort_session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        onnx_output = ort_session.run(None, {"input": sample_input.numpy()})[0]

        # Numerical comparison
        max_diff = np.abs(pytorch_output - onnx_output).max()

        if np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol):
            logging.info(f"   ✔ Numerical validation passed (max diff: {max_diff:.2e})")
            return True
        else:
            logging.warning(f"   ⚠ Numerical mismatch (max diff: {max_diff:.2e})")
            return False

    except Exception as e:
        logging.warning(f"   ⚠ Validation error: {e}")
        return False


def get_onnx_model_info(onnx_path: str) -> dict:
    """
    Get metadata from exported ONNX model.

    Returns:
        Dictionary with model information
    """
    try:
        import onnx

        model = onnx.load(onnx_path)

        input_info = model.graph.input[0]
        output_info = model.graph.output[0]

        return {
            "opset_version": model.opset_import[0].version,
            "input_name": input_info.name,
            "input_shape": [
                d.dim_value or "dynamic" for d in input_info.type.tensor_type.shape.dim
            ],
            "output_name": output_info.name,
            "output_shape": [
                d.dim_value or "dynamic" for d in output_info.type.tensor_type.shape.dim
            ],
            "file_size_mb": Path(onnx_path).stat().st_size / (1024 * 1024),
        }
    except Exception as e:
        return {"error": str(e)}


# ==============================================================================
# METRICS & VISUALIZATION
# ==============================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute comprehensive regression metrics."""
    # Handle single-sample case
    if len(y_true) == 1:
        metrics = {
            "r2_score": float("nan"),  # R² undefined for single sample
            "pearson_corr": float("nan"),  # Correlation undefined for single sample
            "mae_avg": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(np.mean((y_true - y_pred) ** 2)),
        }
    else:
        metrics = {
            "r2_score": r2_score(y_true, y_pred),
            "pearson_corr": calc_pearson(y_true, y_pred),
            "mae_avg": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(np.mean((y_true - y_pred) ** 2)),
        }

    # Per-parameter MAE
    for i in range(y_true.shape[1]):
        metrics[f"mae_p{i}"] = mean_absolute_error(y_true[:, i], y_pred[:, i])

    return metrics


def print_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict[str, float],
    param_names: list | None = None,
    verbose: bool = False,
):
    """Print formatted test results."""
    n_params = y_true.shape[1]

    if param_names is None or len(param_names) != n_params:
        param_names = [f"P{i}" for i in range(n_params)]

    # Overall metrics
    print("\n" + "=" * 80)
    print("OVERALL TEST RESULTS")
    print("=" * 80)
    print(f"Samples:          {len(y_true)}")
    print(f"R² Score:         {metrics['r2_score']:.6f}")
    print(f"Pearson Corr:     {metrics['pearson_corr']:.6f}")
    print(f"RMSE:             {metrics['rmse']:.6f}")
    print(f"MAE (Avg):        {metrics['mae_avg']:.6f}")
    print("=" * 80)

    # Per-parameter MAE
    print("\nPER-PARAMETER MAE:")
    print("-" * 80)
    for i, name in enumerate(param_names):
        print(f"  {name:12s}: {metrics[f'mae_p{i}']:.6f}")
    print("-" * 80)

    # Sample-wise predictions (if verbose)
    if verbose:
        print("\nSAMPLE-WISE PREDICTIONS:")
        print("=" * 80)
        header = "ID   | " + " | ".join([f"{name:>8s}" for name in param_names])
        print(header)
        print("-" * 80)

        for i in range(min(len(y_true), 20)):  # Limit to first 20 samples
            true_str = " | ".join([f"{val:8.4f}" for val in y_true[i]])
            pred_str = " | ".join([f"{val:8.4f}" for val in y_pred[i]])
            print(f"TRUE | {true_str}")
            print(f"PRED | {pred_str}")
            print("-" * 80)

        if len(y_true) > 20:
            print(f"... ({len(y_true) - 20} more samples)")


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    param_names: list | None = None,
):
    """Save predictions to CSV file."""
    n_params = y_true.shape[1]

    if param_names is None or len(param_names) != n_params:
        param_names = [f"P{i}" for i in range(n_params)]

    # Create DataFrame
    columns = [f"True_{name}" for name in param_names] + [
        f"Pred_{name}" for name in param_names
    ]
    data = np.hstack([y_true, y_pred])
    df = pd.DataFrame(data, columns=columns)

    # Add error columns
    for i, name in enumerate(param_names):
        df[f"Error_{name}"] = df[f"Pred_{name}"] - df[f"True_{name}"]
        df[f"AbsError_{name}"] = np.abs(df[f"Error_{name}"])

    df.to_csv(output_path, index=False)
    logging.info(f"   ✔ Predictions saved to: {output_path}")


def plot_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    param_names: list | None = None,
    formats: list = ["png"],
):
    """Generate and save publication-quality plots with consistent LaTeX styling.

    Generates 10 plot types:
    1. Scatter plot (predictions vs ground truth)
    2. Error histogram (error distribution)
    3. Residual plot (residual vs predicted)
    4. Bland-Altman plot (method comparison)
    5. Q-Q plot (normality of errors)
    6. Correlation heatmap (error correlations between parameters)
    7. Relative error plot (% error vs true value)
    8. Cumulative error distribution (CDF)
    9. Prediction vs sample index
    10. Error box plot

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        output_dir: Directory to save plots
        param_names: Optional parameter names for labels
        formats: List of output formats (png, pdf, svg, eps, tiff)
    """
    n_params = y_true.shape[1]

    if param_names is None or len(param_names) != n_params:
        param_names = [f"P{i}" for i in range(n_params)]

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    def save_figure(fig, basename):
        """Save figure in all requested formats."""
        saved_formats = []
        for fmt in formats:
            fmt = fmt.lower().strip(".")
            filepath = output_dir / f"{basename}.{fmt}"
            fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight", format=fmt)
            saved_formats.append(fmt)
        plt.close(fig)
        logging.info(f"   ✔ Saved: {basename} ({', '.join(saved_formats)})")

    # 1. Scatter plot - predictions vs ground truth
    fig = plot_scientific_scatter(y_true, y_pred, param_names=param_names)
    save_figure(fig, "scatter_all")

    # 2. Error histogram - error distribution per parameter
    fig = plot_error_histogram(y_true, y_pred, param_names=param_names)
    save_figure(fig, "error_histogram")

    # 3. Residual plot - residual vs predicted value
    fig = plot_residuals(y_true, y_pred, param_names=param_names)
    save_figure(fig, "residuals")

    # 4. Bland-Altman plot - method agreement analysis
    fig = plot_bland_altman(y_true, y_pred, param_names=param_names)
    save_figure(fig, "bland_altman")

    # 5. Q-Q plot - check error normality
    fig = plot_qq(y_true, y_pred, param_names=param_names)
    save_figure(fig, "qq_plot")

    # 6. Correlation heatmap - error correlations (if multi-output)
    if n_params >= 2:
        fig = plot_correlation_heatmap(y_true, y_pred, param_names=param_names)
        save_figure(fig, "error_correlation")

    # 7. Relative error plot - % error vs true value
    fig = plot_relative_error(y_true, y_pred, param_names=param_names)
    save_figure(fig, "relative_error")

    # 8. Cumulative error distribution (CDF)
    fig = plot_error_cdf(y_true, y_pred, param_names=param_names)
    save_figure(fig, "error_cdf")

    # 9. Prediction vs sample index
    fig = plot_prediction_vs_index(y_true, y_pred, param_names=param_names)
    save_figure(fig, "prediction_vs_index")

    # 10. Error box plot
    fig = plot_error_boxplot(y_true, y_pred, param_names=param_names)
    save_figure(fig, "error_boxplot")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("Tester")

    # Device (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load test data
    X_test, y_test = load_data_for_inference(
        args.data_path,
        format=args.format,
        input_key=args.input_key,
        output_key=args.output_key,
        input_channels=args.input_channels,
    )
    in_shape = tuple(X_test.shape[2:])

    # Determine if we have ground truth targets
    has_targets = y_test is not None

    # Determine output size: from targets, from --out_size, or from scaler
    if args.out_size is not None:
        out_size = args.out_size
        logger.info(f"   Using explicit --out_size={out_size}")
    elif has_targets:
        out_size = y_test.shape[1]
    else:
        # Infer from scaler.pkl
        scaler_path = Path(args.checkpoint) / "scaler.pkl"
        if not scaler_path.exists():
            scaler_path = Path(args.checkpoint).parent / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                temp_scaler = pickle.load(f)
            out_size = len(temp_scaler.scale_)
            logger.info(f"   Inferred out_size={out_size} from scaler.pkl")
        else:
            raise ValueError(
                "Cannot determine output size. Provide targets, --out_size, "
                "or ensure scaler.pkl exists in checkpoint directory."
            )

    # Load model and scaler
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    model, scaler = load_checkpoint(args.checkpoint, in_shape, out_size, args.model)

    # Ensure output directory exists before any file operations (including ONNX export)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle ONNX export if requested
    if args.export == "onnx":
        # Determine output path
        if args.export_path:
            export_path = args.export_path
        else:
            model_name = args.model or Path(args.checkpoint).parent.name.split("_")[0]
            export_path = str(Path(args.output_dir) / f"{model_name}_model.onnx")

        # Export with sample input for tracing
        sample_input = X_test[:1]  # Single sample for tracing
        success = export_to_onnx(
            model=model,
            sample_input=sample_input,
            output_path=export_path,
            opset_version=args.export_opset,
            validate=True,
            scaler=scaler,
            include_denorm=not args.no_denorm,  # De-normalization ON by default
        )

        if success:
            # Print model info
            info = get_onnx_model_info(export_path)
            if "error" not in info:
                logger.info(f"   📊 Model size: {info['file_size_mb']:.2f} MB")
                logger.info(f"   📊 Input: {info['input_name']} {info['input_shape']}")
                logger.info(
                    f"   📊 Output: {info['output_name']} {info['output_shape']}"
                )
            logger.info(f"✅ ONNX export completed: {export_path}")
        else:
            logger.error("❌ ONNX export failed")
            return

        # If only export was requested (no other outputs), exit early
        if not args.save_predictions and not args.plot and not args.verbose:
            logger.info(
                "Export-only mode. Use --save_predictions or --plot for inference."
            )
            return

    # Run inference with timing
    logger.info(f"Running inference on {len(X_test)} samples...")
    import time

    inference_start = time.time()
    y_pred_scaled = run_inference(
        model, X_test, args.batch_size, device, num_workers=args.workers
    )
    inference_time = time.time() - inference_start

    # Calculate timing metrics
    n_samples = len(X_test)
    if n_samples > 0 and inference_time > 0:
        samples_per_sec = n_samples / inference_time
        ms_per_sample = (inference_time / n_samples) * 1000
    else:
        samples_per_sec = 0.0
        ms_per_sample = 0.0
    logger.info(
        f"   Inference time: {inference_time:.2f}s ({ms_per_sample:.2f} ms/sample, {samples_per_sec:.1f} samples/s)"
    )

    # Guard: empty dataset — nothing to evaluate or plot
    if n_samples == 0:
        logger.warning("No samples in test set — skipping evaluation.")
        return

    # Validate scaler dimensions match predictions
    if hasattr(scaler, "scale_") and len(scaler.scale_) != y_pred_scaled.shape[1]:
        raise ValueError(
            f"Scaler output dimension ({len(scaler.scale_)}) doesn't match "
            f"model output dimension ({y_pred_scaled.shape[1]}). "
            f"This may indicate a mismatched checkpoint or scaler.pkl."
        )

    # Inverse transform predictions
    y_pred = scaler.inverse_transform(y_pred_scaled)

    # Validate param_names length if provided
    if args.param_names and len(args.param_names) != y_pred.shape[1]:
        logger.warning(
            f"--param_names has {len(args.param_names)} names but model outputs "
            f"{y_pred.shape[1]} values. Using default names."
        )
        args.param_names = None  # Fall back to default

    if has_targets:
        # === MODE: With ground truth ===
        y_true = y_test.numpy()

        # Compute metrics
        metrics = compute_metrics(y_true, y_pred)

        # Print results
        print_results(y_true, y_pred, metrics, args.param_names, args.verbose)

        # Save predictions (with targets)
        if args.save_predictions:
            output_path = output_dir / "predictions.csv"
            save_predictions(y_true, y_pred, str(output_path), args.param_names)

        # Generate plots
        if args.plot:
            logger.info(f"Generating plots (formats: {', '.join(args.plot_format)})...")
            plot_results(
                y_true,
                y_pred,
                str(output_dir),
                args.param_names,
                formats=args.plot_format,
            )
    else:
        # === MODE: Predictions only (no ground truth) ===
        logger.info("No ground truth targets - skipping metrics and scatter plots")

        # Print prediction summary
        n_params = y_pred.shape[1]
        param_names = args.param_names or [f"P{i}" for i in range(n_params)]

        print("\n" + "=" * 80)
        print("PREDICTION SUMMARY (No Ground Truth)")
        print("=" * 80)
        print(f"Samples:          {len(y_pred)}")
        print(f"Output params:    {n_params}")
        print("-" * 80)
        for i, name in enumerate(param_names):
            print(
                f"  {name:12s}: mean={y_pred[:, i].mean():.6f}, std={y_pred[:, i].std():.6f}"
            )
        print("=" * 80)

        # Save predictions (without targets)
        if args.save_predictions:
            output_path = output_dir / "predictions.csv"
            columns = [f"Pred_{name}" for name in param_names]
            df = pd.DataFrame(y_pred, columns=columns)
            df.to_csv(output_path, index=False)
            logger.info(f"   ✔ Predictions saved to: {output_path}")

        if args.plot:
            logger.warning("Cannot generate scatter plots without ground truth targets")

    logger.info("✅ Testing completed successfully!")


if __name__ == "__main__":
    main()
