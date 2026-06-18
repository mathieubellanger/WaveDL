"""
Unit Tests for the Mamba Selective Scan
=======================================

Verifies that the parallel (single-chunk) and chunked closed-form scans match a
naive sequential implementation of the SSM recurrence

    h_t = A_bar_t · h_{t-1} + BX_t,   y_t = (C_t · h_t) + D · x_t

This guards against off-by-one errors in the prefix-product reconstruction.

Author: Ductho Le (ductho.le@outlook.com)
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

mamba = pytest.importorskip("wavedl.models.mamba")
SelectiveSSM = mamba.SelectiveSSM


def _sequential_reference(x, delta, A, B, C, D):
    """Ground-truth sequential SSM scan (the operator the parallel scan emulates)."""
    A_bar = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
    BX = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)
    bsz, length, d_inner, d_state = A_bar.shape
    h = torch.zeros(bsz, d_inner, d_state, dtype=x.dtype)
    ys = []
    for t in range(length):
        h = A_bar[:, t] * h + BX[:, t]
        y_t = (C[:, t].unsqueeze(1) * h).sum(-1) + D * x[:, t]
        ys.append(y_t)
    return torch.stack(ys, dim=1)


def _make_inputs(length, d_inner=4, d_state=3, bsz=2, seed=0):
    """Well-conditioned random inputs.

    A and delta are kept mild so the discretized A_bar = exp(delta·A) stays near
    1 and the cumulative product does not underflow. This isolates *correctness*
    of the closed-form scan from the (separate, pre-existing) numerical
    sensitivity of the parallel formulation for tiny A_bar over long sequences.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(bsz, length, d_inner, generator=g)
    delta = F.softplus(torch.randn(bsz, length, d_inner, generator=g) * 0.3)  # ~O(0.5)
    A = -(torch.rand(d_state, generator=g) * 0.25 + 0.05)  # in [-0.30, -0.05]
    B = torch.randn(bsz, length, d_state, generator=g)
    C = torch.randn(bsz, length, d_state, generator=g)
    D = torch.randn(d_inner, generator=g)
    return x, delta, A, B, C, D


class TestMambaSelectiveScan:
    """The closed-form scans must reproduce the true SSM recurrence."""

    def test_single_scan_matches_sequential(self):
        ssm = SelectiveSSM(d_model=2, d_state=3, chunk_size=4)  # d_inner = 4
        inputs = _make_inputs(length=6)
        with torch.no_grad():
            y = ssm._selective_scan_single(*inputs)
        y_ref = _sequential_reference(*inputs)
        max_diff = (y - y_ref).abs().max().item()
        assert torch.allclose(y, y_ref, atol=1e-5), f"max abs diff {max_diff:.2e}"

    def test_chunked_scan_matches_sequential(self):
        ssm = SelectiveSSM(d_model=2, d_state=3, chunk_size=4)
        inputs = _make_inputs(length=10)  # spans multiple chunks (> chunk_size)
        with torch.no_grad():
            y = ssm._chunked_selective_scan(*inputs)
        y_ref = _sequential_reference(*inputs)
        max_diff = (y - y_ref).abs().max().item()
        assert torch.allclose(y, y_ref, atol=1e-5), f"max abs diff {max_diff:.2e}"

    def test_single_and_chunked_agree(self):
        ssm = SelectiveSSM(d_model=2, d_state=3, chunk_size=4)
        inputs = _make_inputs(length=9)
        with torch.no_grad():
            y_single = ssm._selective_scan_single(*inputs)
            y_chunked = ssm._chunked_selective_scan(*inputs)
        assert torch.allclose(y_single, y_chunked, atol=1e-5)
