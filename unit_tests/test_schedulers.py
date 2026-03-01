"""
Unit Tests for Learning Rate Schedulers
=======================================

Tests for the scheduler factory and utility functions.

Author: Ductho Le (ductho.le@outlook.com)
"""

import os
import sys

import pytest
import torch.nn as nn
import torch.optim as optim


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavedl.utils.schedulers import get_scheduler, is_epoch_based, list_schedulers


class TestListSchedulers:
    """Tests for list_schedulers function."""

    def test_returns_list(self):
        """list_schedulers should return a list."""
        result = list_schedulers()
        assert isinstance(result, list)

    def test_contains_expected_schedulers(self):
        """list_schedulers should contain all expected scheduler names."""
        result = list_schedulers()
        expected = [
            "plateau",
            "cosine",
            "cosine_restarts",
            "onecycle",
            "step",
            "multistep",
            "exponential",
            "linear_warmup",
        ]
        for sched_name in expected:
            assert sched_name in result


class TestGetScheduler:
    """Tests for get_scheduler factory function."""

    @pytest.fixture
    def optimizer(self):
        model = nn.Linear(10, 5)
        return optim.Adam(model.parameters(), lr=1e-3)

    @pytest.mark.parametrize(
        "sched_name",
        [
            "plateau",
            "cosine",
            "cosine_restarts",
            "step",
            "multistep",
            "exponential",
            "linear_warmup",
        ],
    )
    def test_epoch_based_schedulers_instantiate(self, optimizer, sched_name):
        """All epoch-based schedulers should instantiate without error."""
        scheduler = get_scheduler(sched_name, optimizer, epochs=100)
        assert scheduler is not None

    def test_onecycle_requires_steps_per_epoch(self, optimizer):
        """OneCycleLR should require steps_per_epoch."""
        with pytest.raises(ValueError, match="steps_per_epoch"):
            get_scheduler("onecycle", optimizer, epochs=100)

    def test_onecycle_with_steps_per_epoch(self, optimizer):
        """OneCycleLR should work with steps_per_epoch."""
        scheduler = get_scheduler("onecycle", optimizer, epochs=100, steps_per_epoch=50)
        assert scheduler is not None

    def test_plateau_parameters(self, optimizer):
        """ReduceLROnPlateau should accept custom parameters."""
        scheduler = get_scheduler(
            "plateau", optimizer, patience=5, factor=0.2, min_lr=1e-7
        )
        assert scheduler.patience == 5
        assert scheduler.factor == 0.2

    def test_cosine_parameters(self, optimizer):
        """CosineAnnealingLR should accept T_max and eta_min."""
        scheduler = get_scheduler("cosine", optimizer, epochs=200, min_lr=1e-6)
        assert scheduler.T_max == 200
        assert scheduler.eta_min == 1e-6

    def test_step_parameters(self, optimizer):
        """StepLR should accept step_size and gamma."""
        scheduler = get_scheduler("step", optimizer, step_size=10, gamma=0.5)
        assert scheduler.step_size == 10
        assert scheduler.gamma == 0.5

    def test_multistep_default_milestones(self, optimizer):
        """MultiStepLR should generate default milestones."""
        scheduler = get_scheduler("multistep", optimizer, epochs=100)
        # Default: 30%, 60%, 90% of epochs
        assert 30 in scheduler.milestones
        assert 60 in scheduler.milestones
        assert 90 in scheduler.milestones

    def test_multistep_custom_milestones(self, optimizer):
        """MultiStepLR should accept custom milestones."""
        milestones = [10, 20, 30]
        scheduler = get_scheduler("multistep", optimizer, milestones=milestones)
        assert list(scheduler.milestones) == milestones

    def test_unknown_scheduler_raises_error(self, optimizer):
        """Unknown scheduler name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown scheduler"):
            get_scheduler("unknown_sched", optimizer)


class TestIsEpochBased:
    """Tests for is_epoch_based utility function."""

    def test_onecycle_is_batch_based(self):
        """OneCycleLR should be batch-based (not epoch-based)."""
        assert is_epoch_based("onecycle") is False

    @pytest.mark.parametrize("sched_name", ["plateau", "cosine", "step", "exponential"])
    def test_most_schedulers_are_epoch_based(self, sched_name):
        """Most schedulers should be epoch-based."""
        assert is_epoch_based(sched_name) is True

    def test_case_insensitive(self):
        """is_epoch_based should be case insensitive."""
        assert is_epoch_based("OneCycle") is False
        assert is_epoch_based("PLATEAU") is True


class TestSchedulerStep:
    """Tests for scheduler step functionality."""

    @pytest.fixture
    def optimizer(self):
        model = nn.Linear(10, 5)
        return optim.Adam(model.parameters(), lr=1e-3)

    def test_cosine_lr_decreases(self, optimizer):
        """Cosine scheduler should decrease LR over epochs."""
        initial_lr = optimizer.param_groups[0]["lr"]
        scheduler = get_scheduler("cosine", optimizer, epochs=10)

        # Step through some epochs
        for _ in range(5):
            scheduler.step()

        new_lr = optimizer.param_groups[0]["lr"]
        assert new_lr < initial_lr

    def test_step_lr_decreases_at_step_size(self, optimizer):
        """StepLR should decrease LR at step boundaries."""
        initial_lr = optimizer.param_groups[0]["lr"]
        scheduler = get_scheduler("step", optimizer, step_size=5, gamma=0.5)

        # Before step boundary
        for _ in range(4):
            scheduler.step()
        lr_before = optimizer.param_groups[0]["lr"]
        assert lr_before == pytest.approx(initial_lr)

        # After step boundary
        scheduler.step()
        lr_after = optimizer.param_groups[0]["lr"]
        assert lr_after == pytest.approx(initial_lr * 0.5)

    def test_plateau_requires_metric(self, optimizer):
        """ReduceLROnPlateau requires metric for step."""
        scheduler = get_scheduler("plateau", optimizer, patience=2)

        # Should accept metric
        scheduler.step(0.5)  # Pass validation loss

        # LR should remain same initially
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)

    def test_linear_warmup_increases_lr(self, optimizer):
        """Linear warmup should increase LR towards target."""
        # Start with low LR factor
        scheduler = get_scheduler(
            "linear_warmup", optimizer, warmup_epochs=5, start_factor=0.1
        )

        initial_lr = optimizer.param_groups[0]["lr"]

        # Step through warmup
        scheduler.step()
        lr_after_step = optimizer.param_groups[0]["lr"]

        # LR should have increased
        assert lr_after_step > initial_lr
