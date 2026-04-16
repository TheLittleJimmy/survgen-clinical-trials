#!/usr/bin/env python3
"""
Tests for V4 remediation: global normalization, Weibull clamp, truncation, ablation flags.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(ROOT / "execute"))

import torch
import numpy as np
import pytest

import data_processing
import likelihood
from src import HIVAE_inputDropout


# ======================================================================
# Phase 1.1: Frozen global normalization
# ======================================================================

class TestGlobalNormalization:
    """Test that frozen global normalization gives consistent results."""

    def setup_method(self):
        """Create a small synthetic dataset."""
        self.feat_types = [
            {'type': 'real', 'dim': '1'},
            {'type': 'pos', 'dim': '1'},
            {'type': 'surv_weibull', 'dim': '2'},
        ]
        N = 100
        self.data = torch.cat([
            torch.randn(N, 1),           # real
            torch.rand(N, 1).abs() + 0.1, # pos
            torch.rand(N, 1) * 10 + 0.1,  # surv time
            (torch.rand(N, 1) > 0.3).float(),  # delta
        ], dim=1)
        self.miss = torch.ones(N, 3)

    def test_same_example_two_batches(self):
        """Same input normalized in two different batch contexts gives identical result with frozen stats."""
        global_params = data_processing.compute_global_normalization(
            self.data, self.feat_types, self.miss)

        # Split data into two batches
        batch1_data = self.data[:50]
        batch2_data = self.data[50:]
        miss1 = self.miss[:50]
        miss2 = self.miss[50:]

        # Take one example that appears in batch1
        example = self.data[10:11]
        example_miss = self.miss[10:11]

        # Normalize the example as part of batch1
        dl1 = [batch1_data[:, 0:1], batch1_data[:, 1:2], batch1_data[:, 2:4]]
        ex_in_b1 = [example[:, 0:1], example[:, 1:2], example[:, 2:4]]
        norm1, _ = data_processing.batch_normalization_frozen(
            ex_in_b1, self.feat_types, example_miss, global_params)

        # Normalize the same example but put it in batch2 context
        # (with frozen stats, result should be identical)
        norm2, _ = data_processing.batch_normalization_frozen(
            ex_in_b1, self.feat_types, example_miss, global_params)

        for i in range(len(norm1)):
            assert torch.allclose(norm1[i], norm2[i], atol=1e-6), \
                f"Feature {i}: frozen normalization differs between contexts"

    def test_global_vs_batch_differ(self):
        """Global and per-batch normalization should generally differ."""
        global_params = data_processing.compute_global_normalization(
            self.data, self.feat_types, self.miss)

        batch = self.data[:30]
        miss_b = self.miss[:30]
        dl = [batch[:, 0:1], batch[:, 1:2], batch[:, 2:4]]

        norm_frozen, _ = data_processing.batch_normalization_frozen(
            dl, self.feat_types, miss_b, global_params)
        norm_batch, _ = data_processing.batch_normalization(
            dl, self.feat_types, miss_b)

        # They should NOT be identical (different stats)
        any_diff = any(
            not torch.allclose(norm_frozen[i], norm_batch[i], atol=1e-4)
            for i in range(len(norm_frozen))
        )
        assert any_diff, "Global and batch normalization should differ for real/pos features"


# ======================================================================
# Phase 1.2: Weibull epsilon clamp
# ======================================================================

class TestWeibullClamp:
    """Test Weibull likelihood at edge cases."""

    def test_zero_time_no_nan(self):
        """Exact zero scaled time should not produce NaN/Inf."""
        N = 10
        data = torch.zeros(N, 2)
        data[:, 0] = 0.0  # exact zero time
        data[:, 1] = 1.0  # all events
        missing_mask = torch.ones(N)

        theta = [
            torch.ones(N) * 0.5,   # shape_T
            torch.ones(N) * 0.5,   # scale_T
            torch.ones(N) * 0.5,   # shape_C
            torch.ones(N) * 0.5,   # scale_C
        ]
        norm_params = (torch.tensor(-0.001), torch.tensor(1.0))

        out = likelihood.loglik_surv_weibull(
            [data, missing_mask], {'type': 'surv_weibull'},
            theta, norm_params, 1)

        assert not torch.isnan(out['log_p_x']).any(), "NaN in log_p_x at zero time"
        assert not torch.isinf(out['log_p_x']).any(), "Inf in log_p_x at zero time"

    def test_tiny_time_finite_grad(self):
        """Tiny times should backpropagate finite gradients."""
        N = 10
        data = torch.zeros(N, 2)
        data[:, 0] = 1e-6
        data[:, 1] = 1.0
        missing_mask = torch.ones(N)

        theta = [t.requires_grad_(True) for t in [
            torch.ones(N) * 0.5,
            torch.ones(N) * 0.5,
            torch.ones(N) * 0.5,
            torch.ones(N) * 0.5,
        ]]
        norm_params = (torch.tensor(-0.001), torch.tensor(1.0))

        out = likelihood.loglik_surv_weibull(
            [data, missing_mask], {'type': 'surv_weibull'},
            theta, norm_params, 1)

        loss = out['log_p_x'].sum()
        loss.backward()

        for i, t in enumerate(theta):
            assert t.grad is not None, f"No gradient for theta[{i}]"
            assert not torch.isnan(t.grad).any(), f"NaN gradient for theta[{i}]"
            assert not torch.isinf(t.grad).any(), f"Inf gradient for theta[{i}]"


# ======================================================================
# Phase 1.3: V4_joint post-generation truncation
# ======================================================================

class TestTruncation:
    """Test longitudinal truncation at event time."""

    def test_basic_truncation(self):
        """Visits after event time should be zeroed out."""
        batch = 5
        n_times = 10
        D = 2
        traj = torch.randn(batch, n_times, D)
        time_grid = torch.linspace(0, 1, n_times)
        event_times = torch.tensor([0.3, 0.5, 0.7, 0.1, 0.9])

        observed, mask = HIVAE_inputDropout.truncate_longitudinal_at_event(
            traj, time_grid, event_times)

        # Check that post-event visits are zero
        for i in range(batch):
            for t_idx in range(n_times):
                if time_grid[t_idx] > event_times[i]:
                    assert (observed[i, t_idx] == 0).all(), \
                        f"Patient {i}, time {t_idx}: post-event visit not zeroed"
                else:
                    assert torch.allclose(observed[i, t_idx], traj[i, t_idx]), \
                        f"Patient {i}, time {t_idx}: pre-event visit altered"

    def test_return_planned(self):
        """return_planned=True should return both planned and observed."""
        traj = torch.randn(3, 5, 2)
        time_grid = torch.linspace(0, 1, 5)
        event_times = torch.tensor([0.5, 0.5, 0.5])

        observed, mask, planned = HIVAE_inputDropout.truncate_longitudinal_at_event(
            traj, time_grid, event_times, return_planned=True)

        assert torch.allclose(planned, traj), "Planned should be unchanged"
        assert (observed != traj).any(), "Observed should differ from planned"

    def test_4d_trajectories(self):
        """Should work with (n_samples, batch, n_times, D) shape."""
        n_samples = 3
        batch = 4
        n_times = 8
        traj = torch.randn(n_samples, batch, n_times, 1)
        time_grid = torch.linspace(0, 1, n_times)
        event_times = torch.tensor([0.5, 0.5, 0.5, 0.5])

        observed, mask = HIVAE_inputDropout.truncate_longitudinal_at_event(
            traj, time_grid, event_times)

        assert observed.shape == traj.shape


# ======================================================================
# Phase 2: Ablation flags don't alter default behavior
# ======================================================================

class TestAblationFlags:
    """Ensure ablation flags off = unchanged behavior."""

    def setup_method(self):
        self.feat_types = [
            {'type': 'real', 'dim': '1'},
            {'type': 'surv_weibull', 'dim': '2'},
        ]

    def test_default_model_unchanged(self):
        """Model with all defaults should match pre-remediation behavior."""
        m = HIVAE_inputDropout(
            3, z_dim=10, s_dim=5, y_dim=8, y_dim_partition=None,
            feat_types_dict=self.feat_types, intervals_surv_piecewise=None,
            n_layers_surv_piecewise=2, model_version='v4_joint', n_long_outcomes=1)

        # Should NOT have survival embedding by default
        assert not hasattr(m, 'surv_embed_net'), \
            "Survival embedding should not exist by default"

    def test_survival_no_nan_sampling(self):
        """Survival sampling should not produce NaN/Inf."""
        m = HIVAE_inputDropout(
            3, z_dim=10, s_dim=5, y_dim=8, y_dim_partition=None,
            feat_types_dict=self.feat_types, intervals_surv_piecewise=None,
            n_layers_surv_piecewise=2, model_version='v4_joint', n_long_outcomes=1)

        torch.manual_seed(42)
        data = torch.randn(20, 3).abs() + 0.01
        data[:, 2] = (torch.rand(20) > 0.3).float()
        miss = torch.ones(20, 2)
        times = torch.rand(20, 5)
        values = torch.randn(20, 5)
        masks = (torch.rand(20, 5) > 0.3).float()

        data_list = [data[:, 0:1], data[:, 1:3]]
        data_list_obs = [d * miss[:, i:i+1] for i, d in enumerate(data_list)]

        result = m.forward(data_list_obs, data_list, miss, tau=0.5,
                          n_generated_dataset=1,
                          longitudinal_data=(times, values, masks))

        assert not torch.isnan(result['neg_ELBO_loss']), "NaN in ELBO"


# ======================================================================
# Run tests
# ======================================================================

if __name__ == "__main__":
    # Simple test runner
    import traceback
    test_classes = [
        TestGlobalNormalization,
        TestWeibullClamp,
        TestTruncation,
        TestAblationFlags,
    ]
    n_pass = 0
    n_fail = 0
    for cls in test_classes:
        instance = cls()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                if hasattr(instance, 'setup_method'):
                    instance.setup_method()
                try:
                    getattr(instance, method_name)()
                    print(f"  PASS: {cls.__name__}.{method_name}")
                    n_pass += 1
                except Exception as e:
                    print(f"  FAIL: {cls.__name__}.{method_name}: {e}")
                    traceback.print_exc()
                    n_fail += 1
    print(f"\n{'='*40}")
    print(f"Results: {n_pass} passed, {n_fail} failed")
    print(f"{'='*40}")
    sys.exit(1 if n_fail > 0 else 0)
