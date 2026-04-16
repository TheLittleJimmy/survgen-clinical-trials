#!/usr/bin/env python3
"""
Phase 0 diagnostic instrumentation for multi-version HI-VAE.

Provides:
- Per-epoch loss decomposition logging
- Numerical survival diagnostics
- Gradient-flow monitoring (optional)
- Metrics persistence to JSONL
"""

import torch
import json
import os
from collections import defaultdict


class LossDecomposition:
    """Track and log per-epoch loss components."""

    def __init__(self, output_dir=None):
        self.output_dir = output_dir
        self.epoch_metrics = defaultdict(float)
        self.epoch_counts = defaultdict(int)
        self._log_file = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self._log_file = os.path.join(output_dir, "loss_decomposition.jsonl")

    def update(self, metrics_dict, n_batch=1):
        """Accumulate batch-level metrics."""
        for k, v in metrics_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item() if v.numel() == 1 else v.mean().item()
            self.epoch_metrics[k] += v
            self.epoch_counts[k] = self.epoch_counts.get(k, 0) + n_batch

    def log_epoch(self, epoch, phase="train", print_fn=None):
        """Average and log epoch metrics. Returns the averaged dict."""
        avg = {}
        for k, v in self.epoch_metrics.items():
            n = self.epoch_counts.get(k, 1)
            avg[k] = v / max(n, 1)
        avg["epoch"] = epoch
        avg["phase"] = phase

        if print_fn:
            print_fn(avg)
        if self._log_file:
            with open(self._log_file, "a") as f:
                f.write(json.dumps(avg) + "\n")

        self.epoch_metrics.clear()
        self.epoch_counts.clear()
        return avg


class SurvivalDiagnostics:
    """Log numerical diagnostics for survival features."""

    @staticmethod
    def check_batch(T_surv_scaled, delta, log_p_x=None, prefix=""):
        """Check a batch of survival data for numerical issues."""
        diag = {}
        with torch.no_grad():
            diag[f"{prefix}surv_t_min"] = T_surv_scaled.min().item()
            diag[f"{prefix}surv_t_max"] = T_surv_scaled.max().item()
            diag[f"{prefix}surv_frac_zero"] = (T_surv_scaled == 0).float().mean().item()
            diag[f"{prefix}surv_frac_one"] = (T_surv_scaled == 1).float().mean().item()
            diag[f"{prefix}surv_event_frac"] = delta.float().mean().item()
            diag[f"{prefix}surv_censor_frac"] = (1 - delta).float().mean().item()
            if log_p_x is not None:
                diag[f"{prefix}surv_loss_nan"] = torch.isnan(log_p_x).any().item()
                diag[f"{prefix}surv_loss_inf"] = torch.isinf(log_p_x).any().item()
        return diag


class GradientMonitor:
    """Optional gradient norm logging for parameter groups."""

    def __init__(self, model, enabled=False):
        self.model = model
        self.enabled = enabled
        self._groups = {}

    def register_group(self, name, params):
        """Register a named parameter group for monitoring."""
        self._groups[name] = list(params)

    def auto_register(self):
        """Auto-register common groups based on model structure."""
        m = self.model
        groups = {
            "encoder": [],
            "decoder_baseline": [],
            "longitudinal": [],
            "survival_head": [],
            "v4_seq_summary": [],
        }
        for name, param in m.named_parameters():
            if "s_layer" in name or "z_layer" in name:
                groups["encoder"].append(param)
            elif "y_layer" in name or "z_distribution_layer" in name:
                groups["decoder_baseline"].append(param)
            elif "longitudinal" in name or "time_embed" in name or "long_summary" in name:
                groups["longitudinal"].append(param)
            elif "theta_layer" in name:
                # Check if it's a survival feature
                # Parse feat index from name like "theta_layer.feat_0.theta.weight"
                parts = name.split(".")
                if len(parts) >= 2:
                    feat_key = parts[1]  # e.g., "feat_0"
                    feat_idx = int(feat_key.split("_")[1])
                    if feat_idx < len(m.feat_types_list) and \
                       m.feat_types_list[feat_idx]['type'].startswith('surv'):
                        groups["survival_head"].append(param)
                    else:
                        groups["decoder_baseline"].append(param)
            elif "baseline_summary" in name or "mu_seq" in name:
                groups["v4_seq_summary"].append(param)

        for name, params in groups.items():
            if params:
                self._groups[name] = params

    def compute_norms(self):
        """Compute gradient norms for all registered groups."""
        if not self.enabled:
            return {}
        norms = {}
        for name, params in self._groups.items():
            total_norm = 0.0
            n_params = 0
            for p in params:
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
                    n_params += 1
            if n_params > 0:
                norms[f"grad_norm_{name}"] = total_norm ** 0.5
        return norms


def compute_loss_decomposition(vae_res, model_version, longitudinal_data=None):
    """Extract per-modality loss terms from a forward pass result.

    Returns a dict of raw (unweighted) loss components.
    """
    metrics = {}

    # KL terms
    metrics["KL_s"] = torch.mean(vae_res["KL_s"]).item()
    metrics["KL_z"] = torch.mean(vae_res["KL_z"]).item()

    # Reconstruction: log_p_x is (n_features, batch)
    log_p_x = vae_res["log_p_x"]  # (n_features, batch)

    # Separate baseline vs survival reconstruction
    if hasattr(vae_res.get("_model_ref", None), "feat_types_list"):
        feat_types = vae_res["_model_ref"].feat_types_list
    else:
        feat_types = None

    if feat_types is not None:
        baseline_ll = 0.0
        surv_ll = 0.0
        for i, feat in enumerate(feat_types):
            if i < log_p_x.shape[0]:
                val = torch.mean(log_p_x[i]).item()
                if feat['type'].startswith('surv'):
                    surv_ll += val
                else:
                    baseline_ll += val
        metrics["baseline_reconstruction"] = baseline_ll
        metrics["survival_log_lik"] = surv_ll
    else:
        metrics["total_reconstruction"] = torch.mean(torch.sum(log_p_x, dim=0)).item()

    # Longitudinal
    metrics["longitudinal_loss"] = vae_res.get("longitudinal_loss", torch.tensor(0.0)).item()

    # Total
    metrics["neg_ELBO"] = vae_res["neg_ELBO_loss"].item()

    return metrics
