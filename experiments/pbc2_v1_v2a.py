#!/usr/bin/env python3
"""
Train V1 and V2A (multi-outcome) on PBC2 and generate comparison figures.

Usage:
    cd /project/Stat/s1155202253/myproject/pfizer_projects/survgen-clinical-trials
    conda activate env_2502
    python experiments/pbc2_v1_v2a.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- Path setup ----
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(ROOT / "execute"))

import data_processing
import surv_hivae

DATA_DIR = ROOT / "dataset" / "pbc2"
FIG_DIR = ROOT / "experiments" / "pbc2_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- Hyperparams ----
EPOCHS = 500
N_GEN = 10
BASE_PARAMS = {"lr": 1e-3, "batch_size": 100, "z_dim": 20, "y_dim": 15, "s_dim": 20}

COLS_V1 = ["serBilir", "drug", "sex", "ascites", "hepatomegaly",
           "spiders", "edema", "histologic", "albumin",
           "alkaline", "SGOT", "platelets", "prothrombin", "age"]
COLS_V2A = ["drug", "sex", "ascites", "hepatomegaly",
            "spiders", "edema", "histologic", "serBilir", "albumin",
            "alkaline", "SGOT", "platelets", "prothrombin", "age"]

CAT_FEATS = ["drug", "edema", "histologic"]
CONT_FEATS = ["serBilir", "albumin", "age"]
LONG_OUTCOME_NAMES = ["ascites", "hepatomegaly", "spiders", "edema",
                      "serBilir", "albumin", "alkaline", "SGOT",
                      "platelets", "prothrombin", "histologic"]
# Which longitudinal outcomes are continuous (for trajectory plots)
LONG_CONTINUOUS = ["serBilir", "albumin", "alkaline", "SGOT",
                   "platelets", "prothrombin"]


def plot_feature_comparison(real_df, syn_df, cat_feats, cont_feats, title, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    for i, feat in enumerate(cont_feats):
        ax = axes[i]
        ax.hist(real_df[feat].values, bins=30, alpha=0.5,
                label='Real', density=True, color='steelblue')
        ax.hist(syn_df[feat].values, bins=30, alpha=0.5,
                label='Synthetic', density=True, color='coral')
        ax.set_title(feat, fontweight='bold')
        ax.legend()

    for i, feat in enumerate(cat_feats):
        ax = axes[len(cont_feats) + i]
        real_counts = real_df[feat].value_counts(normalize=True).sort_index()
        syn_counts = syn_df[feat].value_counts(normalize=True).sort_index()
        all_cats = sorted(set(real_counts.index) | set(syn_counts.index))
        x = np.arange(len(all_cats))
        w = 0.35
        ax.bar(x - w / 2, [real_counts.get(c, 0) for c in all_cats],
               w, label='Real', color='steelblue', alpha=0.7)
        ax.bar(x + w / 2, [syn_counts.get(c, 0) for c in all_cats],
               w, label='Synthetic', color='coral', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(c)) for c in all_cats])
        ax.set_title(feat, fontweight='bold')
        ax.legend()

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_longitudinal_trajectories(real_long_df, syn_mu, syn_traj, time_grid,
                                    norm_params, outcome_idx, outcome_name,
                                    save_path):
    """Plot real observed data + synthetic mean trajectory with CI."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Denormalize time grid
    tmin = norm_params["time_min"]
    tmax = norm_params["time_max"]
    tg_orig = time_grid * (tmax - tmin) + tmin

    # Denormalize synthetic mu and trajectories
    vmean = norm_params["value_mean"][outcome_idx]
    vstd = norm_params["value_std"][outcome_idx]
    mu_denorm = syn_mu[:, :, outcome_idx].numpy() * vstd + vmean     # (N, T)
    traj_denorm = syn_traj[:, :, :, outcome_idx].numpy() * vstd + vmean  # (n_gen, N, T)

    # Mean across patients for the mean trajectory
    mu_mean = mu_denorm.mean(axis=0)                                 # (T,)
    # Percentiles from generated trajectories across all samples and patients
    traj_flat = traj_denorm.reshape(-1, traj_denorm.shape[-1])       # (n_gen*N, T)
    q05 = np.percentile(traj_flat, 5, axis=0)
    q95 = np.percentile(traj_flat, 95, axis=0)

    # Plot real observations
    ax.scatter(real_long_df["visit_time"].values,
               real_long_df[outcome_name].values,
               alpha=0.15, s=8, color='steelblue', label='Real observations')

    # Plot synthetic mean + CI
    ax.plot(tg_orig.numpy(), mu_mean, color='coral', linewidth=2, label='Synthetic mean')
    ax.fill_between(tg_orig.numpy(), q05, q95, color='coral', alpha=0.2, label='Synthetic 90% CI')

    ax.set_xlabel("Time (years)")
    ax.set_ylabel(outcome_name)
    ax.set_title(f"V2A: {outcome_name} Longitudinal Trajectories", fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_longitudinal_by_group(real_long_df, syn_mu, syn_traj, time_grid,
                               norm_params, outcome_names, continuous_outcomes,
                               patient_drug, save_path, n_real_lines=15,
                               n_syn_lines=15):
    """Plot real vs synthetic longitudinal trajectories by treatment group.

    Uses median + IQR ribbons instead of individual spaghetti lines for clarity.
    For each continuous outcome, side-by-side panels (D-penicillamine | Placebo).
    Real data: binned median + IQR ribbon.
    Synthetic data: median + IQR ribbon from generated trajectories.
    """
    from scipy.stats import iqr as _iqr  # noqa: F401

    n_out = len(continuous_outcomes)
    fig, axes = plt.subplots(n_out, 2, figsize=(14, 3.6 * n_out), squeeze=False)

    tmin = norm_params["time_min"]
    tmax = norm_params["time_max"]
    tg_orig = (time_grid * (tmax - tmin) + tmin).numpy()

    group_names = {0: "D-penicillamine", 1: "Placebo"}
    REAL_COLOR = "#2166ac"
    SYN_COLOR = "#d6604d"
    N_TIME_BINS = 20  # bins for aggregating real observations

    for row, oname in enumerate(continuous_outcomes):
        oidx = outcome_names.index(oname)
        vmean = norm_params["value_mean"][oidx]
        vstd = norm_params["value_std"][oidx]

        mu_denorm = syn_mu[:, :, oidx].numpy() * vstd + vmean        # (N, T)
        traj_denorm = syn_traj[:, :, :, oidx].numpy() * vstd + vmean # (n_gen, N, T)

        for col, grp in enumerate([0, 1]):
            ax = axes[row, col]

            # --- Identify patients in this group ---
            grp_mask = patient_drug == grp
            grp_pids = np.where(grp_mask)[0]

            # ====== Real data: bin observations and compute median + IQR ======
            real_grp = real_long_df[real_long_df["patient_id"].isin(grp_pids)].copy()
            real_vals = real_grp[oname].values
            real_times = real_grp["visit_time"].values

            # Remove NaNs
            valid = np.isfinite(real_vals) & np.isfinite(real_times)
            real_vals = real_vals[valid]
            real_times = real_times[valid]

            if len(real_vals) > 0:
                bin_edges = np.linspace(real_times.min(), real_times.max(), N_TIME_BINS + 1)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                bin_idx = np.digitize(real_times, bin_edges) - 1
                bin_idx = np.clip(bin_idx, 0, N_TIME_BINS - 1)

                real_med = np.full(N_TIME_BINS, np.nan)
                real_q25 = np.full(N_TIME_BINS, np.nan)
                real_q75 = np.full(N_TIME_BINS, np.nan)
                for b in range(N_TIME_BINS):
                    mask_b = bin_idx == b
                    if mask_b.sum() >= 3:
                        real_med[b] = np.median(real_vals[mask_b])
                        real_q25[b] = np.percentile(real_vals[mask_b], 25)
                        real_q75[b] = np.percentile(real_vals[mask_b], 75)

                ok = np.isfinite(real_med)
                ax.plot(bin_centers[ok], real_med[ok], color=REAL_COLOR,
                        linewidth=2, label="Real median", zorder=4)
                ax.fill_between(bin_centers[ok], real_q25[ok], real_q75[ok],
                                color=REAL_COLOR, alpha=0.18, label="Real IQR", zorder=2)

            # ====== Synthetic data: median + IQR from all generated trajectories ======
            # Pool across all n_gen samples for patients in this group
            grp_traj = traj_denorm[:, grp_mask, :]  # (n_gen, n_grp, T)
            grp_traj_flat = grp_traj.reshape(-1, grp_traj.shape[-1])  # (n_gen*n_grp, T)

            syn_med = np.median(grp_traj_flat, axis=0)
            syn_q25 = np.percentile(grp_traj_flat, 25, axis=0)
            syn_q75 = np.percentile(grp_traj_flat, 75, axis=0)

            ax.plot(tg_orig, syn_med, color=SYN_COLOR, linewidth=2,
                    label="Syn median", zorder=4)
            ax.fill_between(tg_orig, syn_q25, syn_q75, color=SYN_COLOR,
                            alpha=0.18, label="Syn IQR", zorder=2)

            # ====== Robust y-limits (1st–99th percentile of combined data) ======
            all_vals = np.concatenate([real_vals, grp_traj_flat.ravel()])
            ylo = np.percentile(all_vals, 1)
            yhi = np.percentile(all_vals, 99)
            margin = 0.08 * (yhi - ylo)
            ax.set_ylim(ylo - margin, yhi + margin)

            ax.set_xlabel("Time (years)", fontsize=10)
            ax.set_ylabel(oname, fontsize=10)
            if row == 0:
                ax.set_title(group_names[grp], fontsize=13, fontweight="bold")
            ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
            ax.tick_params(labelsize=9)

    fig.suptitle("V2A: Longitudinal Trajectories by Treatment Group",
                 fontsize=15, fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_loss_curves(loss_dict, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, losses in loss_dict.items():
        ax.plot(losses, label=name, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Neg ELBO Loss")
    ax.set_title("Training Loss Evolution", fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    from src import HIVAE_inputDropout

    all_losses = {}

    # ========== V1 (baseline + continuous endpoint) ==========
    print(f"\n{'='*60}")
    print(f"Training V1 (model_version=v1)")
    print(f"{'='*60}")

    surv_hivae.set_seed()
    params_v1 = BASE_PARAMS.copy()

    df_v1, types_v1, miss_v1, true_miss_v1, n_v1 = data_processing.read_data(
        str(DATA_DIR / "data.csv"), str(DATA_DIR / "data_types_v1.csv"),
        "Missing.csv", None
    )
    print(f"  Data shape: {df_v1.shape}, n_samples: {n_v1}")

    batch_v1 = min(params_v1["batch_size"], int(0.9 * n_v1))
    model_v1 = HIVAE_inputDropout(
        input_dim=df_v1.shape[1], z_dim=params_v1["z_dim"],
        y_dim=params_v1["y_dim"], s_dim=params_v1["s_dim"],
        y_dim_partition=None, feat_types_dict=types_v1,
        intervals_surv_piecewise=None, n_layers_surv_piecewise=None,
        model_version="v1"
    )
    data_v1_t = torch.from_numpy(df_v1.values)
    model_v1, loss_v1_train, loss_v1_val = surv_hivae.train_HIVAE(
        model_v1, data_v1_t, miss_v1, true_miss_v1,
        types_v1, batch_v1, params_v1["lr"], EPOCHS
    )
    all_losses["V1"] = loss_v1_train

    gen_v1 = surv_hivae.generate_from_HIVAE(
        model_v1, data_v1_t, miss_v1, true_miss_v1, types_v1, N_GEN
    )

    # Collapse real data
    real_v1 = data_processing.discrete_variables_transformation(data_v1_t, types_v1)

    real_v1_df = pd.DataFrame(real_v1.numpy(), columns=COLS_V1)
    syn_v1_df = pd.DataFrame(gen_v1[0].numpy(), columns=COLS_V1)

    plot_feature_comparison(
        real_v1_df, syn_v1_df, CAT_FEATS, CONT_FEATS,
        "V1: Real vs Synthetic",
        FIG_DIR / "v1_feature_comparison.png",
    )

    # ========== V2A (baseline + all 11 longitudinal outcomes) ==========
    print(f"\n{'='*60}")
    print(f"Training V2A (model_version=v2a, {len(LONG_OUTCOME_NAMES)} longitudinal outcomes)")
    print(f"  Outcomes: {LONG_OUTCOME_NAMES}")
    print(f"{'='*60}")

    surv_hivae.set_seed()
    params_v2a = BASE_PARAMS.copy()

    df_v2a, types_v2a, miss_v2a, true_miss_v2a, n_v2a = data_processing.read_data(
        str(DATA_DIR / "data_v2a.csv"), str(DATA_DIR / "data_types_v2a.csv"),
        "Missing.csv", None
    )
    print(f"  Baseline data shape: {df_v2a.shape}, n_samples: {n_v2a}")

    # Load longitudinal data (all 11 outcomes)
    long_df = pd.read_csv(DATA_DIR / "longitudinal.csv")
    n_long_outcomes = len(LONG_OUTCOME_NAMES)
    times_norm, values_norm, masks, long_norm_params = \
        data_processing.prepare_longitudinal_tensors(
            long_df, patient_id_col="patient_id", time_col="visit_time",
            value_col=LONG_OUTCOME_NAMES,
            n_patients=n_v2a,
        )
    longitudinal_data = (times_norm, values_norm, masks)
    print(f"  Longitudinal shape: times={times_norm.shape}, values={values_norm.shape}")

    # Train V2A
    batch_v2a = min(params_v2a["batch_size"], int(0.9 * n_v2a))
    model_v2a = HIVAE_inputDropout(
        input_dim=df_v2a.shape[1], z_dim=params_v2a["z_dim"],
        y_dim=params_v2a["y_dim"], s_dim=params_v2a["s_dim"],
        y_dim_partition=None, feat_types_dict=types_v2a,
        intervals_surv_piecewise=None, n_layers_surv_piecewise=None,
        model_version="v2a", n_long_outcomes=n_long_outcomes
    )
    data_v2a_t = torch.from_numpy(df_v2a.values)
    model_v2a, loss_v2a_train, loss_v2a_val = surv_hivae.train_HIVAE(
        model_v2a, data_v2a_t, miss_v2a, true_miss_v2a,
        types_v2a, batch_v2a, params_v2a["lr"], EPOCHS,
        longitudinal_data=longitudinal_data
    )
    all_losses["V2A"] = loss_v2a_train

    # Generate baseline synthetic data
    gen_v2a = surv_hivae.generate_from_HIVAE(
        model_v2a, data_v2a_t, miss_v2a, true_miss_v2a, types_v2a, N_GEN
    )
    real_v2a = data_processing.discrete_variables_transformation(data_v2a_t, types_v2a)
    real_v2a_df = pd.DataFrame(real_v2a.numpy(), columns=COLS_V2A)
    syn_v2a_df = pd.DataFrame(gen_v2a[0].numpy(), columns=COLS_V2A)

    plot_feature_comparison(
        real_v2a_df, syn_v2a_df, CAT_FEATS, CONT_FEATS,
        "V2A: Real vs Synthetic (Baseline)",
        FIG_DIR / "v2a_feature_comparison.png",
    )

    # Generate longitudinal trajectories
    # Move to model's device (model may be on GPU after training)
    device = next(model_v2a.parameters()).device
    with torch.no_grad():
        tg = torch.linspace(0, 1, 30)
        # Encode all patients
        data_list_sub, miss_list_sub = data_processing.next_batch(
            data_v2a_t.to(device), types_v2a,
            torch.multiply(miss_v2a, true_miss_v2a).to(device), n_v2a, 0
        )
        data_list_obs = [d * miss_list_sub[:, j].view(n_v2a, 1) for j, d in enumerate(data_list_sub)]
        X_list, _ = data_processing.batch_normalization(data_list_obs, types_v2a, miss_list_sub)
        X = torch.cat(X_list, dim=1)

        long_data_dev = (times_norm.to(device), values_norm.to(device), masks.to(device))
        long_summary = model_v2a._encode_longitudinal_summary(long_data_dev)
        X = torch.cat([X, long_summary], dim=1)
        _, samples = model_v2a.encode(X, tau=1e-3)
        mu, var, trajectories = model_v2a.generate_longitudinal(samples, tg.to(device), n_samples=N_GEN)

    # Move results back to CPU for plotting
    mu = mu.cpu()
    trajectories = trajectories.cpu()
    tg = tg.cpu()

    # Plot trajectories for continuous outcomes only
    for oname in LONG_CONTINUOUS:
        oidx = LONG_OUTCOME_NAMES.index(oname)
        plot_longitudinal_trajectories(
            long_df, mu, trajectories, tg,
            long_norm_params, oidx, oname,
            FIG_DIR / f"v2a_{oname}_trajectories.png"
        )

    # Plot longitudinal trajectories by treatment group (with individual lines)
    # Drug is the first column in data_v2a.csv; after discrete_variables_transformation
    # it stays as 0/1 in COLS_V2A[0] = "drug"
    patient_drug = real_v2a_df["drug"].values.astype(int)
    bygroup_dir = FIG_DIR / "v2a_bygroup"
    bygroup_dir.mkdir(parents=True, exist_ok=True)
    plot_longitudinal_by_group(
        long_df, mu, trajectories, tg,
        long_norm_params, LONG_OUTCOME_NAMES, LONG_CONTINUOUS,
        patient_drug,
        bygroup_dir / "longitudinal_by_group.png",
    )

    # ========== Loss curves ==========
    plot_loss_curves(all_losses, FIG_DIR / "loss_curves.png")

    print(f"\n{'='*60}")
    print(f"All figures saved to: {FIG_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
