#!/usr/bin/env python3
"""
Train V4_joint and V4_seq on PBC2 and generate comprehensive visualization results.

Usage:
    cd /project/Stat/s1155202253/myproject/pfizer_projects/survgen-clinical-trials
    conda activate env_2502
    python experiments/pbc2_v4.py
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
import matplotlib.gridspec as gridspec
from scipy import stats
from lifelines import KaplanMeierFitter

# ---- Path setup ----
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(ROOT / "execute"))

import data_processing
import surv_hivae

DATA_DIR = ROOT / "dataset" / "pbc2"

# Output directories
BASE_FIG = ROOT / "experiments" / "pbc2_v4_figures"
DIR_BASELINE = BASE_FIG / "baseline_covariates"
DIR_LONG = BASE_FIG / "longitudinal_trajectories"
DIR_SURV = BASE_FIG / "survival_analysis"
DIR_COMPARE = BASE_FIG / "version_comparison"
DIR_DIAG = BASE_FIG / "diagnostics"
for d in [DIR_BASELINE, DIR_LONG, DIR_SURV, DIR_COMPARE, DIR_DIAG]:
    d.mkdir(parents=True, exist_ok=True)

# ---- Hyperparams ----
EPOCHS = 500
N_GEN = 10
BASE_PARAMS = {"lr": 5e-4, "batch_size": 50, "z_dim": 20, "y_dim": 15, "s_dim": 20}

COLS_SURV = ["time", "censor", "drug", "sex", "ascites", "hepatomegaly",
             "spiders", "edema", "histologic", "albumin",
             "alkaline", "SGOT", "platelets", "prothrombin", "age"]
COLS_BASELINE = ["drug", "sex", "ascites", "hepatomegaly",
                 "spiders", "edema", "histologic", "albumin",
                 "alkaline", "SGOT", "platelets", "prothrombin", "age"]

CAT_FEATS = ["drug", "edema", "histologic"]
CONT_FEATS = ["albumin", "age", "prothrombin"]
LONG_OUTCOME_NAMES = ["ascites", "hepatomegaly", "spiders", "edema",
                      "serBilir", "albumin", "alkaline", "SGOT",
                      "platelets", "prothrombin", "histologic"]
LONG_CONTINUOUS = ["serBilir", "albumin", "alkaline", "SGOT",
                   "platelets", "prothrombin"]


# =====================================================================
# Visualization helpers
# =====================================================================

def plot_feature_comparison(real_df, syn_df, cat_feats, cont_feats, title, save_path):
    """Compare real vs synthetic distributions for baseline covariates."""
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


def plot_all_continuous_marginals(real_df, syn_df, version_name, save_path):
    """Plot all continuous/pos feature marginals in a grid."""
    feats = [c for c in real_df.columns if c not in ['drug','sex','ascites','hepatomegaly',
             'spiders','edema','histologic','time','censor']]
    n = len(feats)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))
    axes = axes.ravel()
    for i, feat in enumerate(feats):
        ax = axes[i]
        rvals = real_df[feat].dropna().values
        svals = syn_df[feat].dropna().values
        ax.hist(rvals, bins=30, alpha=0.5, label='Real', density=True, color='steelblue')
        ax.hist(svals, bins=30, alpha=0.5, label='Syn', density=True, color='coral')
        # KS test
        ks_stat, ks_p = stats.ks_2samp(rvals, svals)
        ax.set_title(f"{feat}\nKS={ks_stat:.3f} p={ks_p:.3f}", fontsize=9)
        ax.legend(fontsize=7)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"{version_name}: All Continuous Marginals", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_all_categorical_marginals(real_df, syn_df, version_name, save_path):
    """Plot all categorical feature proportions."""
    cat_cols = ['drug','sex','ascites','hepatomegaly','spiders','edema','histologic']
    cat_cols = [c for c in cat_cols if c in real_df.columns]
    n = len(cat_cols)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.ravel()
    for i, feat in enumerate(cat_cols):
        ax = axes[i]
        rc = real_df[feat].value_counts(normalize=True).sort_index()
        sc = syn_df[feat].value_counts(normalize=True).sort_index()
        all_c = sorted(set(rc.index) | set(sc.index))
        x = np.arange(len(all_c)); w = 0.35
        ax.bar(x-w/2, [rc.get(c,0) for c in all_c], w, label='Real', color='steelblue', alpha=0.7)
        ax.bar(x+w/2, [sc.get(c,0) for c in all_c], w, label='Syn', color='coral', alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels([str(int(c)) for c in all_c])
        ax.set_title(feat, fontweight='bold', fontsize=10)
        ax.legend(fontsize=7)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"{version_name}: Categorical Distributions", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_correlation_heatmaps(real_df, syn_df, version_name, save_path):
    """Compare correlation matrices between real and synthetic data."""
    num_cols = [c for c in real_df.columns if c not in ['drug','sex','ascites',
                'hepatomegaly','spiders','time','censor']]
    num_cols = [c for c in num_cols if c in syn_df.columns]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    r_corr = real_df[num_cols].corr()
    s_corr = syn_df[num_cols].corr()
    diff = r_corr - s_corr
    im1 = ax1.imshow(r_corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_title('Real Correlation', fontweight='bold')
    ax1.set_xticks(range(len(num_cols))); ax1.set_xticklabels(num_cols, rotation=45, ha='right', fontsize=7)
    ax1.set_yticks(range(len(num_cols))); ax1.set_yticklabels(num_cols, fontsize=7)
    im2 = ax2.imshow(s_corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax2.set_title('Synthetic Correlation', fontweight='bold')
    ax2.set_xticks(range(len(num_cols))); ax2.set_xticklabels(num_cols, rotation=45, ha='right', fontsize=7)
    ax2.set_yticks(range(len(num_cols))); ax2.set_yticklabels(num_cols, fontsize=7)
    max_diff = max(abs(diff.values.min()), abs(diff.values.max()), 0.01)
    im3 = ax3.imshow(diff.values, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
    ax3.set_title('Difference (Real - Syn)', fontweight='bold')
    ax3.set_xticks(range(len(num_cols))); ax3.set_xticklabels(num_cols, rotation=45, ha='right', fontsize=7)
    ax3.set_yticks(range(len(num_cols))); ax3.set_yticklabels(num_cols, fontsize=7)
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    fig.suptitle(f"{version_name}: Correlation Structure", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_kaplan_meier(real_times, real_events, syn_times, syn_events,
                      version_name, save_path, by_group=None, real_groups=None, syn_groups=None):
    """Kaplan-Meier survival curves: real vs synthetic."""
    fig, ax = plt.subplots(figsize=(10, 7))
    kmf = KaplanMeierFitter()
    if by_group is not None and real_groups is not None:
        colors_real = ['#2166ac', '#4393c3']
        colors_syn = ['#d6604d', '#f4a582']
        for i, grp in enumerate(sorted(np.unique(real_groups))):
            mask_r = real_groups == grp
            mask_s = syn_groups == grp
            grp_name = 'D-penicillamine' if grp == 0 else 'Placebo'
            kmf.fit(real_times[mask_r], real_events[mask_r], label=f'Real {grp_name}')
            kmf.plot_survival_function(ax=ax, color=colors_real[i], linewidth=2)
            kmf.fit(syn_times[mask_s], syn_events[mask_s], label=f'Syn {grp_name}')
            kmf.plot_survival_function(ax=ax, color=colors_syn[i], linewidth=2, linestyle='--')
    else:
        kmf.fit(real_times, real_events, label='Real')
        kmf.plot_survival_function(ax=ax, color='steelblue', linewidth=2)
        kmf.fit(syn_times, syn_events, label='Synthetic')
        kmf.plot_survival_function(ax=ax, color='coral', linewidth=2, linestyle='--')
    ax.set_xlabel("Time (years)", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_title(f"{version_name}: Kaplan-Meier Curves", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_survival_time_distribution(real_times, syn_times, version_name, save_path):
    """Compare survival time distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(real_times, bins=30, alpha=0.5, label='Real', density=True, color='steelblue')
    ax1.hist(syn_times, bins=30, alpha=0.5, label='Synthetic', density=True, color='coral')
    ks, p = stats.ks_2samp(real_times, syn_times)
    ax1.set_title(f'Time Distribution\nKS={ks:.3f}, p={p:.3f}', fontweight='bold')
    ax1.set_xlabel('Time (years)')
    ax1.legend()
    # QQ plot
    real_sorted = np.sort(real_times)
    syn_sorted = np.sort(syn_times)
    n = min(len(real_sorted), len(syn_sorted))
    r_q = np.quantile(real_sorted, np.linspace(0, 1, n))
    s_q = np.quantile(syn_sorted, np.linspace(0, 1, n))
    ax2.scatter(r_q, s_q, s=10, alpha=0.5, color='purple')
    lims = [min(r_q.min(), s_q.min()), max(r_q.max(), s_q.max())]
    ax2.plot(lims, lims, 'k--', linewidth=1, label='y=x')
    ax2.set_xlabel('Real Quantiles')
    ax2.set_ylabel('Synthetic Quantiles')
    ax2.set_title('QQ Plot (Survival Times)', fontweight='bold')
    ax2.legend()
    fig.suptitle(f"{version_name}: Survival Time Analysis", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_censoring_rate(real_events, syn_events, version_name, save_path):
    """Compare censoring rates."""
    fig, ax = plt.subplots(figsize=(6, 5))
    real_rate = real_events.mean()
    syn_rate = syn_events.mean()
    bars = ax.bar(['Real', 'Synthetic'], [real_rate, 1-real_rate],
                  color=['steelblue', 'lightblue'], alpha=0.7, label=['Event', 'Censored'])
    ax.bar(['Real', 'Synthetic'], [0, 0], bottom=[real_rate, syn_rate],
           color=['coral', 'lightsalmon'], alpha=0.7)
    # Stacked bar
    ax.clear()
    x = np.arange(2)
    w = 0.5
    event_rates = [real_rate, syn_rate]
    cens_rates = [1-real_rate, 1-syn_rate]
    ax.bar(x, event_rates, w, label='Event', color='steelblue', alpha=0.7)
    ax.bar(x, cens_rates, w, bottom=event_rates, label='Censored', color='lightcoral', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(['Real', 'Synthetic'])
    ax.set_ylabel('Proportion')
    ax.set_title(f'{version_name}: Event vs Censored\nReal={real_rate:.3f} Syn={syn_rate:.3f}',
                 fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_longitudinal_trajectories(real_long_df, syn_mu, syn_traj, time_grid,
                                    norm_params, outcome_idx, outcome_name,
                                    version_name, save_path):
    """Plot real observed data + synthetic mean trajectory with CI."""
    fig, ax = plt.subplots(figsize=(10, 6))
    tmin = norm_params["time_min"]
    tmax = norm_params["time_max"]
    tg_orig = time_grid * (tmax - tmin) + tmin
    vmean = norm_params["value_mean"][outcome_idx]
    vstd = norm_params["value_std"][outcome_idx]
    mu_denorm = syn_mu[:, :, outcome_idx].numpy() * vstd + vmean
    traj_denorm = syn_traj[:, :, :, outcome_idx].numpy() * vstd + vmean
    mu_mean = mu_denorm.mean(axis=0)
    traj_flat = traj_denorm.reshape(-1, traj_denorm.shape[-1])
    q05 = np.percentile(traj_flat, 5, axis=0)
    q95 = np.percentile(traj_flat, 95, axis=0)
    ax.scatter(real_long_df["visit_time"].values,
               real_long_df[outcome_name].values,
               alpha=0.15, s=8, color='steelblue', label='Real observations')
    ax.plot(tg_orig.numpy(), mu_mean, color='coral', linewidth=2, label='Synthetic mean')
    ax.fill_between(tg_orig.numpy(), q05, q95, color='coral', alpha=0.2, label='Synthetic 90% CI')
    ax.set_xlabel("Time (years)")
    ax.set_ylabel(outcome_name)
    ax.set_title(f"{version_name}: {outcome_name} Trajectories", fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_loss_curves(loss_dict, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, losses in loss_dict.items():
        ax.plot(losses, label=name, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Neg ELBO Loss")
    ax.set_title("Training Loss Comparison", fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# =====================================================================
# Training and generation helper
# =====================================================================

def train_and_generate_v4(model_version, surv_type="weibull"):
    """Train a V4 model and return model, generated data, and longitudinal output."""
    from src import HIVAE_inputDropout

    print(f"\n{'='*60}")
    print(f"Training {model_version} (surv_type={surv_type})")
    print(f"{'='*60}")

    surv_hivae.set_seed()
    params = BASE_PARAMS.copy()

    # Load data with survival feature
    surv_feat_type = f"surv_{surv_type}"
    df, types_dict, miss_mask, true_miss_mask, n_samples = data_processing.read_data(
        str(DATA_DIR / "data_v4.csv"),
        str(DATA_DIR / "data_types_v4.csv"),
        "Missing.csv", None,
        surv_type=surv_feat_type,
    )
    print(f"  Data shape: {df.shape}, n_samples: {n_samples}")
    print(f"  Feature types: {[f['type'] for f in types_dict]}")

    # Load longitudinal data
    long_df = pd.read_csv(DATA_DIR / "longitudinal.csv")
    n_long_outcomes = len(LONG_OUTCOME_NAMES)
    times_norm, values_norm, masks, long_norm_params = \
        data_processing.prepare_longitudinal_tensors(
            long_df, patient_id_col="patient_id", time_col="visit_time",
            value_col=LONG_OUTCOME_NAMES,
            n_patients=n_samples,
        )
    longitudinal_data = (times_norm, values_norm, masks)
    print(f"  Longitudinal: times={times_norm.shape}, values={values_norm.shape}")

    # Create model
    batch_size = min(params["batch_size"], int(0.9 * n_samples))
    model = HIVAE_inputDropout(
        input_dim=df.shape[1], z_dim=params["z_dim"],
        y_dim=params["y_dim"], s_dim=params["s_dim"],
        y_dim_partition=None, feat_types_dict=types_dict,
        intervals_surv_piecewise=None, n_layers_surv_piecewise=None,
        model_version=model_version, n_long_outcomes=n_long_outcomes,
    )
    data_t = torch.from_numpy(df.values)

    # Train
    # Use CPU for v4_seq to avoid GPU numerical instability in skip_surv decode
    train_device = 'cpu' if model_version == 'v4_seq' else None
    model, loss_train, loss_val = surv_hivae.train_HIVAE(
        model, data_t, miss_mask, true_miss_mask,
        types_dict, batch_size, params["lr"], EPOCHS,
        longitudinal_data=longitudinal_data,
        device=train_device,
    )

    # Generate baseline + survival
    gen = surv_hivae.generate_from_HIVAE(
        model, data_t, miss_mask, true_miss_mask, types_dict, N_GEN
    )

    # Generate longitudinal
    device = next(model.parameters()).device
    with torch.no_grad():
        tg = torch.linspace(0, 1, 30)
        data_dev = data_t.to(device)
        miss_dev = torch.multiply(miss_mask, true_miss_mask).to(device)
        data_list_sub, miss_list_sub = data_processing.next_batch(
            data_dev, types_dict, miss_dev, n_samples, 0)
        data_list_obs = [d * miss_list_sub[:, j].view(n_samples, 1)
                         for j, d in enumerate(data_list_sub)]
        X_list, _ = data_processing.batch_normalization(data_list_obs, types_dict, miss_list_sub)
        X = torch.cat(X_list, dim=1)

        long_data_dev = tuple(t.to(device) for t in longitudinal_data)
        long_summary = model._encode_longitudinal_summary(long_data_dev)
        X = torch.cat([X, long_summary], dim=1)
        _, samples = model.encode(X, tau=1e-3)

        if model_version == 'v4_seq':
            # Compute c_X from non-survival features in X_list
            non_surv_x = []
            for fi, feat in enumerate(types_dict):
                if not feat['type'].startswith('surv'):
                    non_surv_x.append(X_list[fi])
            c_X = model.baseline_summary_net(torch.cat(non_surv_x, dim=1))
            mu, var, trajectories = model.generate_longitudinal_seq(
                samples, tg.to(device), c_X, n_samples=N_GEN)
        else:
            mu, var, trajectories = model.generate_longitudinal(
                samples, tg.to(device), n_samples=N_GEN)

    mu = mu.cpu()
    trajectories = trajectories.cpu()
    tg = tg.cpu()

    # Transform real data for comparison
    real_data = data_processing.discrete_variables_transformation(data_t, types_dict)
    real_data = data_processing.survival_variables_transformation(real_data, types_dict)

    return {
        "model": model,
        "loss_train": loss_train,
        "loss_val": loss_val,
        "gen": gen,
        "real_data": real_data,
        "types_dict": types_dict,
        "mu": mu,
        "trajectories": trajectories,
        "time_grid": tg,
        "long_norm_params": long_norm_params,
        "long_df": long_df,
        "n_samples": n_samples,
        "df": df,
    }


# =====================================================================
# Main
# =====================================================================

def main():
    all_losses = {}
    results = {}

    # ---- Train V4_joint ----
    res_joint = train_and_generate_v4("v4_joint", "weibull")
    results["v4_joint"] = res_joint
    all_losses["V4_joint"] = res_joint["loss_train"]

    # ---- Train V4_seq ----
    res_seq = train_and_generate_v4("v4_seq", "weibull")
    results["v4_seq"] = res_seq
    all_losses["V4_seq"] = res_seq["loss_train"]

    # =====================================================================
    # Generate all visualizations
    # =====================================================================

    for vname, res in results.items():
        label = vname.upper().replace("_", " ")
        gen_sample = res["gen"][0]  # first generated dataset
        real = res["real_data"]

        # Map columns
        cols = COLS_SURV
        real_df = pd.DataFrame(real.numpy(), columns=cols)
        syn_df = pd.DataFrame(gen_sample.numpy(), columns=cols)

        # ---- 1. Baseline covariate comparison ----
        plot_feature_comparison(
            real_df, syn_df, CAT_FEATS, CONT_FEATS,
            f"{label}: Real vs Synthetic (Key Features)",
            DIR_BASELINE / f"{vname}_feature_comparison.png")

        plot_all_continuous_marginals(
            real_df, syn_df, label,
            DIR_BASELINE / f"{vname}_all_continuous.png")

        plot_all_categorical_marginals(
            real_df, syn_df, label,
            DIR_BASELINE / f"{vname}_all_categorical.png")

        plot_correlation_heatmaps(
            real_df, syn_df, label,
            DIR_BASELINE / f"{vname}_correlations.png")

        # ---- 2. Survival analysis ----
        real_times = real_df["time"].values
        real_events = real_df["censor"].values
        syn_times = syn_df["time"].values.clip(0)
        syn_events = (syn_df["censor"].values > 0.5).astype(float)

        plot_kaplan_meier(
            real_times, real_events, syn_times, syn_events,
            label, DIR_SURV / f"{vname}_km_overall.png")

        # KM by treatment group
        real_drug = real_df["drug"].values.astype(int)
        syn_drug = syn_df["drug"].values.astype(int)
        plot_kaplan_meier(
            real_times, real_events, syn_times, syn_events,
            label, DIR_SURV / f"{vname}_km_by_drug.png",
            by_group=True, real_groups=real_drug, syn_groups=syn_drug)

        plot_survival_time_distribution(
            real_times, syn_times,
            label, DIR_SURV / f"{vname}_time_dist.png")

        plot_censoring_rate(
            real_events, syn_events,
            label, DIR_SURV / f"{vname}_censoring_rate.png")

        # ---- 3. Longitudinal trajectories ----
        for oname in LONG_CONTINUOUS:
            oidx = LONG_OUTCOME_NAMES.index(oname)
            plot_longitudinal_trajectories(
                res["long_df"], res["mu"], res["trajectories"], res["time_grid"],
                res["long_norm_params"], oidx, oname, label,
                DIR_LONG / f"{vname}_{oname}_trajectories.png")

    # ---- 4. Version comparison plots ----
    plot_loss_curves(all_losses, DIR_COMPARE / "loss_curves.png")

    # KM comparison: both V4 variants vs real
    fig, ax = plt.subplots(figsize=(10, 7))
    kmf = KaplanMeierFitter()
    real_df_j = pd.DataFrame(results["v4_joint"]["real_data"].numpy(), columns=COLS_SURV)
    kmf.fit(real_df_j["time"], real_df_j["censor"], label="Real")
    kmf.plot_survival_function(ax=ax, color='black', linewidth=2.5)
    for vname, color, ls in [("v4_joint", "coral", "--"), ("v4_seq", "forestgreen", "-.")]:
        syn = pd.DataFrame(results[vname]["gen"][0].numpy(), columns=COLS_SURV)
        st = syn["time"].values.clip(0)
        se = (syn["censor"].values > 0.5).astype(float)
        kmf.fit(st, se, label=vname.replace("_", " ").upper())
        kmf.plot_survival_function(ax=ax, color=color, linewidth=2, linestyle=ls)
    ax.set_xlabel("Time (years)", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_title("Survival Curves: Real vs V4_joint vs V4_seq", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(DIR_COMPARE / "km_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {DIR_COMPARE / 'km_comparison.png'}")

    # ---- 5. Diagnostics ----
    for vname, res in results.items():
        label = vname.upper().replace("_", " ")
        # Loss train vs val
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(res["loss_train"], label="Train", linewidth=1.5)
        val_clean = [v for v in res["loss_val"] if not (isinstance(v, float) and np.isnan(v))
                     and not (torch.is_tensor(v) and torch.isnan(v))]
        if val_clean:
            val_epochs = [i for i, v in enumerate(res["loss_val"])
                          if not (isinstance(v, float) and np.isnan(v))
                          and not (torch.is_tensor(v) and torch.isnan(v))]
            ax.plot(val_epochs, val_clean, label="Validation", linewidth=1.5, linestyle='--')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Neg ELBO")
        ax.set_title(f"{label}: Training Diagnostics", fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(DIR_DIAG / f"{vname}_train_val.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {DIR_DIAG / f'{vname}_train_val.png'}")

    print(f"\n{'='*60}")
    print(f"All V4 figures saved to: {BASE_FIG}")
    print(f"  baseline_covariates/  — {len(list(DIR_BASELINE.glob('*.png')))} figures")
    print(f"  longitudinal_trajectories/  — {len(list(DIR_LONG.glob('*.png')))} figures")
    print(f"  survival_analysis/  — {len(list(DIR_SURV.glob('*.png')))} figures")
    print(f"  version_comparison/  — {len(list(DIR_COMPARE.glob('*.png')))} figures")
    print(f"  diagnostics/  — {len(list(DIR_DIAG.glob('*.png')))} figures")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
