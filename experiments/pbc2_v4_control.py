#!/usr/bin/env python3
"""
Train V4_joint and V4_seq on PBC2 **control (placebo) group only**.
Generate comprehensive visualizations evaluating:
  1. Whether generated samples match real control group
  2. Whether generated samples can replace the real control group
     (KM curve comparison, log-rank test, covariate balance)

Usage:
    cd /project/Stat/s1155202253/myproject/pfizer_projects/survgen-clinical-trials
    conda activate env_2502
    python experiments/pbc2_v4_control.py
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
from scipy import stats
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

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
DIR_REPLACE = BASE_FIG / "replacement_analysis"
DIR_COMPARE = BASE_FIG / "version_comparison"
DIR_DIAG = BASE_FIG / "diagnostics"
for d in [DIR_BASELINE, DIR_LONG, DIR_SURV, DIR_REPLACE, DIR_COMPARE, DIR_DIAG]:
    d.mkdir(parents=True, exist_ok=True)

# ---- Hyperparams ----
EPOCHS = 500
N_GEN = 10
BASE_PARAMS = {"lr": 5e-4, "batch_size": 50, "z_dim": 20, "y_dim": 15, "s_dim": 20}

LONG_OUTCOME_NAMES = ["ascites", "hepatomegaly", "spiders", "edema",
                      "serBilir", "albumin", "alkaline", "SGOT",
                      "platelets", "prothrombin", "histologic"]
LONG_CONTINUOUS = ["serBilir", "albumin", "alkaline", "SGOT",
                   "platelets", "prothrombin"]

# Columns after discrete_variables_transformation + survival_variables_transformation
# Note: drug is NOT included since we train on control only (all drug=1=placebo)
# But we keep it for the data format — it will be constant (all 1s)
COLS_OUT = ["time", "censor", "drug", "sex", "ascites", "hepatomegaly",
            "spiders", "edema", "histologic", "albumin",
            "alkaline", "SGOT", "platelets", "prothrombin", "age"]


# =====================================================================
# Data preparation: extract control group
# =====================================================================

def prepare_control_data():
    """Extract placebo (control) group from PBC2 and prepare V4-format data."""
    pbc2 = pd.read_csv(DATA_DIR / "pbc2_id.csv")
    long_df_full = pd.read_csv(DATA_DIR / "longitudinal.csv")

    # Identify placebo patients (0-indexed patient_id)
    placebo_mask = pbc2['drug'] == 'placebo'
    placebo_ids = pbc2.index[placebo_mask].tolist()
    n_placebo = len(placebo_ids)
    print(f"Control (placebo) group: {n_placebo} patients")

    # Encode variables
    sex_map = {'female': 0, 'male': 1}
    yn_map = {'No': 0, 'Yes': 1}
    edema_map = {'No edema': 0, 'edema no diuretics': 1,
                 'untreated or successfully treated': 1,
                 'edema despite diuretics': 2}

    ctrl = pbc2.loc[placebo_mask].copy()
    ctrl['drug_enc'] = 1  # all placebo
    ctrl['sex_enc'] = ctrl['sex'].map(sex_map)
    ctrl['ascites_enc'] = ctrl['ascites'].map(yn_map)
    ctrl['hepatomegaly_enc'] = ctrl['hepatomegaly'].map(yn_map)
    ctrl['spiders_enc'] = ctrl['spiders'].map(yn_map)
    ctrl['edema_enc'] = ctrl['edema'].map(edema_map)
    ctrl['censor'] = ctrl['status2'].astype(int)
    ctrl['time'] = ctrl['years']

    out_cols = ['time', 'censor', 'drug_enc', 'sex_enc', 'ascites_enc',
                'hepatomegaly_enc', 'spiders_enc', 'edema_enc',
                'histologic', 'albumin', 'alkaline', 'SGOT',
                'platelets', 'prothrombin', 'age']
    df_ctrl = ctrl[out_cols].copy()
    # Fill NaN with median
    for c in df_ctrl.columns:
        if df_ctrl[c].isna().any():
            df_ctrl[c] = df_ctrl[c].fillna(df_ctrl[c].median())

    # Save control data
    ctrl_csv = DATA_DIR / "data_v4_control.csv"
    df_ctrl.to_csv(ctrl_csv, index=False, header=False)
    print(f"  Saved: {ctrl_csv}")

    # Extract longitudinal data for control patients
    # Map original patient_ids to 0-based indices for the control subset
    pid_map = {orig: new for new, orig in enumerate(placebo_ids)}
    long_ctrl = long_df_full[long_df_full['patient_id'].isin(placebo_ids)].copy()
    long_ctrl['patient_id'] = long_ctrl['patient_id'].map(pid_map)
    long_ctrl = long_ctrl.sort_values(['patient_id', 'visit_time']).reset_index(drop=True)
    long_ctrl_csv = DATA_DIR / "longitudinal_control.csv"
    long_ctrl.to_csv(long_ctrl_csv, index=False)
    print(f"  Saved: {long_ctrl_csv} ({len(long_ctrl)} records)")

    # Also prepare the FULL dataset (both groups) for replacement analysis
    treat = pbc2.loc[~placebo_mask].copy()
    treat['drug_enc'] = 0
    treat['sex_enc'] = treat['sex'].map(sex_map)
    treat['ascites_enc'] = treat['ascites'].map(yn_map)
    treat['hepatomegaly_enc'] = treat['hepatomegaly'].map(yn_map)
    treat['spiders_enc'] = treat['spiders'].map(yn_map)
    treat['edema_enc'] = treat['edema'].map(edema_map)
    treat['censor'] = treat['status2'].astype(int)
    treat['time'] = treat['years']
    df_treat = treat[out_cols].copy()
    for c in df_treat.columns:
        if df_treat[c].isna().any():
            df_treat[c] = df_treat[c].fillna(df_treat[c].median())

    return df_ctrl, long_ctrl, df_treat, n_placebo, placebo_ids


# =====================================================================
# Visualization helpers
# =====================================================================

def plot_marginals_grid(real_df, syn_df, cols, version, save_path, kind='hist'):
    """Plot marginal distributions for specified columns."""
    n = len(cols)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.5*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.ravel()
    for i, c in enumerate(cols):
        ax = axes[i]
        rv = real_df[c].dropna().values
        sv = syn_df[c].dropna().values
        if kind == 'hist':
            ax.hist(rv, bins=25, alpha=0.5, density=True, color='steelblue', label='Real ctrl')
            ax.hist(sv, bins=25, alpha=0.5, density=True, color='coral', label='Synthetic')
            ks, p = stats.ks_2samp(rv, sv)
            ax.set_title(f"{c}\nKS={ks:.3f} p={p:.3f}", fontsize=9)
        else:  # bar
            rc = real_df[c].value_counts(normalize=True).sort_index()
            sc = syn_df[c].value_counts(normalize=True).sort_index()
            all_c = sorted(set(rc.index) | set(sc.index))
            x = np.arange(len(all_c)); w = 0.35
            ax.bar(x-w/2, [rc.get(v,0) for v in all_c], w, color='steelblue', alpha=0.7, label='Real ctrl')
            ax.bar(x+w/2, [sc.get(v,0) for v in all_c], w, color='coral', alpha=0.7, label='Synthetic')
            ax.set_xticks(x); ax.set_xticklabels([str(int(v)) for v in all_c])
            ax.set_title(c, fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"{version}: Control Group — Generated vs Real", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_correlation_comparison(real_df, syn_df, cols, version, save_path):
    """Correlation heatmap comparison."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    rc = real_df[cols].corr()
    sc = syn_df[cols].corr()
    diff = rc - sc
    for ax, mat, title, cmap, vmin, vmax in [
        (ax1, rc, 'Real Control', 'RdBu_r', -1, 1),
        (ax2, sc, 'Synthetic', 'RdBu_r', -1, 1),
        (ax3, diff, 'Difference', 'RdBu_r', None, None),
    ]:
        if vmin is None:
            mx = max(abs(diff.values.min()), abs(diff.values.max()), 0.01)
            vmin, vmax = -mx, mx
        im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{version}: Correlation Structure (Control)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_km_with_scope(real_t, real_e, syn_t, syn_e, version, save_path,
                       title_suffix="", extra_label_prefix=""):
    """KM curves with x-axis matched to real data scope."""
    fig, ax = plt.subplots(figsize=(10, 7))
    kmf = KaplanMeierFitter()
    real_max = real_t.max()

    # Clip synthetic times to real data range for fair comparison
    syn_t_clip = np.clip(syn_t, 0, real_max * 1.05)

    kmf.fit(real_t, real_e, label=f'{extra_label_prefix}Real control')
    kmf.plot_survival_function(ax=ax, color='steelblue', linewidth=2.5)

    kmf.fit(syn_t_clip, syn_e, label=f'{extra_label_prefix}Synthetic')
    kmf.plot_survival_function(ax=ax, color='coral', linewidth=2, linestyle='--')

    # Log-rank test
    lr = logrank_test(real_t, syn_t_clip, real_e, syn_e)

    ax.set_xlim(0, real_max * 1.05)
    ax.set_xlabel("Time (years)", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_title(f"{version}: KM Curves{title_suffix}\nLog-rank p={lr.p_value:.4f}",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    return lr.p_value


def plot_replacement_km(treat_t, treat_e, real_ctrl_t, real_ctrl_e,
                        syn_ctrl_t, syn_ctrl_e, version, save_path):
    """Compare: Treatment vs Real-Control vs Treatment vs Synthetic-Control."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    kmf = KaplanMeierFitter()

    # Determine shared x scope
    xmax = max(treat_t.max(), real_ctrl_t.max()) * 1.05
    syn_ctrl_t_clip = np.clip(syn_ctrl_t, 0, xmax)

    # Panel 1: Treatment vs Real Control
    kmf.fit(treat_t, treat_e, label='Treatment (D-penicil)')
    kmf.plot_survival_function(ax=ax1, color='#2166ac', linewidth=2.5)
    kmf.fit(real_ctrl_t, real_ctrl_e, label='Real Control (placebo)')
    kmf.plot_survival_function(ax=ax1, color='#b2182b', linewidth=2.5)
    lr_real = logrank_test(treat_t, real_ctrl_t, treat_e, real_ctrl_e)
    ax1.set_xlim(0, xmax)
    ax1.set_xlabel("Time (years)", fontsize=12)
    ax1.set_ylabel("Survival Probability", fontsize=12)
    ax1.set_title(f"Treatment vs Real Control\nLog-rank p={lr_real.p_value:.4f}", fontweight='bold')
    ax1.legend(fontsize=10)

    # Panel 2: Treatment vs Synthetic Control
    kmf.fit(treat_t, treat_e, label='Treatment (D-penicil)')
    kmf.plot_survival_function(ax=ax2, color='#2166ac', linewidth=2.5)
    kmf.fit(syn_ctrl_t_clip, syn_ctrl_e, label='Synthetic Control')
    kmf.plot_survival_function(ax=ax2, color='#d6604d', linewidth=2.5, linestyle='--')
    lr_syn = logrank_test(treat_t, syn_ctrl_t_clip, treat_e, syn_ctrl_e)
    ax2.set_xlim(0, xmax)
    ax2.set_xlabel("Time (years)", fontsize=12)
    ax2.set_ylabel("Survival Probability", fontsize=12)
    ax2.set_title(f"Treatment vs Synthetic Control\nLog-rank p={lr_syn.p_value:.4f}", fontweight='bold')
    ax2.legend(fontsize=10)

    fig.suptitle(f"{version}: Can Synthetic Replace Real Control?", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    return lr_real.p_value, lr_syn.p_value


def plot_replacement_summary(results_dict, save_path):
    """Summary bar chart: log-rank p-values for real vs synthetic control replacement."""
    fig, ax = plt.subplots(figsize=(10, 6))
    versions = list(results_dict.keys())
    p_real = [results_dict[v]['p_real'] for v in versions]
    p_syn = [results_dict[v]['p_syn'] for v in versions]
    x = np.arange(len(versions))
    w = 0.35
    ax.bar(x - w/2, p_real, w, label='Treat vs Real Control', color='steelblue', alpha=0.8)
    ax.bar(x + w/2, p_syn, w, label='Treat vs Synthetic Control', color='coral', alpha=0.8)
    ax.axhline(0.05, color='red', linestyle='--', linewidth=1, label='α=0.05')
    ax.set_xticks(x)
    ax.set_xticklabels([v.upper().replace('_', ' ') for v in versions])
    ax.set_ylabel("Log-rank p-value")
    ax.set_title("Replacement Analysis: Can Synthetic Control Replace Real Control?",
                 fontweight='bold', fontsize=13)
    ax.legend()
    # Annotate
    for i, (pr, ps) in enumerate(zip(p_real, p_syn)):
        ax.text(i - w/2, pr + 0.01, f"{pr:.3f}", ha='center', fontsize=9)
        ax.text(i + w/2, ps + 0.01, f"{ps:.3f}", ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_surv_time_qq(real_t, syn_t, version, save_path):
    """QQ plot + histogram of survival times."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(real_t, bins=25, alpha=0.5, density=True, color='steelblue', label='Real control')
    ax1.hist(syn_t.clip(0, real_t.max()*1.1), bins=25, alpha=0.5, density=True, color='coral', label='Synthetic')
    ks, p = stats.ks_2samp(real_t, syn_t)
    ax1.set_title(f"Survival Time Distribution\nKS={ks:.3f} p={p:.3f}", fontweight='bold')
    ax1.set_xlabel("Time (years)"); ax1.legend()

    n_q = min(len(real_t), len(syn_t))
    q = np.linspace(0, 1, n_q)
    rq = np.quantile(real_t, q); sq = np.quantile(syn_t, q)
    ax2.scatter(rq, sq, s=10, alpha=0.5, color='purple')
    lims = [0, max(rq.max(), sq.max())*1.05]
    ax2.plot(lims, lims, 'k--', linewidth=1)
    ax2.set_xlabel("Real Quantiles"); ax2.set_ylabel("Synthetic Quantiles")
    ax2.set_title("QQ Plot (Survival Times)", fontweight='bold')

    fig.suptitle(f"{version}: Survival Time Analysis (Control)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_longitudinal_traj(long_df, mu, traj, tg, norm_params, oidx, oname, version, save_path):
    """Longitudinal trajectory: real vs synthetic."""
    fig, ax = plt.subplots(figsize=(10, 6))
    tmin, tmax = norm_params["time_min"], norm_params["time_max"]
    tg_orig = tg * (tmax - tmin) + tmin
    vmean = norm_params["value_mean"][oidx]
    vstd = norm_params["value_std"][oidx]
    mu_d = mu[:, :, oidx].numpy() * vstd + vmean
    traj_d = traj[:, :, :, oidx].numpy() * vstd + vmean
    mu_mean = mu_d.mean(axis=0)
    traj_flat = traj_d.reshape(-1, traj_d.shape[-1])
    q05, q95 = np.percentile(traj_flat, 5, axis=0), np.percentile(traj_flat, 95, axis=0)
    ax.scatter(long_df["visit_time"].values, long_df[oname].values,
               alpha=0.15, s=8, color='steelblue', label='Real observations')
    ax.plot(tg_orig.numpy(), mu_mean, color='coral', linewidth=2, label='Synthetic mean')
    ax.fill_between(tg_orig.numpy(), q05, q95, color='coral', alpha=0.2, label='90% CI')
    ax.set_xlabel("Time (years)"); ax.set_ylabel(oname)
    ax.set_title(f"{version}: {oname} (Control Group)", fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_loss_curves(loss_dict, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, losses in loss_dict.items():
        ax.plot(losses, label=name, linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Neg ELBO Loss")
    ax.set_title("Training Loss (Control Group Only)", fontweight='bold')
    ax.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# =====================================================================
# Training
# =====================================================================

def train_v4_control(model_version, n_ctrl, long_ctrl_df):
    """Train a V4 model on control group only."""
    from src import HIVAE_inputDropout

    print(f"\n{'='*60}")
    print(f"Training {model_version} on CONTROL GROUP ({n_ctrl} patients)")
    print(f"{'='*60}")

    surv_hivae.set_seed()
    params = BASE_PARAMS.copy()

    df, types, miss, true_miss, n = data_processing.read_data(
        str(DATA_DIR / "data_v4_control.csv"),
        str(DATA_DIR / "data_types_v4.csv"),
        "Missing.csv", None, surv_type='surv_weibull')
    print(f"  Data shape: {df.shape}, n={n}")

    n_long = len(LONG_OUTCOME_NAMES)
    times_norm, values_norm, masks, long_norm_params = \
        data_processing.prepare_longitudinal_tensors(
            long_ctrl_df, patient_id_col="patient_id", time_col="visit_time",
            value_col=LONG_OUTCOME_NAMES, n_patients=n)
    longitudinal_data = (times_norm, values_norm, masks)
    print(f"  Longitudinal: {times_norm.shape}")

    bs = min(params["batch_size"], int(0.9 * n))
    model = HIVAE_inputDropout(
        input_dim=df.shape[1], z_dim=params["z_dim"], y_dim=params["y_dim"],
        s_dim=params["s_dim"], y_dim_partition=None, feat_types_dict=types,
        intervals_surv_piecewise=None, n_layers_surv_piecewise=None,
        model_version=model_version, n_long_outcomes=n_long)

    data_t = torch.from_numpy(df.values)
    train_device = 'cpu' if model_version == 'v4_seq' else None
    model, loss_train, loss_val = surv_hivae.train_HIVAE(
        model, data_t, miss, true_miss, types, bs, params["lr"], EPOCHS,
        longitudinal_data=longitudinal_data, device=train_device)

    # Generate
    gen = surv_hivae.generate_from_HIVAE(
        model, data_t, miss, true_miss, types, N_GEN, device=train_device)

    # Generate longitudinal
    device = next(model.parameters()).device
    with torch.no_grad():
        tg = torch.linspace(0, 1, 30)
        data_dev = data_t.to(device)
        miss_dev = torch.multiply(miss, true_miss).to(device)
        data_list_sub, miss_list_sub = data_processing.next_batch(data_dev, types, miss_dev, n, 0)
        data_list_obs = [d * miss_list_sub[:, j].view(n, 1) for j, d in enumerate(data_list_sub)]
        X_list, _ = data_processing.batch_normalization(data_list_obs, types, miss_list_sub)
        X = torch.cat(X_list, dim=1)
        long_dev = tuple(t.to(device) for t in longitudinal_data)
        long_summary = model._encode_longitudinal_summary(long_dev)
        X = torch.cat([X, long_summary], dim=1)
        _, samples = model.encode(X, tau=1e-3)

        if model_version == 'v4_seq':
            non_surv_x = [X_list[fi] for fi, f in enumerate(types) if not f['type'].startswith('surv')]
            c_X = model.baseline_summary_net(torch.cat(non_surv_x, dim=1))
            mu, var, trajectories = model.generate_longitudinal_seq(
                samples, tg.to(device), c_X, n_samples=N_GEN)
        else:
            mu, var, trajectories = model.generate_longitudinal(
                samples, tg.to(device), n_samples=N_GEN)

    real_data = data_processing.discrete_variables_transformation(data_t, types)
    real_data = data_processing.survival_variables_transformation(real_data, types)

    return {
        "model": model, "loss_train": loss_train, "loss_val": loss_val,
        "gen": gen, "real_data": real_data, "types": types,
        "mu": mu.cpu(), "trajectories": trajectories.cpu(), "time_grid": tg.cpu(),
        "long_norm_params": long_norm_params, "long_df": long_ctrl_df, "n": n,
    }


# =====================================================================
# Main
# =====================================================================

def main():
    # Prepare data
    df_ctrl, long_ctrl, df_treat, n_ctrl, ctrl_ids = prepare_control_data()

    all_losses = {}
    results = {}
    replacement_results = {}

    for vname in ["v4_joint", "v4_seq"]:
        res = train_v4_control(vname, n_ctrl, long_ctrl)
        results[vname] = res
        all_losses[vname.upper().replace('_',' ')] = res["loss_train"]
        label = vname.upper().replace('_', ' ')

        gen_sample = res["gen"][0]
        real = res["real_data"]
        real_df = pd.DataFrame(real.numpy(), columns=COLS_OUT)
        syn_df = pd.DataFrame(gen_sample.numpy(), columns=COLS_OUT)

        # ---- 1. Baseline covariate similarity ----
        cont_cols = [c for c in COLS_OUT if c not in
                     ['time','censor','drug','sex','ascites','hepatomegaly','spiders','edema','histologic']]
        cat_cols = ['sex','ascites','hepatomegaly','spiders','edema','histologic']

        plot_marginals_grid(real_df, syn_df, cont_cols, label,
                           DIR_BASELINE / f"{vname}_continuous.png", kind='hist')
        plot_marginals_grid(real_df, syn_df, cat_cols, label,
                           DIR_BASELINE / f"{vname}_categorical.png", kind='bar')
        num_cols = [c for c in COLS_OUT if c not in ['drug','sex','ascites','hepatomegaly','spiders','censor']]
        plot_correlation_comparison(real_df, syn_df, num_cols, label,
                                   DIR_BASELINE / f"{vname}_correlations.png")

        # ---- 2. Survival analysis ----
        real_t = real_df["time"].values
        real_e = real_df["censor"].values
        syn_t = syn_df["time"].values
        syn_e = (syn_df["censor"].values > 0.5).astype(float)

        plot_km_with_scope(real_t, real_e, syn_t, syn_e, label,
                          DIR_SURV / f"{vname}_km_control.png",
                          title_suffix=" (Control Group)")
        plot_surv_time_qq(real_t, syn_t, label,
                         DIR_SURV / f"{vname}_time_dist.png")

        # Censoring rate comparison
        fig, ax = plt.subplots(figsize=(6, 5))
        r_rate = real_e.mean(); s_rate = syn_e.mean()
        x = np.arange(2); w = 0.5
        ax.bar(x, [r_rate, s_rate], w, label='Event rate', color='steelblue', alpha=0.7)
        ax.bar(x, [1-r_rate, 1-s_rate], w, bottom=[r_rate, s_rate],
               label='Censored rate', color='lightcoral', alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(['Real Control', 'Synthetic'])
        ax.set_title(f'{label}: Event Rate\nReal={r_rate:.3f} Syn={s_rate:.3f}', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(DIR_SURV / f"{vname}_censoring.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {DIR_SURV / f'{vname}_censoring.png'}")

        # ---- 3. Replacement analysis: can synthetic control replace real control? ----
        treat_t = df_treat['time'].values
        treat_e = df_treat['censor'].values

        p_real, p_syn = plot_replacement_km(
            treat_t, treat_e, real_t, real_e, syn_t, syn_e,
            label, DIR_REPLACE / f"{vname}_replacement_km.png")
        replacement_results[vname] = {'p_real': p_real, 'p_syn': p_syn}
        print(f"  {label} replacement: p_real={p_real:.4f}, p_syn={p_syn:.4f}")

        # ---- 4. Longitudinal trajectories ----
        for oname in LONG_CONTINUOUS:
            oidx = LONG_OUTCOME_NAMES.index(oname)
            plot_longitudinal_traj(
                res["long_df"], res["mu"], res["trajectories"], res["time_grid"],
                res["long_norm_params"], oidx, oname, label,
                DIR_LONG / f"{vname}_{oname}_trajectories.png")

        # ---- 5. Diagnostics ----
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(res["loss_train"], label="Train", linewidth=1.5)
        val_c = [(i,v) for i,v in enumerate(res["loss_val"])
                 if not (isinstance(v, float) and np.isnan(v))
                 and not (torch.is_tensor(v) and torch.isnan(v))]
        if val_c:
            ax.plot([x[0] for x in val_c], [x[1] for x in val_c],
                    label="Validation", linewidth=1.5, linestyle='--')
        ax.set_xlabel("Epoch"); ax.set_ylabel("Neg ELBO")
        ax.set_title(f"{label}: Training (Control Only)", fontweight='bold')
        ax.legend(); plt.tight_layout()
        plt.savefig(DIR_DIAG / f"{vname}_train_val.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {DIR_DIAG / f'{vname}_train_val.png'}")

    # ---- Cross-version comparisons ----
    plot_loss_curves(all_losses, DIR_COMPARE / "loss_curves.png")

    # Combined KM: Real + both synthetics
    fig, ax = plt.subplots(figsize=(10, 7))
    kmf = KaplanMeierFitter()
    real_df_0 = pd.DataFrame(results["v4_joint"]["real_data"].numpy(), columns=COLS_OUT)
    rt, re = real_df_0["time"].values, real_df_0["censor"].values
    xmax = rt.max() * 1.05

    kmf.fit(rt, re, label="Real Control")
    kmf.plot_survival_function(ax=ax, color='black', linewidth=2.5)
    for vn, color, ls in [("v4_joint","coral","--"),("v4_seq","forestgreen","-.")]:
        syn = pd.DataFrame(results[vn]["gen"][0].numpy(), columns=COLS_OUT)
        st = np.clip(syn["time"].values, 0, xmax)
        se = (syn["censor"].values > 0.5).astype(float)
        kmf.fit(st, se, label=vn.upper().replace('_',' '))
        kmf.plot_survival_function(ax=ax, color=color, linewidth=2, linestyle=ls)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Time (years)"); ax.set_ylabel("Survival Probability")
    ax.set_title("Control Group: Real vs V4_joint vs V4_seq", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(DIR_COMPARE / "km_comparison_control.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {DIR_COMPARE / 'km_comparison_control.png'}")

    # Replacement summary
    plot_replacement_summary(replacement_results, DIR_REPLACE / "replacement_summary.png")

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for vn in ["v4_joint", "v4_seq"]:
        rr = replacement_results[vn]
        print(f"  {vn}:")
        print(f"    Treat vs Real Control   log-rank p = {rr['p_real']:.4f}")
        print(f"    Treat vs Synth Control  log-rank p = {rr['p_syn']:.4f}")
        same_conclusion = (rr['p_real'] > 0.05) == (rr['p_syn'] > 0.05)
        print(f"    Same conclusion at α=0.05? {'YES' if same_conclusion else 'NO'}")
    print(f"\nAll figures saved to: {BASE_FIG}")
    for d in [DIR_BASELINE, DIR_LONG, DIR_SURV, DIR_REPLACE, DIR_COMPARE, DIR_DIAG]:
        n_png = len(list(d.glob('*.png')))
        print(f"  {d.name}/ — {n_png} figures")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
