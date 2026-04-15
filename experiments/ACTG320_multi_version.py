#!/usr/bin/env python3
"""
Train V0, V3_weibull, V3_piecewise on ACTG320 and generate comparison figures.

Usage:
    cd /project/Stat/s1155202253/myproject/pfizer_projects/survgen-clinical-trials
    conda activate env_2502
    python experiments/ACTG320_multi_version.py
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
from lifelines import KaplanMeierFitter

# ---- Path setup ----
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(ROOT / "execute"))

import data_processing
import surv_hivae
from src import HIVAE_inputDropout

DATA_DIR = ROOT / "dataset" / "ACTG320"
FIG_DIR = ROOT / "experiments" / "ACTG320_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- Device ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---- Hyperparams ----
EPOCHS = 500
N_GEN = 10
BASE_PARAMS = {"lr": 1e-3, "batch_size": 100, "z_dim": 20, "y_dim": 15, "s_dim": 20}

COLS_V3 = ["time", "censor", "tx", "strat2", "sex", "raceth",
           "ivdrug", "karnof", "cd4", "priorzdv", "age"]
COLS_V0 = ["tx", "strat2", "sex", "raceth", "ivdrug", "karnof",
           "cd4", "priorzdv", "age"]

CAT_FEATS = ["tx", "karnof", "raceth"]
CONT_FEATS = ["cd4", "priorzdv", "age"]


def train_version(name, data_file, types_file, model_version, params, epochs, n_gen):
    """Train one version, return collapsed real data, losses, and generated tensor."""
    print(f"\n{'='*60}")
    print(f"Training {name} (model_version={model_version})")
    print(f"{'='*60}")

    surv_hivae.set_seed()

    df, types_dict, miss_mask, true_miss_mask, n_samples = data_processing.read_data(
        str(data_file), str(types_file), "Missing.csv", None
    )
    print(f"  Data shape (expanded): {df.shape}, n_samples: {n_samples}")

    batch_size = min(params["batch_size"], int(0.9 * n_samples))
    intervals = None
    n_layers = None
    if "n_intervals" in params:
        intervals = surv_hivae.get_intervals(df, params["n_intervals"])
        n_layers = params.get("n_layers_surv_piecewise", 1)

    model = HIVAE_inputDropout(
        input_dim=df.shape[1],
        z_dim=params["z_dim"],
        y_dim=params["y_dim"],
        s_dim=params["s_dim"],
        y_dim_partition=None,
        feat_types_dict=types_dict,
        intervals_surv_piecewise=intervals,
        n_layers_surv_piecewise=n_layers,
        model_version=model_version,
    )

    data_tensor = torch.from_numpy(df.values)

    model, loss_train, loss_val = surv_hivae.train_HIVAE(
        model, data_tensor, miss_mask, true_miss_mask,
        types_dict, batch_size, params["lr"], epochs, device=DEVICE
    )

    gen_data = surv_hivae.generate_from_HIVAE(
        model, data_tensor, miss_mask, true_miss_mask,
        types_dict, n_gen, device=DEVICE
    )

    # Collapse real data — only discrete transform (NOT survival transform).
    # Real data already has (time, event_indicator) format; applying
    # survival_variables_transformation would corrupt it via min(time, 0/1).
    real_collapsed = data_processing.discrete_variables_transformation(data_tensor, types_dict)

    return real_collapsed, types_dict, loss_train, loss_val, gen_data


def plot_feature_comparison(real_df, syn_df, cat_feats, cont_feats, title, save_path):
    """Plot real vs synthetic feature distributions (histograms + bar charts)."""
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


def plot_km_curves(real_df, gen_data_list, col_names, title, save_path, time_max=None):
    """Plot Kaplan-Meier survival curves: real vs synthetic (with confidence band).

    Parameters
    ----------
    real_df : pd.DataFrame with 'time' and 'censor' columns (already correct).
    gen_data_list : list[Tensor] — each tensor is one generated dataset;
        survival columns are already (observed_time, event_indicator) after
        generate_from_HIVAE's internal survival_variables_transformation.
    col_names : list[str] — column names matching generated tensor columns.
    title : str
    save_path : Path
    time_max : float, optional — cap x-axis at this value.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # ---- Real KM ----
    kmf = KaplanMeierFitter()
    kmf.fit(real_df["time"], real_df["censor"], label="Real")
    kmf.plot_survival_function(ax=ax, color='steelblue', linewidth=2)

    # ---- Synthetic KM (multiple samples → median + band) ----
    # Collect survival functions from each generated sample
    eval_times = np.linspace(0, time_max if time_max else real_df["time"].max(), 500)
    surv_curves = []
    for gen_tensor in gen_data_list:
        syn_df = pd.DataFrame(gen_tensor.numpy(), columns=col_names)
        # Generated data already transformed by generate_from_HIVAE:
        # columns are (observed_time, event_indicator), use directly
        syn_time = syn_df["time"].clip(lower=1e-3).values
        syn_event = syn_df["censor"].round().clip(0, 1).astype(int).values
        kmf_s = KaplanMeierFitter()
        kmf_s.fit(syn_time, syn_event)
        surv_at_t = np.interp(eval_times, kmf_s.survival_function_.index,
                              kmf_s.survival_function_.iloc[:, 0])
        surv_curves.append(surv_at_t)

    surv_curves = np.array(surv_curves)
    median_curve = np.median(surv_curves, axis=0)
    lo = np.percentile(surv_curves, 10, axis=0)
    hi = np.percentile(surv_curves, 90, axis=0)

    ax.plot(eval_times, median_curve, color='coral', linewidth=2, linestyle='--',
            label="Synthetic (median)")
    ax.fill_between(eval_times, lo, hi, color='coral', alpha=0.15,
                    label="Synthetic (10-90% band)")

    if time_max:
        ax.set_xlim(0, time_max)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_loss_curves(loss_dict, save_path):
    """Plot training loss evolution for all versions."""
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
    all_losses = {}

    # ========== V0 (baseline only) ==========
    params_v0 = BASE_PARAMS.copy()
    real_v0, _, loss_v0, _, gen_v0 = train_version(
        "V0", DATA_DIR / "data_v0.csv", DATA_DIR / "data_types_v0.csv",
        "v0", params_v0, EPOCHS, N_GEN
    )
    all_losses["V0"] = loss_v0

    real_v0_df = pd.DataFrame(real_v0.numpy(), columns=COLS_V0)
    syn_v0_df = pd.DataFrame(gen_v0[0].numpy(), columns=COLS_V0)
    plot_feature_comparison(
        real_v0_df, syn_v0_df, CAT_FEATS, CONT_FEATS,
        "V0 (Baseline): Real vs Synthetic",
        FIG_DIR / "v0_feature_comparison.png",
    )

    # ---- Read raw ACTG320 data once for real time/censor ----
    raw_df = pd.read_csv(DATA_DIR / "data.csv", header=None)
    real_time_raw = raw_df.iloc[:, 0].values.astype(float)
    real_censor_raw = raw_df.iloc[:, 1].values.astype(float)
    time_max = float(real_time_raw.max())

    # ========== V3 Weibull ==========
    params_w = BASE_PARAMS.copy()
    real_w, _, loss_w, _, gen_w = train_version(
        "V3_weibull", DATA_DIR / "data.csv", DATA_DIR / "data_types_weibull.csv",
        "v3_weibull", params_w, EPOCHS, N_GEN
    )
    all_losses["V3_weibull"] = loss_w

    # Build real DF: time/censor from raw CSV, other features from collapsed tensor
    real_w_df = pd.DataFrame(real_w.numpy(), columns=COLS_V3)
    real_w_df["time"] = real_time_raw
    real_w_df["censor"] = real_censor_raw
    syn_w_df = pd.DataFrame(gen_w[0].numpy(), columns=COLS_V3)
    plot_feature_comparison(
        real_w_df, syn_w_df, CAT_FEATS, CONT_FEATS,
        "V3 Weibull: Real vs Synthetic",
        FIG_DIR / "v3w_feature_comparison.png",
    )
    plot_km_curves(
        real_w_df, gen_w, COLS_V3,
        "V3 Weibull: Kaplan-Meier Survival Curves",
        FIG_DIR / "v3w_km_curves.png",
        time_max=time_max,
    )

    # ========== V3 Piecewise ==========
    params_pw = BASE_PARAMS.copy()
    params_pw["n_intervals"] = 10
    params_pw["n_layers_surv_piecewise"] = 1
    real_pw, _, loss_pw, _, gen_pw = train_version(
        "V3_piecewise", DATA_DIR / "data.csv", DATA_DIR / "data_types_piecewise.csv",
        "v3_piecewise", params_pw, EPOCHS, N_GEN
    )
    all_losses["V3_piecewise"] = loss_pw

    # Build real DF: time/censor from raw CSV, other features from collapsed tensor
    real_pw_df = pd.DataFrame(real_pw.numpy(), columns=COLS_V3)
    real_pw_df["time"] = real_time_raw
    real_pw_df["censor"] = real_censor_raw
    syn_pw_df = pd.DataFrame(gen_pw[0].numpy(), columns=COLS_V3)
    plot_feature_comparison(
        real_pw_df, syn_pw_df, CAT_FEATS, CONT_FEATS,
        "V3 Piecewise: Real vs Synthetic",
        FIG_DIR / "v3pw_feature_comparison.png",
    )
    plot_km_curves(
        real_pw_df, gen_pw, COLS_V3,
        "V3 Piecewise: Kaplan-Meier Survival Curves",
        FIG_DIR / "v3pw_km_curves.png",
        time_max=time_max,
    )

    # ========== Loss curves ==========
    plot_loss_curves(all_losses, FIG_DIR / "loss_curves.png")

    print(f"\n{'='*60}")
    print(f"All figures saved to: {FIG_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
