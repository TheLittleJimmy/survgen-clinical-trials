#!/usr/bin/env python3
"""
Train V1 model on PBC2 and generate ALL visualisations:
  - Overall: distributions, correlations, Q-Q, CDF+KS, pairwise scatter, summary
  - By treatment group: all of the above split by D-penicillamine vs Placebo

Single training run, all plots generated from the same model.

Usage:
    cd /project/Stat/s1155202253/myproject/pfizer_projects/survgen-clinical-trials
    conda activate env_2502
    python experiments/pbc2_v1_detailed.py
"""

import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(ROOT / "execute"))

import data_processing
import surv_hivae

DATA_DIR = ROOT / "dataset" / "pbc2"
FIG_DIR  = ROOT / "experiments" / "pbc2_figures" / "v1_detailed"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── hyper-params ──────────────────────────────────────────────────────────────
EPOCHS = 500
N_GEN  = 10
PARAMS = {"lr": 1e-3, "batch_size": 100, "z_dim": 20, "y_dim": 15, "s_dim": 20}

COLS = ["serBilir", "drug", "sex", "ascites", "hepatomegaly", "spiders",
        "edema", "histologic", "albumin", "alkaline", "SGOT", "platelets",
        "prothrombin", "age"]

CONT_COLS = ["serBilir", "albumin", "alkaline", "SGOT",
             "platelets", "prothrombin", "age"]
CAT_COLS  = ["drug", "sex", "ascites", "hepatomegaly",
             "spiders", "edema", "histologic"]
# For by-group plots, exclude drug (it IS the grouping variable)
CAT_COLS_NOGRP = ["sex", "ascites", "hepatomegaly",
                  "spiders", "edema", "histologic"]

CAT_LABELS = {
    "drug":         {0: "D-penicil", 1: "placebo"},
    "sex":          {0: "female", 1: "male"},
    "ascites":      {0: "No", 1: "Yes"},
    "hepatomegaly": {0: "No", 1: "Yes"},
    "spiders":      {0: "No", 1: "Yes"},
    "edema":        {0: "None", 1: "no diuret", 2: "diuretics"},
    "histologic":   {0: "1", 1: "2", 2: "3", 3: "4"},
}

GROUP_NAMES  = {0: "D-penicillamine", 1: "Placebo"}
GROUP_COLORS = {
    0: {"real": "#2166ac", "syn": "#67a9cf"},   # blues
    1: {"real": "#b2182b", "syn": "#ef8a62"},   # reds
}


# ═══════════════════════════  PLOTTING HELPERS  ═══════════════════════════════

def _savefig(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.name}")


# ─────────────────────── OVERALL (pooled) plots ──────────────────────────────

def plot_loss_curve(train_loss, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_loss, color="steelblue", linewidth=1.2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Neg ELBO")
    ax.set_title("V1 Training Loss", fontweight="bold"); ax.grid(alpha=0.3)
    _savefig(fig, save_path)


def plot_cont_distributions(real, syn, save_path):
    n = len(CONT_COLS); ncols = 4; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.ravel()
    for i, feat in enumerate(CONT_COLS):
        ax = axes[i]
        r, s = real[feat].dropna(), syn[feat].dropna()
        lo, hi = min(r.min(), s.min()), max(r.max(), s.max())
        bins = np.linspace(lo, hi, 40)
        ax.hist(r, bins=bins, density=True, alpha=0.45, color="steelblue", label="Real")
        ax.hist(s, bins=bins, density=True, alpha=0.45, color="coral", label="Syn")
        try:
            xs = np.linspace(lo, hi, 200)
            ax.plot(xs, stats.gaussian_kde(r)(xs), color="steelblue", lw=1.5)
            ax.plot(xs, stats.gaussian_kde(s)(xs), color="coral", lw=1.5)
        except Exception:
            pass
        ks, p = stats.ks_2samp(r.values, s.values)
        ax.text(0.97, 0.93, f"KS={ks:.3f}\np={p:.2e}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox=dict(fc="lightyellow", alpha=0.8, boxstyle="round,pad=0.3"))
        ax.set_title(feat, fontweight="bold"); ax.legend(fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Continuous Feature Distributions (all patients)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(); _savefig(fig, save_path)


def plot_cat_distributions(real, syn, cat_cols, save_path, title_suffix=""):
    n = len(cat_cols); ncols = 4; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, feat in enumerate(cat_cols):
        ax = axes[i]
        labels_map = CAT_LABELS.get(feat, {})
        rc = real[feat].value_counts(normalize=True).sort_index()
        sc = syn[feat].value_counts(normalize=True).sort_index()
        all_cats = sorted(set(rc.index) | set(sc.index))
        x = np.arange(len(all_cats)); w = 0.35
        ax.bar(x - w/2, [rc.get(c, 0) for c in all_cats], w,
               label="Real", color="steelblue", alpha=0.75)
        ax.bar(x + w/2, [sc.get(c, 0) for c in all_cats], w,
               label="Syn", color="coral", alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels([labels_map.get(int(c), str(int(c))) for c in all_cats],
                           fontsize=8)
        ax.set_ylabel("Proportion"); ax.set_title(feat, fontweight="bold")
        ax.legend(fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Categorical Feature Distributions{title_suffix}",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(); _savefig(fig, save_path)


def plot_correlation_matrices(real, syn, save_path, title_suffix=""):
    corr_r = real[CONT_COLS].corr()
    corr_s = syn[CONT_COLS].corr()
    diff   = corr_r - corr_s
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for ax, mat, lbl, cmap in zip(axes,
            [corr_r, corr_s, diff],
            ["Real", "Synthetic", "Diff (R-S)"],
            ["RdBu_r", "RdBu_r", "PiYG"]):
        im = ax.imshow(mat.values, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(CONT_COLS)))
        ax.set_yticks(range(len(CONT_COLS)))
        ax.set_xticklabels(CONT_COLS, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(CONT_COLS, fontsize=8)
        for ii in range(len(CONT_COLS)):
            for jj in range(len(CONT_COLS)):
                ax.text(jj, ii, f"{mat.values[ii, jj]:.2f}", ha="center",
                        va="center", fontsize=7,
                        color="white" if abs(mat.values[ii, jj]) > 0.6 else "black")
        ax.set_title(lbl, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Correlation Matrices{title_suffix}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(); _savefig(fig, save_path)


def plot_qq(real, syn, save_path, title_suffix=""):
    n = len(CONT_COLS); ncols = 4; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = axes.ravel()
    for i, feat in enumerate(CONT_COLS):
        ax = axes[i]
        r = np.sort(real[feat].dropna().values)
        s = np.sort(syn[feat].dropna().values)
        q = np.linspace(0, 1, min(len(r), len(s), 500))
        rq, sq = np.quantile(r, q), np.quantile(s, q)
        ax.scatter(rq, sq, s=6, alpha=0.5, color="darkorchid")
        lo, hi = min(rq.min(), sq.min()), max(rq.max(), sq.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
        ax.set_xlabel("Real quantiles"); ax.set_ylabel("Syn quantiles")
        ax.set_title(feat, fontweight="bold"); ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="box")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Q-Q Plots{title_suffix}", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(); _savefig(fig, save_path)


def plot_marginal_cdf(real, syn, save_path, title_suffix=""):
    n = len(CONT_COLS); ncols = 4; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.ravel()
    for i, feat in enumerate(CONT_COLS):
        ax = axes[i]
        for arr, label, color in [(real[feat], "Real", "steelblue"),
                                   (syn[feat], "Syn", "coral")]:
            xs = np.sort(arr.values)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(xs, ys, color=color, lw=1.5, label=label)
        ks, p = stats.ks_2samp(real[feat].values, syn[feat].values)
        ax.text(0.98, 0.05, f"KS={ks:.3f}\np={p:.2e}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
        ax.set_title(feat, fontweight="bold"); ax.set_ylabel("CDF")
        ax.legend(fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Empirical CDF + KS test{title_suffix}",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(); _savefig(fig, save_path)


def plot_pairwise_scatter(real, syn, feats, save_path):
    n = len(feats)
    fig, axes = plt.subplots(n, n, figsize=(3.5 * n, 3.5 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(real[feats[i]], bins=30, density=True,
                        alpha=0.5, color="steelblue", label="Real")
                ax.hist(syn[feats[i]], bins=30, density=True,
                        alpha=0.5, color="coral", label="Syn")
                ax.set_title(feats[i], fontsize=9, fontweight="bold")
            else:
                ax.scatter(real[feats[j]], real[feats[i]],
                           s=4, alpha=0.3, color="steelblue", label="Real")
                ax.scatter(syn[feats[j]], syn[feats[i]],
                           s=4, alpha=0.3, color="coral", label="Syn")
            if j == 0: ax.set_ylabel(feats[i], fontsize=8)
            if i == n-1: ax.set_xlabel(feats[j], fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Pair-wise Scatter (all patients)", fontsize=14, fontweight="bold")
    fig.tight_layout(); _savefig(fig, save_path)


def plot_summary_table(real, syn, cols, save_path, title_suffix=""):
    rows = []
    for feat in cols:
        r, s = real[feat], syn[feat]
        rows.append({"Feature": feat,
                     "R_mean": f"{r.mean():.3f}", "S_mean": f"{s.mean():.3f}",
                     "R_std":  f"{r.std():.3f}",  "S_std":  f"{s.std():.3f}",
                     "R_med":  f"{r.median():.3f}","S_med": f"{s.median():.3f}",
                     "R_min":  f"{r.min():.3f}",  "S_min":  f"{s.min():.3f}",
                     "R_max":  f"{r.max():.3f}",  "S_max":  f"{s.max():.3f}"})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(18, 0.4 * len(rows) + 2))
    ax.axis("off")
    col_labels = ["Feature", "Mean(R)", "Mean(S)", "Std(R)", "Std(S)",
                  "Med(R)", "Med(S)", "Min(R)", "Min(S)", "Max(R)", "Max(S)"]
    table = ax.table(cellText=df.values, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.3)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        bg = "#D9E2F3" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(bg)
    ax.set_title(f"Summary Statistics{title_suffix}",
                 fontweight="bold", fontsize=13, pad=12)
    _savefig(fig, save_path)


# ─────────────────────── BY-GROUP plots ──────────────────────────────────────

def plot_cont_by_group(real_df, syn_df, save_path):
    n = len(CONT_COLS)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.8 * n), sharey="row")
    for row, feat in enumerate(CONT_COLS):
        for col, grp in enumerate([0, 1]):
            ax = axes[row, col]
            r = real_df.loc[real_df["drug"] == grp, feat].dropna()
            s = syn_df.loc[syn_df["drug"] == grp, feat].dropna()
            lo = min(r.min(), s.min()) if len(s) else r.min()
            hi = max(r.max(), s.max()) if len(s) else r.max()
            bins = np.linspace(lo, hi, 35)
            cr, cs = GROUP_COLORS[grp]["real"], GROUP_COLORS[grp]["syn"]
            ax.hist(r, bins=bins, density=True, alpha=0.40, color=cr,
                    label=f"Real (n={len(r)})")
            ax.hist(s, bins=bins, density=True, alpha=0.40, color=cs,
                    label=f"Syn (n={len(s)})")
            try:
                xs = np.linspace(lo, hi, 200)
                ax.plot(xs, stats.gaussian_kde(r)(xs), color=cr, lw=1.4)
                ax.plot(xs, stats.gaussian_kde(s)(xs), color=cs, lw=1.4,
                        linestyle="--")
            except Exception:
                pass
            if len(s) > 1:
                ks, p = stats.ks_2samp(r.values, s.values)
                ax.text(0.97, 0.93, f"KS={ks:.3f}\np={p:.2e}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=7,
                        bbox=dict(fc="lightyellow", alpha=0.8,
                                  boxstyle="round,pad=0.3"))
            if row == 0:
                ax.set_title(GROUP_NAMES[grp], fontsize=12, fontweight="bold")
            ax.set_ylabel(feat, fontsize=10, fontweight="bold")
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Continuous Features by Treatment Group",
                 fontsize=15, fontweight="bold", y=1.005)
    fig.tight_layout(); _savefig(fig, save_path)


def plot_cat_by_group(real_df, syn_df, save_path):
    cat_cols = CAT_COLS_NOGRP
    n = len(cat_cols)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.2 * n))
    for row, feat in enumerate(cat_cols):
        labels_map = CAT_LABELS.get(feat, {})
        for col, grp in enumerate([0, 1]):
            ax = axes[row, col]
            r = real_df.loc[real_df["drug"] == grp, feat]
            s = syn_df.loc[syn_df["drug"] == grp, feat]
            rc = r.value_counts(normalize=True).sort_index()
            sc = s.value_counts(normalize=True).sort_index()
            all_cats = sorted(set(rc.index) | set(sc.index))
            x = np.arange(len(all_cats)); w = 0.35
            cr, cs = GROUP_COLORS[grp]["real"], GROUP_COLORS[grp]["syn"]
            ax.bar(x - w/2, [rc.get(c, 0) for c in all_cats], w,
                   label=f"Real (n={len(r)})", color=cr, alpha=0.75)
            ax.bar(x + w/2, [sc.get(c, 0) for c in all_cats], w,
                   label=f"Syn (n={len(s)})", color=cs, alpha=0.75)
            ax.set_xticks(x)
            ax.set_xticklabels([labels_map.get(int(c), str(int(c)))
                                for c in all_cats], fontsize=8)
            ax.set_ylabel("Proportion")
            if row == 0:
                ax.set_title(GROUP_NAMES[grp], fontsize=12, fontweight="bold")
            if col == 0:
                ax.annotate(feat, xy=(-0.22, 0.5), xycoords="axes fraction",
                            fontsize=10, fontweight="bold", va="center",
                            rotation=90)
            ax.legend(fontsize=7)
    fig.suptitle("Categorical Features by Treatment Group",
                 fontsize=15, fontweight="bold", y=1.005)
    fig.tight_layout(); _savefig(fig, save_path)


def plot_corr_by_group(real_df, syn_df, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    for row, grp in enumerate([0, 1]):
        r = real_df.loc[real_df["drug"] == grp, CONT_COLS].corr()
        s = syn_df.loc[syn_df["drug"] == grp, CONT_COLS].corr()
        diff = r - s
        for ci, (mat, lbl, cmap) in enumerate([
            (r, "Real", "RdBu_r"), (s, "Synthetic", "RdBu_r"),
            (diff, "Diff (R-S)", "PiYG"),
        ]):
            ax = axes[row, ci]
            im = ax.imshow(mat.values, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
            ax.set_xticks(range(len(CONT_COLS)))
            ax.set_yticks(range(len(CONT_COLS)))
            ax.set_xticklabels(CONT_COLS, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(CONT_COLS, fontsize=7)
            for ii in range(len(CONT_COLS)):
                for jj in range(len(CONT_COLS)):
                    ax.text(jj, ii, f"{mat.values[ii, jj]:.2f}", ha="center",
                            va="center", fontsize=6,
                            color="white" if abs(mat.values[ii, jj]) > 0.55
                            else "black")
            ax.set_title(f"{GROUP_NAMES[grp]} — {lbl}", fontsize=10,
                         fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Correlation Matrices by Treatment Group",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(); _savefig(fig, save_path)


def plot_qq_by_group(real_df, syn_df, save_path):
    n = len(CONT_COLS)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.8 * n))
    for row, feat in enumerate(CONT_COLS):
        for col, grp in enumerate([0, 1]):
            ax = axes[row, col]
            r = np.sort(real_df.loc[real_df["drug"] == grp, feat].dropna().values)
            s = np.sort(syn_df.loc[syn_df["drug"] == grp, feat].dropna().values)
            q = np.linspace(0, 1, min(len(r), len(s), 500))
            rq, sq = np.quantile(r, q), np.quantile(s, q)
            c = GROUP_COLORS[grp]["real"]
            ax.scatter(rq, sq, s=6, alpha=0.5, color=c)
            lo, hi = min(rq.min(), sq.min()), max(rq.max(), sq.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_xlabel("Real quantiles", fontsize=8)
            ax.set_ylabel("Syn quantiles", fontsize=8)
            if row == 0:
                ax.set_title(GROUP_NAMES[grp], fontsize=12, fontweight="bold")
            ax.annotate(feat, xy=(0.03, 0.92), xycoords="axes fraction",
                        fontsize=9, fontweight="bold",
                        bbox=dict(fc="white", alpha=0.7, boxstyle="round,pad=0.2"))
            ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Q-Q Plots by Treatment Group",
                 fontsize=15, fontweight="bold", y=1.005)
    fig.tight_layout(); _savefig(fig, save_path)


def plot_summary_by_group(real_df, syn_df, save_path):
    rows = []
    for grp in [0, 1]:
        gname = GROUP_NAMES[grp]
        rg = real_df[real_df["drug"] == grp]
        sg = syn_df[syn_df["drug"] == grp]
        for feat in COLS:
            rv, sv = rg[feat], sg[feat]
            rows.append({"Group": gname, "Feature": feat,
                         "R_mean": f"{rv.mean():.2f}", "S_mean": f"{sv.mean():.2f}",
                         "R_std":  f"{rv.std():.2f}",  "S_std":  f"{sv.std():.2f}",
                         "R_med":  f"{rv.median():.2f}","S_med": f"{sv.median():.2f}",
                         "R_n": str(len(rv)), "S_n": str(len(sv))})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(18, 0.34 * len(rows) + 2.5))
    ax.axis("off")
    col_labels = ["Group", "Feature", "Mean(R)", "Mean(S)",
                  "Std(R)", "Std(S)", "Med(R)", "Med(S)", "n(R)", "n(S)"]
    table = ax.table(cellText=df.values, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(7); table.scale(1, 1.2)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        is_placebo = df.iloc[i - 1]["Group"] == "Placebo"
        bg = "#fce4e4" if is_placebo else "#dce6f1"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(bg)
    ax.set_title("Summary Statistics by Treatment Group",
                 fontweight="bold", fontsize=13, pad=12)
    _savefig(fig, save_path)


# ═══════════════════════════════  MAIN  ═══════════════════════════════════════

def main():
    from src import HIVAE_inputDropout

    print(f"\n{'='*60}")
    print("V1 — train once, generate all visualisations")
    print(f"{'='*60}")

    surv_hivae.set_seed()

    # ── Load & train ──
    df, types, miss, true_miss, n = data_processing.read_data(
        str(DATA_DIR / "data.csv"),
        str(DATA_DIR / "data_types_v1.csv"),
        "Missing.csv", None,
    )
    print(f"  Data: {df.shape}  n={n}")

    batch = min(PARAMS["batch_size"], int(0.9 * n))
    model = HIVAE_inputDropout(
        input_dim=df.shape[1],
        z_dim=PARAMS["z_dim"], y_dim=PARAMS["y_dim"], s_dim=PARAMS["s_dim"],
        y_dim_partition=None, feat_types_dict=types,
        intervals_surv_piecewise=None, n_layers_surv_piecewise=None,
        model_version="v1",
    )
    data_t = torch.from_numpy(df.values)
    model, train_loss, _ = surv_hivae.train_HIVAE(
        model, data_t, miss, true_miss, types, batch, PARAMS["lr"], EPOCHS,
    )

    # ── Generate ──
    gen = surv_hivae.generate_from_HIVAE(
        model, data_t, miss, true_miss, types, N_GEN,
    )
    real = data_processing.discrete_variables_transformation(data_t, types)
    real_df = pd.DataFrame(real.numpy(), columns=COLS)
    syn_df  = pd.DataFrame(gen[0].numpy(), columns=COLS)

    print(f"\n  Real: {real_df.shape}   Syn: {syn_df.shape}")
    for g in [0, 1]:
        nr = (real_df["drug"] == g).sum()
        ns = (syn_df["drug"] == g).sum()
        print(f"  {GROUP_NAMES[g]:20s}  Real n={nr}  Syn n={ns}")

    # ══════════ OVERALL plots ══════════
    print(f"\n--- Overall plots ---")
    plot_loss_curve(train_loss,
                    FIG_DIR / "01_training_loss.png")
    plot_cont_distributions(real_df, syn_df,
                            FIG_DIR / "02_continuous_distributions.png")
    plot_cat_distributions(real_df, syn_df, CAT_COLS,
                           FIG_DIR / "03_categorical_distributions.png",
                           " (all patients)")
    plot_correlation_matrices(real_df, syn_df,
                              FIG_DIR / "04_correlation_matrices.png",
                              " (all patients)")
    plot_qq(real_df, syn_df,
            FIG_DIR / "05_qq_plots.png", " (all patients)")
    plot_marginal_cdf(real_df, syn_df,
                      FIG_DIR / "06_empirical_cdf_ks.png", " (all patients)")
    plot_pairwise_scatter(real_df, syn_df, CONT_COLS,
                          FIG_DIR / "07_pairwise_scatter.png")
    plot_summary_table(real_df, syn_df, COLS,
                       FIG_DIR / "08_summary_statistics.png", " (all patients)")

    # ══════════ BY-GROUP plots ══════════
    print(f"\n--- By-group plots ---")
    plot_cont_by_group(real_df, syn_df,
                       FIG_DIR / "09_cont_by_group.png")
    plot_cat_by_group(real_df, syn_df,
                      FIG_DIR / "10_cat_by_group.png")
    plot_corr_by_group(real_df, syn_df,
                       FIG_DIR / "11_corr_by_group.png")
    plot_qq_by_group(real_df, syn_df,
                     FIG_DIR / "12_qq_by_group.png")
    plot_summary_by_group(real_df, syn_df,
                          FIG_DIR / "13_summary_by_group.png")

    # ── Save synthetic CSV ──
    syn_df.to_csv(FIG_DIR / "v1_synthetic_sample.csv", index=False)
    print(f"\n  -> v1_synthetic_sample.csv")

    print(f"\n{'='*60}")
    print(f"Done — 13 plots + CSV in {FIG_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
