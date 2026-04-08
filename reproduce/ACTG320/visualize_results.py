#!/usr/bin/env python3
"""
Visualization script for ACTG320 reproduction results.
Generates figures comparing real vs synthetic data and evaluation metrics.

Usage:
    cd survgen-clinical-trials/script
    python3 ../reproduce/ACTG320/visualize_results.py
"""

import sys, os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
module_path = Path(__file__).resolve().parents[2] / 'utils'
sys.path.insert(0, str(module_path))
module_path_exec = Path(__file__).resolve().parents[2] / 'execute'
sys.path.insert(0, str(module_path_exec))

import data_processing
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

sns.set(style="whitegrid", font="STIXGeneral", context="talk", palette="colorblind")

# ── Paths ──────────────────────────────────────────────────────────────────
PROJ = Path(__file__).resolve().parents[2]
DATASET = PROJ / "dataset" / "ACTG320"
REPRO = PROJ / "reproduce" / "ACTG320"
FIGDIR = REPRO / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ── Load original data ────────────────────────────────────────────────────
def load_original_data():
    feat_types_file = DATASET / "data_types_control.csv"
    miss_file = str(DATASET / "Missing10_1.csv")

    df_ctrl_enc, feat_types_dict, miss_ctrl, true_miss_ctrl, _ = \
        data_processing.read_data(str(DATASET / "data_control.csv"), str(feat_types_file), miss_file, None)
    df_treat_enc, _, _, _, _ = \
        data_processing.read_data(str(DATASET / "data_treated.csv"), str(DATASET / "data_types_treated.csv"), miss_file, None)

    fnames = ['time', 'censor'] + pd.read_csv(feat_types_file)["name"].to_list()[1:]

    data_ctrl = data_processing.discrete_variables_transformation(
        torch.from_numpy(df_ctrl_enc.values), feat_types_dict)
    data_treat = data_processing.discrete_variables_transformation(
        torch.from_numpy(df_treat_enc.values), feat_types_dict)

    df_ctrl = pd.DataFrame(data_ctrl.numpy(), columns=fnames)
    df_treat = pd.DataFrame(data_treat.numpy(), columns=fnames)
    df_ctrl["treatment"] = 0
    df_treat["treatment"] = 1
    df_full = pd.concat([df_ctrl, df_treat], ignore_index=True)

    return df_ctrl, df_treat, df_full, fnames, feat_types_dict, df_ctrl_enc, miss_ctrl, true_miss_ctrl


def generate_synthetic(df_ctrl_enc, miss_ctrl, true_miss_ctrl, feat_types_dict, fnames, generator_name="HI-VAE_weibull"):
    """Train one model and generate synthetic control data."""
    import surv_hivae
    # Load best params
    param_dir = DATASET / "optuna_results"
    for f in os.listdir(param_dir):
        if f.endswith(generator_name + '.json') and "traincontrol_ACTG320" in f and "aug" not in f:
            with open(param_dir / f) as fh:
                best_params = json.load(fh)
            break

    feat_dict = [dict(d) for d in feat_types_dict]
    for d in feat_dict:
        if d['name'] == 'survcens':
            if 'weibull' in generator_name:
                d['type'] = 'surv_weibull'
            elif 'piecewise' in generator_name:
                d['type'] = 'surv_piecewise'

    gen_from_prior = 'prior' in generator_name
    n_gen = 10  # fewer for visualization speed
    data_gen = surv_hivae.run(df_ctrl_enc, miss_ctrl, true_miss_ctrl, feat_dict,
                              n_gen, params=best_params, epochs=10000,
                              gen_from_prior=gen_from_prior)

    synth_list = []
    for i in range(n_gen):
        df_syn = pd.DataFrame(data_gen[i].numpy(), columns=fnames)
        df_syn["treatment"] = 0
        synth_list.append(df_syn)
    return synth_list


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Kaplan-Meier curves — real vs synthetic control
# ══════════════════════════════════════════════════════════════════════════
def fig1_km_curves(df_ctrl, synth_list):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Event (censor==1) survival
    kmf_real = KaplanMeierFitter()
    kmf_real.fit(df_ctrl['time'], event_observed=df_ctrl['censor'], label='Real control')

    ax = axes[0]
    kmf_real.plot_survival_function(ax=ax, ci_show=True, color='black', linewidth=2.5)
    for i, df_syn in enumerate(synth_list):
        kmf_syn = KaplanMeierFitter()
        kmf_syn.fit(df_syn['time'], event_observed=df_syn['censor'], label=f'Synthetic {i+1}')
        kmf_syn.plot_survival_function(ax=ax, ci_show=False, alpha=0.4, linewidth=1)
    ax.set_title("Kaplan-Meier: Event-Free Survival", fontweight="bold")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival Probability")
    ax.legend(fontsize=8, loc='lower left')

    # Median overlay
    ax = axes[1]
    kmf_real.plot_survival_function(ax=ax, ci_show=True, color='black', linewidth=2.5)
    # Aggregate: mean KM across synthetics
    times = np.linspace(0, df_ctrl['time'].max(), 200)
    surv_curves = []
    for df_syn in synth_list:
        kmf_s = KaplanMeierFitter()
        kmf_s.fit(df_syn['time'], event_observed=df_syn['censor'])
        surv_at_t = []
        for t in times:
            idx = kmf_s.survival_function_.index
            valid = idx[idx <= t]
            surv_at_t.append(kmf_s.survival_function_.loc[valid].iloc[-1].values[0] if len(valid) > 0 else 1.0)
        surv_curves.append(surv_at_t)
    surv_arr = np.array(surv_curves)
    ax.plot(times, surv_arr.mean(axis=0), 'r--', linewidth=2, label='Synthetic mean')
    ax.fill_between(times, surv_arr.mean(axis=0) - surv_arr.std(axis=0),
                     surv_arr.mean(axis=0) + surv_arr.std(axis=0), alpha=0.2, color='red')
    ax.set_title("KM: Real vs Synthetic Mean ± SD", fontweight="bold")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival Probability")
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig1_km_curves.pdf", bbox_inches='tight')
    fig.savefig(FIGDIR / "fig1_km_curves.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig1_km_curves")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Marginal distributions — real vs one synthetic
# ══════════════════════════════════════════════════════════════════════════
def fig2_marginals(df_ctrl, synth_list, fnames):
    feat_cols = [c for c in fnames if c not in ('time', 'censor')]
    n = len(feat_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    df_syn = synth_list[0].copy()
    df_real = df_ctrl.copy()
    df_real['source'] = 'Real'
    df_syn['source'] = 'Synthetic'
    combined = pd.concat([df_real, df_syn], ignore_index=True)

    cat_cols = {'strat2', 'sex', 'raceth', 'ivdrug', 'karnof'}

    for i, col in enumerate(feat_cols):
        ax = axes[i]
        if col in cat_cols:
            sns.countplot(data=combined, x=col, hue='source', ax=ax, stat='percent', palette=['steelblue', 'coral'])
            ax.set_title(col, fontweight='bold')
        else:
            sns.kdeplot(data=df_real, x=col, ax=ax, label='Real', color='steelblue', fill=True, alpha=0.3)
            sns.kdeplot(data=df_syn, x=col, ax=ax, label='Synthetic', color='coral', fill=True, alpha=0.3)
            ax.set_title(col, fontweight='bold')
            ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Marginal Distributions: Real vs Synthetic (Control)", fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(FIGDIR / "fig2_marginals.pdf", bbox_inches='tight')
    fig.savefig(FIGDIR / "fig2_marginals.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig2_marginals")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Survival time distributions — real vs synthetic
# ══════════════════════════════════════════════════════════════════════════
def fig3_time_distributions(df_ctrl, synth_list):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Time histogram
    ax = axes[0]
    ax.hist(df_ctrl['time'], bins=30, alpha=0.5, label='Real', color='steelblue', density=True)
    ax.hist(synth_list[0]['time'], bins=30, alpha=0.5, label='Synthetic', color='coral', density=True)
    ax.set_title("Survival Time Distribution", fontweight='bold')
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Density")
    ax.legend()

    # Censoring rate
    ax = axes[1]
    labels = ['Real'] + [f'Syn {i+1}' for i in range(len(synth_list))]
    rates = [df_ctrl['censor'].mean()] + [s['censor'].mean() for s in synth_list]
    colors = ['steelblue'] + ['coral'] * len(synth_list)
    bars = ax.bar(labels, rates, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=df_ctrl['censor'].mean(), color='black', linestyle='--', linewidth=1.5, label=f'Real rate={df_ctrl["censor"].mean():.3f}')
    ax.set_title("Event Rate (Censoring Indicator)", fontweight='bold')
    ax.set_ylabel("Event Rate")
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=45)

    # QQ plot of survival times
    ax = axes[2]
    real_sorted = np.sort(df_ctrl['time'].values)
    syn_sorted = np.sort(synth_list[0]['time'].values)
    n = min(len(real_sorted), len(syn_sorted))
    real_q = np.quantile(real_sorted, np.linspace(0, 1, n))
    syn_q = np.quantile(syn_sorted, np.linspace(0, 1, n))
    ax.scatter(real_q, syn_q, s=10, alpha=0.6, color='coral')
    lim = max(real_q.max(), syn_q.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1)
    ax.set_xlabel("Real Quantiles")
    ax.set_ylabel("Synthetic Quantiles")
    ax.set_title("Q-Q Plot: Survival Time", fontweight='bold')
    ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig3_time_distributions.pdf", bbox_inches='tight')
    fig.savefig(FIGDIR / "fig3_time_distributions.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig3_time_distributions")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: General metric scores (boxplots from metric_results)
# ══════════════════════════════════════════════════════════════════════════
def fig4_metric_boxplots():
    scores_file = REPRO / "metric_results" / "traincontrol_general_scores_df.csv"
    if not scores_file.exists():
        scores_file = DATASET / "metric_results" / "traincontrol_general_scores_df.csv"
    scores = pd.read_csv(scores_file)

    metrics = [('J-S distance', 'min'), ('KS test', 'max'),
               ('Survival curves distance', 'min'), ('Detection XGB', 'min'),
               ('NNDR', 'max'), ('K-map score', 'max')]

    # Only plot metrics that exist in the data
    metrics = [(m, d) for m, d in metrics if m in scores.columns]
    n = len(metrics)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(6 * ((n + 1) // 2), 10))
    axes = axes.flatten()

    for i, (metric, direction) in enumerate(metrics):
        ax = axes[i]
        sns.boxplot(data=scores, x='generator', y=metric, ax=ax,
                    palette='colorblind', linewidth=1.5)
        arrow = '↓' if direction == 'min' else '↑'
        ax.set_title(f"{metric} ({arrow} better)", fontweight='bold', fontsize=12)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=30, labelsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("General Metric Scores (traincontrol)", fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(FIGDIR / "fig4_metric_boxplots.pdf", bbox_inches='tight')
    fig.savefig(FIGDIR / "fig4_metric_boxplots.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig4_metric_boxplots")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Replicability scores (line plots from metric_results)
# ══════════════════════════════════════════════════════════════════════════
def fig5_replicability():
    rep_file = REPRO / "metric_results" / "traincontrol_replicability_scores_df.csv"
    if not rep_file.exists():
        rep_file = DATASET / "metric_results" / "traincontrol_replicability_scores_df.csv"
    rep = pd.read_csv(rep_file)

    metric_cols = [c for c in rep.columns if c not in ('Generator', 'Nb generated datasets')]
    n = len(metric_cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for i, col in enumerate(metric_cols):
        ax = axes[i]
        sns.lineplot(data=rep, x='Nb generated datasets', y=col,
                     hue='Generator', ax=ax, palette='colorblind', marker='o')
        ax.set_ylim(0, 1.05)
        ax.set_title(col, fontweight='bold')
        ax.legend(fontsize=8)

    plt.suptitle("Replicability Scores vs Number of Generated Datasets", fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGDIR / "fig5_replicability.pdf", bbox_inches='tight')
    fig.savefig(FIGDIR / "fig5_replicability.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig5_replicability")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Cox model CI comparison (forest plot from error_df)
# ══════════════════════════════════════════════════════════════════════════
def fig6_cox_forest():
    err_file = REPRO / "metric_results" / "traincontrol_error_df.csv"
    if not err_file.exists():
        err_file = DATASET / "metric_results" / "traincontrol_error_df.csv"
    err = pd.read_csv(err_file)

    fig, ax = plt.subplots(figsize=(8, max(6, len(err) * 0.4)))
    for i, row in err.iterrows():
        ax.errorbar(x=row['midpoints'], y=i, xerr=row['errors'],
                     fmt='o', capsize=4, color=row['colors'], markersize=6, linewidth=1.5)
    ax.axvline(x=0, color='grey', linestyle=':', linewidth=1)
    # Mark initial estimate
    init_row = err[err['label'] == 'Init']
    if len(init_row) > 0:
        ax.axvline(x=init_row.iloc[0]['midpoints'], color='red', linestyle='--',
                    linewidth=1, alpha=0.7, label='Real estimate')
    ax.set_yticks(range(len(err)))
    ax.set_yticklabels(err['label'], fontsize=9)
    ax.set_xlabel("Treatment Effect (Cox HR log-coefficient)", fontweight='bold')
    ax.set_title("Cox Model: Real vs Synthetic Treatment Effect CIs", fontweight='bold')
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig6_cox_forest.pdf", bbox_inches='tight')
    fig.savefig(FIGDIR / "fig6_cox_forest.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig6_cox_forest")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 7: Pairwise correlation heatmaps — real vs synthetic
# ══════════════════════════════════════════════════════════════════════════
def fig7_correlation(df_ctrl, synth_list, fnames):
    feat_cols = [c for c in fnames if c not in ('time', 'censor')]
    cols = ['time'] + feat_cols

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    corr_real = df_ctrl[cols].corr()
    corr_syn = synth_list[0][cols].corr()
    corr_diff = corr_real - corr_syn

    sns.heatmap(corr_real, ax=axes[0], cmap='RdBu_r', vmin=-1, vmax=1,
                annot=True, fmt='.2f', square=True, linewidths=0.5, annot_kws={'size': 7})
    axes[0].set_title("Real Data Correlation", fontweight='bold')

    sns.heatmap(corr_syn, ax=axes[1], cmap='RdBu_r', vmin=-1, vmax=1,
                annot=True, fmt='.2f', square=True, linewidths=0.5, annot_kws={'size': 7})
    axes[1].set_title("Synthetic Data Correlation", fontweight='bold')

    sns.heatmap(corr_diff, ax=axes[2], cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                annot=True, fmt='.2f', square=True, linewidths=0.5, annot_kws={'size': 7})
    axes[2].set_title("Difference (Real - Synthetic)", fontweight='bold')

    plt.suptitle("Pairwise Correlation Comparison", fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGDIR / "fig7_correlation.pdf", bbox_inches='tight')
    fig.savefig(FIGDIR / "fig7_correlation.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig7_correlation")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 8: KM curves — full synthetic trial (treated real + synthetic control)
# ══════════════════════════════════════════════════════════════════════════
def fig8_full_trial_km(df_full, df_treat, synth_list):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Real trial
    ax = axes[0]
    for treat_val, label, color in [(0, 'Control', 'steelblue'), (1, 'Treated', 'coral')]:
        subset = df_full[df_full['treatment'] == treat_val]
        kmf = KaplanMeierFitter()
        kmf.fit(subset['time'], event_observed=subset['censor'], label=label)
        kmf.plot_survival_function(ax=ax, color=color, linewidth=2)
    ax.set_title("Real Trial: Control vs Treated", fontweight='bold')
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival Probability")

    # Synthetic trial (real treated + synthetic control)
    ax = axes[1]
    kmf_treat = KaplanMeierFitter()
    kmf_treat.fit(df_treat['time'], event_observed=df_treat['censor'], label='Treated (real)')
    kmf_treat.plot_survival_function(ax=ax, color='coral', linewidth=2)

    for i, df_syn in enumerate(synth_list[:3]):
        kmf_syn = KaplanMeierFitter()
        kmf_syn.fit(df_syn['time'], event_observed=df_syn['censor'], label=f'Control (syn {i+1})')
        kmf_syn.plot_survival_function(ax=ax, alpha=0.6, linewidth=1.5)
    ax.set_title("Synthetic Trial: Real Treated + Synthetic Control", fontweight='bold')
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival Probability")

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig8_full_trial_km.pdf", bbox_inches='tight')
    fig.savefig(FIGDIR / "fig8_full_trial_km.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved fig8_full_trial_km")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Output directory: {FIGDIR}")
    print("Loading original data...")
    df_ctrl, df_treat, df_full, fnames, feat_types_dict, df_ctrl_enc, miss_ctrl, true_miss_ctrl = load_original_data()
    print(f"  Control: {df_ctrl.shape}, Treated: {df_treat.shape}")

    print("Generating synthetic data (HI-VAE_weibull, 10 datasets)...")
    synth_list = generate_synthetic(df_ctrl_enc, miss_ctrl, true_miss_ctrl, feat_types_dict, fnames)
    print(f"  Generated {len(synth_list)} synthetic datasets of shape {synth_list[0].shape}")

    print("\nGenerating figures...")
    fig1_km_curves(df_ctrl, synth_list)
    fig2_marginals(df_ctrl, synth_list, fnames)
    fig3_time_distributions(df_ctrl, synth_list)
    fig4_metric_boxplots()
    fig5_replicability()
    fig6_cox_forest()
    fig7_correlation(df_ctrl, synth_list, fnames)
    fig8_full_trial_km(df_full, df_treat, synth_list)

    print(f"\nAll figures saved to: {FIGDIR}")
    print("Files:")
    for f in sorted(FIGDIR.iterdir()):
        print(f"  {f.name}")
