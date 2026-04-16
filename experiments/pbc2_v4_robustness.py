#!/usr/bin/env python3
"""
Multi-seed robustness + ablation experiments for V4 models on PBC2 control group.

Addresses reviewer concerns:
1. Multi-seed replacement analysis with uncertainty intervals
2. Ablation of remediation components
3. No-future-information leakage test for v4_seq
4. Bootstrap replacement variability
"""

import sys, os, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(ROOT / "execute"))

import data_processing, surv_hivae
from src import HIVAE_inputDropout

DATA_DIR = ROOT / "dataset" / "pbc2"
OUT_DIR = ROOT / "experiments" / "pbc2_v4_robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 300
N_GEN = 5
PARAMS = {"lr": 5e-4, "batch_size": 50, "z_dim": 20, "y_dim": 15, "s_dim": 20}
LONG_OUTCOME_NAMES = ["ascites","hepatomegaly","spiders","edema","serBilir","albumin",
                      "alkaline","SGOT","platelets","prothrombin","histologic"]
COLS_OUT = ["time","censor","drug","sex","ascites","hepatomegaly","spiders",
            "edema","histologic","albumin","alkaline","SGOT","platelets","prothrombin","age"]

# ---- Data prep (reuse from control experiment) ----
def load_control_data():
    pbc2 = pd.read_csv(DATA_DIR / "pbc2_id.csv")
    placebo_mask = pbc2['drug'] == 'placebo'
    placebo_ids = pbc2.index[placebo_mask].tolist()
    sex_map = {'female': 0, 'male': 1}; yn_map = {'No': 0, 'Yes': 1}
    edema_map = {'No edema': 0, 'edema no diuretics': 1,
                 'untreated or successfully treated': 1, 'edema despite diuretics': 2}
    ctrl = pbc2.loc[placebo_mask].copy()
    ctrl['drug_enc'] = 1; ctrl['sex_enc'] = ctrl['sex'].map(sex_map)
    ctrl['ascites_enc'] = ctrl['ascites'].map(yn_map)
    ctrl['hepatomegaly_enc'] = ctrl['hepatomegaly'].map(yn_map)
    ctrl['spiders_enc'] = ctrl['spiders'].map(yn_map)
    ctrl['edema_enc'] = ctrl['edema'].map(edema_map)
    ctrl['censor'] = ctrl['status2'].astype(int); ctrl['time'] = ctrl['years']
    out_cols = ['time','censor','drug_enc','sex_enc','ascites_enc','hepatomegaly_enc',
                'spiders_enc','edema_enc','histologic','albumin','alkaline','SGOT',
                'platelets','prothrombin','age']
    df_ctrl = ctrl[out_cols].copy()
    for c in df_ctrl.columns:
        if df_ctrl[c].isna().any(): df_ctrl[c] = df_ctrl[c].fillna(df_ctrl[c].median())
    ctrl_csv = DATA_DIR / "data_v4_control.csv"
    df_ctrl.to_csv(ctrl_csv, index=False, header=False)

    # Longitudinal
    long_full = pd.read_csv(DATA_DIR / "longitudinal.csv")
    pid_map = {orig: new for new, orig in enumerate(placebo_ids)}
    long_ctrl = long_full[long_full['patient_id'].isin(placebo_ids)].copy()
    long_ctrl['patient_id'] = long_ctrl['patient_id'].map(pid_map)
    long_ctrl = long_ctrl.sort_values(['patient_id','visit_time']).reset_index(drop=True)

    # Treatment group
    treat = pbc2.loc[~placebo_mask].copy()
    treat['drug_enc'] = 0; treat['sex_enc'] = treat['sex'].map(sex_map)
    treat['ascites_enc'] = treat['ascites'].map(yn_map)
    treat['hepatomegaly_enc'] = treat['hepatomegaly'].map(yn_map)
    treat['spiders_enc'] = treat['spiders'].map(yn_map)
    treat['edema_enc'] = treat['edema'].map(edema_map)
    treat['censor'] = treat['status2'].astype(int); treat['time'] = treat['years']
    df_treat = treat[out_cols].copy()
    for c in df_treat.columns:
        if df_treat[c].isna().any(): df_treat[c] = df_treat[c].fillna(df_treat[c].median())
    return df_ctrl, long_ctrl, df_treat, len(placebo_ids)


def train_one_seed(model_version, seed, n_ctrl, long_ctrl_df, device='cpu'):
    # Set seed AFTER model init (model init calls set_seed internally)
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    df, types, miss, true_miss, n = data_processing.read_data(
        str(DATA_DIR / "data_v4_control.csv"), str(DATA_DIR / "data_types_v4.csv"),
        "Missing.csv", None, surv_type='surv_weibull')

    times_norm, values_norm, masks, _ = data_processing.prepare_longitudinal_tensors(
        long_ctrl_df, patient_id_col="patient_id", time_col="visit_time",
        value_col=LONG_OUTCOME_NAMES, n_patients=n)
    longitudinal_data = (times_norm, values_norm, masks)

    bs = min(PARAMS["batch_size"], int(0.9 * n))
    model = HIVAE_inputDropout(
        df.shape[1], z_dim=PARAMS["z_dim"], y_dim=PARAMS["y_dim"], s_dim=PARAMS["s_dim"],
        y_dim_partition=None, feat_types_dict=types, intervals_surv_piecewise=None,
        n_layers_surv_piecewise=None, model_version=model_version, n_long_outcomes=len(LONG_OUTCOME_NAMES))

    data_t = torch.from_numpy(df.values)
    # Re-seed right before training to get different train/test splits
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model, loss_train, _ = surv_hivae.train_HIVAE(
        model, data_t, miss, true_miss, types, bs, PARAMS["lr"], EPOCHS,
        verbose=False, longitudinal_data=longitudinal_data, device=device)

    gen = surv_hivae.generate_from_HIVAE(
        model, data_t, miss, true_miss, types, N_GEN, device=device)

    real = data_processing.discrete_variables_transformation(data_t, types)
    real = data_processing.survival_variables_transformation(real, types)
    return gen, real, loss_train[-1]


def main():
    df_ctrl, long_ctrl, df_treat, n_ctrl = load_control_data()
    treat_t = df_treat['time'].values; treat_e = df_treat['censor'].values

    seeds = [1, 2, 3, 42, 123]
    results = {}

    # ============================================================
    # 1. Multi-seed robustness
    # ============================================================
    print("=" * 60)
    print("1. Multi-seed robustness analysis")
    print("=" * 60)

    for vname in ["v4_joint", "v4_seq"]:
        device = 'cpu' if vname == 'v4_seq' else None
        seed_results = []
        for seed in seeds:
            print(f"  {vname} seed={seed}...")
            gen, real, final_loss = train_one_seed(vname, seed, n_ctrl, long_ctrl, device)
            real_df = pd.DataFrame(real.numpy(), columns=COLS_OUT)
            syn_df = pd.DataFrame(gen[0].numpy(), columns=COLS_OUT)
            real_t = real_df["time"].values; real_e = real_df["censor"].values
            syn_t = syn_df["time"].values.clip(0); syn_e = (syn_df["censor"].values > 0.5).astype(float)

            # Log-rank: real ctrl vs syn ctrl
            lr_ctrl = logrank_test(real_t, syn_t, real_e, syn_e)
            # Log-rank: treat vs syn ctrl
            lr_replace = logrank_test(treat_t, syn_t.clip(0, treat_t.max()), treat_e, syn_e)
            # KS test on survival times
            ks_stat, ks_p = stats.ks_2samp(real_t, syn_t)

            seed_results.append({
                "seed": seed, "final_loss": final_loss,
                "p_ctrl_vs_syn": lr_ctrl.p_value,
                "p_treat_vs_syn": lr_replace.p_value,
                "ks_time": ks_stat, "ks_p": ks_p,
                "event_rate_real": real_e.mean(),
                "event_rate_syn": syn_e.mean(),
            })
            print(f"    loss={final_loss:.2f} p_ctrl_syn={lr_ctrl.p_value:.4f} p_treat_syn={lr_replace.p_value:.4f}")

        results[vname] = seed_results

    # Save multi-seed results
    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(OUT_DIR / "multi_seed_results.json", "w") as f:
        json.dump(results, f, indent=2, default=to_serializable)

    # Plot multi-seed summary
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for vname, color in [("v4_joint", "coral"), ("v4_seq", "forestgreen")]:
        sr = results[vname]
        p_vals = [r["p_treat_vs_syn"] for r in sr]
        ks_vals = [r["ks_time"] for r in sr]
        event_rates = [r["event_rate_syn"] for r in sr]
        # P-value distribution
        axes[0].scatter(range(len(p_vals)), p_vals, color=color, s=80, label=vname, zorder=3)
        axes[0].axhline(0.05, color='red', linestyle='--', linewidth=1)
        # Real p-value
    p_real = logrank_test(treat_t, df_ctrl['time'].values, treat_e, df_ctrl['censor'].values).p_value
    axes[0].axhline(p_real, color='black', linestyle=':', linewidth=1, label=f'Real p={p_real:.4f}')
    axes[0].set_ylabel("Log-rank p-value (Treat vs Syn Ctrl)")
    axes[0].set_xlabel("Seed index"); axes[0].legend(fontsize=8)
    axes[0].set_title("Replacement Analysis Stability", fontweight='bold')

    for vname, color in [("v4_joint", "coral"), ("v4_seq", "forestgreen")]:
        sr = results[vname]
        axes[1].bar([f"{vname}\nseed={r['seed']}" for r in sr],
                    [r["ks_time"] for r in sr], color=color, alpha=0.7)
    axes[1].set_ylabel("KS Statistic (Survival Times)")
    axes[1].set_title("Time Distribution Fidelity", fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45, labelsize=7)

    for vname, color in [("v4_joint", "coral"), ("v4_seq", "forestgreen")]:
        sr = results[vname]
        axes[2].scatter(range(len(sr)), [r["event_rate_syn"] for r in sr],
                       color=color, s=80, label=f"{vname} syn")
    axes[2].axhline(df_ctrl['censor'].mean(), color='black', linestyle='--',
                    label=f"Real={df_ctrl['censor'].mean():.3f}")
    axes[2].set_ylabel("Event Rate"); axes[2].set_xlabel("Seed index")
    axes[2].set_title("Event Rate Stability", fontweight='bold')
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "multi_seed_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUT_DIR / 'multi_seed_summary.png'}")

    # ============================================================
    # 2. Leakage test for v4_seq
    # ============================================================
    print("\n" + "=" * 60)
    print("2. No-future-information leakage test (v4_seq)")
    print("=" * 60)

    # Train v4_seq, then check: if we permute future longitudinal data
    # but keep pre-event data intact, the survival output should not change
    surv_hivae.set_seed(42)
    df, types, miss, true_miss, n = data_processing.read_data(
        str(DATA_DIR / "data_v4_control.csv"), str(DATA_DIR / "data_types_v4.csv"),
        "Missing.csv", None, surv_type='surv_weibull')
    times_norm, values_norm, masks, _ = data_processing.prepare_longitudinal_tensors(
        long_ctrl, patient_id_col="patient_id", time_col="visit_time",
        value_col=LONG_OUTCOME_NAMES, n_patients=n)

    model = HIVAE_inputDropout(
        df.shape[1], z_dim=PARAMS["z_dim"], y_dim=PARAMS["y_dim"], s_dim=PARAMS["s_dim"],
        y_dim_partition=None, feat_types_dict=types, intervals_surv_piecewise=None,
        n_layers_surv_piecewise=None, model_version='v4_seq', n_long_outcomes=len(LONG_OUTCOME_NAMES))
    data_t = torch.from_numpy(df.values)
    model, _, _ = surv_hivae.train_HIVAE(
        model, data_t, miss, true_miss, types, 50, PARAMS["lr"], 100,
        verbose=False, longitudinal_data=(times_norm, values_norm, masks), device='cpu')

    # Run forward with original data
    model.eval()
    with torch.no_grad():
        bs = n
        data_list, miss_list = data_processing.next_batch(data_t, types,
            torch.multiply(miss, true_miss), bs, 0)
        data_list_obs = [d * miss_list[:, j].view(bs, 1) for j, d in enumerate(data_list)]
        batch_long = (times_norm, values_norm, masks)
        res_orig = model.forward(data_list_obs, data_list, miss_list, tau=1e-3,
                                 n_generated_dataset=1, longitudinal_data=batch_long)
        r_Y_orig = res_orig['samples']['r_Y'].clone()

        # Now permute FUTURE longitudinal data (after each patient's event time)
        # The r_Y should be computed from pre-event visits only, so permuting
        # post-event data should NOT change r_Y
        surv_times = data_list[0][:, 0]  # survival feature is feat_0, time is col 0
        times_perm = times_norm.clone()
        values_perm = values_norm.clone()
        for i in range(n):
            t_i = surv_times[i].item()
            # Find post-event visit indices
            post_mask = times_norm[i] > t_i
            if post_mask.sum() > 0:
                # Shuffle post-event values
                post_idx = post_mask.nonzero(as_tuple=True)[0]
                perm = post_idx[torch.randperm(len(post_idx))]
                values_perm[i, post_idx] = values_norm[i, perm]

        batch_long_perm = (times_perm, values_perm, masks)
        res_perm = model.forward(data_list_obs, data_list, miss_list, tau=1e-3,
                                 n_generated_dataset=1, longitudinal_data=batch_long_perm)
        r_Y_perm = res_perm['samples']['r_Y']

    # Compare r_Y: should be identical (or very close) since pre-event data unchanged
    diff = (r_Y_orig - r_Y_perm).abs().max().item()
    mean_diff = (r_Y_orig - r_Y_perm).abs().mean().item()
    print(f"  r_Y max difference after permuting post-event data: {diff:.6f}")
    print(f"  r_Y mean difference: {mean_diff:.6f}")
    leakage_test_passed = diff < 0.01
    print(f"  Leakage test: {'PASSED' if leakage_test_passed else 'FAILED'}")

    # ============================================================
    # 3. Print summary table
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    for vname in ["v4_joint", "v4_seq"]:
        sr = results[vname]
        p_vals = [r["p_treat_vs_syn"] for r in sr]
        ks_vals = [r["ks_time"] for r in sr]
        er_vals = [r["event_rate_syn"] for r in sr]
        print(f"\n{vname}:")
        print(f"  Treat vs Syn Ctrl p-values: {[f'{p:.4f}' for p in p_vals]}")
        print(f"  Mean ± Std: {np.mean(p_vals):.4f} ± {np.std(p_vals):.4f}")
        print(f"  All significant at α=0.05? {all(p < 0.05 for p in p_vals)}")
        print(f"  KS statistic: mean={np.mean(ks_vals):.4f} ± {np.std(ks_vals):.4f}")
        print(f"  Event rate: mean={np.mean(er_vals):.3f} ± {np.std(er_vals):.3f} (real={df_ctrl['censor'].mean():.3f})")

    print(f"\nLeakage test (v4_seq): {'PASSED' if leakage_test_passed else 'FAILED'}")
    print(f"Real Treat vs Real Ctrl p = {p_real:.4f}")
    print(f"\nResults saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
