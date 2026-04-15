#!/usr/bin/env python3
"""
Unified CLI entry point for the multi-version HI-VAE framework.

Model versions
--------------
v0             Static HI-VAE for baseline / pre-randomization variables only.
v1             HI-VAE for baseline variables + one continuous post-randomization endpoint.
v2a            Lightweight longitudinal HI-VAE (shared patient latent + time-conditioned
               Gaussian decoder, NO autoregressive recurrence).
v3_weibull     Survival HI-VAE with Weibull event/censoring model (existing).
v3_piecewise   Survival HI-VAE with piecewise-constant hazard model (existing).

Example commands
----------------
# V0 – baseline only
python run_unified.py --model_version v0 \
    --data_file ../dataset/my_trial/data.csv \
    --types_file ../dataset/my_trial/data_types_baseline.csv \
    --miss_file ../dataset/my_trial/Missing.csv \
    --epochs 1000 --n_generated_dataset 100 --output_dir ./output_v0

# V1 – baseline + one continuous endpoint
python run_unified.py --model_version v1 \
    --data_file ../dataset/my_trial/data.csv \
    --types_file ../dataset/my_trial/data_types_v1.csv \
    --miss_file ../dataset/my_trial/Missing.csv \
    --endpoint_column week36_score \
    --epochs 1000 --n_generated_dataset 100 --output_dir ./output_v1

# V2A – baseline + longitudinal repeated measures
python run_unified.py --model_version v2a \
    --data_file ../dataset/my_trial/data_baseline.csv \
    --types_file ../dataset/my_trial/data_types_baseline.csv \
    --miss_file ../dataset/my_trial/Missing.csv \
    --longitudinal_file ../dataset/my_trial/longitudinal.csv \
    --patient_id_col patient_id --time_col visit_time --longitudinal_value_col value \
    --epochs 1000 --n_generated_dataset 10 --output_dir ./output_v2a

# V3 – survival (Weibull)
python run_unified.py --model_version v3_weibull \
    --data_file ../dataset/my_trial/data.csv \
    --types_file ../dataset/my_trial/data_types.csv \
    --miss_file ../dataset/my_trial/Missing.csv \
    --epochs 1000 --n_generated_dataset 200 --output_dir ./output_v3w

# V3 – survival (piecewise)
python run_unified.py --model_version v3_piecewise \
    --data_file ../dataset/my_trial/data.csv \
    --types_file ../dataset/my_trial/data_types.csv \
    --miss_file ../dataset/my_trial/Missing.csv \
    --epochs 1000 --n_generated_dataset 200 --output_dir ./output_v3pw
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add parent directories to path
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent / "utils"))
sys.path.insert(0, str(_SCRIPT_DIR))

import data_processing
import surv_hivae


def parse_args():
    p = argparse.ArgumentParser(
        description="Unified multi-version HI-VAE for clinical trial data generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- required ----
    p.add_argument("--model_version", type=str, required=True,
                    choices=["v0", "v1", "v2a", "v3_weibull", "v3_piecewise",
                             "v4_joint", "v4_seq"],
                    help="Model version to train and generate from.")
    p.add_argument("--data_file", type=str, required=True,
                    help="Path to the baseline / full data CSV.")
    p.add_argument("--types_file", type=str, required=True,
                    help="Path to the data_types CSV describing each feature.")
    p.add_argument("--miss_file", type=str, default="Missing.csv",
                    help="Path to the missing-data index file.")

    # ---- optional common ----
    p.add_argument("--true_miss_file", type=str, default=None,
                    help="Path to the true-missing mask file (for NaN data).")
    p.add_argument("--output_dir", type=str, default="./output",
                    help="Directory to save generated data.")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--n_generated_dataset", type=int, default=100,
                    help="Number of synthetic datasets to generate.")
    p.add_argument("--n_generated_sample", type=int, default=None,
                    help="Number of samples per generated dataset (default = N).")
    p.add_argument("--gen_from_prior", action="store_true", default=False)
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=1)

    # ---- hyperparameters ----
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--z_dim", type=int, default=20)
    p.add_argument("--y_dim", type=int, default=15)
    p.add_argument("--s_dim", type=int, default=20)
    p.add_argument("--n_layers_surv_piecewise", type=int, default=1)
    p.add_argument("--n_intervals", type=int, default=10)

    # ---- V1 ----
    p.add_argument("--endpoint_column", type=str, default=None,
                    help="Name of the continuous endpoint column (V1 only, for documentation).")

    # ---- V2A ----
    p.add_argument("--longitudinal_file", type=str, default=None,
                    help="Long-format CSV with repeated measures (V2A).")
    p.add_argument("--patient_id_col", type=str, default="patient_id",
                    help="Patient ID column in longitudinal CSV (0-based row index).")
    p.add_argument("--time_col", type=str, default="visit_time",
                    help="Visit-time column in longitudinal CSV.")
    p.add_argument("--longitudinal_value_col", type=str, nargs="+", default=None,
                    help="Outcome value column(s) in longitudinal CSV. Multiple columns for multi-outcome V2A.")
    p.add_argument("--longitudinal_mask_col", type=str, default=None,
                    help="Optional per-visit mask column in longitudinal CSV.")
    p.add_argument("--time_grid", type=float, nargs="+", default=None,
                    help="Normalised time grid for V2A generation.")

    # ---- misc ----
    p.add_argument("--use_controls_only", action="store_true", default=False)

    # ---- V4 ----
    p.add_argument("--surv_type", type=str, default="weibull",
                    choices=["weibull", "piecewise"],
                    help="Survival family for V4 models (default: weibull).")

    return p.parse_args()


def main():
    args = parse_args()
    surv_hivae.set_seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load baseline / static data
    # ------------------------------------------------------------------
    surv_type = None
    if args.model_version == "v3_weibull":
        surv_type = "surv_weibull"
    elif args.model_version == "v3_piecewise":
        surv_type = "surv_piecewise"
    elif args.model_version in ("v4_joint", "v4_seq"):
        surv_type = "surv_weibull" if args.surv_type == "weibull" else "surv_piecewise"

    df, types_dict, miss_mask, true_miss_mask, n_samples = data_processing.read_data(
        args.data_file, args.types_file, args.miss_file, args.true_miss_file,
        surv_type=surv_type,
    )

    print(f"[run_unified] model_version = {args.model_version}")
    print(f"[run_unified] n_samples = {n_samples},  n_features (types) = {len(types_dict)}")
    print(f"[run_unified] data shape = {df.shape}")

    # ------------------------------------------------------------------
    # 2. Version-specific data validation
    # ------------------------------------------------------------------
    if args.model_version in ("v0", "v1", "v2a"):
        # Warn if survival features are present
        surv_feats = [d for d in types_dict if d["type"].startswith("surv")]
        if surv_feats:
            print(f"[WARNING] {len(surv_feats)} survival feature(s) found in types file "
                  f"but model_version={args.model_version}. They will be modelled as-is.")
    if args.model_version == "v1" and args.endpoint_column:
        print(f"[run_unified] V1 endpoint column: {args.endpoint_column}")
    if args.model_version in ("v2a", "v4_joint", "v4_seq") and args.longitudinal_file is None:
        raise ValueError(f"{args.model_version} requires --longitudinal_file")

    # ------------------------------------------------------------------
    # 3. V2A: load longitudinal data
    # ------------------------------------------------------------------
    longitudinal_data = None
    long_norm_params = None
    n_long_outcomes = 1
    if args.model_version in ("v2a", "v4_joint", "v4_seq") and args.longitudinal_file is not None:
        long_df = pd.read_csv(args.longitudinal_file)
        long_value_cols = args.longitudinal_value_col if args.longitudinal_value_col is not None else ["value"]
        n_long_outcomes = len(long_value_cols)
        # Pass single string for backward compat, list for multi-outcome
        value_col_arg = long_value_cols[0] if n_long_outcomes == 1 else long_value_cols
        print(f"[run_unified] Longitudinal file: {args.longitudinal_file}  "
              f"({len(long_df)} records, {long_df[args.patient_id_col].nunique()} patients, "
              f"{n_long_outcomes} outcome(s): {long_value_cols})")
        times_norm, values_norm, masks, long_norm_params = \
            data_processing.prepare_longitudinal_tensors(
                long_df,
                patient_id_col=args.patient_id_col,
                time_col=args.time_col,
                value_col=value_col_arg,
                mask_col=args.longitudinal_mask_col,
                n_patients=n_samples,
            )
        longitudinal_data = (times_norm, values_norm, masks)
        print(f"[run_unified] Longitudinal padded shape: {times_norm.shape}  "
              f"(max_visits={long_norm_params['max_visits']})")

    # ------------------------------------------------------------------
    # 4. Build hyperparameter dict
    # ------------------------------------------------------------------
    params = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "z_dim": args.z_dim,
        "y_dim": args.y_dim,
        "s_dim": args.s_dim,
    }
    if args.model_version == "v3_piecewise" or \
       (args.model_version in ("v4_joint", "v4_seq") and args.surv_type == "piecewise"):
        params["n_layers_surv_piecewise"] = args.n_layers_surv_piecewise
        params["n_intervals"] = args.n_intervals

    # ------------------------------------------------------------------
    # 5. Run
    # ------------------------------------------------------------------
    result = surv_hivae.run(
        df, miss_mask, true_miss_mask, types_dict,
        n_generated_dataset=args.n_generated_dataset,
        n_generated_sample=args.n_generated_sample,
        params=params,
        epochs=args.epochs,
        verbose=args.verbose,
        gen_from_prior=args.gen_from_prior,
        model_version=args.model_version,
        longitudinal_data=longitudinal_data,
        time_grid=args.time_grid,
        n_long_outcomes=n_long_outcomes,
    )

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_version in ("v2a", "v4_joint", "v4_seq"):
        # result is (baseline_tensor, longitudinal_dict)
        baseline_result, long_result = result

        # Save baseline
        _save_baseline(baseline_result, types_dict, df.columns.tolist(),
                       args.output_dir, args.n_generated_dataset)

        # Save longitudinal trajectories
        tg = long_result["time_grid"].numpy()
        mu_np = long_result["mu"].numpy()       # (N, n_times, D) or (N, n_times)
        var_np = long_result["var"].numpy()
        traj_np = long_result["trajectories"].numpy()  # (n_gen, N, n_times, D)

        # Denormalise if norm_params available
        if long_norm_params is not None:
            vmean = long_norm_params["value_mean"]
            vstd = long_norm_params["value_std"]
            tmin = long_norm_params["time_min"]
            tmax = long_norm_params["time_max"]
            # vmean/vstd can be scalar or list depending on multi-outcome
            if isinstance(vmean, list):
                vmean = np.array(vmean)
                vstd = np.array(vstd)
            mu_np = mu_np * vstd + vmean
            traj_np = traj_np * vstd + vmean
            tg_original = tg * (tmax - tmin) + tmin
        else:
            tg_original = tg

        np.save(os.path.join(args.output_dir, "longitudinal_mu.npy"), mu_np)
        np.save(os.path.join(args.output_dir, "longitudinal_var.npy"), var_np)
        np.save(os.path.join(args.output_dir, "longitudinal_trajectories.npy"), traj_np)
        np.save(os.path.join(args.output_dir, "time_grid.npy"), tg_original)
        print(f"[run_unified] Saved longitudinal outputs to {args.output_dir}")
    else:
        _save_baseline(result, types_dict, df.columns.tolist(),
                       args.output_dir, args.n_generated_dataset)

    print("[run_unified] Done.")


def _save_baseline(tensor_result, types_dict, columns, output_dir, n_generated_dataset):
    """Save generated baseline/tabular data to CSV files."""
    if isinstance(tensor_result, list):
        # List of tensors (one per n_generated_sample entry)
        for k, t in enumerate(tensor_result):
            for j in range(t.shape[0]):
                df_gen = pd.DataFrame(t[j].numpy(), columns=columns)
                df_gen.to_csv(os.path.join(output_dir, f"gen_set{k}_sample{j}.csv"), index=False)
    else:
        for j in range(min(tensor_result.shape[0], n_generated_dataset)):
            df_gen = pd.DataFrame(tensor_result[j].numpy(), columns=columns)
            df_gen.to_csv(os.path.join(output_dir, f"gen_sample{j}.csv"), index=False)
    print(f"[run_unified] Saved baseline synthetic data to {output_dir}")


if __name__ == "__main__":
    main()
