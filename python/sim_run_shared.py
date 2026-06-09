"""
Entry point for SupMultiviewShared simulation sweep.

Data must be pre-generated in R (generate_factor_data_shared / generate_factor_data_shared_testing)
and saved as:
  sim/data_ortho/k.<K>/train/<cond_dir>/sim_data_shared_rep<i>.rds
  sim/data_ortho/k.<K>/test/<cond_dir>/sim_data_shared_rep<i>.rds

where <cond_dir> follows the pattern n{n}p{p}_snr{snr_x}.{snr_y}_sparse{sparsity}.
"""

import argparse
import os
import itertools
from pathlib import Path
import time

import rpy2.robjects as robjects
from rpy2.robjects import r
from rpy2 import rinterface

import torch
import pyro
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
# Set to "mps" for Apple Silicon GPU, "cuda" for NVIDIA GPU, "cpu" for CPU.
# Falls back to CPU automatically if the requested device is unavailable.
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

from data_utils import load_rds_rep
from run_methods import (
    process_data_shared,
    load_model_shared,
    train_model_shared,
    train_model_locally_shared,
    evaluate_fitted_model_shared,
)

# ---------------------------------------------------------------------------
# Data paths  (edit K to match the simulation design)
# ---------------------------------------------------------------------------
K = 5
train_sim_data_path = f"~/multiomic_integration/sim/data/multi_view_shared/k.{K}/train/"
test_sim_data_path  = f"~/multiomic_integration/sim/data/multi_view_shared/k.{K}/test/"

# ---------------------------------------------------------------------------
# Simulation grid
# ---------------------------------------------------------------------------
n_array              = (100, 500, 1000, 5000)
p_array              = (50, 100, 1000, 5000)
snr_x_array          = [2]
snr_y_array          = [2]
reps                 = 10
loading_sparsity     = [0.25]
k_deltas             = [0, 10]       # over-specification of k at inference time

sim_grid = list(itertools.product(
    n_array, p_array, snr_x_array, snr_y_array, loading_sparsity, k_deltas
))

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
N_POSTERIOR_SAMPLES  = 200
MINIBATCH_FRACTION   = 0.1   # fraction of training set size; converted per run
MIN_EPOCHS           = 100
# MIN_EPOCHS_HIGH     = 500   # for n=50
MIN_EPOCHS_LOCAL    = 100
NUM_EPOCHS          = 1000
NUM_PARTICLES       = 1
CONVERGENCE_CRITERION = "elbo_snr"
WINDOW = 50
SNR_THRESHOLD = 1

# AdagradRMSProp defaults (match single-view model)
ETA   = 0.25
DELTA = 1e-16
T     = 0.75

TRAINING_SPLIT = False
TRAINING_SIZE  = 0.8
RANDOM_SEED    = 123

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
model_out_path  = os.path.expanduser(
    "~/multiomic_integration/sim/results/integration_shared/models")
metric_out_path = os.path.expanduser(
    "~/multiomic_integration/sim/results/integration_shared/metrics")
Path(model_out_path).mkdir(parents=True, exist_ok=True)
Path(metric_out_path).mkdir(parents=True, exist_ok=True)

file_dir_base           = "n{}p{}_snr{}.{}_sparse{}"
file_name_base          = "sim_data_shared_rep{}"
model_out_filename_base = "shared_n{}p{}_snr{}.{}_sparse{}_deltak{}_rep{}"
metric_out_filename_base = model_out_filename_base

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
metric_list = []

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-id", type=int, default=None,
        help="SGE array task ID (1-indexed). Selects one grid element; omit to run all.",
    )
    args = parser.parse_args()

    grid_to_run = (
        [sim_grid[args.task_id - 1]] if args.task_id is not None else sim_grid
    )

    # Build filename dicts once per data root
    def _build_filename_dict(path):
        expanded = os.path.expanduser(path)
        dirs = [os.path.join(path, d)
                for d in os.listdir(expanded)
                if os.path.isdir(os.path.join(expanded, d))]
        return {os.path.basename(d): os.listdir(os.path.expanduser(d)) for d in dirs}

    filename_dict      = _build_filename_dict(train_sim_data_path)
    test_filename_dict = _build_filename_dict(test_sim_data_path)

    for n, p, snr_x, snr_y, sparsity, k_delta in grid_to_run:
        pyro.clear_param_store()

        cond_str = file_dir_base.format(n, p, snr_x, snr_y, sparsity)
        print(f"\n=== Condition: {cond_str}  k_delta={k_delta} ===")

        # Skip if all reps already have metrics
        check_metric = os.path.join(
            metric_out_path,
            metric_out_filename_base.format(n, p, snr_x, snr_y, sparsity, k_delta, reps))
        if os.path.isfile(check_metric + ".csv"):
            print("All reps done, skipping.")
            continue

        for rep in range(1, reps + 1):
            model_out_filename = os.path.join(
                model_out_path,
                model_out_filename_base.format(n, p, snr_x, snr_y, sparsity, k_delta, rep))
            metric_out_filename = os.path.join(
                metric_out_path,
                metric_out_filename_base.format(n, p, snr_x, snr_y, sparsity, k_delta, rep))

            if os.path.isfile(metric_out_filename + ".csv"):
                print(f"  rep {rep}: metrics already computed, skipping.")
                continue

            rep_stem      = file_name_base.format(rep)
            sim_data      = load_rds_rep(cond_str, train_sim_data_path, filename_dict,      rep_stem)
            sim_data_test = load_rds_rep(cond_str, test_sim_data_path,  test_filename_dict, rep_stem)

            # ------- data prep -------
            train_subset, test_subset, data_package = process_data_shared(
                sim_data, sim_data_test,
                training_split=TRAINING_SPLIT,
                training_size=TRAINING_SIZE,
                seed=RANDOM_SEED,
            )

            k_fit = int(sim_data.get("K")) + k_delta

            model_config = {"k": k_fit, "dense_model": False}
            train_config = {
                "eta":           ETA,
                "delta":         DELTA,
                "t":             T,
                "num_particles": NUM_PARTICLES,
                "minibatch_fraction": MINIBATCH_FRACTION,
                "min_epochs":    MIN_EPOCHS,
                "max_epochs":    NUM_EPOCHS,
                "convergence_criterion": CONVERGENCE_CRITERION,
                "window": WINDOW,
                "snr_threshold": SNR_THRESHOLD,
            }

            try:
                if os.path.isfile(model_out_filename + ".pth"):
                    # Model already trained — restore param store and skip to eval
                    print(f"  rep {rep}: model found, re-running evaluation.")
                    factor_model, train_handler, global_state = load_model_shared(
                        model_out_filename, model_config, len(train_subset), train_config,
                        device=DEVICE,
                    )
                else:
                    # ------- global training -------
                    pyro.clear_param_store()
                    factor_model, train_handler, global_state = train_model_shared(
                        model_config, train_config, train_subset,
                        model_out_filename, opt="adagrad", verbose=False, write=True,
                        device=DEVICE,
                    )

                # ------- local inference on test set -------
                local_config = {**train_config, "min_epochs": MIN_EPOCHS_LOCAL}
                factor_model, test_handler, local_state = train_model_locally_shared(
                    factor_model, local_config, test_subset, opt="adagrad",
                    device=DEVICE,
                )

                # ------- evaluation -------
                rep_config = {
                    "n": n, "p_l": p, "snr_x": snr_x, "snr_y": snr_y,
                    "rep": rep, "sparsity": sparsity, "k_delta": k_delta,
                }

                eval_table, cov_table, _, _ = evaluate_fitted_model_shared(
                    rep_config, data_package,
                    train_handler, test_handler,
                    global_state, local_state,
                    N_POSTERIOR_SAMPLES, metric_out_filename,
                )
                eval_table.to_csv(metric_out_filename + ".csv")
                if cov_table is not None:
                    cov_table.to_csv(metric_out_filename + "_cov.csv")
                metric_list.append(eval_table)
                print(f"  rep {rep}: done. "
                      f"outcome_mse={eval_table['outcome_mse'].iloc[0]:.4f}  "
                      f"mean_test_rse={eval_table['test_rse'].mean():.4f}")

            except Exception as e:
                print(f"  rep {rep}: FAILED — {e}")
