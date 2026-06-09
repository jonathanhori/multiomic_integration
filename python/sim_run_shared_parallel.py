"""
Entry point for SupMultiviewShared simulation sweep — parallel rep variant.

Identical to sim_run_shared.py except that reps within each simulation
condition are dispatched to a pool of worker processes (joblib/loky) when
the condition's data size is small enough that running MAX_WORKERS reps
simultaneously will not cause memory pressure.

Conditions where n * p >= LARGE_DATA_NP_THRESHOLD run their reps
sequentially, matching the behaviour of the original script.

Data must be pre-generated in R (generate_factor_data_shared /
generate_factor_data_shared_testing) and saved as:
  sim/data_ortho/k.<K>/train/<cond_dir>/sim_data_shared_rep<i>.rds
  sim/data_ortho/k.<K>/test/<cond_dir>/sim_data_shared_rep<i>.rds

where <cond_dir> follows the pattern n{n}p{p}_snr{snr_x}.{snr_y}_sparse{sparsity}.
"""

import argparse
import gc
import os
import itertools
from pathlib import Path

import torch
import pyro

DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Data paths  (edit K to match the simulation design)
# ---------------------------------------------------------------------------
K = 5
train_sim_data_path = f"~/multiomic_integration/sim/data/multi_view_shared/k.{K}/train/"
test_sim_data_path  = f"~/multiomic_integration/sim/data/multi_view_shared/k.{K}/test/"

# ---------------------------------------------------------------------------
# Simulation grid
# ---------------------------------------------------------------------------
n_array          = (100, 500, 1000)
p_array          = (50, 100, 1000, 5000)
snr_x_array      = [2]
snr_y_array      = [2]
reps             = 10
loading_sparsity = [0.25]
k_deltas         = [0, 10]

sim_grid = list(itertools.product(
    n_array, p_array, snr_x_array, snr_y_array, loading_sparsity, k_deltas
))

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
N_POSTERIOR_SAMPLES   = 100
MIN_EPOCHS            = 200
MIN_EPOCHS_LOCAL      = 100
NUM_EPOCHS            = 500
CONVERGENCE_CRITERION = "elbo_snr"
WINDOW                = 50
SNR_THRESHOLD         = 5

MINIBATCH_FRACTION    = 0.15   # fraction of training set size; converted per run
NUM_PARTICLES         = 1

ETA   = 0.25
DELTA = 1e-16
T     = 0.1

SVD_INIT = True   # set True to initialise loc_Z / loc_Lambda from SVD of stacked X

TRAINING_SPLIT = False
TRAINING_SIZE  = 0.8
RANDOM_SEED    = 123

# ---------------------------------------------------------------------------
# Parallelism settings
# ---------------------------------------------------------------------------
# Maximum worker processes for parallel rep execution within one condition.
MAX_WORKERS = 4

# Conditions where n*p >= this threshold run reps sequentially to limit peak
# RAM usage.  Each rep loads roughly 16 * n * p bytes of data tensors into
# memory (2 views, train + test), so MAX_WORKERS concurrent reps need
# ~MAX_WORKERS * 16 * n * p bytes.
#
# At the threshold of 1_000_000 elements per view:
#   ~16 MB/rep → ~64 MB for 4 workers  (fine)
# Above the threshold, e.g. n=1000, p=5000 → 5 M elements:
#   ~80 MB/rep → ~320 MB for 4 workers (serial is safer)
LARGE_DATA_NP_THRESHOLD = 1_000_000

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
model_out_path  = os.path.expanduser(
    "~/multiomic_integration/sim/results/integration_shared/models")
metric_out_path = os.path.expanduser(
    "~/multiomic_integration/sim/results/integration_shared/metrics")
Path(model_out_path).mkdir(parents=True, exist_ok=True)
Path(metric_out_path).mkdir(parents=True, exist_ok=True)

file_dir_base            = "n{}p{}_snr{}.{}_sparse{}"
file_name_base           = "sim_data_shared_rep{}"
model_out_filename_base  = "shared_n{}p{}_snr{}.{}_sparse{}_deltak{}_rep{}"
metric_out_filename_base = model_out_filename_base

# ---------------------------------------------------------------------------
# Asymmetric (misaligned) mode — data in multi_view_shared_misaligned/
# ---------------------------------------------------------------------------
ASYM_N_ARRAY      = (100, 500, 1000)
ASYM_P_ARRAY      = (50, 100, 500, 1000)
ASYM_ALIGN_LABELS = ("symmetric", "mild_misalignment", "strong_misalignment", "anti_aligned")
ASYM_K_DELTAS     = [0, 10]
ASYM_SNR_X        = 2
ASYM_SNR_Y        = 2
ASYM_SPARSITY     = 0.25

asym_train_data_path = f"~/multiomic_integration/sim/data/multi_view_shared_misaligned/k.{K}/train/"
asym_test_data_path  = f"~/multiomic_integration/sim/data/multi_view_shared_misaligned/k.{K}/test/"

asym_model_out_path  = os.path.expanduser(
    "~/multiomic_integration/sim/results/integration_shared_misaligned/models")
asym_metric_out_path = os.path.expanduser(
    "~/multiomic_integration/sim/results/integration_shared_misaligned/metrics")

# cond_dir name matches R: n{n}p{p}_snr{snr_x}.{snr_y}_sparse{s}_align_{label}
asym_cond_dir_base            = "n{}p{}_snr{}.{}_sparse{}_align_{}"
asym_model_out_filename_base  = "shared_asym_n{}p{}_align{}_deltak{}_rep{}"
asym_metric_out_filename_base = asym_model_out_filename_base


# ---------------------------------------------------------------------------
# Module-level worker functions
# These must live at module scope (not inside __main__) so that loky's
# spawn-based workers can import and pickle them.
# ---------------------------------------------------------------------------

def _build_filename_dict(path):
    """Return {condition_dir: [filenames]} for every subdirectory of path."""
    expanded = os.path.expanduser(path)
    dirs = [os.path.join(path, d)
            for d in os.listdir(expanded)
            if os.path.isdir(os.path.join(expanded, d))]
    return {os.path.basename(d): os.listdir(os.path.expanduser(d)) for d in dirs}


def _run_rep(rep_args):
    """Train and evaluate SupMultiviewShared for one simulation rep.

    Fully self-contained so it runs correctly whether called in the main
    process (sequential) or in a loky worker subprocess (parallel).  All
    imports are local so the function works in a freshly spawned process.

    Parameters
    ----------
    rep_args : dict
        Keys: rep, n, p, snr_x, snr_y, sparsity, k_delta, device,
        cond_str, train_sim_data_path, test_sim_data_path, filename_dict,
        test_filename_dict, rep_stem, model_out_filename, metric_out_filename,
        training_split, training_size, random_seed, n_posterior_samples,
        min_epochs_local, train_config.

    Returns
    -------
    dict
        ``{"status": "done",    "rep": int, "outcome_mse": float, "mean_test_rse": float}``
        ``{"status": "skipped", "rep": int}``
        ``{"_error": str,       "rep": int}``
    """
    import gc
    import os
    import traceback as tb
    import torch
    import pyro
    from data_utils import load_rds_rep
    from run_methods import (
        process_data_shared,
        load_model_shared,
        train_model_shared,
        train_model_locally_shared,
        evaluate_fitted_model_shared,
    )

    rep                 = rep_args["rep"]
    n                   = rep_args["n"]
    p                   = rep_args["p"]
    snr_x               = rep_args["snr_x"]
    snr_y               = rep_args["snr_y"]
    sparsity            = rep_args["sparsity"]
    k_delta             = rep_args["k_delta"]
    device              = torch.device(rep_args["device"])
    model_out_filename  = rep_args["model_out_filename"]
    metric_out_filename = rep_args["metric_out_filename"]
    train_config        = rep_args["train_config"]

    if os.path.isfile(metric_out_filename + ".csv"):
        print(f"  rep {rep}: metrics already computed, skipping.")
        return {"status": "skipped", "rep": rep}

    try:
        pyro.clear_param_store()

        sim_data = load_rds_rep(
            rep_args["cond_str"],
            rep_args["train_sim_data_path"],
            rep_args["filename_dict"],
            rep_args["rep_stem"],
        )
        sim_data_test = load_rds_rep(
            rep_args["cond_str"],
            rep_args["test_sim_data_path"],
            rep_args["test_filename_dict"],
            rep_args["rep_stem"],
        )

        train_subset, test_subset, data_package = process_data_shared(
            sim_data, sim_data_test,
            training_split=rep_args["training_split"],
            training_size=rep_args["training_size"],
            seed=rep_args["random_seed"],
        )

        k_fit        = int(sim_data.get("K")) + k_delta
        model_config = {"k": k_fit, "dense_model": False}

        if os.path.isfile(model_out_filename + ".pth"):
            print(f"  rep {rep}: model found, re-running evaluation.")
            factor_model, train_handler, global_state = load_model_shared(
                model_out_filename, model_config, len(train_subset), train_config,
                device=device,
            )
        else:
            factor_model, train_handler, global_state = train_model_shared(
                model_config, train_config, train_subset,
                model_out_filename, opt="adagrad", verbose=False, write=True,
                device=device, svd_init=rep_args["svd_init"],
            )

        local_config = {**train_config, "min_epochs": rep_args["min_epochs_local"]}
        factor_model, test_handler, local_state = train_model_locally_shared(
            factor_model, local_config, test_subset, opt="adagrad", device=device,
        )

        rep_config = {
            "n": n, "p_l": p, "snr_x": snr_x, "snr_y": snr_y,
            "rep": rep, "sparsity": sparsity, "k_delta": k_delta,
        }

        eval_table, cov_table, _, _ = evaluate_fitted_model_shared(
            rep_config, data_package,
            train_handler, test_handler,
            global_state, local_state,
            rep_args["n_posterior_samples"], metric_out_filename,
        )

        # Append training/model config so every output row is self-describing.
        config_cols = {
            "k_fit":                 k_fit,
            "dense_model":           model_config["dense_model"],
            "svd_init":              rep_args["svd_init"],
            "eta":                   train_config["eta"],
            "delta":                 train_config["delta"],
            "t":                     train_config["t"],
            "num_particles":         train_config["num_particles"],
            "minibatch_fraction":    train_config["minibatch_fraction"],
            "min_epochs":            train_config["min_epochs"],
            "max_epochs":            train_config["max_epochs"],
            "convergence_criterion": train_config["convergence_criterion"],
            "window":                train_config["window"],
            "snr_threshold":         train_config["snr_threshold"],
        }
        eval_table = eval_table.assign(**config_cols)
        if cov_table is not None:
            cov_table = cov_table.assign(**config_cols)

        # Tag asymmetric runs with their alignment scenario label.
        align_label = rep_args.get("align_label")
        if align_label is not None:
            eval_table = eval_table.assign(align_label=align_label)
            if cov_table is not None:
                cov_table = cov_table.assign(align_label=align_label)

        eval_table.to_csv(metric_out_filename + ".csv")
        if cov_table is not None:
            cov_table.to_csv(metric_out_filename + "_cov.csv")

        result = {
            "status":         "done",
            "rep":            rep,
            "outcome_mse":    float(eval_table["outcome_mse"].iloc[0]),
            "mean_train_rse": float(eval_table["joint_rse"].mean()),
            "mean_test_rse":  float(eval_table["test_rse"].mean()),
            "mean_cov_frob":  float(cov_table["cov_frob"].mean()) if cov_table is not None else float("nan"),
        }
        cov_str = f"  cov_frob={result['mean_cov_frob']:.4f}" if cov_table is not None else ""
        print(
            f"  rep {rep}: done.  "
            f"outcome_mse={result['outcome_mse']:.4f}  "
            f"train_rse={result['mean_train_rse']:.4f}  "
            f"test_rse={result['mean_test_rse']:.4f}"
            f"{cov_str}"
        )
        return result

    except Exception as e:
        print(f"  rep {rep}: FAILED — {e}")
        return {"_error": f"{e}\n{tb.format_exc()}", "rep": rep}

    finally:
        pyro.clear_param_store()
        gc.collect()


def _run_rep_worker(args):
    """Single-argument wrapper so joblib.Parallel can dispatch _run_rep."""
    return _run_rep(args)


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
    parser.add_argument(
        "--mode", choices=["standard", "asymmetric"], default="standard",
        help=(
            "'standard' runs the symmetric shared-multiview grid "
            "(multi_view_shared/); 'asymmetric' runs the beta-misalignment grid "
            "(multi_view_shared_misaligned/)."
        ),
    )
    cli_args = parser.parse_args()
    mode = cli_args.mode

    # Mode-specific active grid and I/O paths.
    if mode == "standard":
        active_grid       = sim_grid
        active_train_path = train_sim_data_path
        active_test_path  = test_sim_data_path
        active_model_out  = model_out_path
        active_metric_out = metric_out_path
    else:
        active_grid = list(itertools.product(
            ASYM_N_ARRAY, ASYM_P_ARRAY, ASYM_ALIGN_LABELS, ASYM_K_DELTAS
        ))
        active_train_path = asym_train_data_path
        active_test_path  = asym_test_data_path
        active_model_out  = asym_model_out_path
        active_metric_out = asym_metric_out_path
        Path(active_model_out).mkdir(parents=True, exist_ok=True)
        Path(active_metric_out).mkdir(parents=True, exist_ok=True)

    grid_to_run = (
        [active_grid[cli_args.task_id - 1]] if cli_args.task_id is not None
        else active_grid
    )

    filename_dict      = _build_filename_dict(active_train_path)
    test_filename_dict = _build_filename_dict(active_test_path)

    # Build train_config once; shared across all conditions and reps.
    train_config = {
        "eta":                   ETA,
        "delta":                 DELTA,
        "t":                     T,
        "num_particles":         NUM_PARTICLES,
        "minibatch_fraction":    MINIBATCH_FRACTION,
        "min_epochs":            MIN_EPOCHS,
        "max_epochs":            NUM_EPOCHS,
        "convergence_criterion": CONVERGENCE_CRITERION,
        "window":                WINDOW,
        "snr_threshold":         SNR_THRESHOLD,
    }

    for row in grid_to_run:
        # Unpack the grid row and derive the condition string / filenames.
        if mode == "standard":
            n, p, snr_x, snr_y, sparsity, k_delta = row
            align_label = None
            cond_str    = file_dir_base.format(n, p, snr_x, snr_y, sparsity)
            check_stem  = metric_out_filename_base.format(n, p, snr_x, snr_y, sparsity, k_delta, reps)
            def _make_stem(rep, _n=n, _p=p, _sx=snr_x, _sy=snr_y, _sp=sparsity, _kd=k_delta):
                return model_out_filename_base.format(_n, _p, _sx, _sy, _sp, _kd, rep)
        else:
            n, p, align_label, k_delta = row
            snr_x    = ASYM_SNR_X
            snr_y    = ASYM_SNR_Y
            sparsity = ASYM_SPARSITY
            cond_str   = asym_cond_dir_base.format(n, p, snr_x, snr_y, sparsity, align_label)
            check_stem = asym_metric_out_filename_base.format(n, p, align_label, k_delta, reps)
            def _make_stem(rep, _n=n, _p=p, _al=align_label, _kd=k_delta):
                return asym_model_out_filename_base.format(_n, _p, _al, _kd, rep)

        print(f"\n=== Condition: {cond_str}  k_delta={k_delta} ===")

        # Skip if all reps already have metrics.
        check_metric = os.path.join(active_metric_out, check_stem)
        if os.path.isfile(check_metric + ".csv"):
            print("All reps done, skipping.")
            continue

        # Decide parallelism based on data size per rep.
        use_parallel   = (n * p < LARGE_DATA_NP_THRESHOLD)
        n_workers_this = MAX_WORKERS if use_parallel else 1
        if use_parallel:
            print(f"  n*p={n*p:,} — dispatching reps to up to {n_workers_this} workers.")
        else:
            print(
                f"  n*p={n*p:,} >= {LARGE_DATA_NP_THRESHOLD:,} — "
                f"running reps sequentially to limit memory."
            )

        # Build per-rep argument dicts.  Everything must be picklable for loky.
        rep_args_list = []
        for rep in range(1, reps + 1):
            stem                = _make_stem(rep)
            model_out_filename  = os.path.join(active_model_out,  stem)
            metric_out_filename = os.path.join(active_metric_out, stem)

            rep_args_list.append({
                "rep":                 rep,
                "n":                   n,
                "p":                   p,
                "snr_x":               snr_x,
                "snr_y":               snr_y,
                "sparsity":            sparsity,
                "k_delta":             k_delta,
                "align_label":         align_label,
                "device":              str(DEVICE),
                "cond_str":            cond_str,
                "train_sim_data_path": active_train_path,
                "test_sim_data_path":  active_test_path,
                "filename_dict":       filename_dict,
                "test_filename_dict":  test_filename_dict,
                "rep_stem":            file_name_base.format(rep),
                "model_out_filename":  model_out_filename,
                "metric_out_filename": metric_out_filename,
                "training_split":      TRAINING_SPLIT,
                "training_size":       TRAINING_SIZE,
                "random_seed":         RANDOM_SEED,
                "n_posterior_samples": N_POSTERIOR_SAMPLES,
                "min_epochs_local":    MIN_EPOCHS_LOCAL,
                "train_config":        train_config,
                "svd_init":            SVD_INIT,
            })

        if n_workers_this == 1:
            results = [_run_rep_worker(a) for a in rep_args_list]
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_workers_this, backend="loky")(
                delayed(_run_rep_worker)(a) for a in rep_args_list
            )

        for r in results:
            if r and r.get("status") == "done":
                metric_list.append(r)
            elif r and "_error" in r:
                print(f"  rep {r.get('rep', '?')}: FAILED — {r['_error']}")
