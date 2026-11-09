"""
Re-evaluate saved SupMultiviewShared models from integration_shared_misaligned
to add credible interval width metrics (outcome_95p_width, joint_95p_width).

Scans the models/ directory, parses each filename to reconstruct simulation
conditions, loads data, runs local inference, and writes updated metric CSVs
alongside the existing ones.  Already-updated files (those containing the
outcome_95p_width column) are skipped so the script is safe to re-run.

Usage
-----
  python python/eval_intervals_shared.py [--force]

  --force   Overwrite even files that already have outcome_95p_width.

Data paths assume the standard project layout:
  sim/data/multi_view_shared_misaligned/k.5/{train,test}/
  sim/results/integration_shared_misaligned/{models,metrics}/
"""

import argparse
import gc
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch
import pyro

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
MODELS_DIR     = PROJECT_ROOT / "sim/results/integration_shared_misaligned/models"
METRICS_DIR    = PROJECT_ROOT / "sim/results/integration_shared_misaligned/metrics"
TRAIN_DATA_ROOT = PROJECT_ROOT / "sim/data/multi_view_shared_misaligned/k.5/train"
TEST_DATA_ROOT  = PROJECT_ROOT / "sim/data/multi_view_shared_misaligned/k.5/test"

# Fixed simulation parameters for this sweep (not encoded in model filename)
SNR_X    = 2
SNR_Y    = 2
SPARSITY = 0.25
K_TRUE   = 5

N_POSTERIOR_SAMPLES = 100
MIN_EPOCHS_LOCAL    = 100


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# Pattern: shared_asym_n{n}p{p}_align{align_label}_deltak{k_delta}_rep{rep}
_STEM_RE = re.compile(
    r"^shared_asym_n(\d+)p(\d+)_align(.+)_deltak(\d+)_rep(\d+)$"
)


def parse_model_stem(stem: str):
    """Return (n, p, align_label, k_delta, rep) parsed from a model file stem."""
    m = _STEM_RE.match(stem)
    if m is None:
        return None
    return (
        int(m.group(1)),   # n
        int(m.group(2)),   # p
        m.group(3),        # align_label
        int(m.group(4)),   # k_delta
        int(m.group(5)),   # rep
    )


def collect_model_stems(models_dir: Path):
    """Return list of (stem, pth_path) for every primary model file."""
    stems = []
    for f in sorted(models_dir.iterdir()):
        if f.suffix != ".pth":
            continue
        stem = f.stem
        # Skip paramstore / local files
        if stem.endswith("_paramstore") or "_local" in stem:
            continue
        parsed = parse_model_stem(stem)
        if parsed is None:
            print(f"  [warn] cannot parse filename: {f.name}, skipping")
            continue
        stems.append((stem, f, parsed))
    return stems


# ---------------------------------------------------------------------------
# Data path helpers
# ---------------------------------------------------------------------------

def condition_dir(n, p, snr_x, snr_y, sparsity, align_label):
    return f"n{n}p{p}_snr{snr_x}.{snr_y}_sparse{sparsity}_align_{align_label}"


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(stem, model_path, parsed, force=False):
    import numpy as np
    from data_utils import load_rds_rep
    from run_methods import (
        process_data_shared,
        load_model_shared,
        train_model_locally_shared,
        evaluate_fitted_model_shared,
    )

    n, p, align_label, k_delta, rep = parsed

    metric_path = METRICS_DIR / f"{stem}.csv"
    cov_path    = METRICS_DIR / f"{stem}_cov.csv"

    # Skip if already up-to-date
    if metric_path.exists() and not force:
        try:
            existing = pd.read_csv(metric_path, nrows=1)
            if "outcome_95p_width" in existing.columns:
                print(f"  [{stem}] already has outcome_95p_width, skipping")
                return
        except Exception:
            pass

    cond_str = condition_dir(n, p, SNR_X, SNR_Y, SPARSITY, align_label)
    rep_stem = f"sim_data_shared_rep{rep}"

    train_path = TRAIN_DATA_ROOT / cond_str
    test_path  = TEST_DATA_ROOT  / cond_str

    if not train_path.exists() or not test_path.exists():
        print(f"  [{stem}] data not found at {train_path}, skipping")
        return

    # Build filename_dict the same way sim_run_shared_parallel._build_filename_dict does
    def _file_list(directory):
        d = directory.expanduser().resolve()
        return {directory.name: [f.name for f in d.iterdir() if f.is_file()]}

    train_filename_dict = _file_list(train_path)
    test_filename_dict  = _file_list(test_path)

    pyro.clear_param_store()

    print(f"  [{stem}] loading data")
    sim_data      = load_rds_rep(cond_str, str(TRAIN_DATA_ROOT) + "/", train_filename_dict, rep_stem)
    sim_data_test = load_rds_rep(cond_str, str(TEST_DATA_ROOT)  + "/", test_filename_dict,  rep_stem)

    train_subset, test_subset, data_package = process_data_shared(
        sim_data, sim_data_test,
        training_split=False,
        training_size=0.8,
        seed=123,
    )

    k_fit        = K_TRUE + k_delta
    model_config = {"k": k_fit, "dense_model": False}

    # Read train_config from the saved global state
    global_state = torch.load(str(model_path), weights_only=False)
    train_config = global_state["config"]

    print(f"  [{stem}] loading model")
    factor_model, train_handler, _ = load_model_shared(
        str(model_path.with_suffix("")),   # strip .pth — function appends it
        model_config, len(train_subset), train_config, device="cpu",
    )

    print(f"  [{stem}] running local inference")
    local_config = {**train_config, "min_epochs": MIN_EPOCHS_LOCAL}
    factor_model, test_handler, local_state = train_model_locally_shared(
        factor_model, local_config, test_subset, opt="adagrad", device="cpu",
    )

    rep_config = {
        "n": n, "p_l": p,
        "snr_x": SNR_X, "snr_y": SNR_Y,
        "rep": rep, "sparsity": SPARSITY, "k_delta": k_delta,
    }

    # Re-use global_state for runtime bookkeeping (inference_time, svd_time)
    eval_table, cov_table, _, _ = evaluate_fitted_model_shared(
        rep_config, data_package,
        train_handler, test_handler,
        global_state, local_state,
        N_POSTERIOR_SAMPLES, str(metric_path.with_suffix("")),
    )

    # Re-attach config columns that sim_run_shared_parallel adds after the call
    config_cols = {
        "k_fit":                 k_fit,
        "dense_model":           model_config["dense_model"],
        "svd_init":              train_config.get("svd_init", True),
        "eta":                   train_config["eta"],
        "delta":                 train_config["delta"],
        "t":                     train_config["t"],
        "num_particles":         train_config["num_particles"],
        "minibatch_fraction":    train_config.get("minibatch_fraction",
                                                   train_config.get("minibatch_size", "?")),
        "min_epochs":            train_config["min_epochs"],
        "max_epochs":            train_config["max_epochs"],
        "convergence_criterion": train_config["convergence_criterion"],
        "window":                train_config["window"],
        "snr_threshold":         train_config["snr_threshold"],
        "align_label":           align_label,
    }
    eval_table = eval_table.assign(**config_cols)
    if cov_table is not None:
        cov_table = cov_table.assign(**config_cols)

    eval_table.to_csv(metric_path)
    if cov_table is not None:
        cov_table.to_csv(cov_path)

    outcome_width = eval_table["outcome_95p_width"].iloc[0]
    outcome_cov   = eval_table["outcome_95p_coverage"].iloc[0]
    print(
        f"  [{stem}] done.  "
        f"outcome_95p_coverage={outcome_cov:.3f}  "
        f"outcome_95p_width={outcome_width:.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite CSVs that already have outcome_95p_width")
    args = parser.parse_args()

    # Add python/ to sys.path so imports (data_utils, run_methods, …) resolve
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    model_stems = collect_model_stems(MODELS_DIR)
    print(f"Found {len(model_stems)} model files to process")

    errors = []
    for stem, model_path, parsed in model_stems:
        try:
            run_eval(stem, model_path, parsed, force=args.force)
        except Exception as e:
            import traceback
            print(f"  [{stem}] FAILED: {e}")
            traceback.print_exc()
            errors.append(stem)
        finally:
            pyro.clear_param_store()
            gc.collect()

    if errors:
        print(f"\n{len(errors)} reps failed: {errors}")
    else:
        print("\nAll reps completed successfully.")


if __name__ == "__main__":
    main()
