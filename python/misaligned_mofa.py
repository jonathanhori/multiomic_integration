"""
MOFA2 + OLS on the misaligned simulation datasets, scored with the same
Python metric code as the other comparators.

For each dataset (k_delta = 10, so K + 10 = 15 factors): preprocess with
run_methods.process_data, export the standardised views, fit MOFA2 + OLS with
analysis/tcga_coadread/fit_mofa.R, and compute every metric with
misaligned_baselines.factor_method_metrics. The implied covariance uses MOFA's
weights and the second moment of its fitted training factors
(W Sigma_Z W^T; MOFA's factors need not have unit variance).

Replaces the June 2026 MOFA runs (archived in archive/metrics_mofa*), which
scored covariance as W W^T and saved no fitted factors.

Outputs (sim/results/integration_shared_misaligned/mofa/)
    <dataset>.csv        one row of metrics per dataset
    <dataset>.npz        MOFA weights per view, training and test factors
    ../mofa.csv          all rows combined (written at the end)

Runs N_WORKERS fits in parallel (THREADS_PER_FIT threads each) and skips datasets
already done. Run times are measured under this parallel load, so they are not
comparable with serially measured run times.

usage: python misaligned_mofa.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(1, "/Users/jonathanhori/multiomic_integration/python")
warnings.filterwarnings("ignore")

DATA_ROOT = os.path.expanduser(
    "~/multiomic_integration/sim/data/multi_view_shared_misaligned/k.5")
OUT_DIR = os.path.expanduser(
    "~/multiomic_integration/sim/results/integration_shared_misaligned/mofa")
FIT_MOFA = os.path.expanduser("~/multiomic_integration/analysis/tcga_coadread/fit_mofa.R")
K_FIT = 15            # K + k_delta, as for the shared model
N_REPS = 10
N_WORKERS = 4
THREADS_PER_FIT = 2     # N_WORKERS x THREADS_PER_FIT = cores
COND_PATTERN = re.compile(r"n(\d+)p(\d+)_snr([\d.]+)\.([\d.]+)_sparse([\d.]+)_align_(.+)")


def run_one(cond, rep):
    """Fit and score MOFA on one dataset; returns the metric row."""
    from run_methods import load_rds, process_data
    from misaligned_baselines import factor_method_metrics

    m = COND_PATTERN.match(cond)
    n, p, label = int(m[1]), int(m[2]), m[6]
    name = f"mofa_n{n}p{p}_align{label}_rep{rep}"
    out_csv = os.path.join(OUT_DIR, f"{name}.csv")
    if os.path.exists(out_csv):
        return None

    fname = f"sim_data_shared_rep{rep}.rds"
    tr = load_rds(f"{DATA_ROOT}/train/{cond}/{fname}")
    te = load_rds(f"{DATA_ROOT}/test/{cond}/{fname}")
    _, _, dp = process_data(tr, te, model_type="shared")
    L = len(dp["train_X_l_list_clean"])

    work = tempfile.mkdtemp(prefix=f"{name}_")
    try:
        for l, (a, b) in enumerate(zip(dp["train_X_l_list_clean"], dp["test_X_l_list_clean"]), 1):
            np.savetxt(os.path.join(work, f"x_train_view{l}.csv"), a.numpy(), delimiter=",")
            np.savetxt(os.path.join(work, f"x_test_view{l}.csv"), b.numpy(), delimiter=",")
        np.savetxt(os.path.join(work, "y_train.csv"), dp["train_y_clean"].reshape(-1).numpy(), delimiter=",")
        # Limit each MOFA fit's linear-algebra threads so N_WORKERS fits share the cores.
        env = {**os.environ, **{v: str(THREADS_PER_FIT) for v in
               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
        subprocess.run(["Rscript", FIT_MOFA, work, str(L), str(K_FIT), "123"], check=True, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        read = lambda f: pd.read_csv(os.path.join(work, f)).values
        pred = pd.read_csv(os.path.join(work, "mofa_predictions.csv"))
        W = [read(f"mofa_weights_view{l}.csv") for l in range(1, L + 1)]
        Z_train, Z_test = read("mofa_z_train.csv"), read("mofa_z_test.csv")
        runtime = float(open(os.path.join(work, "mofa_runtime.txt")).read())
    finally:
        shutil.rmtree(work, ignore_errors=True)

    Sigma_Z = Z_train.T @ Z_train / Z_train.shape[0]
    metrics = factor_method_metrics(Z_train, Z_test, W, W, Sigma_Z,
                                    pred["fit"].values, pred["lwr"].values, pred["upr"].values, dp)
    row = {"n": n, "p": p, "align_label": label, "rep": rep, "method": "mofa",
           **metrics, "runtime": runtime, "n_factors": Z_train.shape[1]}
    np.savez(os.path.join(OUT_DIR, f"{name}.npz"), Z_train=Z_train, Z_test=Z_test,
             **{f"weights_view{l}": w for l, w in enumerate(W)})
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    print(f"{name}: mse={row['outcome_mse']:.3f} cov={row['cov_within']:.4f} ({runtime:.0f}s)", flush=True)
    return row


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = [(cond, rep) for cond in sorted(os.listdir(f"{DATA_ROOT}/train")) if COND_PATTERN.match(cond)
            for rep in range(1, N_REPS + 1)
            if os.path.exists(f"{DATA_ROOT}/test/{cond}/sim_data_shared_rep{rep}.rds")]
    Parallel(n_jobs=N_WORKERS, backend="loky")(delayed(run_one)(c, r) for c, r in jobs)
    files = [f for f in os.listdir(OUT_DIR) if f.endswith(".csv")]
    combined = pd.concat([pd.read_csv(os.path.join(OUT_DIR, f)) for f in files], ignore_index=True)
    combined.to_csv(os.path.join(os.path.dirname(OUT_DIR), "mofa.csv"), index=False)
    print(f"\n{len(combined)} datasets -> mofa.csv")


if __name__ == "__main__":
    main()
