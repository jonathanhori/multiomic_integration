"""
Score every method on the controlled-strength datasets (weak-factor study, Phase 2).

Per replicate (data: R/generate_controlled_data.R; generator documented in
docs/controlled_strength_data.md):
  1. MOFA2 + OLS with K_FIT = 15 factors, fitted as in misaligned_mofa.py
     (analysis/tcga_coadread/fit_mofa.R on the standardised views); the fit is
     saved to results/.../mofa/<stem>.npz and reused when present.
  2. misaligned_exact_risk.score_methods: exact risk R(b, c), (R - R*)/R*,
     kappa2_hat, g_hat and test outcome MSE for
        oracle, viper_k15 (closed-form prediction from the saved fit of
        sim_run_shared_controlled.py, alpha = 1), viper_k15_mc, pca5_ols, pca15_ols,
        mofa, ridge, elastic_net; E_q[tau] per VIPER column (Etau_view<l>).
  3. oracle_known_z: y_hat = beta^T z with the true test scores. Not a function of x,
     so its "risk" is the exact error variance sigma_y^2 + theta^T Psi theta (below R*).
Extra columns: c, g_target, g_true (realised), s, gamma, sqrt_gamma, g_over_sqrt_gamma.

Output: sim/results/integration_shared_controlled/controlled_scores.csv
(appended in batches; datasets already present are skipped).

usage: python controlled_scoring.py [--only n,P,c] [--reps 1 2] [--alphas 1] [--workers 4]
       [--no-enet] [--out PATH]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(1, "/Users/jonathanhori/multiomic_integration/python")
warnings.filterwarnings("ignore")

DATA_ROOT = os.path.expanduser("~/multiomic_integration/sim/data/multi_view_shared_controlled/k.5")
RESULTS = os.path.expanduser("~/multiomic_integration/sim/results/integration_shared_controlled")
FIT_MOFA = os.path.expanduser("~/multiomic_integration/analysis/tcga_coadread/fit_mofa.R")
N_ARRAY = (100, 200, 500)
P_ARRAY = (600, 3000)
C_ARRAY = (0.3, 0.6, 1, 1.5, 2, 3, 10)
K_FIT = 15
THREADS_PER_FIT = 2


def cond_name(n, P, c):
    return f"n{n}_P{P}_c{c:g}"


def fit_mofa(dp, out_npz, k_fit=K_FIT, threads=THREADS_PER_FIT, seed=123):
    """MOFA2 fit on the standardised views (as misaligned_mofa.run_one); saves
    weights_view<l>, Z_train, Z_test and the fit time to out_npz."""
    L = dp["L"]
    work = tempfile.mkdtemp(prefix="mofa_ctrl_")
    try:
        for l, (a, b) in enumerate(zip(dp["train_X_l_list_clean"], dp["test_X_l_list_clean"]), 1):
            np.savetxt(os.path.join(work, f"x_train_view{l}.csv"), a.numpy(), delimiter=",")
            np.savetxt(os.path.join(work, f"x_test_view{l}.csv"), b.numpy(), delimiter=",")
        np.savetxt(os.path.join(work, "y_train.csv"), dp["train_y_clean"].reshape(-1).numpy(), delimiter=",")
        env = {**os.environ, **{v: str(threads) for v in
               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
        subprocess.run(["Rscript", FIT_MOFA, work, str(L), str(k_fit), str(seed)], check=True, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        read = lambda f: pd.read_csv(os.path.join(work, f)).values
        W = [read(f"mofa_weights_view{l}.csv") for l in range(1, L + 1)]
        runtime = float(open(os.path.join(work, "mofa_runtime.txt")).read())
        np.savez(out_npz, Z_train=read("mofa_z_train.csv"), Z_test=read("mofa_z_test.csv"),
                 runtime=runtime, **{f"weights_view{l}": w for l, w in enumerate(W)})
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return runtime


def score_one(n, P, c, rep, alphas, with_enet):
    import torch
    from run_methods import load_rds, process_data
    from misaligned_exact_risk import score_methods
    import risk_metrics as rm

    cond = cond_name(n, P, c)
    fname = f"sim_data_controlled_{cond}_rep{rep}.rds"
    tr = load_rds(f"{DATA_ROOT}/train/{cond}/{fname}")
    te = load_rds(f"{DATA_ROOT}/test/{cond}/{fname}")

    os.makedirs(os.path.join(RESULTS, "mofa"), exist_ok=True)
    npz = os.path.join(RESULTS, "mofa", f"mofa_ctrl_{cond}_rep{rep}.npz")
    if not os.path.exists(npz):
        _, _, dp = process_data(tr, te, model_type="shared")
        fit_mofa(dp, npz)
    mofa_runtime = float(np.load(npz)["runtime"])

    viper_fits = {("viper_k15" if a == 1 else f"viper_k15_alpha{a:g}"):
                  os.path.join(RESULTS, "models", f"shared_ctrl_{cond}_alpha{a:g}_rep{rep}_paramstore.pth")
                  for a in alphas}
    g_true = float(tr["g_true"].item())
    key = {"n": n, "P_grid": P, "c": c, "rep": rep, "g_target": float(tr["g_target"].item()),
           "g_true": g_true, "s": int(tr["s"].item()),
           "g_over_sqrt_gamma": g_true / np.sqrt(P / n)}
    rows = score_methods(tr, te, key, viper_fits, npz, with_enet=with_enet, seed=rep)
    for r in rows:
        if r["method"] == "mofa":
            r["runtime"] = mofa_runtime

    # oracle_known_z
    theta = rm.true_params(tr)
    orc = rm.oracle(theta)
    err_var = theta["sigma2_y"] + (0.0 if theta["theta"] is None else
                                   float(sum((p * t ** 2).sum() for p, t in zip(theta["psi"], theta["theta"]))))
    y_raw = tr["y"].double().reshape(-1)
    mu, sd = y_raw.mean(), y_raw.std()
    y_te = (te["y"].double().reshape(-1) - mu) / sd
    pred = (te["Z"].double() @ theta["beta"] - mu) / sd
    base = {k: v for k, v in rows[0].items() if k in key or k in ("P", "R_star", "var_y", "g_y", "gamma", "sqrt_gamma")}
    rows.append({**base, "method": "oracle_known_z", "risk": err_var,
                 "risk_rel": (err_var - orc["R_star"]) / orc["R_star"],
                 "risk_over_var_y": err_var / orc["var_y"],
                 "outcome_mse": float(((pred - y_te) ** 2).mean())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="n,P,c")
    ap.add_argument("--reps", nargs="*", type=int, default=list(range(1, 11)))
    ap.add_argument("--alphas", nargs="*", type=float, default=[1])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-enet", action="store_true")
    ap.add_argument("--out", default=os.path.join(RESULTS, "controlled_scores.csv"))
    args = ap.parse_args()

    cells = [(n, P, c) for n in N_ARRAY for P in P_ARRAY for c in C_ARRAY]
    if args.only:
        n, P, c = (float(v) for v in args.only.split(","))
        cells = [(int(n), int(P), c)]
    done = set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        done = set(zip(prev["n"], prev["P_grid"], prev["c"], prev["rep"]))
    jobs = [(n, P, c, r) for n, P, c in cells for r in args.reps if (n, P, c, r) not in done]
    print(f"{len(jobs)} datasets to score", flush=True)

    from joblib import Parallel, delayed
    t0 = time.time()
    batch = 2 * args.workers
    for i in range(0, len(jobs), batch):
        res = Parallel(n_jobs=args.workers, backend="loky")(
            delayed(score_one)(n, P, c, r, args.alphas, not args.no_enet) for n, P, c, r in jobs[i:i + batch])
        new = pd.DataFrame([row for rows in res for row in rows])
        if os.path.exists(args.out):     # keep the file's column order; batches may lack some columns
            cols = list(pd.read_csv(args.out, nrows=0).columns)
            extra = [c for c in new.columns if c not in cols]
            if extra:
                old = pd.read_csv(args.out)
                pd.concat([old, new], ignore_index=True).to_csv(args.out, index=False)
                continue
            new = new.reindex(columns=cols)
        new.to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
        print(f"{min(i + batch, len(jobs))}/{len(jobs)} datasets, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
