"""
Exact prediction risk, direction recovery and strength estimates for every
linear method on the misaligned datasets (sim/data/multi_view_shared_misaligned).

Definitions: risk_metrics.py and docs/evaluation_metrics.md ("Exact risk and
factor strength"). Every method below predicts with a linear function of the
standardised views, so its raw-scale coefficients (b, c) are recovered and its
exact risk R(b, c) is computed from the replicate's true parameters, with no
test-set noise.

Methods (one row per dataset and method)
    viper_k5, viper_k15    saved SupMultiviewShared fits (k_delta 0 / 10), prediction
                           in closed form (viper_predict.closed_form_scores, Option 1)
    viper_k5_mc, viper_k15_mc   the same fits, Option 2 (Monte Carlo over q_g, 200 draws)
    pca5_ols, pca15_ols    refitted as in misaligned_baselines.pca_ols
    mofa                   saved MOFA fits (mofa/*.npz, 10-15 factors): OLS of y on the
                           training factors without intercept (as fit_mofa.R), test
                           factors X W (W^T W)^-1, so b_s = W (W^T W)^-1 coef
    ridge                  refitted as in misaligned_baselines.penalised
    elastic_net            only with --with-enet (refit is slow: ~10 h for all datasets)
    oracle                 b*; its risk is R* by construction (a check)

Columns: risk, risk_rel = (R - R*)/R*, risk_over_var_y, risk_excess, risk_intercept,
outcome_mse (test set, standardised; compare with earlier result files),
kappa2_y, kappa2_min_rel (min over true factors with beta_k != 0), span_rank,
g_hat, g_hat_mu, g_hat_overlap, g_hat_rank, g_hat_above_threshold, and per-dataset
R_star, var_y, g_y, gamma, sqrt_gamma. g_hat needs a fitted noise variance: VIPER
uses 1/E_q[1/psi] per feature, PCA the PPCA maximum-likelihood isotropic noise, the oracle
the true Psi; MOFA's noise is not saved and ridge / EN have none (NaN).

Output: sim/results/integration_shared_misaligned/exact_risk.csv (resumes by dataset)

usage: python misaligned_exact_risk.py [--with-enet] [--only COND_DIR] [--reps 1 2] [--workers 4]
       [--out PATH]
"""
import argparse
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(1, "/Users/jonathanhori/multiomic_integration/python")
warnings.filterwarnings("ignore")

DATA_ROOT = os.path.expanduser("~/multiomic_integration/sim/data/multi_view_shared_misaligned/k.5")
RESULTS = os.path.expanduser("~/multiomic_integration/sim/results/integration_shared_misaligned")
COND_PATTERN = re.compile(r"n(\d+)p(\d+)_snr([\d.]+)\.([\d.]+)_sparse([\d.]+)_align_(.+)")
N_REPS = 10
MC_DRAWS = 200


def score_dataset(cond, rep, with_enet=False):
    """Score every method on one misaligned dataset (paths of the saved fits)."""
    from run_methods import load_rds

    m = COND_PATTERN.match(cond)
    n, p, label = int(m[1]), int(m[2]), m[6]
    fname = f"sim_data_shared_rep{rep}.rds"
    tr = load_rds(f"{DATA_ROOT}/train/{cond}/{fname}")
    te = load_rds(f"{DATA_ROOT}/test/{cond}/{fname}")
    viper_fits = {f"viper_k{5 + kd}": os.path.join(
        RESULTS, "models", f"shared_asym_n{n}p{p}_align{label}_deltak{kd}_rep{rep}_paramstore.pth")
        for kd in (0, 10)}
    mofa_npz = os.path.join(RESULTS, "mofa", f"mofa_n{n}p{p}_align{label}_rep{rep}.npz")
    return score_methods(tr, te, {"n": n, "p": p, "align_label": label, "rep": rep},
                         viper_fits, mofa_npz, with_enet=with_enet, seed=rep)


def score_methods(tr, te, key, viper_fits, mofa_npz=None, with_enet=False, seed=0,
                  pca_components=(5, 15)):
    """Exact-risk rows for every linear method on one dataset.

    tr, te      training / test datasets (dicts from run_methods.load_rds)
    key         identifying columns added to every row (must include "n")
    viper_fits  {method label: path of a saved SupMultiviewShared _paramstore.pth};
                missing files are skipped. Each gives rows <label> (closed form) and
                <label>_mc (Monte Carlo, seed `seed`), plus E_q[tau] per column.
    mofa_npz    saved MOFA fit (weights_view<l>, Z_train), or None
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import ElasticNetCV, RidgeCV
    from run_methods import process_data
    from misaligned_baselines import RIDGE_ALPHAS, ENET_L1_RATIOS
    from viper_predict import closed_form_scores, load_guide_moments, mc_scores, expected_tau
    import pyro
    import risk_metrics as rm

    n = key["n"]
    _, _, dp = process_data(tr, te, model_type="shared")
    L = dp["L"]
    y_raw = tr["y"].double().reshape(-1)
    y_mean, y_sd = float(y_raw.mean()), float(y_raw.std())
    theta = rm.true_params(tr)
    orc = rm.oracle(theta)
    rel = [k for k in range(theta["beta"].numel()) if float(theta["beta"][k]) != 0]

    X_tr_s, X_te_s = dp["train_X_l_list_clean"], dp["test_X_l_list_clean"]
    X_tr = torch.cat(X_tr_s, 1).double().numpy()
    X_te = torch.cat(X_te_s, 1).double().numpy()
    y_tr = dp["train_y_clean"].double().reshape(-1).numpy()
    y_te = dp["test_y_clean"].double().reshape(-1).numpy()
    P = X_tr.shape[1]
    split = lambda A: list(torch.split(torch.as_tensor(A, dtype=torch.float64), [x.shape[1] for x in X_tr_s]))

    key = {**key, "P": P,
           "R_star": orc["R_star"], "var_y": orc["var_y"], "g_y": orc["g_y"],
           "gamma": P / n, "sqrt_gamma": np.sqrt(P / n)}
    rows = []

    def add(method, coef_s, intercept_s=0.0, load_s=None, noise_s=None, pred=None, runtime=None):
        row = {**key, "method": method, **rm.risk_summary(coef_s, dp, y_mean, y_sd, theta, orc, intercept_s)}
        if pred is None:
            pred = intercept_s + X_te @ torch.cat([torch.as_tensor(np.asarray(c), dtype=torch.float64).reshape(-1)
                                                   for c in (coef_s if isinstance(coef_s, list) else [coef_s])]).numpy()
        row["outcome_mse"] = float(np.mean((np.asarray(pred) - y_te) ** 2))
        if load_s is not None:
            k2 = rm.kappa2_hat(rm.raw_loadings(load_s, dp), theta)
            row.update(kappa2_y=k2["kappa2_y"], kappa2_min_rel=float(min(k2["kappa2_k"][k] for k in rel)),
                       span_rank=k2["span_rank"])
        if noise_s is not None:
            row.update({k: v for k, v in rm.g_hat(X_tr_s, noise_s, theta, dp).items() if k != "gamma"})
        row["runtime"] = runtime
        rows.append(row)

    # Oracle (check: risk_rel must be 0)
    b_star_s = [(b[f] * sd.double().reshape(-1) / y_sd) for b, f, sd in
                zip(orc["b_star"], dp["X_l_list_column_filters"], dp["X_l_sd_list"])]
    c_star_s = (float(sum((b[f] @ mu.double().reshape(-1)) for b, f, mu in
                          zip(orc["b_star"], dp["X_l_list_column_filters"], dp["X_l_mean_list"]))) - y_mean) / y_sd
    psi_s = [ps[f] / sd.double().reshape(-1) ** 2 for ps, f, sd in
             zip(theta["psi"], dp["X_l_list_column_filters"], dp["X_l_sd_list"])]
    lam_s = [Lm[f] / sd.double().reshape(-1)[:, None] for Lm, f, sd in
             zip(theta["Lambda"], dp["X_l_list_column_filters"], dp["X_l_sd_list"])]
    add("oracle", b_star_s, c_star_s, load_s=lam_s, noise_s=psi_s)

    # VIPER-Shared, saved fits
    for label, stem in viper_fits.items():
        if not os.path.exists(stem):
            continue
        mom = load_guide_moments(stem, L)
        # E_q[tau_lk] per view and column (MGP pruning check), as "v0;v1;..." per view
        tau = {f"Etau_view{l}": ";".join(f"{v:.4g}" for v in t.tolist())
               for l, t in enumerate(expected_tau(pyro.get_param_store(), L))}
        noise = [b / a for a, b in zip(mom["a_psi"], mom["b_psi"])]       # 1 / E_q[1/psi]
        t0 = time.time()
        cf = closed_form_scores(mom, X_te_s)
        t_cf = time.time() - t0
        add(label, cf["coef"], load_s=mom["M"], noise_s=noise, pred=cf["yhat"].numpy(), runtime=t_cf)
        rows[-1].update(tau)
        t0 = time.time()
        mc = mc_scores(mom, X_te_s, n_draws=MC_DRAWS, seed=seed)
        add(f"{label}_mc", mc["coef"], load_s=mom["M"], pred=mc["yhat"].numpy(), runtime=time.time() - t0)

    # PCA + OLS (as misaligned_baselines.pca_ols)
    for kc in pca_components:
        t0 = time.time()
        # random_state fixed: sklearn's default solver is randomised for these sizes
        # (misaligned_baselines.pca_ols leaves it unseeded; differences ~1e-3 in risk_rel)
        pca = PCA(n_components=kc, random_state=0).fit(X_tr)
        T_tr = pca.transform(X_tr)
        D = np.column_stack([np.ones(n), T_tr])
        coef, *_ = np.linalg.lstsq(D, y_tr, rcond=None)
        V = pca.components_.T
        bs = V @ coef[1:]
        cs = coef[0] - pca.mean_ @ bs
        # PPCA maximum-likelihood noise (Tipping & Bishop): discarded variance over P - K.
        # sklearn's noise_variance_ averages only the min(n, P) - K computed eigenvalues,
        # which overstates the noise when P > n.
        sigma2_ml = (X_tr.var(0, ddof=1).sum() - pca.explained_variance_.sum()) / (P - kc)
        add(f"pca{kc}_ols", split(bs), cs, load_s=split(V), noise_s=float(sigma2_ml),
            runtime=time.time() - t0)

    # MOFA + OLS (saved fits)
    if mofa_npz is not None and os.path.exists(mofa_npz):
        d = np.load(mofa_npz)
        W = [d[f"weights_view{l}"] for l in range(L)]
        Wc = np.vstack(W)
        coef, *_ = np.linalg.lstsq(d["Z_train"], y_tr, rcond=None)
        bs = Wc @ np.linalg.solve(Wc.T @ Wc, coef)
        add("mofa", split(bs), 0.0, load_s=W)
        rows[-1]["n_factors"] = W[0].shape[1]

    # Ridge / elastic net (as misaligned_baselines.penalised)
    t0 = time.time()
    fit = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_tr, y_tr)
    add("ridge", split(fit.coef_), float(fit.intercept_), runtime=time.time() - t0)
    if with_enet:
        t0 = time.time()
        fit = ElasticNetCV(l1_ratio=ENET_L1_RATIOS, n_alphas=50, cv=5, max_iter=5000, n_jobs=1).fit(X_tr, y_tr)
        add("elastic_net", split(fit.coef_), float(fit.intercept_), runtime=time.time() - t0)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-enet", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="condition directories to score")
    ap.add_argument("--reps", nargs="*", type=int, default=list(range(1, N_REPS + 1)))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exact_risk.csv"))
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        done = set(zip(prev["n"], prev["p"], prev["align_label"], prev["rep"]))
    conds = [c for c in sorted(os.listdir(f"{DATA_ROOT}/train")) if COND_PATTERN.match(c)
             and (args.only is None or c in args.only)]
    jobs = []
    for c in conds:
        m = COND_PATTERN.match(c)
        for r in args.reps:
            if (int(m[1]), int(m[2]), m[6], r) not in done and os.path.exists(f"{DATA_ROOT}/test/{c}/sim_data_shared_rep{r}.rds"):
                jobs.append((c, r))
    print(f"{len(jobs)} datasets to score", flush=True)

    from joblib import Parallel, delayed
    t0 = time.time()
    batch = 4 * args.workers
    for i in range(0, len(jobs), batch):
        res = Parallel(n_jobs=args.workers, backend="loky")(
            delayed(score_dataset)(c, r, args.with_enet) for c, r in jobs[i:i + batch])
        new = pd.DataFrame([row for rows in res for row in rows])
        new.to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
        print(f"{min(i + batch, len(jobs))}/{len(jobs)} datasets, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
