"""
Reference predictors and comparison methods for the misaligned simulation scenarios.

For every misaligned dataset (sim/data/multi_view_shared_misaligned) this script
computes the oracles and fits the comparison methods that are not run by the
main sweeps (the shared model and MOFA have their own result folders). All
methods use exactly the pipeline's preprocessing — run_methods.process_data:
all-zero columns dropped, views and outcome standardised with training
statistics — and the same metric definitions (docs/evaluation_metrics.md).

Oracles (baselines.csv)
    oracle_known_z  y_hat = beta^T z with the true test scores (outcome noise only)
    oracle_from_x   y_hat = beta^T E[z | x] with the true Lambda, Psi, beta: the
                    lowest MSE achievable when scores are inferred from the views
    ell_min_outcome_factor, sqrt_P_over_n
                    strength of the weakest outcome-relevant factor and the
                    detection threshold (see docs/evaluation_metrics.md)

Comparison methods (comparators.csv, one row per dataset and method)
    pca5_ols, pca15_ols  PCA of the stacked views (5 = true K, 15 = K fitted by the
                         factor models), then OLS of y on the scores. Signal
                         reconstruction uses the rank-K PCA reconstruction;
                         covariance uses probabilistic-PCA loadings
                         W = V_K (Lambda_K - sigma^2 I)^1/2; intervals are OLS
                         prediction intervals on the scores (as for MOFA + OLS).
    ridge                ridge regression on the stacked views, penalty by 5-fold CV
    elastic_net          elastic net on the stacked views, penalty and mix by 5-fold CV
Metrics: joint_rse, test_rse, cov_within, cov_cross, outcome_mse,
outcome_95p_coverage, outcome_95p_width, runtime. Metrics a method does not
define (factor metrics and intervals for ridge / elastic net) are left empty.

The script resumes: datasets already in comparators.csv are skipped.

usage: python misaligned_baselines.py
"""
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV

sys.path.insert(1, "/Users/jonathanhori/multiomic_integration/python")
from run_methods import load_rds, process_data
from data_utils import eval_rse, implied_cov_errors, scale_sim_struct

warnings.filterwarnings("ignore")

DATA_ROOT = os.path.expanduser(
    "~/multiomic_integration/sim/data/multi_view_shared_misaligned/k.5")
OUT_DIR = os.path.expanduser(
    "~/multiomic_integration/sim/results/integration_shared_misaligned")
N_REPS = 10
RIDGE_ALPHAS = np.logspace(-2, 5, 30)
ENET_L1_RATIOS = [0.1, 0.5, 0.9, 1.0]     # same grid as analysis/tcga_coadread/compare_methods.py
COND_PATTERN = re.compile(r"n(\d+)p(\d+)_snr([\d.]+)\.([\d.]+)_sparse([\d.]+)_align_(.+)")


def oracles(tr, te, snr_x, std_y):
    """Oracle predictions and per-factor strength from the true parameters."""
    # Features with all-zero loadings carry no signal and have zero noise variance.
    keep = [(L ** 2).sum(1) > 0 for L in tr["Lambda_l"]]
    Lam = [L.double()[k] for L, k in zip(tr["Lambda_l"], keep)]
    X_te = [x.double()[:, k] for x, k in zip(te["X_l"], keep)]
    psi = [(L ** 2).sum(1) / snr_x for L in Lam]              # simulated noise variances
    beta = tr["beta"].double().reshape(-1)
    K = Lam[0].shape[1]

    precision = sum(L.T @ (L / s[:, None]) for L, s in zip(Lam, psi))   # sum_l Lam^T Psi^-1 Lam
    post_cov = torch.linalg.inv(torch.eye(K, dtype=torch.float64) + precision)
    ez = sum((x / s) @ L for x, L, s in zip(X_te, Lam, psi)) @ post_cov  # E[z | x]

    y_te = std_y(te["y"].double().reshape(-1))
    ell = torch.diag(precision)
    outcome_factors = [k for k in range(K) if beta[k] != 0]
    return {
        "oracle_known_z": float(((std_y(te["Z"].double() @ beta) - y_te) ** 2).mean()),
        "oracle_from_x":  float(((std_y(ez @ beta) - y_te) ** 2).mean()),
        "ell_min_outcome_factor": float(min(ell[k] for k in outcome_factors)),
    }


def ols_prediction_interval(T_tr, y_tr, T_te, level=0.95):
    """OLS of y on scores (with intercept) and its prediction interval."""
    D_tr = np.column_stack([np.ones(len(T_tr)), T_tr])
    D_te = np.column_stack([np.ones(len(T_te)), T_te])
    coef, *_ = np.linalg.lstsq(D_tr, y_tr, rcond=None)
    dof = len(y_tr) - D_tr.shape[1]
    s2 = float(((y_tr - D_tr @ coef) ** 2).sum() / dof)
    lev = np.einsum("ij,jk,ik->i", D_te, np.linalg.pinv(D_tr.T @ D_tr), D_te)
    half = stats.t.ppf(0.5 + level / 2, dof) * np.sqrt(s2 * (1 + lev))
    pred = D_te @ coef
    return pred, pred - half, pred + half


def factor_method_metrics(Z_train, Z_test, recon_loadings, cov_loadings, Sigma_Z,
                          pred, lo, hi, dp):
    """Metrics shared by the factor-based comparators (PCA + OLS, MOFA + OLS).

    Z_train, Z_test   scores (n x K) used for signal reconstruction with recon_loadings
                      (list of p_l x K): J_hat^(l) = Z recon_loadings[l]^T
    cov_loadings      loadings for the implied covariance, with Sigma_Z the scores'
                      second moment (None = identity); see data_utils.implied_cov_errors
    pred, lo, hi      standardised test-outcome predictions and 95% interval bounds
    """
    X_tr_list, X_te_list = dp["train_X_l_list_clean"], dp["test_X_l_list_clean"]
    L = len(X_tr_list)
    sim = dp["train_SIM_decomp"]
    truth = scale_sim_struct(
        [sim["SIM_Z"] @ sim["SIM_Lambda_l_list"][l][dp["X_l_list_column_filters"][l]].T for l in range(L)],
        dp["X_l_mean_list"], dp["X_l_sd_list"])
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)
    joint_rse = np.mean([eval_rse(t(Z_train) @ t(recon_loadings[l]).T, truth[l].double()).item()
                         for l in range(L)])
    test_rse = np.mean([eval_rse(t(Z_test) @ t(recon_loadings[l]).T, X_te_list[l].double()).item()
                        for l in range(L)])
    cov = implied_cov_errors([t(W) for W in cov_loadings], Sigma_Z, dp)
    y_te = dp["test_y_clean"].reshape(-1).numpy()
    return {
        "joint_rse": joint_rse, "test_rse": test_rse,
        "cov_within": float(cov.loc[cov["cov_type"] == "within", "cov_frob"].mean()),
        "cov_cross": float(cov.loc[cov["cov_type"] == "cross", "cov_frob"].mean()),
        "outcome_mse": float(np.mean((pred - y_te) ** 2)),
        "outcome_95p_coverage": float(np.mean((y_te >= lo) & (y_te <= hi))),
        "outcome_95p_width": float(np.mean(hi - lo)),
    }


def pca_ols(n_comp, dp):
    """PCA + OLS with the same metrics as the factor models."""
    X_tr_list, X_te_list = dp["train_X_l_list_clean"], dp["test_X_l_list_clean"]
    bounds = np.cumsum([0] + [x.shape[1] for x in X_tr_list])
    X_tr = torch.cat(X_tr_list, 1).numpy()
    X_te = torch.cat(X_te_list, 1).numpy()
    y_tr = dp["train_y_clean"].reshape(-1).numpy()

    t0 = time.time()
    pca = PCA(n_components=n_comp).fit(X_tr)
    T_tr, T_te = pca.transform(X_tr), pca.transform(X_te)
    pred, lo, hi = ols_prediction_interval(T_tr, y_tr, T_te)
    runtime = time.time() - t0

    # Reconstruction: rank-K PCA (scores x components). Covariance: probabilistic-PCA
    # loadings W = V_K (eigenvalue - noise variance)^1/2, whose factors have unit
    # variance by construction (Sigma_Z = identity).
    V = pca.components_.T
    W = V * np.sqrt(np.clip(pca.explained_variance_ - pca.noise_variance_, 0, None))
    split = lambda A: [A[bounds[l]:bounds[l + 1]] for l in range(len(X_tr_list))]
    return {**factor_method_metrics(T_tr, T_te, split(V), split(W), None, pred, lo, hi, dp),
            "runtime": runtime}


def penalised(method, dp):
    X_tr = torch.cat(dp["train_X_l_list_clean"], 1).numpy()
    X_te = torch.cat(dp["test_X_l_list_clean"], 1).numpy()
    y_tr = dp["train_y_clean"].reshape(-1).numpy()
    y_te = dp["test_y_clean"].reshape(-1).numpy()
    t0 = time.time()
    if method == "ridge":
        fit = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_tr, y_tr)
    else:
        fit = ElasticNetCV(l1_ratio=ENET_L1_RATIOS, n_alphas=50, cv=5, max_iter=5000,
                           n_jobs=-1).fit(X_tr, y_tr)
    pred = fit.predict(X_te)
    return {"outcome_mse": float(np.mean((pred - y_te) ** 2)), "runtime": time.time() - t0}


def main():
    path_base = os.path.join(OUT_DIR, "baselines.csv")
    path_comp = os.path.join(OUT_DIR, "comparators.csv")
    comp = pd.read_csv(path_comp) if os.path.exists(path_comp) else pd.DataFrame()
    base = pd.read_csv(path_base) if os.path.exists(path_base) and len(comp) else pd.DataFrame()
    done = set(zip(comp.get("n", []), comp.get("p", []), comp.get("align_label", []), comp.get("rep", [])))

    for cond in sorted(os.listdir(f"{DATA_ROOT}/train")):
        m = COND_PATTERN.match(cond)
        if not m:
            continue
        n, p, snr_x, label = int(m[1]), int(m[2]), float(m[3]), m[6]
        for rep in range(1, N_REPS + 1):
            fname = f"sim_data_shared_rep{rep}.rds"
            if (n, p, label, rep) in done or not os.path.exists(f"{DATA_ROOT}/test/{cond}/{fname}"):
                continue
            tr = load_rds(f"{DATA_ROOT}/train/{cond}/{fname}")
            te = load_rds(f"{DATA_ROOT}/test/{cond}/{fname}")
            _, _, dp = process_data(tr, te, model_type="shared")
            y_raw = tr["y"].double().reshape(-1)
            mu, sd = y_raw.mean(), y_raw.std()
            key = {"n": n, "p": p, "align_label": label, "rep": rep}

            P = sum(x.shape[1] for x in dp["train_X_l_list_clean"])
            base = pd.concat([base, pd.DataFrame([{
                **key, "P": P, "sqrt_P_over_n": float(np.sqrt(P / n)),
                **oracles(tr, te, snr_x, lambda v: (v - mu) / sd)}])], ignore_index=True)
            rows = [{**key, "method": f"pca{k}_ols", **pca_ols(k, dp)} for k in (5, 15)]
            rows += [{**key, "method": mth, **penalised(mth, dp)} for mth in ("ridge", "elastic_net")]
            comp = pd.concat([comp, pd.DataFrame(rows)], ignore_index=True)
            base.to_csv(path_base, index=False)
            comp.to_csv(path_comp, index=False)
        print(f"{cond}: done", flush=True)

    print(f"\n{len(base)} datasets -> {path_base}\n{len(comp)} method fits -> {path_comp}")


if __name__ == "__main__":
    main()
