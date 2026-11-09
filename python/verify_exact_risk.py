"""
Check that the exact risk R(b, c) (risk_metrics.exact_risk) equals the expected
squared prediction error of a linear predictor on new samples, and how it maps to
the pipeline's outcome_mse.

Part 1 (Monte Carlo, raw scale). For a few controlled-strength datasets and
methods (oracle b*, VIPER-Shared closed form, PCA5 + OLS, ridge), draw N_MC fresh
samples from the replicate's true parameters (z ~ N(0, I), x_l = Lambda_l z + eps_l
with eps ~ N(0, Psi), y = beta^T z + theta^T eps + e) and compare the empirical mean
squared error of y_hat = c + b^T x with R(b, c); report the Monte Carlo standard
error and the z-score (R - MSE_mc) / se.

Part 2 (pipeline scale). outcome_mse is computed on the standardised outcome
(y - ybar) / sd_y with the training sd_y, so E[outcome_mse] = R / sd_y^2. Over all
Phase 2 rows (controlled_scores.csv) compare outcome_mse (100 test samples) with
R / sd_y^2: the mean ratio should be ~1 and the z-scores, using the sampling
variance of a 100-sample mean of squared Gaussian errors (2 (R / sd_y^2)^2 / 100),
roughly standard normal.

Outputs: sim/results/integration_shared_controlled/exact_risk_check_mc.csv and
exact_risk_check_pipeline.csv

usage: python verify_exact_risk.py [--n-mc 200000] [--workers 4]
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(1, "/Users/jonathanhori/multiomic_integration/python")
warnings.filterwarnings("ignore")

RESULTS = os.path.expanduser("~/multiomic_integration/sim/results/integration_shared_controlled")
MC_DATASETS = [(100, 600, 1.0, 1), (100, 3000, 10.0, 1), (200, 600, 2.0, 1),
               (200, 3000, 1.0, 1), (500, 600, 10.0, 1), (500, 3000, 2.0, 1)]


def draw(theta, N, gen):
    """N fresh samples (raw scale) from the true parameters."""
    Lam, psi, beta = theta["Lambda"], theta["psi"], theta["beta"]
    K = beta.numel()
    z = torch.randn(N, K, generator=gen, dtype=torch.float64)
    eps = [torch.randn(N, L.shape[0], generator=gen, dtype=torch.float64) * p.sqrt() for L, p in zip(Lam, psi)]
    X = [z @ L.T + e for L, e in zip(Lam, eps)]
    y = z @ beta + torch.randn(N, generator=gen, dtype=torch.float64) * np.sqrt(theta["sigma2_y"])
    if theta.get("theta") is not None:
        y = y + sum(e @ t for e, t in zip(eps, theta["theta"]))
    return X, y


def mc_one(n, P, c, rep, n_mc):
    import pyro
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    import risk_metrics as rm
    from run_methods import process_data
    from misaligned_baselines import RIDGE_ALPHAS
    from controlled_viper_diagnosis import cond_name, load_dataset
    from viper_predict import closed_form_scores, guide_moments

    tr, te = load_dataset(n, P, c, rep)
    _, _, dp = process_data(tr, te, model_type="shared")
    theta = rm.true_params(tr)
    orc = rm.oracle(theta)
    y_raw = tr["y"].double().reshape(-1)
    ym, ysd = float(y_raw.mean()), float(y_raw.std())
    X_tr = torch.cat(dp["train_X_l_list_clean"], 1).double().numpy()
    y_tr = dp["train_y_clean"].double().reshape(-1).numpy()
    sizes = [x.shape[1] for x in dp["train_X_l_list_clean"]]
    split = lambda v: list(torch.split(torch.as_tensor(v, dtype=torch.float64), sizes))

    preds = {"oracle": (orc["b_star"], 0.0)}
    stem = os.path.join(RESULTS, "models", f"shared_ctrl_{cond_name(n, P, c)}_alpha1_rep{rep}_paramstore.pth")
    pyro.clear_param_store()
    pyro.get_param_store().set_state(torch.load(stem, weights_only=False, map_location="cpu"))
    cf = closed_form_scores(guide_moments(pyro.get_param_store(), dp["L"]), dp["test_X_l_list_clean"])
    preds["viper_k15"] = rm.to_raw_scale(cf["coef"], dp, ym, ysd)
    pca = PCA(n_components=5, random_state=0).fit(X_tr)
    D = np.column_stack([np.ones(len(X_tr)), pca.transform(X_tr)])
    coef, *_ = np.linalg.lstsq(D, y_tr, rcond=None)
    bs = pca.components_.T @ coef[1:]
    preds["pca5_ols"] = rm.to_raw_scale(split(bs), dp, ym, ysd, coef[0] - pca.mean_ @ bs)
    fit = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_tr, y_tr)
    preds["ridge"] = rm.to_raw_scale(split(fit.coef_), dp, ym, ysd, float(fit.intercept_))

    gen = torch.Generator().manual_seed(1000 + rep)
    rows = []
    sq = {m: [] for m in preds}
    for _ in range(n_mc // 20000):                    # in blocks to bound memory
        X, y = draw(theta, 20000, gen)
        for m, (b, c0) in preds.items():
            yhat = c0 + sum(x @ bl for x, bl in zip(X, b))
            sq[m].append(((y - yhat) ** 2).numpy())
    for m, (b, c0) in preds.items():
        s = np.concatenate(sq[m])
        R = rm.exact_risk(b, c0, theta, orc)["risk"]
        se = s.std() / np.sqrt(len(s))
        rows.append({"n": n, "P_grid": P, "c": c, "rep": rep, "method": m, "R_formula": R,
                     "mse_mc": s.mean(), "se_mc": se, "z": (R - s.mean()) / se,
                     "rel_diff": (R - s.mean()) / s.mean(), "n_mc": len(s)})
    return rows


def train_sd(n, P, c, rep):
    from controlled_viper_diagnosis import load_dataset
    tr, _ = load_dataset(n, P, c, rep)
    return {"n": n, "P_grid": P, "c": c, "rep": rep, "sd_y": float(tr["y"].double().reshape(-1).std())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mc", type=int, default=200000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    from joblib import Parallel, delayed

    mc = Parallel(n_jobs=args.workers, backend="loky")(delayed(mc_one)(*d, args.n_mc) for d in MC_DATASETS)
    mc = pd.DataFrame([r for rows in mc for r in rows])
    mc.to_csv(os.path.join(RESULTS, "exact_risk_check_mc.csv"), index=False)
    pd.set_option("display.width", 200)
    print("Part 1: exact risk vs Monte Carlo MSE on fresh samples (raw scale)")
    print(mc.round(4).to_string(index=False))

    sc = pd.read_csv(os.path.join(RESULTS, "controlled_scores.csv"))
    keys = sc[["n", "P_grid", "c", "rep"]].drop_duplicates().itertuples(index=False)
    sds = pd.DataFrame(Parallel(n_jobs=args.workers, backend="loky")(delayed(train_sd)(*k) for k in keys))
    d = sc.merge(sds, on=["n", "P_grid", "c", "rep"])
    d = d[d.method != "oracle_known_z"].copy()
    d["R_std"] = d.risk / d.sd_y ** 2
    d["ratio"] = d.outcome_mse / d.R_std
    d["z"] = (d.outcome_mse - d.R_std) / np.sqrt(2 * d.R_std ** 2 / 100)
    d.to_csv(os.path.join(RESULTS, "exact_risk_check_pipeline.csv"), index=False)
    print("\nPart 2: test outcome_mse (100 samples, standardised y) vs R / sd_y^2")
    print(d.groupby("method").agg(rows=("z", "size"), mean_ratio=("ratio", "mean"),
                                  median_ratio=("ratio", "median"), mean_z=("z", "mean"),
                                  sd_z=("z", "std"), corr=("outcome_mse", lambda s: np.corrcoef(s, d.loc[s.index, "R_std"])[0, 1]))
          .round(3).to_string())


if __name__ == "__main__":
    main()
