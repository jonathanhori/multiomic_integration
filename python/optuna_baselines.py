"""
Comparison methods on the exact CV folds of the Optuna studies (optuna_controlled.py),
so their objectives can be set next to the 40 VIPER-Shared configurations.

For each tuning dataset the folds are rebuilt as in optuna_controlled.py (zero-variance
columns dropped on the full training file, make_cv_folds(n, 3, seed 42), each fold
standardised with its own training statistics). Methods, as in the earlier simulations
(misaligned_baselines.py, misaligned_mofa.py):
    pca5_ols, pca15_ols  PCA of the stacked views + OLS (with intercept)
    mofa                 MOFA2 (15 factors) + OLS without intercept (analysis/tcga_coadread/fit_mofa.R)
    ridge, elastic_net   penalised regression, penalties by 5-fold CV within the fold
Per fold: validation MSE of the standardised outcome, covariance error (PCA:
probabilistic-PCA loadings with Sigma_Z = I; MOFA: weights with its training factors'
second moment; none for ridge / elastic net), wall time of the fit, and the exact
relative excess risk of the fitted linear predictor (risk_metrics).

Wall times: run with the same thread limit as the Optuna studies (OMP_NUM_THREADS=2)
and the three datasets in parallel, as the studies were.

Output: sim/results/integration_shared_controlled/optuna/<study>_methods.csv

usage: OMP_NUM_THREADS=2 python optuna_baselines.py --data controlled:500,3000,1,11 --study c1
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(1, "/Users/jonathanhori/multiomic_integration/python")
warnings.filterwarnings("ignore")


def main():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import ElasticNetCV, RidgeCV
    from data_utils import normalize_tensor_by_col, implied_cov_errors, zero_variance_col_filter
    from run_methods import make_cv_folds
    from misaligned_baselines import RIDGE_ALPHAS, ENET_L1_RATIOS
    from controlled_scoring import fit_mofa
    from optuna_controlled import OUT_DIR, load_training
    import risk_metrics as rm

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--study", required=True)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sim = load_training(args.data)
    filters = [zero_variance_col_filter(X) for X in sim["X_l"]]
    X_raw = [X[:, f] for X, f in zip(sim["X_l"], filters)]
    y_raw = sim["y"].reshape(-1)
    theta = rm.true_params(sim)
    orc = rm.oracle(theta)
    L = len(X_raw)
    rows = []
    for i, (tr, va) in enumerate(make_cv_folds(X_raw[0].shape[0], n_folds=args.folds, seed=args.seed)):
        norms = [normalize_tensor_by_col(X[tr]) for X in X_raw]
        Xtr = [d["data_clean"] for d in norms]
        means, sds = [d["means"] for d in norms], [d["sds"] for d in norms]
        Xva = [normalize_tensor_by_col(X[va], m, s)["data_clean"] for X, m, s in zip(X_raw, means, sds)]
        ym, ysd = y_raw[tr].mean(), y_raw[tr].std()
        ytr = ((y_raw[tr] - ym) / ysd).double().numpy()
        yva = ((y_raw[va] - ym) / ysd).double().numpy()
        A_tr = torch.cat(Xtr, 1).double().numpy()
        A_va = torch.cat(Xva, 1).double().numpy()
        sizes = [x.shape[1] for x in Xtr]
        split = lambda v: list(torch.split(torch.as_tensor(np.asarray(v), dtype=torch.float64), sizes))
        dp = {"SIM_decomp": {"SIM_Lambda_l_list": sim["Lambda_l"]}, "X_l_list_column_filters": filters,
              "X_l_sd_list": sds, "X_l_mean_list": means, "L": L,
              "train_X_l_list_clean": Xtr, "test_X_l_list_clean": Xva,
              "train_y_clean": torch.as_tensor(ytr)}

        def add(method, bs, cs, seconds, cov=None):
            pred = cs + A_va @ np.concatenate([np.asarray(b).reshape(-1) for b in (bs if isinstance(bs, list) else [bs])])
            r = rm.risk_summary(split(np.concatenate([np.asarray(b).reshape(-1) for b in (bs if isinstance(bs, list) else [bs])])),
                                dp, float(ym), float(ysd), theta, orc, float(cs))
            rows.append({"fold": i, "method": method, "outcome_mse": float(np.mean((pred - yva) ** 2)),
                         "cov_frob": cov, "train_time": seconds, "risk_rel": r["risk_rel"]})

        for k in (5, 15):
            t0 = time.time()
            pca = PCA(n_components=k, random_state=0).fit(A_tr)
            D = np.column_stack([np.ones(len(A_tr)), pca.transform(A_tr)])
            coef, *_ = np.linalg.lstsq(D, ytr, rcond=None)
            secs = time.time() - t0
            V = pca.components_.T
            bs = V @ coef[1:]
            W = V * np.sqrt(np.clip(pca.explained_variance_ - pca.noise_variance_, 0, None))
            cov = float(implied_cov_errors(split(W), None, dp).cov_frob.mean())
            add(f"pca{k}_ols", bs, coef[0] - pca.mean_ @ bs, secs, cov)

        t0 = time.time()
        fit = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(A_tr, ytr)
        add("ridge", fit.coef_, float(fit.intercept_), time.time() - t0)
        t0 = time.time()
        fit = ElasticNetCV(l1_ratio=ENET_L1_RATIOS, n_alphas=50, cv=5, max_iter=5000, n_jobs=1).fit(A_tr, ytr)
        add("elastic_net", fit.coef_, float(fit.intercept_), time.time() - t0)

        npz = os.path.join(OUT_DIR, f"{args.study}_mofa_fold{i}.npz")
        if not os.path.exists(npz):
            fit_mofa(dp, npz, k_fit=15, threads=2)
        d = np.load(npz)
        W = [d[f"weights_view{l}"] for l in range(L)]
        Wc = np.vstack(W)
        coef, *_ = np.linalg.lstsq(d["Z_train"], ytr, rcond=None)
        Sigma_Z = d["Z_train"].T @ d["Z_train"] / d["Z_train"].shape[0]
        cov = float(implied_cov_errors([torch.as_tensor(w) for w in W], torch.as_tensor(Sigma_Z), dp).cov_frob.mean())
        add("mofa", Wc @ np.linalg.solve(Wc.T @ Wc, coef), 0.0, float(d["runtime"]), cov)
        print(f"fold {i} done", flush=True)

    out = pd.DataFrame(rows).assign(study=args.study, data=args.data)
    out.to_csv(os.path.join(OUT_DIR, f"{args.study}_methods.csv"), index=False)
    print(out.groupby("method")[["outcome_mse", "cov_frob", "train_time", "risk_rel"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
