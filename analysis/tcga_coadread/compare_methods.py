"""
Compare prediction methods on TCGA COAD/READ: log(1 + TMB) from gene expression
and TSS200 methylation, over several random 80/20 train/test splits.

Every method sees exactly the same data in each split: samples, genes (the most
variable genes chosen on the training samples), imputation and standardisation
(training statistics), as produced by coadread_data.load_coadread(...,
preprocess_on_train_only=True) and run_methods.process_data.

Methods
    shared       supervised shared-factor model (SupMultiviewShared), K factors
    mofa         MOFA2 (unsupervised, K factors) + OLS of y on the factors   [fit_mofa.R]
    pca_ols      PCA of the stacked views (K components) + OLS
    ridge        ridge regression on the stacked views, penalty by 5-fold CV
    elastic_net  elastic net on the stacked views, penalty and mix by 5-fold CV
    mean         predicts the training mean (R² = 0 reference)

Metrics per split and method (outcome on the log(1 + TMB) scale)
    mse, r2 (1 - SSE / SST of the test outcome), coverage and width of 95%
    intervals (shared: posterior predictive; mofa: OLS prediction interval),
    test_rse (relative squared error of the reconstructed test views, factor
    methods only), runtime (seconds).

Outputs (results/)
    method_comparison_splits.csv   one row per split and method
    method_predictions.csv         one row per split, method and test sample
    splits/seed<k>/                the matrices exported for MOFA and its outputs, and
                                   <method>_factors.npz for shared, mofa and pca_ols:
                                   training scores, test scores, loadings per view, and
                                   outcome coefficients (used by the factor comparison)

The script resumes: (split, method) pairs already in method_comparison_splits.csv
are skipped.

usage: python compare_methods.py
"""
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pyro
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV

sys.path.insert(1, os.path.expanduser("~/multiomic_integration/python"))
from coadread_data import load_coadread, make_view_list
from run_methods import process_data, train_model, train_model_locally
from data_utils import summarise_post_samples
from constants import Sites, Params

warnings.filterwarnings("ignore")

HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS     = os.path.join(HERE, "results")
SPLIT_SEEDS = [42, 1, 2, 3, 4]        # 42 is the split used in tcga_coadread.qmd
VIEWS       = ["expr", "meth_tss200"]
N_RNA, N_METH = 2000, 1000
K           = 20
N_POSTERIOR_SAMPLES = 300
METHODS     = ["shared", "mofa", "pca_ols", "ridge", "elastic_net", "mean"]

SHARED_MODEL_CONFIG = {"k": K, "dense_model": False, "outcome_weight": 1.0}
SHARED_TRAIN_CONFIG = {
    "eta": 0.25, "delta": 1e-16, "t": 0.1, "num_particles": 1,
    "minibatch_fraction": 0.15, "min_epochs": 200, "max_epochs": 2000,   # see convergence_check.py
    "convergence_criterion": "elbo_snr", "window": 50, "snr_threshold": 2,
}


def load_split(seed):
    """Standardised train/test views and outcome for one split."""
    data = load_coadread(top_rna_genes=N_RNA, top_meth_genes=N_METH, log1p_tmb=True,
                         train_frac=0.8, random_seed=seed, preprocess_on_train_only=True)
    real = lambda split, y: {"X_l": make_view_list(data, VIEWS, split=split), "y": y,
                             "K": torch.tensor(K), "Z": None}
    train_subset, test_subset, dp = process_data(
        real("train", data["y"]), real("test", data["y_test"]), model_type="shared")
    return data, train_subset, test_subset, dp


def rse(X_list, Xhat_list):
    """Relative squared error over all views."""
    num = sum(float(((X - Xh) ** 2).sum()) for X, Xh in zip(X_list, Xhat_list))
    return num / sum(float((X ** 2).sum()) for X in X_list)


def split_dir(seed):
    d = os.path.join(RESULTS, "splits", f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    return d


def save_factors(seed, method, Z_train, Z_test, loadings, beta):
    """Factor outputs used by the factor comparison in tcga_coadread.qmd."""
    np.savez(os.path.join(split_dir(seed), f"{method}_factors.npz"),
             Z_train=np.asarray(Z_train), Z_test=np.asarray(Z_test), beta=np.asarray(beta),
             **{f"loadings_view{l}": np.asarray(L) for l, L in enumerate(loadings)})


def ols_beta(Z_train, dp):
    """OLS coefficients of the standardised outcome on training scores (no intercept)."""
    y = dp["train_y_clean"].reshape(-1).numpy()
    return np.linalg.lstsq(np.asarray(Z_train), y, rcond=None)[0]


def fit_shared(seed, train_subset, test_subset, dp):
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    t0 = time.time()
    fm, th, _ = train_model(SHARED_MODEL_CONFIG, SHARED_TRAIN_CONFIG, train_subset,
                            model_type="shared", write=False, svd_init=True)
    Lambda = [pyro.param(Params.loc_Lambda_l.format(l=l)).detach().squeeze() for l in range(len(VIEWS))]
    Z_train = pyro.param(Params.loc_Z).detach()
    beta = pyro.param(Params.loc_beta).detach().reshape(-1)
    fm, teh, _ = train_model_locally(fm, SHARED_TRAIN_CONFIG, test_subset, model_type="shared")
    runtime = time.time() - t0
    post = teh.predict(dp["test_X_l_list_clean"], N_POSTERIOR_SAMPLES, [Sites.y_pred, Sites.Z_pred])
    s = summarise_post_samples(post[Sites.y_pred].squeeze().cpu())
    Z_test = post[Sites.Z_pred].mean(0).squeeze().cpu()
    save_factors(seed, "shared", Z_train, Z_test, Lambda, beta)
    xhat = [Z_test @ L.T for L in Lambda]
    return s["mean"], s["q2.5"], s["q97.5"], rse(dp["test_X_l_list_clean"], xhat), runtime


def fit_mofa(seed, dp):
    split_dir_ = split_dir(seed)
    for l, (Xtr, Xte) in enumerate(zip(dp["train_X_l_list_clean"], dp["test_X_l_list_clean"]), start=1):
        np.savetxt(os.path.join(split_dir_, f"x_train_view{l}.csv"), Xtr.numpy(), delimiter=",")
        np.savetxt(os.path.join(split_dir_, f"x_test_view{l}.csv"), Xte.numpy(), delimiter=",")
    np.savetxt(os.path.join(split_dir_, "y_train.csv"), dp["train_y_clean"].reshape(-1).numpy(), delimiter=",")
    subprocess.run(["Rscript", os.path.join(HERE, "fit_mofa.R"), split_dir_, str(len(VIEWS)), str(K)],
                   check=True)
    pred = pd.read_csv(os.path.join(split_dir_, "mofa_predictions.csv"))
    Z_test = pd.read_csv(os.path.join(split_dir_, "mofa_z_test.csv")).values
    W = [pd.read_csv(os.path.join(split_dir_, f"mofa_weights_view{l}.csv")).values
         for l in range(1, len(VIEWS) + 1)]
    xhat = [torch.tensor(Z_test @ w.T, dtype=torch.float32) for w in W]
    runtime = float(open(os.path.join(split_dir_, "mofa_runtime.txt")).read())
    Z_train = pd.read_csv(os.path.join(split_dir_, "mofa_z_train.csv")).values
    save_factors(seed, "mofa", Z_train, Z_test, W, ols_beta(Z_train, dp))
    t = lambda c: torch.tensor(pred[c].values, dtype=torch.float32)
    return t("fit"), t("lwr"), t("upr"), rse(dp["test_X_l_list_clean"], xhat), runtime


def fit_sklearn(method, dp, seed):
    X_tr = torch.cat(dp["train_X_l_list_clean"], 1).numpy()
    X_te = torch.cat(dp["test_X_l_list_clean"], 1).numpy()
    y_tr = dp["train_y_clean"].reshape(-1).numpy()
    t0 = time.time()
    test_rse = np.nan
    if method == "pca_ols":
        pca = PCA(n_components=K).fit(X_tr)
        ols = LinearRegression().fit(pca.transform(X_tr), y_tr)
        p_views = np.cumsum([0] + [x.shape[1] for x in dp["train_X_l_list_clean"]])
        save_factors(seed, "pca_ols", pca.transform(X_tr), pca.transform(X_te),
                     [pca.components_.T[a:b] for a, b in zip(p_views[:-1], p_views[1:])], ols.coef_)
        pred = ols.predict(pca.transform(X_te))
        xhat = pca.inverse_transform(pca.transform(X_te))
        test_rse = float(((X_te - xhat) ** 2).sum() / (X_te ** 2).sum())
    elif method == "ridge":
        pred = RidgeCV(alphas=np.logspace(-1, 5, 40), cv=5).fit(X_tr, y_tr).predict(X_te)
    elif method == "elastic_net":
        pred = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], n_alphas=50, cv=5,
                            max_iter=5000, n_jobs=-1).fit(X_tr, y_tr).predict(X_te)
    elif method == "mean":
        pred = np.full(X_te.shape[0], y_tr.mean())
    runtime = time.time() - t0
    nan = torch.full((X_te.shape[0],), float("nan"))
    return torch.tensor(pred, dtype=torch.float32), nan, nan, test_rse, runtime


def main():
    os.makedirs(RESULTS, exist_ok=True)
    out_metrics = os.path.join(RESULTS, "method_comparison_splits.csv")
    out_preds   = os.path.join(RESULTS, "method_predictions.csv")
    metrics = pd.read_csv(out_metrics) if os.path.exists(out_metrics) else pd.DataFrame()
    preds   = pd.read_csv(out_preds) if os.path.exists(out_preds) else pd.DataFrame()
    done = set(zip(metrics.get("seed", []), metrics.get("method", [])))

    for seed in SPLIT_SEEDS:
        todo = [m for m in METHODS if (seed, m) not in done]
        if not todo:
            continue
        data, train_subset, test_subset, dp = load_split(seed)
        y_mean, y_sd = float(data["y"].mean()), float(data["y"].std())
        y_true = data["y_test"].reshape(-1)
        for method in todo:
            if method == "shared":
                m, lo, hi, test_rse, runtime = fit_shared(seed, train_subset, test_subset, dp)
            elif method == "mofa":
                m, lo, hi, test_rse, runtime = fit_mofa(seed, dp)
            else:
                m, lo, hi, test_rse, runtime = fit_sklearn(method, dp, seed)
            # back to the log(1 + TMB) scale
            m, lo, hi = (v * y_sd + y_mean for v in (m, lo, hi))
            resid = m - y_true
            row = {
                "seed": seed, "method": method, "n_train": len(train_subset), "n_test": len(y_true),
                "mse": float((resid ** 2).mean()),
                "r2": float(1 - (resid ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()),
                "coverage_95": float(((y_true >= lo) & (y_true <= hi)).float().mean()) if not torch.isnan(lo).all() else np.nan,
                "width_95": float((hi - lo).mean()) if not torch.isnan(lo).all() else np.nan,
                "test_rse": test_rse, "runtime": runtime,
            }
            metrics = pd.concat([metrics, pd.DataFrame([row])], ignore_index=True)
            preds = pd.concat([preds, pd.DataFrame({
                "seed": seed, "method": method, "sample_id": data["sample_ids_test"],
                "y_true": y_true.numpy(), "y_pred": m.numpy(), "lwr": lo.numpy(), "upr": hi.numpy()})],
                ignore_index=True)
            metrics.to_csv(out_metrics, index=False)
            preds.to_csv(out_preds, index=False)
            print(f"seed {seed} {method:<12} R²={row['r2']:.3f} MSE={row['mse']:.3f} "
                  f"({runtime:.0f}s)", flush=True)

    print(metrics.groupby("method")[["r2", "mse", "coverage_95", "width_95", "test_rse", "runtime"]]
          .agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
