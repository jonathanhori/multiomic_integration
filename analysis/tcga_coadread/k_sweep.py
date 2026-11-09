"""
Number-of-factors sweep on TCGA COAD/READ: refit the factor methods with K = 20,
50 and 100 on the five splits of compare_methods.py and score their predictions.

The question: the factor methods tie each other at R² ≈ 0.44 with K = 20, below
ridge (0.63), and PCA + OLS approaches ridge only with 150-300 components. Does
fitting more factors close the gap for the shared model, and how many does its
MGP prior keep active?

Data, preprocessing and training settings are those of compare_methods.py; only
K changes (outcome_weight stays 1). Methods: shared (1000 epochs), shared_ma (the
same run at the MA w25 p50 stop), mofa, pca_ols. The K = 20 shared fits are those
of compare_methods.py (copied, not refitted), so run compare_methods.py first.

Outputs (results/k_sweep/)
    k_sweep_splits.csv               one row per split, method and K: mse, r2,
                                     coverage_95, width_95, test_rse, runtime,
                                     epochs (shared), active factors
    seed<s>/k<K>/<method>_factors.npz  training and test scores, loadings per view,
                                     outcome coefficients; for the shared model also
                                     the posterior sd of the loadings (scale_view<l>),
                                     E_q[tau] per view (tau_view<l>), the param store
                                     (<method>_paramstore.pt), shared_loss.csv and
                                     shared_curve.csv (closed-form test MSE every 25 epochs)

Resumes: (split, method, K) rows already in k_sweep_splits.csv are skipped.

usage: python k_sweep.py
"""
import os
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pyro
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, HERE)
import compare_methods as cm
from run_methods import train_model, train_model_locally
from data_utils import summarise_post_samples
from constants import Sites, Params

warnings.filterwarnings("ignore")

OUT     = os.path.join(cm.RESULTS, "k_sweep")
K_GRID  = [int(k) for k in os.environ.get("K_SWEEP_GRID", "20,50,100").split(",")]
METHODS = ["shared", "shared_ma", "mofa", "pca_ols"]


def fit_dir(seed, k):
    d = os.path.join(OUT, f"seed{seed}", f"k{k}")
    os.makedirs(d, exist_ok=True)
    return d


def save_factors(d, method, Z_train, Z_test, loadings, beta, **extra):
    np.savez(os.path.join(d, f"{method}_factors.npz"),
             Z_train=np.asarray(Z_train), Z_test=np.asarray(Z_test), beta=np.asarray(beta),
             **{f"loadings_view{l}": np.asarray(L) for l, L in enumerate(loadings)},
             **{k: np.asarray(v) for k, v in extra.items()})


def n_active(Z_train, loadings):
    """Factors explaining more than 1% of the data (as in tcga_coadread.qmd)."""
    Z = np.asarray(Z_train)
    n, P = Z.shape[0], sum(np.asarray(L).shape[0] for L in loadings)
    energy = np.sum(Z ** 2, 0) * sum(np.sum(np.asarray(L) ** 2, 0) for L in loadings)
    return int(np.sum(energy / (n * P) > 0.01))


def fit_shared(k, seed, train_subset, test_subset, dp):
    """Both checkpoints (1000 epochs and the MA w25 p50 stop) of one shared run:
    {"shared": res, "shared_ma": res}. K = cm.K reuses compare_methods.py's run."""
    if k == cm.K:
        return reuse_compare_methods(seed, dp)
    rng = int(os.environ["K_SWEEP_RNG"]) if "K_SWEEP_RNG" in os.environ else None   # replacement run
    res, loss, curve = cm.fit_shared_two_stops(k, seed, train_subset, test_subset, dp, rng_seed=rng)
    d = fit_dir(seed, k)
    pd.DataFrame({"epoch": np.arange(1, len(loss) + 1), "loss_per_sample": loss}).to_csv(
        os.path.join(d, "shared_loss.csv"), index=False)
    curve.assign(ma_stop=res["ma"]["ma_stop"]).to_csv(os.path.join(d, "shared_curve.csv"), index=False)
    out = {}
    for method, name in (("shared", "1000"), ("shared_ma", "ma")):
        r = res[name]
        if "failed_at" in r:
            out[method] = {"failed_at": r["failed_at"], "ma_stop": r["ma_stop"]}
            continue
        cm.save_shared(d, method, r)
        out[method] = (*r["pred"], r["test_rse"], r["train_seconds"], r["epoch"],
                       n_active(r["Z_train"], r["Lambda"]))
    return out


def reuse_compare_methods(seed, dp):
    """The K = cm.K shared fits of compare_methods.py (same data, settings and seed):
    copy their files and read their predictions on the standardised outcome scale."""
    src, d = cm.split_dir(seed), fit_dir(seed, cm.K)
    metrics = pd.read_csv(os.path.join(cm.RESULTS, "method_comparison_splits.csv"))
    preds = pd.read_csv(os.path.join(cm.RESULTS, "method_predictions.csv"))
    out = {}
    for method in ("shared", "shared_ma"):
        for f in (f"{method}_factors.npz", f"{method}_paramstore.pt"):
            shutil.copy(os.path.join(src, f), os.path.join(d, f))
        f = np.load(os.path.join(src, f"{method}_factors.npz"))
        m = metrics[(metrics.seed == seed) & (metrics.method == method)].iloc[0]
        p = preds[(preds.seed == seed) & (preds.method == method)]
        out[method] = {"orig_scale": (torch.tensor(p.y_pred.values), torch.tensor(p.lwr.values),
                                      torch.tensor(p.upr.values)),
                       "test_rse": m.test_rse, "runtime": m.runtime, "epochs": m.epochs,
                       "active": n_active(f["Z_train"], [f[f"loadings_view{l}"] for l in range(len(cm.VIEWS))])}
    for f in ("shared_loss.csv", "shared_curve.csv"):
        shutil.copy(os.path.join(src, f), os.path.join(d, f))
    return out


def fit_mofa(k, seed, dp):
    d = fit_dir(seed, k)
    for l, (Xtr, Xte) in enumerate(zip(dp["train_X_l_list_clean"], dp["test_X_l_list_clean"]), start=1):
        np.savetxt(os.path.join(d, f"x_train_view{l}.csv"), Xtr.numpy(), delimiter=",")
        np.savetxt(os.path.join(d, f"x_test_view{l}.csv"), Xte.numpy(), delimiter=",")
    np.savetxt(os.path.join(d, "y_train.csv"), dp["train_y_clean"].reshape(-1).numpy(), delimiter=",")
    subprocess.run(["Rscript", os.path.join(HERE, "fit_mofa.R"), d, str(len(cm.VIEWS)), str(k)],
                   check=True, stdout=subprocess.DEVNULL)
    read = lambda f: pd.read_csv(os.path.join(d, f)).values
    pred = pd.read_csv(os.path.join(d, "mofa_predictions.csv"))
    W = [read(f"mofa_weights_view{l}.csv") for l in range(1, len(cm.VIEWS) + 1)]
    Z_train, Z_test = read("mofa_z_train.csv"), read("mofa_z_test.csv")
    runtime = float(open(os.path.join(d, "mofa_runtime.txt")).read())
    for f in os.listdir(d):                      # the exported matrices are large; keep only outputs
        if f.startswith("x_"):
            os.remove(os.path.join(d, f))
    save_factors(d, "mofa", Z_train, Z_test, W, cm.ols_beta(Z_train, dp))
    xhat = [torch.tensor(Z_test @ w.T, dtype=torch.float32) for w in W]
    t = lambda c: torch.tensor(pred[c].values, dtype=torch.float32)
    return (t("fit"), t("lwr"), t("upr"), cm.rse(dp["test_X_l_list_clean"], xhat), runtime,
            np.nan, n_active(Z_train, W))


def fit_pca(k, seed, dp):
    X_tr = torch.cat(dp["train_X_l_list_clean"], 1).numpy()
    X_te = torch.cat(dp["test_X_l_list_clean"], 1).numpy()
    y_tr = dp["train_y_clean"].reshape(-1).numpy()
    t0 = time.time()
    pca = PCA(n_components=k).fit(X_tr)
    ols = LinearRegression().fit(pca.transform(X_tr), y_tr)
    runtime = time.time() - t0
    p_views = np.cumsum([0] + [x.shape[1] for x in dp["train_X_l_list_clean"]])
    loadings = [pca.components_.T[a:b] for a, b in zip(p_views[:-1], p_views[1:])]
    save_factors(fit_dir(seed, k), "pca_ols", pca.transform(X_tr), pca.transform(X_te), loadings, ols.coef_)
    xhat = pca.inverse_transform(pca.transform(X_te))
    nan = torch.full((X_te.shape[0],), float("nan"))
    return (torch.tensor(ols.predict(pca.transform(X_te)), dtype=torch.float32), nan, nan,
            float(((X_te - xhat) ** 2).sum() / (X_te ** 2).sum()), runtime, np.nan, k)


def main():
    os.makedirs(OUT, exist_ok=True)
    # K_SWEEP_SEEDS / K_SWEEP_CSV: run a subset of splits into a separate file (parallel runs;
    # merge into k_sweep_splits.csv afterwards)
    out_csv = os.path.join(OUT, os.environ.get("K_SWEEP_CSV", "k_sweep_splits.csv"))
    rows = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
    done = set(zip(rows.get("seed", []), rows.get("method", []), rows.get("k", [])))

    seeds = [int(x) for x in os.environ["K_SWEEP_SEEDS"].split(",")] if "K_SWEEP_SEEDS" in os.environ else cm.SPLIT_SEEDS
    for seed in seeds:
        todo = [(m, k) for k in K_GRID for m in METHODS if (seed, m, k) not in done]
        if not todo:
            continue
        data, train_subset, test_subset, dp = cm.load_split(seed)
        y_mean, y_sd = float(data["y"].mean()), float(data["y"].std())
        y_true = data["y_test"].reshape(-1)
        shared_out = {}
        for method, k in todo:
            orig_scale = None
            if method in ("shared", "shared_ma"):
                if k not in shared_out:                   # one run gives both checkpoints
                    try:
                        shared_out[k] = fit_shared(k, seed, train_subset, test_subset, dp)
                    except ValueError as e:      # NaN parameters: record the failure and move on
                        print(f"seed {seed} shared   K={k:<4} FAILED: {str(e).splitlines()[0][:120]}", flush=True)
                        shared_out[k] = None
                if shared_out[k] is None:
                    rows = pd.concat([rows, pd.DataFrame([{"seed": seed, "method": method, "k": k,
                                                           "failed": True}])], ignore_index=True)
                    rows.to_csv(out_csv, index=False)
                    continue
                res = shared_out[k][method]
                if isinstance(res, dict) and "failed_at" in res:   # NaN before this checkpoint
                    rows = pd.concat([rows, pd.DataFrame([{"seed": seed, "method": method, "k": k, "failed": True,
                                                           "failed_at": res["failed_at"], "ma_stop": res["ma_stop"]}])],
                                     ignore_index=True)
                    rows.to_csv(out_csv, index=False)
                    print(f"seed {seed} {method:<8} K={k:<4} FAILED after epoch {res['failed_at']}", flush=True)
                    continue
                if isinstance(res, dict):                # reused from compare_methods.py
                    orig_scale = res["orig_scale"]
                    res = (None, None, None, res["test_rse"], res["runtime"], res["epochs"], res["active"])
            elif method == "mofa":
                res = fit_mofa(k, seed, dp)
            else:
                res = fit_pca(k, seed, dp)
            m, lo, hi, test_rse, runtime, epochs, active = res
            if orig_scale is not None:
                m, lo, hi = (v.float() for v in orig_scale)
            else:
                m, lo, hi = (v * y_sd + y_mean for v in (m, lo, hi))   # back to the log(1 + TMB) scale
            resid = m - y_true
            has_int = not torch.isnan(lo).all()
            row = {"seed": seed, "method": method, "k": k,
                   "mse": float((resid ** 2).mean()),
                   "r2": float(1 - (resid ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()),
                   "coverage_95": float(((y_true >= lo) & (y_true <= hi)).float().mean()) if has_int else np.nan,
                   "width_95": float((hi - lo).mean()) if has_int else np.nan,
                   "test_rse": test_rse, "runtime": runtime, "epochs": epochs, "active_factors": active,
                   "rng_seed": int(os.environ.get("K_SWEEP_RNG", seed)) if method.startswith("shared") else np.nan}
            rows = pd.concat([rows, pd.DataFrame([row])], ignore_index=True)
            rows.to_csv(out_csv, index=False)
            print(f"seed {seed} {method:<8} K={k:<4} R²={row['r2']:.3f} active={active} "
                  f"epochs={epochs} ({runtime:.0f}s)", flush=True)

    print(rows[rows.get("failed", pd.Series(False, index=rows.index)) != True].groupby(["method", "k"])[["r2", "mse", "coverage_95", "test_rse", "active_factors", "runtime"]]
          .agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
