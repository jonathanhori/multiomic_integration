"""
Compare prediction methods on TCGA COAD/READ: log(1 + TMB) from gene expression
and TSS200 methylation, over several random 80/20 train/test splits.

Every method sees exactly the same data in each split: samples, genes (the most
variable genes chosen on the training samples), imputation and standardisation
(training statistics), as produced by coadread_data.load_coadread(...,
preprocess_on_train_only=True) and run_methods.process_data.

Methods
    shared       supervised shared-factor model (SupMultiviewShared), K factors,
                 trained for 1000 epochs (final setting; see SHARED_TRAIN_CONFIG)
    shared_ma    the same run, scored at the MA w25 p50 stopping rule's stop
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
                                   outcome coefficients (used by the factor comparison);
                                   for shared / shared_ma also the loading sd, E[tau],
                                   the param store (<method>_paramstore.pt) and the
                                   training loss (shared_loss.csv) and the closed-form
                                   test MSE every 25 epochs (shared_curve.csv)

The script resumes: (split, method) pairs already in method_comparison_splits.csv
are skipped.

usage: python compare_methods.py
"""
import copy
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
from run_methods import process_data, train_model, train_model_locally, _minibatch_size
from stopping_rules import ma_patience
from viper_predict import expected_tau, closed_form_scores, guide_moments
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
METHODS     = ["shared", "shared_ma", "mofa", "pca_ols", "ridge", "elastic_net", "mean"]

# Final VIPER-Shared setting from the November 2026 simulation studies
# (sim/exploration/final_benchmark.qmd, config fg_eta0.22_t0.02_mb0.08): low-variance
# guide start (initial guide sd of Lambda 0.01, of Z 0.1), AdagradRMSProp eta 0.22,
# t 0.02, minibatch fraction 0.08, trained for exactly 1000 epochs. Each fit is also
# scored at the epoch where the MA w25 p50 stopping rule (moving average over 25
# epochs, patience 50, no minimum) would have stopped it: method "shared_ma".
# Earlier settings: results/archive_default_guide/, results/archive_lowvar_old_optimizer/.
GUIDE_START = {"init_scale_Lambda": 0.01, "init_scale_Z": 0.1}
SHARED_MODEL_CONFIG = {"k": K, "dense_model": False, "outcome_weight": 1.0, **GUIDE_START}
SHARED_TRAIN_CONFIG = {
    "eta": 0.22, "delta": 1e-16, "t": 0.02, "num_particles": 1,
    "minibatch_fraction": 0.08, "min_epochs": 1000, "max_epochs": 1000,
    "convergence_criterion": "elbo_snr", "window": 50, "snr_threshold": 2,
}
# Local inference of the test scores (loadings frozen): same optimiser, stopping rule
# as before (elbo_snr after at least 200 epochs).
LOCAL_TRAIN_CONFIG = {**SHARED_TRAIN_CONFIG, "min_epochs": 200, "max_epochs": 2000}
MA_RULE = {"window": 25, "patience": 50, "min_epochs": 0}
CHECKPOINT_EVERY = 5     # epochs; the MA fit is the last checkpoint at or before the stop
CURVE_EVERY = 25         # epochs between closed-form test scores (the test MSE curve)


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


def _score_shared(fm, test_subset, dp):
    """Factor outputs and test predictions of the fit in the current param store."""
    L = len(VIEWS)
    store = pyro.get_param_store()
    out = {
        "Lambda": [pyro.param(Params.loc_Lambda_l.format(l=l)).detach().squeeze().clone() for l in range(L)],
        "Lambda_sd": [pyro.param(Params.scale_Lambda_l.format(l=l)).detach().squeeze().clone() for l in range(L)],
        "Z_train": pyro.param(Params.loc_Z).detach().clone(),
        "beta": pyro.param(Params.loc_beta).detach().reshape(-1).clone(),
        "tau": [t.numpy() for t in expected_tau(store, L)],
    }
    fm, teh, _ = train_model_locally(fm, LOCAL_TRAIN_CONFIG, test_subset, model_type="shared")
    post = teh.predict(dp["test_X_l_list_clean"], N_POSTERIOR_SAMPLES, [Sites.y_pred, Sites.Z_pred])
    s = summarise_post_samples(post[Sites.y_pred].squeeze().cpu())
    out["Z_test"] = post[Sites.Z_pred].mean(0).squeeze().cpu()
    out["pred"] = (s["mean"], s["q2.5"], s["q97.5"])
    out["test_rse"] = rse(dp["test_X_l_list_clean"], [out["Z_test"] @ Lv.T for Lv in out["Lambda"]])
    return out


def fit_shared_two_stops(k, seed, train_subset, test_subset, dp, rng_seed=None):
    """Train the shared model for SHARED_TRAIN_CONFIG["max_epochs"] epochs in chunks of
    CHECKPOINT_EVERY, and score two fits of the same run: the fit at the end ("1000")
    and the last checkpoint at or before the epoch where the MA rule would have
    stopped ("ma"; the end fit if the rule never fires). Returns {"1000": ..., "ma": ...}
    (outputs of _score_shared plus epoch, train_seconds, param store state), the
    per-epoch training loss, and the test curve: every CURVE_EVERY epochs, the test
    MSE (standardised outcome) of the closed-form predictions from the guide
    (viper_predict.closed_form_scores; cheaper than local SVI, used only for the curve).
    If training fails with NaN parameters (ValueError) before the end, res["1000"] is
    {"failed_at": epoch}; the MA fit is still scored if the rule fired before the
    failure (otherwise res["ma"] is {"failed_at": epoch} too)."""
    pyro.clear_param_store()
    pyro.set_rng_seed(seed if rng_seed is None else rng_seed)   # rng_seed: replacement run after a failure
    cfg = {**SHARED_TRAIN_CONFIG, "min_epochs": CHECKPOINT_EVERY, "max_epochs": CHECKPOINT_EVERY}
    t0 = time.time()
    fm, th, _ = train_model({**SHARED_MODEL_CONFIG, "k": k}, cfg, train_subset,
                            model_type="shared", write=False, svd_init=True)
    store = pyro.get_param_store()
    y_test = dp["test_y_clean"].reshape(-1).double()
    curve = []

    def score_curve(ep):
        cf = closed_form_scores(guide_moments(store, len(VIEWS)), dp["test_X_l_list_clean"])
        curve.append({"epoch": ep, "test_mse_std": float(((cf["yhat"] - y_test) ** 2).mean()),
                      "loss_per_sample": float(np.mean(fm.loss_history[-CHECKPOINT_EVERY:])) / len(train_subset)})

    score_curve(CHECKPOINT_EVERY)
    epoch, ma_stop, ma_ckpt = CHECKPOINT_EVERY, None, None
    prev = (epoch, copy.deepcopy(store.get_state()), time.time() - t0)
    failed_at = None
    while epoch < SHARED_TRAIN_CONFIG["max_epochs"]:
        try:
            th.do_inference(train_dataset=train_subset, min_epochs=CHECKPOINT_EVERY, epochs=CHECKPOINT_EVERY,
                            minibatch_flag=True, minibatch_size=_minibatch_size(cfg, len(train_subset)),
                            variational_diff_func=np.mean, verbose=False,
                            convergence_criterion=cfg["convergence_criterion"])
        except ValueError as e:              # NaN parameters
            failed_at = epoch
            print(f"  K={k} seed {seed}: training failed after epoch {epoch}: {str(e).splitlines()[0][:100]}", flush=True)
            break
        epoch += CHECKPOINT_EVERY
        if epoch % CURVE_EVERY == 0:
            score_curve(epoch)
        now = (epoch, copy.deepcopy(store.get_state()), time.time() - t0)
        if ma_stop is None:
            ma_stop = ma_patience(fm.loss_history, MA_RULE["min_epochs"], MA_RULE["window"], MA_RULE["patience"])
            if ma_stop is not None:
                ma_ckpt = now if ma_stop >= epoch else prev
        prev = now
    end = prev if failed_at is None else None
    if ma_ckpt is None:
        ma_ckpt = end
    loss = np.asarray(fm.loss_history, float) / len(train_subset)
    res = {}
    for name, ckpt in (("1000", end), ("ma", ma_ckpt)):
        if ckpt is None:
            res[name] = {"failed_at": failed_at, "ma_stop": ma_stop}
            continue
        ep, state, secs = ckpt
        pyro.clear_param_store()
        store.set_state(copy.deepcopy(state))
        res[name] = {**_score_shared(fm, test_subset, dp), "epoch": ep, "ma_stop": ma_stop,
                     "train_seconds": secs, "state": state}
    return res, loss, pd.DataFrame(curve)


def save_shared(d, method, r):
    """Factor outputs of one shared fit (both checkpoints use the same layout), with
    the posterior sd of the loadings (scale_view<l>), E_q[tau] per view (tau_view<l>)
    and the param store."""
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, f"{method}_factors.npz"),
             Z_train=r["Z_train"].numpy(), Z_test=r["Z_test"].numpy(), beta=r["beta"].numpy(),
             epoch=r["epoch"],
             **{f"loadings_view{l}": Lv.numpy() for l, Lv in enumerate(r["Lambda"])},
             **{f"scale_view{l}": sd.numpy() for l, sd in enumerate(r["Lambda_sd"])},
             **{f"tau_view{l}": t for l, t in enumerate(r["tau"])})
    torch.save(r["state"], os.path.join(d, f"{method}_paramstore.pt"))


def fit_shared(seed, train_subset, test_subset, dp):
    """Both checkpoints of one K = K run for compare_methods: returns
    {"shared": (m, lo, hi, test_rse, runtime, epoch), "shared_ma": (...)}."""
    res, loss, curve = fit_shared_two_stops(K, seed, train_subset, test_subset, dp)
    d = split_dir(seed)
    pd.DataFrame({"epoch": np.arange(1, len(loss) + 1), "loss_per_sample": loss}).to_csv(
        os.path.join(d, "shared_loss.csv"), index=False)
    curve.assign(ma_stop=res["ma"]["ma_stop"]).to_csv(os.path.join(d, "shared_curve.csv"), index=False)
    out = {}
    for method, name in (("shared", "1000"), ("shared_ma", "ma")):
        r = res[name]
        save_shared(d, method, r)
        out[method] = (*r["pred"], r["test_rse"], r["train_seconds"], r["epoch"])
    return out


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
        shared_out = None
        for method in todo:
            epochs = np.nan
            if method in ("shared", "shared_ma"):
                if shared_out is None:          # one run gives both checkpoints
                    shared_out = fit_shared(seed, train_subset, test_subset, dp)
                m, lo, hi, test_rse, runtime, epochs = shared_out[method]
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
                "test_rse": test_rse, "runtime": runtime, "epochs": epochs,
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
