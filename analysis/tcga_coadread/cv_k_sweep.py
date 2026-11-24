"""
5-fold cross-validation of the prediction methods on TCGA COAD/READ, with a sweep
over the shared model's maximum number of factors K_max (and the K of the other
factor methods). Read in tcga_coadread_cv.qmd.

Folds: the 530 aligned samples shuffled with seed CV_SEED and cut into 5 folds
(coadread_data.load_coadread(n_folds=5, fold=f)); no stratification. Within each
fold, gene selection, imputation and standardisation use the training samples
only, so the test fold never influences preprocessing or fitting.

Methods (data, settings and outputs as in compare_methods.py, outcome_weight 1)
    shared       VIPER-Shared with the package defaults (run_methods.with_defaults,
                 November 2026: eta 0.3, t 0.005, minibatch 0.1, avg_param_change
                 stopping rule, SVD init, guide start 0.01 / 0.1), K_max in K_GRID;
                 the fit at the rule's stop is scored (test scores by local inference
                 with the same package defaults). The fit is also scored every 25
                 epochs in closed form (checkpoint_recorder; the trajectory is unchanged):
                 fold<f>/k<K>/shared_default_checkpoints.csv; the param-store state every
                 25 epochs, the rule's block means and beta per epoch are saved to
                 fold<f>/k<K>/shared_default_states.pt (local inference and any stopping
                 rule can be evaluated later from them: tcga_rule_replay.py)
    shared_ma    VIPER-Shared (SupMultiviewShared, MGP prior), K_max in K_GRID, final
                 setting, scored at the MA w25 p50 stop. Training continues EXTRA
                 epochs past the stop (capped at 1000) only to draw the test-MSE curve;
                 the scored fit is the one at the stop.
    mofa         MOFA2 (K factors) + OLS                                [fit_mofa.R]
    pca_ols      PCA of the stacked views (K components) + OLS
    ridge, elastic_net, mean   K-independent, once per fold

The K sweep is a sensitivity analysis (how results change with K_max); K is not
chosen on the test folds.

Outputs (results/cv/)
    rows/<method>_fold<f>_k<K>.csv     one metrics row per fit (resumable: existing
                                       files are skipped)
    preds/<method>_fold<f>_k<K>.csv    test-fold predictions (log(1 + TMB) scale)
    fold<f>/                           train/test sample ids, gene lists
    fold<f>/k<K>/                      <method>_factors.npz; for shared_ma also the
                                       param store, shared_loss.csv, shared_curve.csv
    cv_results.csv, cv_predictions.csv all rows / predictions (written at the end)

usage: OMP_NUM_THREADS=1 python cv_k_sweep.py [--k 20 50 100] [--folds 0 1 2 3 4]
                                              [--workers 4] [--smoke]
"""
import argparse
import copy
import glob
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, HERE)
warnings.filterwarnings("ignore")

OUT = os.path.join(HERE, "results", "cv")
N_FOLDS, CV_SEED = 5, 42
K_GRID = [20, 50, 100]
EXTRA = 150            # epochs past the MA stop, for the curve only
MAX_EPOCHS = 1000


def load_fold(fold):
    import torch
    import compare_methods as cm
    from coadread_data import load_coadread, make_view_list
    from run_methods import process_data
    data = load_coadread(top_rna_genes=cm.N_RNA, top_meth_genes=cm.N_METH, log1p_tmb=True,
                         random_seed=CV_SEED, n_folds=N_FOLDS, fold=fold,
                         preprocess_on_train_only=True)
    real = lambda split, y: {"X_l": make_view_list(data, cm.VIEWS, split=split), "y": y,
                             "K": torch.tensor(cm.K), "Z": None}
    train_subset, test_subset, dp = process_data(
        real("train", data["y"]), real("test", data["y_test"]), model_type="shared")
    d = os.path.join(OUT, f"fold{fold}")
    os.makedirs(d, exist_ok=True)
    for name, ids in (("train_samples.txt", data["sample_ids_train"]), ("test_samples.txt", data["sample_ids_test"])):
        if not os.path.exists(os.path.join(d, name)):
            open(os.path.join(d, name), "w").write("\n".join(ids))
    for l, v in enumerate(cm.VIEWS):
        key = {"expr": "gene_names_expr", "meth_tss200": "gene_names_tss200"}[v]
        path = os.path.join(d, f"genes_view{l}.txt")
        if not os.path.exists(path):
            open(path, "w").write("\n".join(data[key]))
    return data, train_subset, test_subset, dp


def fit_shared(k, fold, train_subset, test_subset, dp, d):
    """Train in chunks of cm.CHECKPOINT_EVERY epochs until EXTRA epochs past the MA
    stop (or MAX_EPOCHS); score the checkpoint at the stop (as compare_methods.py)."""
    import pyro
    import compare_methods as cm
    from k_sweep import n_active
    from run_methods import train_model, _minibatch_size
    from stopping_rules import ma_patience
    from viper_predict import closed_form_scores, guide_moments
    E = cm.CHECKPOINT_EVERY
    pyro.clear_param_store()
    pyro.set_rng_seed(CV_SEED + fold)
    cfg = {**cm.SHARED_TRAIN_CONFIG, "min_epochs": E, "max_epochs": E}
    t0 = time.time()
    fm, th, _ = train_model({**cm.SHARED_MODEL_CONFIG, "k": k}, cfg, train_subset,
                            model_type="shared", write=False, svd_init=True)
    store = pyro.get_param_store()
    y_test = dp["test_y_clean"].reshape(-1).double()
    curve = []

    def score_curve(ep):
        cf = closed_form_scores(guide_moments(store, len(cm.VIEWS)), dp["test_X_l_list_clean"])
        curve.append({"epoch": ep, "test_mse_std": float(((cf["yhat"] - y_test) ** 2).mean()),
                      "loss_per_sample": float(np.mean(fm.loss_history[-E:])) / len(train_subset)})

    score_curve(E)
    epoch, ma_stop, ma_ckpt, failed_at = E, None, None, None
    prev = (epoch, copy.deepcopy(store.get_state()), time.time() - t0)
    while epoch < (MAX_EPOCHS if ma_stop is None else min(MAX_EPOCHS, ma_stop + EXTRA)):
        try:
            th.do_inference(train_dataset=train_subset, min_epochs=E, epochs=E, minibatch_flag=True,
                            minibatch_size=_minibatch_size(cfg, len(train_subset)),
                            variational_diff_func=np.mean, verbose=False,
                            convergence_criterion=cfg["convergence_criterion"])
        except ValueError as e:                     # NaN parameters
            failed_at = epoch
            print(f"  fold {fold} K={k}: failed after epoch {epoch}: {str(e).splitlines()[0][:100]}", flush=True)
            break
        epoch += E
        if epoch % cm.CURVE_EVERY == 0:
            score_curve(epoch)
        now = (epoch, copy.deepcopy(store.get_state()), time.time() - t0)
        if ma_stop is None:
            ma_stop = ma_patience(fm.loss_history, cm.MA_RULE["min_epochs"], cm.MA_RULE["window"],
                                  cm.MA_RULE["patience"])
            if ma_stop is not None:
                ma_ckpt = now if ma_stop >= epoch else prev
        prev = now
    if ma_ckpt is None and failed_at is None:      # the rule never fired: the fit at MAX_EPOCHS
        ma_ckpt = prev
    loss = np.asarray(fm.loss_history, float) / len(train_subset)
    pd.DataFrame({"epoch": np.arange(1, len(loss) + 1), "loss_per_sample": loss}).to_csv(
        os.path.join(d, "shared_loss.csv"), index=False)
    pd.DataFrame(curve).assign(ma_stop=ma_stop).to_csv(os.path.join(d, "shared_curve.csv"), index=False)
    if ma_ckpt is None:
        return {"failed_at": failed_at, "ma_stop": ma_stop}
    ep, state, secs = ma_ckpt
    pyro.clear_param_store()
    store.set_state(copy.deepcopy(state))
    r = {**cm._score_shared(fm, test_subset, dp), "epoch": ep, "state": state}
    cm.save_shared(d, "shared_ma", r)
    return (*r["pred"], r["test_rse"], secs, ep, n_active(r["Z_train"], r["Lambda"]), ma_stop)


def fit_shared_default(k, fold, train_subset, test_subset, dp, d):
    """VIPER-Shared with the package defaults; scored at the default rule's stop, with
    local inference of the test scores also at the package defaults. Closed-form test
    scores every 25 epochs go to shared_default_checkpoints.csv."""
    import pyro
    import torch
    import compare_methods as cm
    from k_sweep import n_active
    from run_methods import train_model
    from checkpoint_recorder import record_checkpoints
    from viper_predict import closed_form_scores, guide_moments
    from constants import Params
    L = len(cm.VIEWS)
    Xtr = [x.double() for x in dp["train_X_l_list_clean"]]
    tot = sum(float((x ** 2).sum()) for x in Xtr)
    ytr = dp["train_y_clean"].double().reshape(-1)
    yte = dp["test_y_clean"].double().reshape(-1)

    def score(store, epoch):
        Z = store[Params.loc_Z].detach().double()
        Lam = [store[Params.loc_Lambda_l.format(l=l)].detach().double().squeeze() for l in range(L)]
        pve = (sum((Z ** 2).mean(0) * (M ** 2).sum(0) for M in Lam) * Z.shape[0] / tot).numpy()
        beta = store[Params.loc_beta].detach().double().reshape(-1)
        cf = closed_form_scores(guide_moments(store, L), dp["test_X_l_list_clean"])
        mu, sd = cf["yhat"], cf["pred_var"].sqrt()
        return {"test_mse_std": float(((mu - yte) ** 2).mean()),
                "cov95": float(((yte - mu).abs() <= 1.959964 * sd).double().mean()),
                "width95_std": float((2 * 1.959964 * sd).mean()),
                "n_active": int((pve > 0.01).sum()),
                "train_r2": float(1 - ((Z @ beta - ytr) ** 2).sum() / (ytr ** 2).sum()),
                **{f"pve{j}": float(v) for j, v in enumerate(pve)},
                **{f"beta{j}": float(b) for j, b in enumerate(beta)}}

    rec = record_checkpoints(25, score)
    pyro.clear_param_store()
    pyro.set_rng_seed(CV_SEED + fold)
    t0 = time.time()
    try:
        fm, th, _ = train_model({"k": k, "dense_model": False, "outcome_weight": 1.0}, {}, train_subset,
                                model_type="shared", write=False)
    finally:
        rec.restore()
    secs = time.time() - t0
    rows = rec.rows
    rec.save(os.path.join(d, "shared_default_states.pt"), loss=np.asarray(fm.loss_history, float),
             stop_epoch=fm.total_epochs, n_train=len(train_subset))
    store = pyro.get_param_store()
    state = copy.deepcopy(store.get_state())
    pd.DataFrame(rows).assign(stop_epoch=fm.total_epochs, fit_seconds=secs).to_csv(
        os.path.join(d, "shared_default_checkpoints.csv"), index=False)
    loss = np.asarray(fm.loss_history, float) / len(train_subset)
    pd.DataFrame({"epoch": np.arange(1, len(loss) + 1), "loss_per_sample": loss}).to_csv(
        os.path.join(d, "shared_default_loss.csv"), index=False)
    pd.DataFrame([{"epoch": e, **st} for e, st in th.avg_tracker.history]).to_csv(
        os.path.join(d, "shared_default_rule.csv"), index=False)
    r = {**cm._score_shared(fm, test_subset, dp, local_config={}), "epoch": fm.total_epochs, "state": state}
    cm.save_shared(d, "shared", r)
    return (*r["pred"], r["test_rse"], secs, fm.total_epochs, n_active(r["Z_train"], r["Lambda"]), np.nan)


def fit_mofa(k, d, dp):
    import torch
    import compare_methods as cm
    from k_sweep import n_active, save_factors
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
    for f in os.listdir(d):                        # exported matrices are large
        if f.startswith("x_"):
            os.remove(os.path.join(d, f))
    save_factors(d, "mofa", Z_train, Z_test, W, cm.ols_beta(Z_train, dp))
    xhat = [torch.tensor(Z_test @ w.T, dtype=torch.float32) for w in W]
    t = lambda c: torch.tensor(pred[c].values, dtype=torch.float32)
    return (t("fit"), t("lwr"), t("upr"), cm.rse(dp["test_X_l_list_clean"], xhat), runtime,
            np.nan, n_active(Z_train, W), np.nan)


def fit_simple(method, k, d, dp):
    import torch
    import compare_methods as cm
    from k_sweep import save_factors
    from sklearn.decomposition import PCA
    from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
    X_tr = torch.cat(dp["train_X_l_list_clean"], 1).numpy()
    X_te = torch.cat(dp["test_X_l_list_clean"], 1).numpy()
    y_tr = dp["train_y_clean"].reshape(-1).numpy()
    t0, test_rse, active = time.time(), np.nan, np.nan
    if method == "pca_ols":
        pca = PCA(n_components=k).fit(X_tr)
        ols = LinearRegression().fit(pca.transform(X_tr), y_tr)
        p_views = np.cumsum([0] + [x.shape[1] for x in dp["train_X_l_list_clean"]])
        save_factors(d, "pca_ols", pca.transform(X_tr), pca.transform(X_te),
                     [pca.components_.T[a:b] for a, b in zip(p_views[:-1], p_views[1:])], ols.coef_)
        pred = ols.predict(pca.transform(X_te))
        xhat = pca.inverse_transform(pca.transform(X_te))
        test_rse, active = float(((X_te - xhat) ** 2).sum() / (X_te ** 2).sum()), k
    elif method == "ridge":
        pred = RidgeCV(alphas=np.logspace(-1, 5, 40), cv=5).fit(X_tr, y_tr).predict(X_te)
    elif method == "elastic_net":
        pred = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], n_alphas=50, cv=5, max_iter=5000,
                            n_jobs=1).fit(X_tr, y_tr).predict(X_te)
    else:
        pred = np.full(X_te.shape[0], y_tr.mean())
    nan = torch.full((X_te.shape[0],), float("nan"))
    return torch.tensor(pred, dtype=torch.float32), nan, nan, test_rse, time.time() - t0, np.nan, active, np.nan


def run_job(method, fold, k, smoke=False):
    tag = f"{method}_fold{fold}_k{k}"
    row_path = os.path.join(OUT, "rows", tag + ".csv")
    if os.path.exists(row_path) and not smoke:
        return tag
    import torch
    torch.set_num_threads(1)
    import compare_methods as cm
    if smoke:
        global MAX_EPOCHS, EXTRA
        MAX_EPOCHS, EXTRA = 60, 10
        cm.MA_RULE.update(window=5, patience=10)
        cm.LOCAL_TRAIN_CONFIG.update(min_epochs=20, max_epochs=40)
        cm.N_POSTERIOR_SAMPLES = 20
    data, train_subset, test_subset, dp = load_fold(fold)
    d = os.path.join(OUT, f"fold{fold}", f"k{k}" if k else "base")
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    try:
        if method == "shared_ma":
            res = fit_shared(k, fold, train_subset, test_subset, dp, d)
        elif method == "shared":
            res = fit_shared_default(k, fold, train_subset, test_subset, dp, d)
        elif method == "mofa":
            res = fit_mofa(k, d, dp)
        else:
            res = fit_simple(method, k, d, dp)
    except Exception as e:                          # record and move on
        res = {"failed_at": f"{type(e).__name__}: {str(e).splitlines()[0][:150]}", "ma_stop": np.nan}
    key = {"fold": fold, "method": method, "k": k if k else np.nan,
           "n_train": len(train_subset), "n_test": len(data["y_test"])}
    if isinstance(res, dict):
        row = {**key, "failed": str(res["failed_at"]), "ma_stop": res["ma_stop"], "wall": time.time() - t0}
        pd.DataFrame([row]).to_csv(row_path, index=False)
        print(f"{tag} FAILED {row['failed']}", flush=True)
        return tag
    m, lo, hi, test_rse, runtime, epochs, active, ma_stop = res
    y_mean, y_sd = float(data["y"].mean()), float(data["y"].std())
    m, lo, hi = (v * y_sd + y_mean for v in (m, lo, hi))       # log(1 + TMB) scale
    y = data["y_test"].reshape(-1)
    resid = m - y
    has_int = not torch.isnan(lo).all()
    row = {**key, "mse": float((resid ** 2).mean()),
           "r2": float(1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()),
           "coverage_95": float(((y >= lo) & (y <= hi)).float().mean()) if has_int else np.nan,
           "width_95": float((hi - lo).mean()) if has_int else np.nan,
           "test_rse": test_rse, "runtime": runtime, "epochs": epochs, "ma_stop": ma_stop,
           "active_factors": active, "wall": time.time() - t0}
    pd.DataFrame({"fold": fold, "method": method, "k": key["k"], "sample_id": data["sample_ids_test"],
                  "y_true": y.numpy(), "y_pred": m.numpy(), "lwr": lo.numpy(), "upr": hi.numpy()}).to_csv(
        os.path.join(OUT, "preds", tag + ".csv"), index=False)
    pd.DataFrame([row]).to_csv(row_path, index=False)
    print(f"{tag:<26} R²={row['r2']:.3f} MSE={row['mse']:.3f} epochs={epochs} active={active} "
          f"({row['wall']:.0f}s)", flush=True)
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", nargs="+", type=int, default=K_GRID)
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(N_FOLDS)))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="one short shared fit on fold 0, K = 20")
    args = ap.parse_args()
    global OUT
    if args.smoke:
        OUT = os.path.join(OUT, "smoke")
    for sub in ("rows", "preds"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    if args.smoke:
        for m in ("shared_ma", "pca_ols", "ridge"):
            run_job(m, 0, 20 if m != "ridge" else 0, smoke=True)
        return
    # longest jobs first
    jobs = [("shared", f, k) for k in sorted(args.k, reverse=True) for f in args.folds]
    jobs += [("shared_ma", f, k) for k in sorted(args.k, reverse=True) for f in args.folds]
    jobs += [("mofa", f, k) for k in sorted(args.k, reverse=True) for f in args.folds]
    jobs += [("pca_ols", f, k) for k in args.k for f in args.folds]
    jobs += [(m, f, 0) for m in ("ridge", "elastic_net", "mean") for f in args.folds]
    print(f"{len(jobs)} jobs, {args.workers} workers", flush=True)
    from joblib import Parallel, delayed
    t0 = time.time()
    Parallel(n_jobs=args.workers, backend="loky", verbose=5)(delayed(run_job)(*j) for j in jobs)
    for sub, name in (("rows", "cv_results.csv"), ("preds", "cv_predictions.csv")):
        files = glob.glob(os.path.join(OUT, sub, "*.csv"))
        pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv(os.path.join(OUT, name), index=False)
    print(f"done in {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
