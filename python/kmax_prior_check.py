"""
Does the MGP global precision tau_k fall over a long range of columns as K_max
grows, in simulations as on TCGA? And does factor recovery degrade with it?

Each fit uses the TCGA pipeline's training (compare_methods.py): SVD
initialisation, GUIDE_START scales, AdagradRMSProp (eta 0.22, t 0.02),
minibatch fraction 0.08, outcome_weight 1, stopped by the MA w25 p50 rule
(checkpoints every 5 epochs, at most 1000 epochs).

Datasets (training split only is used for fitting; true Z and loadings known)
  mis    June misaligned study, n = 500, p = 1000 per view x 3, symmetric
         alignment, true K = 5 (reps 1, 2)
  ctrl   controlled-strength study, n = 500, P = 3000, c = 1, true K = 5 (reps 1, 2)
  many   TCGA-like: n = 424, two views (2000 + 1000 features), true K = 35 with
         a smoothly decaying spectrum (factor 1: 10% PVE, factors 2-35: 2.5%
         falling geometrically to 0.25%; about 44% signal in total, as in the
         TCGA fits), loadings Gaussian on a random 30% of features, unit noise;
         y = Z beta + e on factors 1, 3, 10, 25 (R^2 about 0.5). Seeds 1, 2.

Data "tcga" uses CV fold <rep> of analysis/tcga_coadread (cv_k_sweep.load_fold).

Optimiser (--opt): "default", or "gslow<f>" = step size f * eta for every global
parameter (loadings, delta, rho, psi, beta, sigma) and eta for the per-sample
scores; AdagradRMSProp's first step is eta * sign(gradient) for every parameter.

Priors: "mgp" (default MGP), "ard" (dense_model=True: one inverse-gamma
variance per column, not cumulative), and MGP variants set through
--prior mgp:a1=..,a2=.. (attributes set on the built model; core code
unchanged).

Per fit, saved to sim/results/kmax_prior_check/<tag>.npz and one CSV row:
E_q[tau_k] per column (view 0), PVE per column, RMS of score / loading
columns, counts of columns above 1% and 0.1% PVE, recovery of each true
factor (best |corr| of its scores with any fitted column) and canonical
correlations between the true score space and the fitted top-K_true space.

usage: python kmax_prior_check.py [--data mis ctrl many] [--k 15 50 100]
       [--prior mgp ard] [--reps 1 2] [--workers 4]
"""
import argparse
import copy
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/multiomic_integration")
OUT = f"{ROOT}/sim/results/kmax_prior_check"
sys.path.insert(0, f"{ROOT}/python")
sys.path.insert(0, f"{ROOT}/analysis/tcga_coadread")

MIS = f"{ROOT}/sim/data/multi_view_shared_misaligned/k.5/{{split}}/n500p1000_snr2.2_sparse0.25_align_symmetric/sim_data_shared_rep{{rep}}.rds"
CTRL = f"{ROOT}/sim/data/multi_view_shared_controlled/k.5/{{split}}/n500_P3000_c1/sim_data_controlled_n500_P3000_c1_rep{{rep}}.rds"
MAX_EPOCHS = 1000


def many_factor_data(seed, n_train=424, n_test=106, p=(2000, 1000), k=35):
    import torch
    rng = np.random.default_rng(seed)
    pve = np.r_[0.10, 0.025 * 0.1 ** (np.arange(k - 1) / (k - 2))]
    s2 = pve / (1 - pve.sum()) / 0.3                     # 0.3 = share of features loaded
    n = n_train + n_test
    Z = rng.standard_normal((n, k))
    Lam = [rng.standard_normal((pl, k)) * (rng.random((pl, k)) < 0.3) * np.sqrt(s2) for pl in p]
    X = [Z @ L.T + rng.standard_normal((n, L.shape[0])) for L in Lam]
    beta = np.zeros(k); beta[[0, 2, 9, 24]] = [0.6, 0.5, 0.4, 0.3]
    y = Z @ beta + rng.standard_normal(n) * np.sqrt((beta ** 2).sum())
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    mk = lambda idx: {"X_l": [t(x[idx]) for x in X], "y": t(y[idx]).reshape(-1, 1), "K": torch.tensor(k),
                      "Z": t(Z[idx]), "Lambda_l": [t(L) for L in Lam]}
    return mk(slice(0, n_train)), mk(slice(n_train, n))


def get_data(name, rep):
    from run_methods import load_rds
    if name == "tcga":                      # rep = CV fold (already processed; see fit_one)
        return None, None
    if name == "many":
        return many_factor_data(rep)
    path = MIS if name == "mis" else CTRL
    return load_rds(path.format(split="train", rep=rep)), load_rds(path.format(split="test", rep=rep))


def parse_prior(prior):
    kind, _, rest = prior.partition(":")
    over = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in rest.split(",") if kv}
    return kind, over


def patch_optimiser(opt):
    """opt = "default" or "gslow<f>": the global parameters (everything except the
    per-sample loc_Z / scale_Z) get step size f * eta. Done by wrapping the
    AdagradRMSProp that run_methods builds (per-parameter optim_args)."""
    import run_methods
    import pyro.optim
    if opt == "default":
        return
    f = float(opt.replace("gslow", ""))

    def factory(args):
        def per_param(name):
            return args if name in ("loc_Z", "scale_Z") else {**args, "eta": args["eta"] * f}
        return pyro.optim.AdagradRMSProp(per_param)
    run_methods.AdagradRMSProp = factory


def fit_one(name, rep, k, prior, opt="default"):
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import torch
    torch.set_num_threads(2)
    import pyro
    import compare_methods as cm
    import multiview_shared
    from run_methods import process_data, train_model, _minibatch_size
    from stopping_rules import ma_patience

    tag = f"{name}_rep{rep}_k{k}_{prior.replace(':', '_').replace(',', '_').replace('=', '')}"
    if opt != "default":
        tag += f"_{opt}"
    patch_optimiser(opt)
    if os.path.exists(f"{OUT}/{tag}.npz"):
        return None
    kind, over = parse_prior(prior)

    # hyperparameter overrides on the built model (no change to the model code)
    orig_init = multiview_shared.SupMultiviewShared.__init__

    def patched(self, *a, **kw):
        orig_init(self, *a, **kw)
        for key, v in over.items():
            setattr(self, {"a1": "a1_sigma_joint", "a2": "a2_sigma_joint", "alpha": "alpha_joint"}[key], v)
    multiview_shared.SupMultiviewShared.__init__ = patched

    tr, te = get_data(name, rep)
    if name == "tcga":
        from cv_k_sweep import load_fold
        _, train_subset, test_subset, dp = load_fold(rep)
        tr = {"X_l": list(train_subset.tensors[1:-1]), "Z": None}
    else:
        train_subset, test_subset, dp = process_data(tr, te, model_type="shared")
    pyro.clear_param_store()
    pyro.set_rng_seed(100 + rep)
    E = cm.CHECKPOINT_EVERY
    cfg = {**cm.SHARED_TRAIN_CONFIG, "min_epochs": E, "max_epochs": E}
    mcfg = {**cm.SHARED_MODEL_CONFIG, "k": k, "dense_model": kind == "ard"}
    t0 = time.time()
    fm, th, _ = train_model(mcfg, cfg, train_subset, model_type="shared", write=False, svd_init=True)
    store = pyro.get_param_store()
    epoch, stop, ckpt, prev, failed = E, None, None, None, None
    prev = (epoch, copy.deepcopy(store.get_state()))
    while stop is None and epoch < MAX_EPOCHS:
        try:
            th.do_inference(train_dataset=train_subset, min_epochs=E, epochs=E, minibatch_flag=True,
                            minibatch_size=_minibatch_size(cfg, len(train_subset)),
                            variational_diff_func=np.mean, verbose=False,
                            convergence_criterion=cfg["convergence_criterion"])
        except ValueError:
            failed = epoch
            break
        epoch += E
        now = (epoch, copy.deepcopy(store.get_state()))
        stop = ma_patience(fm.loss_history, cm.MA_RULE["min_epochs"], cm.MA_RULE["window"], cm.MA_RULE["patience"])
        if stop is not None:
            ckpt = now if stop >= epoch else prev
        prev = now
    ckpt = ckpt or prev
    pyro.clear_param_store()
    store.set_state(copy.deepcopy(ckpt[1]))
    s = {n_: pyro.param(n_).detach().double().numpy() for n_ in store.keys()}
    from viper_predict import closed_form_scores, guide_moments
    L = len(tr["X_l"])
    if kind == "mgp":
        cf = closed_form_scores(guide_moments(store, L), dp["test_X_l_list_clean"])
        test_mse = float(((cf["yhat"] - dp["test_y_clean"].reshape(-1).double()) ** 2).mean())
    else:
        test_mse = np.nan
    Z = s["loc_Z"]
    W = np.vstack([s[f"loc_Lambda_l{l}"] for l in range(L)])
    pve = (Z ** 2).sum(0) * (W ** 2).sum(0) / (Z.shape[0] * W.shape[0])
    if kind == "ard":
        # E_q[1 / sigma2_k] under the inverse-gamma guide (shape a, rate b): a / b
        tau = s["a_sigma_lambda_l0"] / s["b_sigma_lambda_l0"] if "a_sigma_lambda_l0" in s else np.full(k, np.nan)
    else:
        tau = np.exp(np.cumsum(np.log(s["a_delta_lambda_l0"] / s["b_delta_lambda_l0"])))
    rho = (s["a_rho_lambda_l0"] / s["b_rho_lambda_l0"]).mean(0) if "a_rho_lambda_l0" in s else np.full(k, np.nan)

    if tr["Z"] is not None:
        Zt = tr["Z"].numpy().astype(float)
        kt = Zt.shape[1]
        C = np.nan_to_num(np.abs(np.corrcoef(Zt.T, Z.T)[:kt, kt:]))
        top = Z[:, np.argsort(pve)[::-1][:kt]]
        qa, _ = np.linalg.qr(Zt - Zt.mean(0)); qb, _ = np.linalg.qr(top - top.mean(0))
        cc = np.linalg.svd(qa.T @ qb, compute_uv=False)
    else:
        kt, C, cc = 0, np.full((1, 1), np.nan), np.full(1, np.nan)
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/{tag}.npz", Z=Z, W=W, pve=pve, tau=tau, rho=rho, best_corr=C.max(1), canon=cc,
             scale_Z=s["scale_Z"], epoch=ckpt[0])
    row = {"data": name, "rep": rep, "K_max": k, "prior": prior, "opt": opt, "test_mse_std": test_mse,
           "K_true": kt, "epoch": ckpt[0],
           "failed_at": failed, "seconds": time.time() - t0,
           "cols > 1%": int((pve > .01).sum()), "cols > 0.1%": int((pve > .001).sum()),
           "true matched > 0.9": int((C.max(1) > .9).sum()), "mean best |corr|": C.max(1).mean(),
           "mean canon corr": cc.mean(), "min canon corr": cc.min(),
           "log10 min tau": np.log10(tau.min()) if np.isfinite(tau).all() else np.nan,
           "argmin tau": int(np.argmin(tau)) + 1 if np.isfinite(tau).all() else np.nan,
           "cols with tau < tau_1": int((tau < tau[0]).sum()) if np.isfinite(tau).all() else np.nan}
    pd.DataFrame([row]).to_csv(f"{OUT}/{tag}.csv", index=False)
    print(tag, {k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in row.items()}, flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*", default=["mis", "ctrl", "many"])
    ap.add_argument("--k", nargs="*", type=int, default=[15, 50, 100])
    ap.add_argument("--prior", nargs="*", default=["mgp"])
    ap.add_argument("--reps", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--opt", nargs="*", default=["default"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from joblib import Parallel, delayed
    jobs = sorted(itertools.product(a.data, a.reps, a.k, a.prior, a.opt), key=lambda j: -j[2])
    Parallel(n_jobs=a.workers, backend="loky")(delayed(fit_one)(*j) for j in jobs)
    import glob
    rows = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/*.csv") if not f.endswith("summary.csv")])
    rows.to_csv(f"{OUT}/summary.csv", index=False)


if __name__ == "__main__":
    main()
