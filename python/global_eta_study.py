"""
Smaller step size for the global parameters: prediction over epochs, factor
recovery, and the MGP global precision tau, as K_max varies.

Background (docs/mgp_shrinkage_collapse.md): with the current optimiser
(AdagradRMSProp, eta = 0.22) the first steps move every loading and every
q(delta) parameter by about eta, whatever its gradient, and E_q[tau_k] collapses
over a long range of columns at large K_max. Here the global parameters
(loadings, delta, rho, psi, beta, sigma) get step size f * eta, the per-sample
scores (loc_Z, scale_Z) keep eta.

Grid
  opt     default   : current optimiser (AdagradRMSProp, eta for every parameter)
          gslow0.3  : step 0.3 * eta for the global parameters, eta for the scores
          warm5     : eta warmed up linearly from 0 over the first 5 epochs (per
                      optimiser step, every parameter), then the current optimiser
          dfreeze25 : q(delta) (the MGP global-precision guide) held at its starting
                      value for the first 25 epochs while everything else trains with
                      the current optimiser ("delayed shrinkage"), then trained as usual
          <opt>_qfix : q(delta) started at its closed-form shape (as qcavi) and the shape
                      held fixed (never stepped); the rate is trained
          <opt>_allfix : as qfix, and q(rho), q(psi), q(sigma2_y), q(sigma2_beta) also
                      started at their closed-form shapes, held fixed
                      (docs/conjugate_guide_shapes.md); needs a warm/dfreeze <opt>
          <opt>-rand : any of these with random guide means (seed_random_locs)
                      instead of the SVD start (does the MGP order the columns itself?)
          <opt>_<start> : as <opt>, with q(delta_h) = Gamma(a, b) started at
                      qprior : the prior, Gamma(a_h, 1) (a_1 = 2.1, a_h = 3.1)
                      qflat  : tightly at 1, Gamma(50, 50)
                      qc<c>  : Gamma(c, c) (mean 1, sd 1/sqrt(c))
                      qp<c>  : Gamma(c * a_h, c) (the prior's mean, concentration c)
                      qcavi  : the closed-form (CAVI) shape a_h + p_l (K - h + 1) / 2,
                               rate = shape (mean 1)
                      instead of the code's Gamma(1, 1) (set in the param store before
                      training; model code unchanged)
          dense     : the dense prior (dense_model=True: per-view, per-column
                      inverse-gamma variance, no cumulative product), current optimiser
  data    tcga: CV folds 0-4 of analysis/tcga_coadread (cv_k_sweep.load_fold),
                K_max in {20, 50, 100}
          ctrl: controlled-strength sims, n = 500, P = 3000, c = 1 (reps 1, 2)
          mis : June misaligned sims, n = 500, p = 1000 x 3, symmetric (reps 1, 2)
                sims: K_max in {15, 50, 100}
Everything else as the TCGA pipeline (compare_methods.py): MGP prior, SVD
initialisation, GUIDE_START scales, outcome weight 1, minibatch fraction 0.08.

Training runs a fixed 1000 epochs (no early stop). Every 25 epochs
(and at epoch 5) the script records
  - closed-form test predictions (viper_predict.closed_form_scores): test MSE,
    95% coverage and width (standardised outcome), and the predictions
    themselves (to pool out-of-fold R^2 over the TCGA folds);
  - E_q[log10 tau_k] per view and column;
  - columns above 1% / 0.1% PVE of the training views;
  - sims only: recovery of the true factors (mean canonical correlation of the
    true score space with the fitted top-K_true PVE columns, and how many true
    factors have a fitted column with |corr| > 0.9).
The MA w25 p50 stop is found from the loss afterwards. Loadings and scores
are saved at the MA stop's checkpoint, at epoch 500 and at epoch 1000.

Outputs: sim/results/global_eta_study/<tag>_{curve.csv,tau.csv,preds.npz,fits.npz}
         (TCGA fits are real-data results and go to
          analysis/tcga_coadread/results/global_eta_study/)
usage: python global_eta_study.py [--data tcga ctrl mis] [--k ...] [--opt default gslow0.3]
       [--workers 4]
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
sys.path.insert(0, f"{ROOT}/python")
sys.path.insert(0, f"{ROOT}/analysis/tcga_coadread")
OUT_SIM = os.environ.get("GLOBAL_ETA_OUT", f"{ROOT}/sim/results/global_eta_study")
OUT_TCGA = os.environ.get("GLOBAL_ETA_OUT", f"{ROOT}/analysis/tcga_coadread/results/global_eta_study")
EPOCHS = int(os.environ.get("GLOBAL_ETA_EPOCHS", 1000))     # --epochs sets it for the workers
E, EVERY, SAVE_AT = 5, 25, (500, EPOCHS)
K_GRID = {"tcga": [20, 50, 100], "ctrl": [15, 50, 100], "mis": [15, 50, 100]}
REPS = {"tcga": [0, 1, 2, 3, 4], "ctrl": [1, 2], "mis": [1, 2]}


def patch_schedule(opt, steps_per_epoch, frozen=None):
    """warm<E>: linear warm-up of eta over E epochs; dfreeze<E>: q(delta) frozen for
    E epochs. Implemented as a subclass of the torch AdagradRMSProp that Pyro wraps
    (one optimiser per parameter, so each counts its own steps); the model code and
    run_methods are not changed, only the factory run_methods calls.
    frozen: optional predicate on the parameter name; those parameters are never
    stepped (held at their starting value for the whole run)."""
    import pyro.optim
    import run_methods
    from pyro.optim.adagrad_rmsprop import AdagradRMSProp as TorchAdagradRMSProp

    class Scheduled(TorchAdagradRMSProp):
        def __init__(self, params, warmup_steps=0, freeze_steps=0, **kw):
            super().__init__(params, **kw)
            self.warmup_steps, self.freeze_steps, self.n = warmup_steps, freeze_steps, 0
            self.base = [g["eta"] for g in self.param_groups]

        def step(self, closure=None):
            self.n += 1
            if self.n <= self.freeze_steps:
                return None                      # parameter held; optimiser state untouched
            f = min(1.0, self.n / self.warmup_steps) if self.warmup_steps else 1.0
            for g, b in zip(self.param_groups, self.base):
                g["eta"] = b * f
            return super().step(closure)

    never = 10 ** 12
    hold = (lambda name: never if frozen is not None and frozen(name) else 0)
    if opt.startswith("warm"):
        w = int(opt[4:]) * steps_per_epoch
        per_param = lambda args: (lambda name: {**args, "warmup_steps": w, "freeze_steps": hold(name)})
    elif opt.startswith("dfreeze"):
        fz = int(opt[7:]) * steps_per_epoch
        per_param = lambda args: (lambda name: {**args, "freeze_steps": max(fz if "delta" in name else 0, hold(name))})
    else:
        raise ValueError(opt)
    run_methods.AdagradRMSProp = lambda args: pyro.optim.PyroOptim(Scheduled, per_param(args))


def seed_random_locs(k, n, p):
    """Random start for the guide means instead of the SVD: loc_Z ~ N(0, 1) (the
    SVD start's score scale) and loc_Lambda_l ~ N(0, 0.1^2), so no column starts
    out stronger than another. Uses the RNG seeded just before."""
    import torch, pyro
    from constants import Params
    pyro.param(Params.loc_Z, torch.randn(n, k))
    for l, p_l in enumerate(p):
        pyro.param(Params.loc_Lambda_l.format(l=l), 0.1 * torch.randn(p_l, k))


CONJ_SHAPES = ("a_delta_lambda_l", "a_rho_lambda_l", "a_psi_l", "a_sigma_y", "a_sigma_beta")


def seed_conjugate(k, n, p, outcome_weight=1.0):
    """q(rho), q(psi), q(sigma2_y), q(sigma2_beta) started at their closed-form
    (mean-field coordinate-optimal) shapes; derivation and the choice of rates in
    docs/conjugate_guide_shapes.md. The rate keeps the expectation that enters the
    other factors' updates at the current start's value: E[rho] = 1, and
    E[1/psi] = E[1/sigma2_y] = E[1/sigma2_beta] = 3 (InvGamma(3, 1) start)."""
    import inspect, torch, pyro
    import pyro.distributions as dist
    from constants import Params
    from multiview_shared import SupMultiviewShared
    d = {kk: v.default for kk, v in inspect.signature(SupMultiviewShared.__init__).parameters.items()}
    pos = dist.constraints.positive
    nu2 = d["alpha_joint"] / 2
    for l, p_l in enumerate(p):
        a = torch.full((p_l, k), nu2 + 0.5)
        pyro.param(Params.a_rho_lambda_l.format(l=l), a, constraint=pos)
        pyro.param(Params.b_rho_lambda_l.format(l=l), a.clone(), constraint=pos)          # E[rho] = 1
        a = torch.full((p_l,), d["a_psi"] + n / 2)
        pyro.param(Params.a_psi_l.format(l=l), a, constraint=pos)
        pyro.param(Params.b_psi_l.format(l=l), a / 3, constraint=pos)                      # E[1/psi] = 3
    a = torch.tensor(d["a_sigma_y"] + outcome_weight * n / 2)
    pyro.param(Params.a_sigma_y, a, constraint=pos)
    pyro.param(Params.b_sigma_y, a / 3, constraint=pos)                                    # E[1/sigma2_y] = 3
    a = torch.full((k,), d["a_sigma_beta"] + 0.5)
    pyro.param(Params.a_sigma_beta, a, constraint=pos)
    pyro.param(Params.b_sigma_beta, a / 3, constraint=pos)                                 # E[1/sigma2_beta] = 3


def seed_delta(k, p, start):
    """Starting values of q(delta_h) = Gamma(a_h, b_h) for every view (p: features per view)."""
    import torch, pyro
    import pyro.distributions as dist
    from constants import Params
    a_prior = torch.full((k,), 3.1)
    a_prior[0] = 2.1
    for l, p_l in enumerate(p):
        if start == "qflat":
            a, b = torch.full((k,), 50.0), torch.full((k,), 50.0)
        elif start == "qprior":
            a, b = a_prior.clone(), torch.ones(k)
        elif start in ("qcavi", "qfix", "allfix"):
            a = a_prior + p_l * (k - torch.arange(k, dtype=torch.float32)) / 2
            b = a.clone()
        elif start.startswith("qc"):
            c = float(start[2:])
            a, b = torch.full((k,), c), torch.full((k,), c)
        elif start.startswith("qp"):
            c = float(start[2:])
            a, b = c * a_prior, torch.full((k,), c)
        else:
            raise ValueError(start)
        pyro.param(Params.a_delta_lambda_l.format(l=l), a, constraint=dist.constraints.positive)
        pyro.param(Params.b_delta_lambda_l.format(l=l), b, constraint=dist.constraints.positive)


def snapshot(store, L, Ztrue, dp):
    import torch, pyro
    from scipy.special import digamma
    from viper_predict import closed_form_scores, guide_moments
    P = lambda n: pyro.param(n).detach().double().numpy()
    cf = closed_form_scores(guide_moments(store, L), dp["test_X_l_list_clean"])
    y = dp["test_y_clean"].reshape(-1).double().numpy()
    yhat, sd = cf["yhat"].numpy(), np.sqrt(cf["pred_var"].numpy())
    Z = P("loc_Z")
    W = np.vstack([P(f"loc_Lambda_l{l}") for l in range(L)])
    pve = (Z ** 2).sum(0) * (W ** 2).sum(0) / (Z.shape[0] * W.shape[0])
    row = {"test_mse_std": float(((yhat - y) ** 2).mean()),
           "coverage_95": float((np.abs(yhat - y) <= 1.96 * sd).mean()),
           "width_95": float((2 * 1.96 * sd).mean()),
           "cols > 1%": int((pve > .01).sum()), "cols > 0.1%": int((pve > .001).sum())}
    if Ztrue is not None:
        kt = Ztrue.shape[1]
        C = np.nan_to_num(np.abs(np.corrcoef(Ztrue.T, Z.T)[:kt, kt:]))
        top = Z[:, np.argsort(pve)[::-1][:kt]]
        qa, _ = np.linalg.qr(Ztrue - Ztrue.mean(0)); qb, _ = np.linalg.qr(top - top.mean(0))
        cc = np.linalg.svd(qa.T @ qb, compute_uv=False)
        row.update({"recovery (mean canon corr)": float(cc.mean()), "recovery (min canon corr)": float(cc.min()),
                    "true matched > 0.9": int((C.max(1) > .9).sum())})
    taus = ([np.cumsum(digamma(P(f"a_delta_lambda_l{l}")) - np.log(P(f"b_delta_lambda_l{l}"))) / np.log(10)
             for l in range(L)] if "a_delta_lambda_l0" in store else [])
    fit = {"Z": Z, "W": W, "pve": pve, "beta": P("loc_beta")}
    if "a_delta_lambda_l0" in pyro.get_param_store():
        for l in range(L):
            fit[f"a_delta_l{l}"] = P(f"a_delta_lambda_l{l}")
            fit[f"b_delta_l{l}"] = P(f"b_delta_lambda_l{l}")
    st = pyro.get_param_store()
    for nm in [f"{ab}_psi_l{l}" for ab in "ab" for l in range(L)] + ["a_sigma_y", "b_sigma_y", "a_sigma_beta", "b_sigma_beta"]:
        if nm in st:
            fit[nm] = P(nm)
    for l in range(L):
        if f"a_rho_lambda_l{l}" in st:
            fit[f"median_a_rho_l{l}"] = np.median(P(f"a_rho_lambda_l{l}"))
            fit[f"median_b_rho_l{l}"] = np.median(P(f"b_rho_lambda_l{l}"))
    return row, yhat, sd, taus, fit


def _fit(data, rep, k, opt, seed):
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import torch, pyro
    torch.set_num_threads(2)
    import compare_methods as cm
    from kmax_prior_check import get_data, patch_optimiser
    from run_methods import process_data, train_model, _minibatch_size
    from stopping_rules import ma_patience

    out = OUT_TCGA if data == "tcga" else OUT_SIM
    tag = f"{data}_rep{rep}_k{k}_{opt}"
    if os.path.exists(f"{out}/{tag}_curve.csv"):
        return
    os.makedirs(out, exist_ok=True)
    import pyro.optim, run_methods
    run_methods.AdagradRMSProp = pyro.optim.AdagradRMSProp   # undo a patch left by an earlier job in this worker
    if data == "tcga":
        from cv_k_sweep import load_fold
        _, train_subset, _, dp = load_fold(rep)
        Ztrue = None
    else:
        tr, te = get_data(data, rep)
        train_subset, _, dp = process_data(tr, te, model_type="shared")
        Ztrue = tr["Z"].numpy().astype(float)
    L = len(train_subset.tensors) - 2
    # this study applies its own warm-up / shape freezing (patch_schedule, seed_*), so the
    # model defaults added afterwards (warm-up, fixed conjugate shapes) are switched off
    cfg = {**cm.SHARED_TRAIN_CONFIG, "min_epochs": E, "max_epochs": E, "warmup_epochs": 0}
    mb = _minibatch_size(cfg, len(train_subset))
    opt_main, _, init = opt.partition("-")             # "<opt>-rand": random start instead of the SVD
    base, _, start = opt_main.partition("_")
    # qfix: q(delta) shape held at its closed form; allfix: every conjugate shape held
    frozen = ({"qfix": lambda nm: nm.startswith("a_delta_lambda_l"),
               "allfix": lambda nm: nm.startswith(CONJ_SHAPES)}.get(start))
    if base.startswith(("warm", "dfreeze")):
        patch_schedule(base, int(np.ceil(len(train_subset) / mb)), frozen)
    elif base != "dense":
        patch_optimiser(base)
    pyro.clear_param_store()
    if frozen is not None and not base.startswith(("warm", "dfreeze")):
        raise ValueError("qfix / allfix need a warm<E> or dfreeze<E> optimiser")
    if start:
        seed_delta(k, [t.shape[1] for t in train_subset.tensors[1:-1]], start)
    if start == "allfix":
        seed_conjugate(k, len(train_subset), [t.shape[1] for t in train_subset.tensors[1:-1]],
                       cm.SHARED_MODEL_CONFIG.get("outcome_weight", 1.0))
    pyro.set_rng_seed(seed)
    if init == "rand":
        seed_random_locs(k, len(train_subset), [t.shape[1] for t in train_subset.tensors[1:-1]])
    elif init:
        raise ValueError(init)
    t0 = time.time()
    fm, th, _ = train_model({**cm.SHARED_MODEL_CONFIG, "k": k, "dense_model": base == "dense", "conjugate_shapes": False}, cfg, train_subset,
                            model_type="shared", write=False, svd_init=(init != "rand"))
    store = pyro.get_param_store()
    curve, taus, preds, states, fits = [], [], {}, {}, {}
    epoch, failed = E, None
    while True:
        if epoch == E or epoch % EVERY == 0:
            row, yhat, sd, tau, fit = snapshot(store, L, Ztrue, dp)
            curve.append({"epoch": epoch, "seconds": time.time() - t0, **row})
            for l, t in enumerate(tau):
                taus += [{"epoch": epoch, "view": l, "column": j + 1, "E log10 tau": t[j]} for j in range(k)]
            preds[epoch] = np.c_[yhat, sd]
            states[epoch] = copy.deepcopy(store.get_state())
            if epoch in SAVE_AT:
                fits[epoch] = fit
        if epoch >= EPOCHS:
            break
        try:
            th.do_inference(train_dataset=train_subset, min_epochs=E, epochs=E, minibatch_flag=True,
                            minibatch_size=mb,
                            variational_diff_func=np.mean, verbose=False,
                            convergence_criterion=cfg["convergence_criterion"])
        except ValueError:
            failed = epoch
            break
        epoch += E
    stop = ma_patience(fm.loss_history, cm.MA_RULE["min_epochs"], cm.MA_RULE["window"], cm.MA_RULE["patience"])
    if stop is not None:
        # the MA checkpoint: the last recorded checkpoint at or before the stop
        ck = max(e for e in states if e <= max(stop, E))
        pyro.clear_param_store(); store.set_state(states[ck])
        fits["ma"] = {**snapshot(store, L, Ztrue, dp)[4], "epoch": ck}
    loss = np.asarray(fm.loss_history, float) / len(train_subset)
    pd.DataFrame(curve).assign(data=data, rep=rep, K_max=k, opt=opt, ma_stop=stop, failed_at=failed, seed=seed).to_csv(
        f"{out}/{tag}_curve.csv", index=False)
    pd.DataFrame(taus).assign(data=data, rep=rep, K_max=k, opt=opt).to_csv(f"{out}/{tag}_tau.csv", index=False)
    np.savez(f"{out}/{tag}_preds.npz", epochs=np.array(sorted(preds)), y=dp["test_y_clean"].reshape(-1).numpy(),
             pred=np.stack([preds[e] for e in sorted(preds)]), loss=loss)
    np.savez(f"{out}/{tag}_fits.npz", **{f"{key}_{name}": np.asarray(v) for key, f in fits.items()
                                          for name, v in f.items()})
    print(f"{tag}: MA stop {stop}, failed {failed}, {time.time() - t0:.0f}s, "
          f"test MSE at epoch {curve[-1]['epoch']}: {curve[-1]['test_mse_std']:.3f}", flush=True)


def fit_one(data, rep, k, opt, max_tries=3):
    """Run _fit; if the parameters become NaN (ValueError from a distribution's
    argument check, e.g. in the first epoch), log it to failures.csv and retry
    with a new seed, up to max_tries seeds. Never raises, so one failure does
    not stop the batch."""
    out = OUT_TCGA if data == "tcga" else OUT_SIM
    tag = f"{data}_rep{rep}_k{k}_{opt}"
    if os.path.exists(f"{out}/{tag}_curve.csv"):
        return
    for attempt in range(max_tries):
        seed = 1000 + rep + 100 * attempt
        try:
            _fit(data, rep, k, opt, seed)
            return
        except (ValueError, RuntimeError) as e:
            msg = str(e).splitlines()[0][:200]
            print(f"{tag}: attempt {attempt + 1} (seed {seed}) failed: {msg}", flush=True)
            row = pd.DataFrame([{"data": data, "rep": rep, "K_max": k, "opt": opt, "seed": seed,
                                 "attempt": attempt + 1, "error": msg}])
            f = f"{out}/failures.csv"
            row.to_csv(f, mode="a", header=not os.path.exists(f), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*", default=["tcga", "ctrl", "mis"])
    ap.add_argument("--k", nargs="*", type=int, default=None)
    ap.add_argument("--opt", nargs="*", default=["default", "gslow0.3"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reps", nargs="*", type=int, default=None, help="override the replicate / fold list")
    ap.add_argument("--outdir", default=None, help="write everything here instead of the study folders")
    ap.add_argument("--epochs", type=int, default=None, help="training length (default 1000)")
    a = ap.parse_args()
    if a.epochs:
        # _fit is defined in __main__, so loky pickles it by value with the
        # parent's globals: set them here, not only in the environment
        global EPOCHS, SAVE_AT
        EPOCHS, SAVE_AT = a.epochs, (500, a.epochs)
        os.environ["GLOBAL_ETA_EPOCHS"] = str(a.epochs)
    if a.outdir:                       # read by the worker processes at import
        os.environ["GLOBAL_ETA_OUT"] = a.outdir
        global OUT_SIM, OUT_TCGA
        OUT_SIM = OUT_TCGA = a.outdir
    jobs = [(d, r, k, o) for d in a.data for r in (a.reps if a.reps is not None else REPS[d])
            for k in (a.k or K_GRID[d]) for o in a.opt]
    jobs.sort(key=lambda j: -j[2])                     # longest first
    from joblib import Parallel, delayed
    Parallel(n_jobs=a.workers, backend="loky")(delayed(fit_one)(*j) for j in jobs)


if __name__ == "__main__":
    main()
