"""
Multi-objective Optuna search for SupMultiviewShared's optimiser and stopping-rule
settings (rerun of the study in sim/exploration/shared_cv_optuna.qmd).

Objectives (all minimised, averaged over K-fold CV on one training dataset), as
in the original study:
    outcome_mse   validation-fold MSE of the standardised outcome
    cov_frob      covariance reconstruction error of the training-fold fit
    train_time    wall-clock seconds of global training per fold (incl. SVD init)

Changes from python/optuna_cv.py (which is left unchanged):
  - stopping rule: Tran, Nguyen & Dao (2021) moving-average loss with patience
    (convergence_criterion "ma_patience"), its window and patience tuned;
    min_epochs fixed per study (run once with 0 and once with 200);
  - SVD initialisation of the scores and loadings, as in the simulation runs;
  - validation predictions in closed form from the trained guide
    (viper_predict.closed_form_scores) by default; --predict local_svi uses the
    previous local SVI (needed for non-Gaussian outcomes, e.g. right censoring,
    for which no closed form is implemented);
  - covariance error from the implied covariance Lambda Sigma_Z Lambda^T, Sigma_Z the
    fitted training scores' second moment (docs/evaluation_metrics.md) by default;
    --cov loadings gives the previous Lambda Lambda^T;
  - per-run seeding that is actually applied (multiview_shared resets the seed
    when first imported, so it is imported before seeding).
Diagnostics stored per trial (not objectives): mean epochs run per fold and, for
simulated data, the mean exact relative excess risk of the fold models
(risk_metrics), so the Pareto front can be read against the truth.

Search space (TODO(user): ranges): eta, t, minibatch_fraction as in the original
study; ma_window and ma_patience (log-uniform integers). Fixed: delta 1e-16, one
particle, max_epochs (--max-epochs, default 1500), K_fit = K + 10, MGP prior,
outcome_weight 1.

Low-variance study (November 2026): --space lv --init-scale-lambda 0.01
--init-scale-z 0.1 searches the narrowed optimiser range around the validated
settings with the low-variance guide start (defaults reproduce the earlier studies).

Full-length study (November 2026; --full-length --chunk 50 --space full): every fold
trains for exactly --max-epochs with no early stopping (training in chunks of --chunk
epochs, the optimiser state carried over, as in controlled_training_length.py). After
every chunk the fold's fit is scored: validation outcome MSE, covariance error, the
diagnostic exact risk, and optimisation time so far (SVD init + SVI, scoring
excluded). The objectives are the values at the last epoch. The per-checkpoint scores
go to <study>_curves.csv and each fold's per-epoch loss to
<study>_losses/trial<i>_fold<f>.npy, so stopping rules can be replayed afterwards
(optuna_stopping_replay.py) and every (setting, rule) pair scored by CV MSE and time.
Window and patience are therefore not searched (space "full": eta, t, minibatch only).
--sampler tpe uses multivariate TPE with --n-startup random trials first (Optuna's
recommended budget for TPE is 100-1000 trials, for NSGA-II 100-10000; NSGA-II's first
generation, 50 trials by default, is pure random sampling).

Seeding (--seed-from OLD_trials.csv, with --full-length): before a new study's first
trial, the earlier study's trials whose folds all ran the full --max-epochs (so its
stopping rule never fired) and none failed are added as completed trials
(study.add_trial), with their eta, t, minibatch fraction and objectives. They are the
same experiment as a new full-length trial when the data, folds, seeds, guide start and
epoch budget match. The CSV does not record the data or the guide start, so these are
not checked: the caller must pass the earlier study on the same data and settings.
Their recorded wall times were distorted by load, so the time objective is replaced by
epochs x fitted seconds per epoch (a + b / minibatch fraction, least squares over the
old study's trials, refitted twice without trials more than 1.5x above the fit). Each is
tagged with user attributes seeded_from, old_trial, old_ma_window, old_ma_patience.
With seeding, TPE's random warm-up is n_seeded + --n-startup trials, so --n-startup
random new trials are still drawn over the full range. Seeding happens only when the
study has no trials (resuming does not add them again). Seeded trials have no curves
or losses, so optuna_stopping_replay.py covers only the study's own trials.

Results: an SQLite study (--storage) plus a CSV of all trials next to it.

usage: python optuna_controlled.py --data controlled:200,3000,1,11 --min-epochs 0
           --trials 50 --folds 5 --study NAME [--storage sqlite:///PATH.db]
       python optuna_controlled.py --data misaligned:anti_aligned,500,1000,1 --min-epochs 200 ...
       (original = the aligned dataset of the first study: multi_view_shared/k.5,
        n500p1000_snr2.2_sparse0.25, rep 1)
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

SIM = os.path.expanduser("~/multiomic_integration/sim")
OUT_DIR = os.path.join(SIM, "results", "integration_shared_controlled", "optuna")
MAX_EPOCHS = 1500
K_DELTA = 10

# TODO(user): search ranges. t and minibatch_fraction as in the original study; eta's
# lower bound raised from 0.001 to 0.01 (smaller steps never trigger the rule within
# the epoch cap, and each such trial costs the full cap).
PARAM_SPACE = {
    "eta":                {"type": "float", "low": 0.01,  "high": 1.0, "log": True},
    "t":                  {"type": "float", "low": 0.01,  "high": 1.0, "log": True},
    "minibatch_fraction": {"type": "float", "low": 0.05,  "high": 0.5, "log": True},
    "ma_window":          {"type": "int",   "low": 10,    "high": 200, "log": True},
    "ma_patience":        {"type": "int",   "low": 10,    "high": 500, "log": True},
}


# Low-variance guide start (November 2026; --space lv): the optimiser over the range
# spanned by the settings validated so far (current: eta 0.25, t 0.1, minibatch 0.15;
# long_eta25: 0.25, 0.025, 0.05; long: 0.35, 0.025, 0.05), widened below to eta 0.15.
# TODO(user): ranges, including the stopping rule's (narrower than the original
# space's because max_epochs is 1000 here).
PARAM_SPACE_LV = {
    "eta":                {"type": "float", "low": 0.15,  "high": 0.35, "log": True},
    "t":                  {"type": "float", "low": 0.025, "high": 0.1,  "log": True},
    "minibatch_fraction": {"type": "float", "low": 0.05,  "high": 0.15, "log": True},
    "ma_window":          {"type": "int",   "low": 10,    "high": 200,  "log": True},
    "ma_patience":        {"type": "int",   "low": 10,    "high": 300,  "log": True},
}
# Full-length study (--space full, with --full-length): the optimiser only (the stopping
# rule is replayed afterwards), widened below and above the low-variance study's range,
# whose best trials sat at the lower edges of t and the minibatch fraction
# (sim/exploration/optuna_lowvar.qmd). TODO(user): ranges.
PARAM_SPACE_FULL = {
    "eta":                {"type": "float", "low": 0.15,  "high": 0.40, "log": True},
    "t":                  {"type": "float", "low": 0.01,  "high": 0.1,  "log": True},
    "minibatch_fraction": {"type": "float", "low": 0.03,  "high": 0.20, "log": True},
}
SPACES = {"original": PARAM_SPACE, "lv": PARAM_SPACE_LV, "full": PARAM_SPACE_FULL}


# The setting used so far (sim_run_shared_parallel.py): elbo_snr, window 50, threshold 5.
BASELINE = {"eta": 0.25, "t": 0.1, "minibatch_fraction": 0.15,
            "convergence_criterion": "elbo_snr", "window": 50, "snr_threshold": 5.0}


def load_training(spec):
    """Training dataset for the study. spec: 'original', 'controlled:n,P,c,rep' or
    'misaligned:label,n,p_l,rep'."""
    from run_methods import load_rds
    if spec == "original":
        return load_rds(f"{SIM}/data/multi_view_shared/k.5/train/n500p1000_snr2.2_sparse0.25/sim_data_shared_rep1.rds")
    if spec.startswith("misaligned:"):
        label, n, p, rep = spec.split(":")[1].split(",")
        return load_rds(f"{SIM}/data/multi_view_shared_misaligned/k.5/train/"
                        f"n{int(n)}p{int(p)}_snr2.2_sparse0.25_align_{label}/sim_data_shared_rep{int(rep)}.rds")
    n, P, c, rep = spec.split(":")[1].split(",")
    cond = f"n{int(n)}_P{int(P)}_c{float(c):g}"
    return load_rds(f"{SIM}/data/multi_view_shared_controlled/k.5/train/{cond}/"
                    f"sim_data_controlled_{cond}_rep{int(rep)}.rds")


def run_fold(sim, X_raw, y_raw, filters, tr_idx, va_idx, cfg, opts, seed):
    """Train on one fold and return its metrics."""
    import pyro
    import multiview_shared  # noqa: F401  (import before seeding; see module docstring)
    from torch.utils.data import TensorDataset
    from data_utils import normalize_tensor_by_col, implied_cov_errors
    from run_methods import train_model, train_model_locally
    from constants import Params, Sites
    from viper_predict import closed_form_scores, guide_moments
    import risk_metrics as rm

    Xtr = [X[tr_idx] for X in X_raw]
    Xva = [X[va_idx] for X in X_raw]
    norms = [normalize_tensor_by_col(X) for X in Xtr]
    Xtr_s = [d["data_clean"] for d in norms]
    means, sds = [d["means"] for d in norms], [d["sds"] for d in norms]
    Xva_s = [normalize_tensor_by_col(X, m, s)["data_clean"] for X, m, s in zip(Xva, means, sds)]
    ym, ysd = y_raw[tr_idx].mean(), y_raw[tr_idx].std()
    ytr, yva = (y_raw[tr_idx] - ym) / ysd, (y_raw[va_idx] - ym) / ysd
    train_ds = TensorDataset(torch.arange(len(tr_idx)), *Xtr_s, ytr)

    train_config = {"convergence_criterion": "ma_patience", **cfg,
                    "min_epochs": opts["min_epochs"], "max_epochs": opts["max_epochs"],
                    "delta": 1e-16, "num_particles": 1}
    model_config = {"k": int(sim["K"]) + K_DELTA, "dense_model": False,
                    "init_scale_Lambda": opts.get("init_scale_Lambda", 1.0),
                    "init_scale_Z": opts.get("init_scale_Z", 1.0)}
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    if opts.get("chunk"):
        return _run_fold_full(sim, filters, Xva_s, yva, ym, ysd, means, sds, train_ds, model_config,
                              train_config, opts)
    t0 = time.time()
    model, handler, state = train_model(model_config, train_config, train_ds, None, model_type="shared",
                                        opt="adagrad", write=False, svd_init=True)
    train_time = time.time() - t0                    # SVD initialisation + SVI
    epochs = model.total_epochs
    store = pyro.get_param_store()

    if opts["predict"] == "closed_form":
        mom = guide_moments(store, len(X_raw))
        cf = closed_form_scores(mom, Xva_s)
        yhat, coef = cf["yhat"].float(), cf["coef"]
    else:
        va_ds = TensorDataset(torch.arange(len(va_idx)), *Xva_s, yva)
        model, vh, _ = train_model_locally(model, train_config, va_ds, model_type="shared", opt="adagrad")
        post = vh.predict(Xva_s, 100, [Sites.y_pred])
        yhat, coef = post[Sites.y_pred].mean(0).reshape(-1), None
    mse = float(((yhat.reshape(-1) - yva.reshape(-1).float()) ** 2).mean())

    out = {"outcome_mse": mse, "train_time": train_time, "epochs": epochs}
    if sim.get("Lambda_l") is not None:
        dp = {"SIM_decomp": {"SIM_Lambda_l_list": sim["Lambda_l"]}, "X_l_list_column_filters": filters,
              "X_l_sd_list": sds, "X_l_mean_list": means}
        Lam = [store[Params.loc_Lambda_l.format(l=l)].detach().double().squeeze() for l in range(len(X_raw))]
        if opts["cov"] == "implied":
            M = store[Params.loc_Z].detach().double()
            S = store[Params.scale_Z].detach().double()
            Sigma_Z = (M.T @ M + torch.diag((S ** 2).sum(0))) / M.shape[0]
        else:
            Sigma_Z = None
        out["cov_frob"] = float(implied_cov_errors(Lam, Sigma_Z, dp).cov_frob.mean())
        if coef is not None:                         # exact risk of this fold's model (diagnostic)
            theta = rm.true_params(sim)
            out["risk_rel"] = rm.risk_summary(coef, dp, float(ym), float(ysd), theta, rm.oracle(theta))["risk_rel"]
    pyro.clear_param_store()
    return out


def _score_full(store, sim, filters, Xva_s, yva, ym, ysd, means, sds, L):
    """Validation MSE (closed form), implied-covariance error and exact risk of the
    current fit, for the full-length mode."""
    import risk_metrics as rm
    from constants import Params
    from data_utils import implied_cov_errors
    from viper_predict import closed_form_scores, guide_moments
    cf = closed_form_scores(guide_moments(store, L), Xva_s)
    out = {"outcome_mse": float(((cf["yhat"].float().reshape(-1) - yva.reshape(-1).float()) ** 2).mean())}
    if sim.get("Lambda_l") is not None:
        dp = {"SIM_decomp": {"SIM_Lambda_l_list": sim["Lambda_l"]}, "X_l_list_column_filters": filters,
              "X_l_sd_list": sds, "X_l_mean_list": means}
        Lam = [store[Params.loc_Lambda_l.format(l=l)].detach().double().squeeze() for l in range(L)]
        M = store[Params.loc_Z].detach().double()
        S = store[Params.scale_Z].detach().double()
        Sigma_Z = (M.T @ M + torch.diag((S ** 2).sum(0))) / M.shape[0]
        out["cov_frob"] = float(implied_cov_errors(Lam, Sigma_Z, dp).cov_frob.mean())
        theta = rm.true_params(sim)
        out["risk_rel"] = rm.risk_summary(cf["coef"], dp, float(ym), float(ysd), theta, rm.oracle(theta))["risk_rel"]
    return out


def _run_fold_full(sim, filters, Xva_s, yva, ym, ysd, means, sds, train_ds, model_config, train_config, opts):
    """Full-length mode (opts["chunk"]): train exactly opts["max_epochs"] epochs in chunks,
    scoring after each chunk. Returns the last checkpoint's metrics plus "curve" (one
    dict per checkpoint) and "loss" (per-epoch training loss). The stopping rule in
    train_config never fires: each chunk runs with min_epochs = epochs = chunk."""
    import pyro
    from run_methods import train_model, _minibatch_size
    chunk, max_epochs, L = opts["chunk"], opts["max_epochs"], len(Xva_s)
    # optional finer checkpoints early on: every opts["early_chunk"] epochs until opts["early_until"]
    step = lambda e: opts["early_chunk"] if opts.get("early_chunk") and e < opts.get("early_until", 0) else chunk
    first = step(0)
    cfg = {**train_config, "min_epochs": first, "max_epochs": first}
    t0 = time.time()
    model, handler, _ = train_model(model_config, cfg, train_ds, None, model_type="shared",
                                    opt="adagrad", write=False, svd_init=True)
    train_time = time.time() - t0                    # SVD initialisation + SVI, scoring excluded
    curve, epoch = [], first
    while True:
        curve.append({"epoch": epoch, "train_time": train_time,
                      **_score_full(pyro.get_param_store(), sim, filters, Xva_s, yva, ym, ysd, means, sds, L)})
        if epoch >= max_epochs:
            break
        nxt = step(epoch)
        t1 = time.time()
        handler.do_inference(train_dataset=train_ds, min_epochs=nxt, epochs=nxt, minibatch_flag=True,
                             minibatch_size=_minibatch_size(cfg, len(train_ds)), variational_diff_func=np.mean,
                             verbose=False, convergence_criterion=cfg["convergence_criterion"],
                             ma_window=cfg.get("ma_window", 50), ma_patience=cfg.get("ma_patience", 50))
        train_time += time.time() - t1
        epoch += nxt
    loss = np.asarray(model.loss_history, float)
    pyro.clear_param_store()
    return {**curve[-1], "epochs": model.total_epochs, "curve": curve, "loss": loss}


def seed_trials(old_csv, space, max_epochs):
    """Completed-trial records from an earlier study's trials CSV for seeding (see the
    module docstring): trials whose folds all ran max_epochs, with the fitted time."""
    from optuna.distributions import FloatDistribution, IntDistribution
    from optuna.trial import create_trial
    d = pd.read_csv(old_csv)
    d = d[d.state == "COMPLETE"].copy()
    names = {"eta": "params_eta", "t": "params_t", "minibatch_fraction": "params_minibatch_fraction"}
    assert set(space) == set(names), "seeding is implemented for the eta / t / minibatch space"
    spe = d.values_2 / d.user_attrs_mean_epochs
    X = np.c_[np.ones(len(d)), 1 / d.params_minibatch_fraction]
    ok = np.ones(len(d), bool)
    for _ in range(3):
        coef = np.linalg.lstsq(X[ok], spe.values[ok], rcond=None)[0]
        ok = spe.values < 1.5 * (X @ coef)
    d["fitted_time"] = d.user_attrs_mean_epochs * (X @ coef)
    keep = d[(d.user_attrs_mean_epochs == max_epochs) & (d.user_attrs_folds_ok == d.user_attrs_folds_ok.max())]
    dist = lambda v: (IntDistribution if v["type"] == "int" else FloatDistribution)(
        v["low"], v["high"], log=v.get("log", False))
    dists = {k: dist(v) for k, v in space.items()}
    out = []
    for _, r in keep.iterrows():
        params = {k: float(r[c]) for k, c in names.items()}
        if not all(space[k]["low"] <= v <= space[k]["high"] for k, v in params.items()):
            continue
        out.append(create_trial(
            params=params, distributions=dists, values=[float(r.values_0), float(r.values_1), float(r.fitted_time)],
            user_attrs={"seeded_from": os.path.basename(old_csv), "old_trial": int(r.number),
                        "old_ma_window": int(r.params_ma_window), "old_ma_patience": int(r.params_ma_patience),
                        "old_wall_time": float(r.values_2), "mean_epochs": float(r.user_attrs_mean_epochs),
                        "mean_risk_rel": float(r.user_attrs_mean_risk_rel), "folds_ok": int(r.user_attrs_folds_ok)}))
    return out


def suggest(trial, space):
    """trial.suggest_* for each entry (integers and floats, optionally on a log scale)."""
    out = {}
    for name, spec in space.items():
        f = trial.suggest_int if spec["type"] == "int" else trial.suggest_float
        out[name] = f(name, spec["low"], spec["high"], log=spec.get("log", False))
    return out


def main():
    import optuna
    from run_methods import make_cv_folds
    from data_utils import zero_variance_col_filter

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--min-epochs", type=int, required=True)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--predict", choices=["closed_form", "local_svi"], default="closed_form")
    ap.add_argument("--cov", choices=["implied", "loadings"], default="implied")
    ap.add_argument("--space", choices=list(SPACES), default="original")
    ap.add_argument("--init-scale-lambda", type=float, default=1.0, help="initial guide sd of the loadings")
    ap.add_argument("--init-scale-z", type=float, default=1.0, help="initial guide sd of the training scores")
    ap.add_argument("--study", required=True)
    ap.add_argument("--baseline", action="store_true",
                    help="evaluate the current pipeline setting (BASELINE) with the same folds; "
                         "writes <study>_baseline.csv instead of running a study")
    ap.add_argument("--storage", default=None)
    ap.add_argument("--full-length", action="store_true",
                    help="train every fold for exactly --max-epochs, scoring every --chunk epochs; "
                         "writes <study>_curves.csv and <study>_losses/ for stopping-rule replays")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--sampler", choices=["nsga2", "tpe"], default="nsga2")
    ap.add_argument("--n-startup", type=int, default=20,
                    help="random trials before TPE starts modelling (with --seed-from: random new trials "
                         "after the seeded ones)")
    ap.add_argument("--seed-from", default=None,
                    help="earlier study's trials CSV whose full-length trials seed a new study")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    storage = args.storage or f"sqlite:///{os.path.join(OUT_DIR, args.study)}.db"

    sim = load_training(args.data)
    filters = [zero_variance_col_filter(X) for X in sim["X_l"]]      # as in the original study
    X_raw = [X[:, f] for X, f in zip(sim["X_l"], filters)]
    y_raw = sim["y"].reshape(-1)
    folds = make_cv_folds(X_raw[0].shape[0], n_folds=args.folds, seed=args.seed)
    opts = {"min_epochs": args.min_epochs, "max_epochs": args.max_epochs, "predict": args.predict, "cov": args.cov,
            "init_scale_Lambda": args.init_scale_lambda, "init_scale_Z": args.init_scale_z,
            "chunk": args.chunk if args.full_length else None}
    if args.full_length:
        loss_dir = os.path.join(OUT_DIR, f"{args.study}_losses")
        os.makedirs(loss_dir, exist_ok=True)
        curves_csv = os.path.join(OUT_DIR, f"{args.study}_curves.csv")

    if args.baseline:
        rows = [{"fold": i, **run_fold(sim, X_raw, y_raw, filters, tr, va, BASELINE, opts, seed=args.seed + i)}
                for i, (tr, va) in enumerate(folds)]
        out = pd.DataFrame(rows).assign(data=args.data, min_epochs=args.min_epochs, **BASELINE)
        out.to_csv(os.path.join(OUT_DIR, f"{args.study}_baseline.csv"), index=False)
        print(out.mean(numeric_only=True).round(4).to_string())
        return

    def objective(trial):
        cfg = suggest(trial, SPACES[args.space])
        if args.full_length:                         # never fires: see _run_fold_full
            cfg = {**cfg, "convergence_criterion": "ma_patience"}
        res = []
        for i, (tr, va) in enumerate(folds):
            try:
                r = run_fold(sim, X_raw, y_raw, filters, tr, va, cfg, opts, seed=args.seed + i)
            except Exception as e:                   # e.g. a diverging optimiser setting
                print(f"  trial {trial.number} fold {i} failed: {repr(e)[:120]}", flush=True)
                continue
            if args.full_length:
                np.save(os.path.join(loss_dir, f"trial{trial.number}_fold{i}.npy"), r.pop("loss"))
                cur = pd.DataFrame(r.pop("curve")).assign(trial=trial.number, fold=i,
                                                          **{k: v for k, v in cfg.items() if k in SPACES[args.space]})
                cur.to_csv(curves_csv, mode="a", header=not os.path.exists(curves_csv), index=False)
            res.append(r)
        if not res:
            raise optuna.TrialPruned()
        mean = lambda k: float(np.mean([r[k] for r in res if k in r]))
        for k in ("epochs", "risk_rel"):
            if any(k in r for r in res):
                trial.set_user_attr(f"mean_{k}", mean(k))
        trial.set_user_attr("folds_ok", len(res))
        print(f"trial {trial.number}: mse {mean('outcome_mse'):.3f} cov {mean('cov_frob'):.4f} "
              f"time {mean('train_time'):.0f}s epochs {mean('epochs'):.0f} {cfg}", flush=True)
        return mean("outcome_mse"), mean("cov_frob"), mean("train_time")

    study = optuna.create_study(study_name=args.study, storage=storage, load_if_exists=True,
                                directions=["minimize", "minimize", "minimize"],
                                sampler=optuna.samplers.NSGAIISampler(seed=args.seed))
    if args.seed_from:
        assert args.full_length, "--seed-from needs --full-length (seeded trials ran the full budget)"
        if not study.trials:
            seeds = seed_trials(args.seed_from, SPACES[args.space], args.max_epochs)
            study.add_trials(seeds)
            print(f"seeded {len(seeds)} full-length trials from {args.seed_from}", flush=True)
    n_seeded = sum("seeded_from" in t.user_attrs for t in study.trials)
    if args.sampler == "tpe":
        study.sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True,
                                                   n_startup_trials=n_seeded + args.n_startup)
    for k, v in {"data": args.data, "min_epochs": args.min_epochs, "predict": args.predict,
                 "cov": args.cov, "folds": args.folds, "max_epochs": args.max_epochs, "space": args.space,
                 "init_scale_Lambda": args.init_scale_lambda, "init_scale_Z": args.init_scale_z,
                 "full_length": args.full_length, "chunk": args.chunk, "sampler": args.sampler,
                 "n_startup": args.n_startup, "seed_from": args.seed_from, "n_seeded": n_seeded}.items():
        study.set_user_attr(k, v)
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)
    df = study.trials_dataframe()
    df.to_csv(os.path.join(OUT_DIR, f"{args.study}_trials.csv"), index=False)
    print(f"{len(df)} trials -> {OUT_DIR}/{args.study}_trials.csv")


if __name__ == "__main__":
    main()
