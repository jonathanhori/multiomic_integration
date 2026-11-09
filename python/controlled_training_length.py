"""
Training-length study for VIPER-Shared on the controlled-strength data: does more
optimisation close the gap to PCA + OLS, and when should training stop?

Every Phase 2 fit stopped at epoch 202-205: the elbo_snr rule (window 50,
snr_threshold 5) fires as soon as min_epochs = 200 is passed, while the loss was
still falling ~2-3% per 50 epochs. Here each dataset is trained from the same
SVD initialisation and settings as sim_run_shared_controlled.py, but with no
early stopping, for MAX_EPOCHS epochs in chunks of CHUNK epochs (the optimiser
and its AdagradRMSProp state carry over between chunks). After every chunk the
current fit is scored with controlled_viper_diagnosis.viper_variants (exact risk
of the own / ols_xonly / ols_locZ predictors, kappa2_hat).

Outputs (sim/results/integration_shared_controlled/training_length/<label>/)
    checkpoints.csv   one row per dataset x checkpoint x variant; coef_rel_change is
                      ||b_t - b_{t-1}|| / ||b_t|| for the model's own coefficients
                      (a truth-free signal for stopping rules)
                      with cfg["cov"] (controlled_optimizer_grid.py --cov) also
                      train_time (optimisation seconds so far, scoring excluded) and
                      cov_within, cov_cross, cov_frob (covariance reconstruction, _cov_errors)
                      with cfg["test_mse"] (controlled_optimizer_grid.py --test-mse) also
                      test_mse (held-out outcome MSE of the closed-form predictions, _test_mse)
                      with cfg["recovery"] (controlled_optimizer_grid.py --recovery) also
                      factor recovery rec_*, cross-method factor comparison xm_* and 95%
                      coverage cov95_* / width95_* (checkpoint_metrics.py), metrics_time
    loss_<stem>.npy   per-epoch training loss (summed over minibatches), for
                      replaying candidate stopping rules offline

Optimiser settings can be overridden (--eta, --delta, --t, --minibatch-fraction,
--num-particles), as can the guide's initial sds (--init-scale-lambda, --init-scale-z, --init-param); --label names the configuration's output folder. A run whose
optimisation fails (e.g. NaN parameters) keeps its checkpoints so far and adds a
row with column "failed". controlled_optimizer_grid.py runs many configurations.

usage: python controlled_training_length.py --cells 500,3000,1 [...] --reps 1 2
       [--max-epochs 2000] [--chunk 100] [--workers 4] [--label base] [--eta ...]
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(1, "/Users/jonathanhori/multiomic_integration/python")
warnings.filterwarnings("ignore")

RESULTS = os.path.expanduser("~/multiomic_integration/sim/results/integration_shared_controlled/training_length")


def _train_per_param(model_config, train_config, train_subset, eta_loc):
    """As run_methods.train_model (SVD initialisation, first chunk of epochs), but with
    AdagradRMSProp step scale eta_loc for the location and scale parameters (names
    starting loc_ / scale_: loadings, scores, beta) and train_config["eta"] for the
    rest (the Gamma / inverse-gamma shape and rate parameters a_*, b_*)."""
    import numpy as np
    from pyro.infer import Trace_ELBO
    from pyro.optim import AdagradRMSProp
    from handler import ModelHandler
    from run_methods import _build_model, _convergence_kwargs, _minibatch_size, compute_svd_init

    X_l_list = list(train_subset.tensors[1:-1])
    Z_init, Lambda_init = compute_svd_init(X_l_list, model_config["k"])
    model = _build_model("shared", model_config, "cpu", Z_init, Lambda_init)

    def optim_args(name):
        big = name.startswith("loc_") or name.startswith("scale_")
        return {"eta": eta_loc if big else train_config["eta"], "delta": train_config["delta"],
                "t": train_config["t"]}

    handler = ModelHandler("train", model, opt=AdagradRMSProp(optim_args),
                           loss=Trace_ELBO(num_particles=train_config["num_particles"]),
                           opt_scheduler=None, orthogonal_projection=True, device="cpu")
    handler.do_inference(train_dataset=train_subset, min_epochs=train_config["min_epochs"],
                         epochs=train_config["max_epochs"], minibatch_flag=True,
                         minibatch_size=_minibatch_size(train_config, len(train_subset)),
                         variational_diff_func=np.mean, verbose=False, **_convergence_kwargs(train_config))
    return model, handler


def _cov_errors(dp):
    """Covariance reconstruction of the current fit (cfg["cov"] = True): mean squared
    entrywise error of Lambda_l Sigma_Z Lambda_m^T against the true factor covariance,
    standardised scale (data_utils.implied_cov_errors, as in the Optuna studies), with
    Lambda the guide means and Sigma_Z = (M^T M + sum_i diag(s_i^2)) / n from the
    training-score guide. Returns cov_within / cov_cross (means over view pairs) and
    cov_frob (mean over all pairs, the Optuna objective)."""
    import pyro
    import torch
    from constants import Params
    from data_utils import implied_cov_errors
    st = pyro.get_param_store()
    M = st[Params.loc_Z].detach().double()
    s2 = st[Params.scale_Z].detach().double() ** 2
    Sigma_Z = (M.T @ M + torch.diag(s2.sum(0))) / M.shape[0]
    Lam = [st[Params.loc_Lambda_l.format(l=l)].detach().double() for l in range(dp["L"])]
    t = implied_cov_errors(Lam, Sigma_Z, dp)
    return {"cov_within": float(t[t.cov_type == "within"].cov_frob.mean()),
            "cov_cross": float(t[t.cov_type == "cross"].cov_frob.mean()),
            "cov_frob": float(t.cov_frob.mean())}


def _test_mse(dp):
    """Held-out outcome MSE of the current fit (cfg["test_mse"] = True): closed-form
    predictions from the guide (viper_predict.closed_form_scores, as in the Optuna
    studies' validation folds) on the dataset's test set, standardised outcome scale
    (training mean and sd). Needs no true parameters."""
    import pyro
    from viper_predict import closed_form_scores, guide_moments
    cf = closed_form_scores(guide_moments(pyro.get_param_store(), dp["L"]), dp["test_X_l_list_clean"])
    y = dp["test_y_clean"].double().reshape(-1)
    return {"test_mse": float(((cf["yhat"].double().reshape(-1) - y) ** 2).mean())}


def run_one(n, P, c, rep, cfg, out_dir):
    import pyro
    import torch
    import risk_metrics as rm
    import sim_run_shared_parallel as base
    from run_methods import process_data, train_model, _minibatch_size
    from controlled_viper_diagnosis import cond_name, load_dataset, viper_variants

    stem = f"{cond_name(n, P, c)}_rep{rep}"
    tr, te = load_dataset(n, P, c, rep)
    train_subset, _, dp = process_data(tr, te, model_type="shared")
    theta = rm.true_params(tr)
    orc = rm.oracle(theta)
    rel = [k for k in range(theta["beta"].numel()) if float(theta["beta"][k]) != 0]
    y_raw = tr["y"].double().reshape(-1)
    y_mean, y_sd = float(y_raw.mean()), float(y_raw.std())

    chunk = cfg["chunk"]
    # Optional finer checkpoints early in training (cfg["early_chunk"] epochs until
    # epoch cfg["early_until"], then cfg["chunk"]), e.g. to score fits at the time a
    # fast comparison method takes; the training itself is unchanged.
    step = lambda e: cfg["early_chunk"] if cfg.get("early_chunk") and e < cfg.get("early_until", 0) else chunk
    first = step(0)
    train_config = {
        "eta": cfg["eta"], "delta": cfg.get("delta", base.DELTA), "t": cfg["t"],
        "num_particles": cfg["num_particles"],
        "minibatch_fraction": cfg["minibatch_fraction"], "min_epochs": first, "max_epochs": first,
        "convergence_criterion": base.CONVERGENCE_CRITERION, "window": base.WINDOW,
        "snr_threshold": base.SNR_THRESHOLD,
    }
    model_config = {"k": int(tr["K"]) + 10, "dense_model": False, "outcome_weight": cfg["alpha"],
                    "init_scale_Lambda": cfg.get("init_scale_Lambda", 1.0), "init_scale_Z": cfg.get("init_scale_Z", 1.0),
                    "init_param": cfg.get("init_param", 1.0)}
    key = {"n": n, "P_grid": P, "c": c, "rep": rep, "label": cfg["label"],
           "eta": train_config["eta"], "delta": train_config["delta"], "t": train_config["t"],
           "minibatch_fraction": train_config["minibatch_fraction"],
           "eta_loc": cfg.get("eta_loc"), "num_particles": train_config["num_particles"],
           "init_scale_Lambda": model_config["init_scale_Lambda"], "init_scale_Z": model_config["init_scale_Z"],
           "init_param": model_config["init_param"],
           "g_over_sqrt_gamma": float(tr["g_true"].item()) / np.sqrt(P / n)}

    # multiview_shared calls pyro.set_rng_seed(0) when first imported, which
    # run_methods does lazily on the first model build in a process; import it
    # before seeding so the seed below is the one the run uses.
    import multiview_shared  # noqa: F401
    pyro.clear_param_store()
    pyro.set_rng_seed(rep)
    t0 = time.time()
    from viper_predict import closed_form_scores, guide_moments
    rows = []
    epoch = first
    prev_b = None
    refs = None
    if cfg.get("recovery"):                       # comparison methods' factors, once per dataset
        from checkpoint_metrics import reference_fits
        from controlled_scoring import RESULTS as _CR
        refs = reference_fits(dp, os.path.join(_CR, "mofa", f"mofa_ctrl_{stem}.npz"))
    t_train = time.time()
    try:
        if cfg.get("eta_loc") is None:
            model, handler, _ = train_model(model_config, train_config, train_subset, None, model_type="shared",
                                            opt="adagrad", write=False, svd_init=True)
        else:
            model, handler = _train_per_param(model_config, train_config, train_subset, cfg["eta_loc"])
    except Exception as e:                       # e.g. a diverging optimiser configuration
        pyro.clear_param_store()
        return [{**key, "epoch": 0, "failed": repr(e)[:200]}]
    train_time = time.time() - t_train            # optimisation only (scoring excluded)
    while True:
        lh = np.asarray(model.loss_history, float)
        # Truth-free convergence signal for stopping rules: relative change of the
        # model's own linear predictor coefficients (standardised scale) since the
        # previous checkpoint.
        b = torch.cat(closed_form_scores(guide_moments(pyro.get_param_store(), dp["L"]),
                                         [x[:1] for x in dp["train_X_l_list_clean"]])["coef"])
        coef_change = float((b - prev_b).norm() / b.norm()) if prev_b is not None else float("nan")
        prev_b = b
        extra = {"train_time": train_time, **_cov_errors(dp)} if cfg.get("cov") else {}
        if cfg.get("test_mse"):
            extra.update(_test_mse(dp))
        if cfg.get("recovery"):
            # opt-in: factor recovery, cross-method comparison and coverage (checkpoint_metrics);
            # the Monte Carlo outcome interval only every cfg["mc_every"] epochs and at the end
            from checkpoint_metrics import viper_metrics
            mc = cfg.get("mc_draws", 0) if (epoch % cfg.get("mc_every", 100) == 0 or epoch >= cfg["max_epochs"]) else 0
            t_m = time.time()
            extra.update(viper_metrics(pyro.get_param_store(), dp, theta, refs, mc_draws=mc))
            extra["metrics_time"] = time.time() - t_m
        try:
            for r in viper_variants(pyro.get_param_store(), dp, theta, orc, y_mean, y_sd, rel):
                rows.append({**key, "epoch": epoch, "loss_last_chunk": float(lh[-chunk:].mean()),
                             "coef_rel_change": coef_change, "elapsed": time.time() - t0, **extra, **r})
            if epoch >= cfg["max_epochs"]:
                break
            nxt = step(epoch)
            t_train = time.time()
            handler.do_inference(train_dataset=train_subset, min_epochs=nxt, epochs=nxt, minibatch_flag=True,
                                 minibatch_size=_minibatch_size(train_config, len(train_subset)),
                                 variational_diff_func=np.mean, verbose=False,
                                 convergence_criterion=train_config["convergence_criterion"],
                                 window=train_config["window"], snr_threshold=train_config["snr_threshold"])
            train_time += time.time() - t_train
        except Exception as e:                   # NaN parameters etc.: keep the checkpoints so far
            rows.append({**key, "epoch": epoch, "failed": repr(e)[:200]})
            break
        epoch += nxt
    tag = "" if cfg["label"] == os.path.basename(out_dir) else f"{cfg['label']}_"
    np.save(os.path.join(out_dir, f"loss_{tag}{stem}.npy"), np.asarray(model.loss_history, float))
    pyro.clear_param_store()
    return rows


def main():
    import sim_run_shared_parallel as base
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True, help="n,P,c triples")
    ap.add_argument("--reps", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--max-epochs", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--label", default="base")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=base.ETA)
    ap.add_argument("--t", type=float, default=base.T)
    ap.add_argument("--delta", type=float, default=base.DELTA)
    ap.add_argument("--minibatch-fraction", type=float, default=base.MINIBATCH_FRACTION)
    ap.add_argument("--num-particles", type=int, default=base.NUM_PARTICLES)
    ap.add_argument("--init-scale-lambda", type=float, default=1.0, help="initial guide sd of the loadings")
    ap.add_argument("--init-scale-z", type=float, default=1.0, help="initial guide sd of the scores")
    ap.add_argument("--init-param", type=float, default=1.0,
                    help="initial shape/rate of the MGP Gamma guides and initial guide sd of beta")
    args = ap.parse_args()

    cfg = {"label": args.label, "chunk": args.chunk, "max_epochs": args.max_epochs, "alpha": args.alpha,
           "eta": args.eta, "delta": args.delta, "t": args.t, "minibatch_fraction": args.minibatch_fraction,
           "num_particles": args.num_particles,
           "init_scale_Lambda": args.init_scale_lambda, "init_scale_Z": args.init_scale_z,
           "init_param": args.init_param}
    out_dir = os.path.join(RESULTS, args.label)
    os.makedirs(out_dir, exist_ok=True)
    pd.Series(cfg).to_csv(os.path.join(out_dir, "config.csv"))
    out = os.path.join(out_dir, "checkpoints.csv")
    done = set()
    if os.path.exists(out):
        prev = pd.read_csv(out)
        done = set(zip(prev.n, prev.P_grid, prev.c, prev.rep))
    jobs = []
    for cell in args.cells:
        n, P, c = cell.split(",")
        for r in args.reps:
            if (int(n), int(P), float(c), r) not in done:
                jobs.append((int(n), int(P), float(c), r))
    print(f"{len(jobs)} runs, up to {args.max_epochs} epochs each ({args.label})", flush=True)

    from joblib import Parallel, delayed
    t0 = time.time()
    for i in range(0, len(jobs), args.workers):
        res = Parallel(n_jobs=args.workers, backend="loky")(
            delayed(run_one)(*j, cfg, out_dir) for j in jobs[i:i + args.workers])
        new = pd.DataFrame([r for rows in res for r in rows])
        new.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        print(f"{min(i + args.workers, len(jobs))}/{len(jobs)} runs, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
