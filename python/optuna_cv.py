"""Optuna-based multi-objective hyperparameter search for SupMultiviewShared.

Replaces the exhaustive grid search in run_cv_grid_shared with Bayesian
optimisation (NSGA-II sampler) and early pruning (MedianPruner on ELBO).

Usage
-----
    from optuna_cv import run_optuna_study

    study, labels = run_optuna_study(
        X_l_cv_raw, y_cv_raw,
        model_config={"k": K_FIT, "dense_model": False},
        sim_decomp=cv_sim_decomp,
        column_filters=cv_col_filters,
        minibatch_choices=[32, 64, 128],
        n_trials=100,
        storage="sqlite:///optuna_cv_shared.db",
        study_name="shared_cv_run1",
    )
    # Inspect Pareto-optimal configs
    for t in study.best_trials:
        print(t.number, t.values, t.params)
"""

import numpy as np
import torch
import pyro
import optuna
from torch.utils.data import TensorDataset

from constants import Sites
from data_utils import (
    normalize_tensor_by_col,
    eval_cov, calc_cov, scale_sim_cov,
    summarise_post_samples,
)
from run_methods import (
    train_model_shared,
    train_model_locally_shared,
    make_cv_folds,
)


# ---------------------------------------------------------------------------
# Single CV fold
# ---------------------------------------------------------------------------

def _run_fold(
    X_l_raw_list, y_raw, train_idx, val_idx,
    model_config, train_config,
    n_posterior_samples, opt,
    optuna_trial=None, optuna_step_offset=0,
    sim_decomp=None, column_filters=None,
):
    """Train + evaluate one CV fold.

    optuna_trial is threaded through to do_inference so that intermediate
    ELBO values are reported for pruning.  Pass None to disable pruning
    (used for all folds after the pilot fold).
    """
    # Convert fractional minibatch spec to an absolute size for this fold,
    # then apply the halving safety-net cap.
    n_train = len(train_idx)
    if "minibatch_fraction" in train_config:
        train_config = {
            **train_config,
            "minibatch_size": max(1, int(n_train * train_config["minibatch_fraction"])),
        }
    train_mb = int(train_config["minibatch_size"])
    while train_mb > n_train:
        train_mb //= 2
    train_config = {**train_config, "minibatch_size": train_mb}

    # --- data prep ---
    X_l_tr_raw = [X[train_idx] for X in X_l_raw_list]
    X_l_va_raw = [X[val_idx]   for X in X_l_raw_list]
    y_tr_raw   = y_raw[train_idx]
    y_va_raw   = y_raw[val_idx]

    train_norms = [normalize_tensor_by_col(X) for X in X_l_tr_raw]
    X_l_tr   = [d["data_clean"] for d in train_norms]
    X_l_mean = [d["means"]      for d in train_norms]
    X_l_sd   = [d["sds"]        for d in train_norms]

    val_norms = [normalize_tensor_by_col(X, m, s)
                 for X, m, s in zip(X_l_va_raw, X_l_mean, X_l_sd)]
    X_l_va = [d["data_clean"] for d in val_norms]

    y_mean = torch.mean(y_tr_raw)
    y_std  = torch.std(y_tr_raw)
    y_tr   = ((y_tr_raw - y_mean) / y_std).squeeze()
    y_va   = ((y_va_raw - y_mean) / y_std).squeeze()

    train_ds = TensorDataset(torch.arange(len(train_idx)), *X_l_tr, y_tr)
    val_ds   = TensorDataset(torch.arange(len(val_idx)),   *X_l_va, y_va)

    pyro.clear_param_store()

    # --- global training (pruning active on pilot fold only) ---
    factor_model, train_handler, global_state = train_model_shared(
        model_config, train_config, train_ds,
        model_out_filename="", opt=opt, verbose=False, write=False,
        optuna_trial=optuna_trial,
        optuna_step_offset=optuna_step_offset,
    )

    # --- local inference on validation fold ---
    factor_model, val_handler, local_state = train_model_locally_shared(
        factor_model, train_config, val_ds, opt=opt, verbose=False,
    )

    # --- outcome MSE ---
    post_val    = val_handler.predict(X_l_va, n_posterior_samples, [Sites.y_pred])
    pred_mean   = summarise_post_samples(post_val[Sites.y_pred])["mean"]
    outcome_mse = float(torch.mean((pred_mean - y_va) ** 2).item())
    train_time  = float(global_state["inference_time"])

    metrics = {"outcome_mse": outcome_mse, "train_time": train_time}

    # --- covariance Frobenius error (streamed pair-by-pair to save memory) ---
    if sim_decomp is not None and column_filters is not None:
        L = len(X_l_raw_list)
        lambda_sites = [Sites.Lambda_l.format(l=l) for l in range(L)]
        post_train = train_handler.predict(X_l_tr, n_posterior_samples, lambda_sites)
        est_Lambda_list = [
            post_train[Sites.Lambda_l.format(l=l)].mean(dim=0).squeeze().cpu()
            for l in range(L)
        ]
        sim_Lambda_list = [
            sim_decomp["SIM_Lambda_l_list"][l][column_filters[l]]
            for l in range(L)
        ]

        cov_frob_vals = []
        for l in range(L):
            for m in range(l, L):
                loading2_est = est_Lambda_list[m] if l != m else None
                loading2_sim = sim_Lambda_list[m] if l != m else None
                est_c = calc_cov(est_Lambda_list[l], loading2_est)
                raw_c = calc_cov(sim_Lambda_list[l], loading2_sim)
                sim_c = scale_sim_cov(raw_c, X_l_sd[l], X_l_sd[m])
                cov_frob_vals.append(float(eval_cov(est_c, sim_c).item()))
                del est_c, raw_c, sim_c

        metrics["cov_frob"] = float(np.mean(cov_frob_vals))

    return metrics


# ---------------------------------------------------------------------------
# Parameter space helpers
# ---------------------------------------------------------------------------

def _suggest_params(trial, param_space):
    """Build a train_config dict from param_space by calling the appropriate
    trial.suggest_* for each tunable entry or reading the value for fixed ones.

    param_space format — each key maps to a spec dict with a mandatory "type":
        {"type": "fixed",       "value": <v>}
        {"type": "float",       "low": <l>, "high": <h>, "log": <bool>}
        {"type": "int",         "low": <l>, "high": <h>}
        {"type": "categorical", "choices": [<v>, ...]}
    """
    config = {}
    for name, spec in param_space.items():
        t = spec["type"]
        if t == "fixed":
            config[name] = spec["value"]
        elif t == "float":
            config[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif t == "int":
            config[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "categorical":
            config[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown param_space type {t!r} for {name!r}")
    return config


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def _make_objective(
    X_l_raw_list, y_raw, model_config,
    folds, n_posterior_samples, opt,
    sim_decomp, column_filters,
    param_space,
):
    has_cov = sim_decomp is not None and column_filters is not None

    def objective(trial):
        config = _suggest_params(trial, param_space)

        fold_metrics = []
        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            try:
                m = _run_fold(
                    X_l_raw_list, y_raw, tr_idx, va_idx,
                    model_config, config,
                    n_posterior_samples, opt,
                    optuna_trial=None,   # trial.report unsupported for multi-objective
                    optuna_step_offset=0,
                    sim_decomp=sim_decomp,
                    column_filters=column_filters,
                )
                fold_metrics.append(m)
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                print(f"  [trial {trial.number}] fold {fold_idx} FAILED: {exc}")

        if not fold_metrics:
            raise optuna.TrialPruned()

        mean_mse  = float(np.mean([m["outcome_mse"] for m in fold_metrics]))
        mean_time = float(np.mean([m["train_time"]  for m in fold_metrics]))

        trial.set_user_attr("mean_outcome_mse", mean_mse)
        trial.set_user_attr("mean_train_time",  mean_time)

        if has_cov:
            mean_cov = float(np.mean([m["cov_frob"] for m in fold_metrics
                                       if "cov_frob" in m]))
            trial.set_user_attr("mean_cov_frob", mean_cov)
            return mean_mse, mean_cov, mean_time

        return mean_mse, mean_time

    return objective


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_optuna_study(
    X_l_raw_list, y_raw, model_config,
    param_space,
    n_trials=100,
    n_folds=5,
    seed=42,
    n_posterior_samples=100,
    opt="adagrad",
    sim_decomp=None,
    column_filters=None,
    storage="sqlite:///optuna_cv_shared.db",
    study_name="cv_shared",
):
    """Run a multi-objective Optuna study for SupMultiviewShared HPO.

    Objectives minimised
    --------------------
    - outcome_mse  : mean CV test-set MSE
    - cov_frob     : mean CV covariance reconstruction error (when sim_decomp
                     and column_filters are provided)
    - train_time   : mean wall-clock training time per fold

    Sampler / pruner
    ----------------
    NSGAIISampler finds the Pareto front across all objectives.
    MedianPruner aborts trials whose pilot-fold ELBO is below the median of
    completed trials at the same epoch, saving time on unpromising configs.

    Parameters
    ----------
    param_space : dict
        Search space definition. Each key is a train_config parameter name;
        each value is a spec dict with a mandatory "type" field:
            {"type": "fixed",       "value": <v>}
            {"type": "float",       "low": <l>, "high": <h>, "log": <bool>}
            {"type": "int",         "low": <l>, "high": <h>}
            {"type": "categorical", "choices": [<v>, ...]}
        Must include all keys expected by train_model_shared / do_inference.
    storage : str or None
        SQLite URL for persistence, e.g. "sqlite:///optuna.db".  The study
        resumes automatically if study_name already exists in that database.
        Pass None for in-memory storage (no persistence).
    n_startup_trials : int
        Random trials before NSGA-II guided search begins.
    n_warmup_steps : int
        Epochs that must complete before pruning is considered.

    Returns
    -------
    study : optuna.Study
    direction_labels : list of str
        Names of the objectives in the order returned by the study.
    """
    has_cov = sim_decomp is not None and column_filters is not None
    directions       = ["minimize", "minimize", "minimize"] if has_cov else ["minimize", "minimize"]
    direction_labels = ["outcome_mse", "cov_frob", "train_time"]  if has_cov else ["outcome_mse", "train_time"]

    sampler = optuna.samplers.NSGAIISampler(seed=seed)

    study = optuna.create_study(
        study_name=study_name,
        directions=directions,
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    n = X_l_raw_list[0].shape[0]
    folds = make_cv_folds(n, n_folds=n_folds, seed=seed)

    objective = _make_objective(
        X_l_raw_list, y_raw, model_config,
        folds=folds,
        n_posterior_samples=n_posterior_samples,
        opt=opt,
        sim_decomp=sim_decomp,
        column_filters=column_filters,
        param_space=param_space,
    )

    print(f"Objectives : {direction_labels}")
    print(f"Sampler    : NSGAIISampler  (no pruning — unsupported for multi-objective)")
    print(f"Storage    : {storage}  study={study_name}")
    print(f"Trials     : {n_trials}  folds={n_folds}")
    print()

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    pareto = study.best_trials
    print(f"\n{len(pareto)} Pareto-optimal trial(s):")
    for t in pareto:
        vals = {k: f"{v:.4f}" for k, v in zip(direction_labels, t.values)}
        print(f"  trial {t.number:>4d}  {vals}  {t.params}")

    return study, direction_labels


# ---------------------------------------------------------------------------
# Post-hoc helpers
# ---------------------------------------------------------------------------

def study_to_dataframe(study, direction_labels):
    """Convert a completed study to a tidy DataFrame for analysis."""
    import pandas as pd
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial": t.number}
        row.update({k: v for k, v in zip(direction_labels, t.values)})
        row.update(t.params)
        row.update(t.user_attrs)
        rows.append(row)
    return pd.DataFrame(rows)


def pareto_dataframe(study, direction_labels):
    """Return only the Pareto-optimal trials as a DataFrame."""
    import pandas as pd
    rows = []
    for t in study.best_trials:
        row = {"trial": t.number}
        row.update({k: v for k, v in zip(direction_labels, t.values)})
        row.update(t.params)
        rows.append(row)
    return pd.DataFrame(rows)
