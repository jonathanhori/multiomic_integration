"""
Replay stopping rules over a full-length Optuna study (optuna_controlled.py
--full-length), so every (optimiser setting, stopping rule) pair is scored by CV
outcome MSE and training time.

Every fold of every trial was trained for the full epoch budget with no early
stopping, scored every CHUNK epochs (<study>_curves.csv), and its per-epoch loss saved
(<study>_losses/trial<i>_fold<f>.npy). A stopping rule only decides when to stop, so the
fit at a rule's stop is exactly the fit a run with that rule would have produced (same
seed, data and fold). For each trial x fold x rule (the candidates of
stopping_rule_sweep.rules(): ma_patience windows x patiences x minimum epochs, the
current elbo_snr rule, fixed budgets):
    stop          epochs run at the stop (the budget if the rule never fired)
    censored      True if the rule had not fired by the budget
    outcome_mse   validation MSE              } at the last checkpoint <= stop
    cov_frob      covariance error            } (lags the stop by up to CHUNK - 1
    risk_rel      exact risk (diagnostic)     }  epochs, so slightly pessimistic)
    train_time    optimisation seconds to the stop, = stop x the fold's seconds per
                  epoch over the whole run (scoring excluded)
then averaged over folds per trial x rule (folds_ok counts the folds that ran).
Selection is by CV outcome MSE (exact risk is a diagnostic only). Trials seeded from an
earlier study (--seed-from) have no curves or losses, so they are not replayed.

Outputs, next to the study:
    <study>_replay_folds.csv   one row per trial x fold x rule
    <study>_replay.csv         one row per trial x rule, fold means, with the trial's
                               optimiser settings

usage: python optuna_stopping_replay.py --study full_n500_c1
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(1, "/Users/jonathanhori/multiomic_integration/python")
from optuna_controlled import OUT_DIR
from stopping_rule_sweep import rules

SCORES = ["outcome_mse", "cov_frob", "risk_rel"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    args = ap.parse_args()
    cur = pd.read_csv(os.path.join(OUT_DIR, f"{args.study}_curves.csv"))
    loss_dir = os.path.join(OUT_DIR, f"{args.study}_losses")
    budget = int(cur.epoch.max())
    R = {k: f for k, f in rules().items()
         if not (k.startswith("fixed_") and int(k.split("_")[1]) > budget)}
    settings = [c for c in cur.columns if c not in {"epoch", "train_time", "trial", "fold", *SCORES}]
    rows = []
    for (trial, fold), g in cur.groupby(["trial", "fold"]):
        f = os.path.join(loss_dir, f"trial{trial}_fold{fold}.npy")
        if not os.path.exists(f):
            continue
        L = np.load(f)
        g = g.sort_values("epoch")
        sec_per_epoch = g.train_time.iloc[-1] / g.epoch.iloc[-1]
        for name, rule in R.items():
            e = rule(L)
            stop = e if e is not None else len(L)
            at = g[g.epoch <= stop]
            if at.empty:                         # stopped before the first checkpoint
                continue
            last = at.iloc[-1]
            rows.append({"trial": trial, "fold": fold, "rule": name, "stop": stop, "censored": e is None,
                         "score_epoch": int(last.epoch), "train_time": stop * sec_per_epoch,
                         **{s: last[s] for s in SCORES if s in last}, **{s: last[s] for s in settings}})
    folds = pd.DataFrame(rows)
    folds.to_csv(os.path.join(OUT_DIR, f"{args.study}_replay_folds.csv"), index=False)
    num = ["stop", "train_time", "score_epoch", *[s for s in SCORES if s in folds]]
    agg = folds.groupby(["trial", "rule"]).agg(
        **{s: (s, "mean") for s in num}, censored=("censored", "mean"), folds_ok=("fold", "nunique"),
        **{s: (s, "first") for s in settings}).reset_index()
    agg.to_csv(os.path.join(OUT_DIR, f"{args.study}_replay.csv"), index=False)
    print(f"{len(folds)} fold rows, {len(agg)} trial x rule rows -> {OUT_DIR}/{args.study}_replay.csv")
    best = agg.sort_values("outcome_mse").groupby("rule").head(1).sort_values("outcome_mse")
    print(best[["rule", "trial", "outcome_mse", "train_time", "stop", *settings]].head(15).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
