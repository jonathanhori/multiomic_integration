"""
Offline sweep of stopping rules over a controlled_optimizer_grid.py run.

For every configuration x dataset in training_length/grid_<name>/ (fits trained with
no early stopping, scored every CHUNK epochs, with --cov), replay each candidate rule
on the saved per-epoch loss (stopping_rules.py) and record what a run using that rule
would have returned:
    stop        epochs run at the stop (the cap if the rule never fired)
    censored    True if the rule had not fired by the cap
    test_mse    held-out outcome MSE (grids run with --test-mse; the  }
                selection criterion)                                  }
    risk_rel    relative excess risk of the model's own predictor   } at the last
    cov_frob    covariance reconstruction error (also within/cross) } checkpoint <= stop
    train_time  optimisation seconds to the stop, = stop x (seconds per epoch of that
                fit, from its last checkpoint); excludes scoring, but wall times
                depend on the machine's load while the grid ran
Because a rule only decides when to stop, the fit at the stop is exactly the fit a
run with that rule would have produced (same seed and data). Scores are taken at the
checkpoint at or before the stop, so with CHUNK = 50 they lag by up to 49 epochs.

Rules (TODO(user): candidate windows, patiences, minimum epochs and caps):
    ma_patience_w{W}_p{P}_min{M}  for W in WINDOWS, P in PATIENCES, M in MIN_EPOCHS
    elbo_snr_min200               the current setting's rule (window 50, threshold 5)
    fixed_{E}                     fixed budgets, for reference
each with a cap of every value in CAPS.

Output: training_length/grid_<name>/stopping_sweep.csv (one row per configuration x
dataset x rule x cap).

usage: python stopping_rule_sweep.py --name lv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(1, "/Users/jonathanhori/multiomic_integration/python")
import stopping_rules as sr
from controlled_training_length import RESULTS
from controlled_viper_diagnosis import cond_name

WINDOWS = [25, 50, 100, 200]
PATIENCES = [10, 20, 50, 100, 200, 300]
MIN_EPOCHS = [0, 200]
FIXED = [200, 300, 500, 700, 1000, 1500]
CAPS = [500, 1000, 1500]            # those above the grid's epoch budget are skipped
KEY = ["label", "n", "P_grid", "c", "rep"]
SCORES = ["test_mse", "risk_rel", "cov_frob", "cov_within", "cov_cross"]


def rules():
    out = {}
    for w in WINDOWS:
        for p in PATIENCES:
            for m in MIN_EPOCHS:
                out[f"ma_patience_w{w}_p{p}_min{m}"] = lambda L, w=w, p=p, m=m: sr.ma_patience(L, m, w, p)
    out["elbo_snr_min200"] = lambda L: sr.elbo_snr(L, min_epochs=200)
    for e in FIXED:
        out[f"fixed_{e}"] = lambda L, e=e: sr.fixed_budget(L, e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    d = os.path.join(RESULTS, f"grid_{args.name}")
    ck = pd.read_csv(os.path.join(d, "checkpoints.csv"))
    failed = ck[ck.get("failed", pd.Series(dtype=object)).notna()] if "failed" in ck else ck.iloc[:0]
    ck = ck[ck.variant == "own"]
    R = rules()
    grid_epochs = int(ck.epoch.max())          # caps beyond the grid's budget are skipped
    rows = []
    for key, g in ck.groupby(KEY):
        label, n, P, c, rep = key
        f = os.path.join(d, f"loss_{label}_{cond_name(n, P, c)}_rep{rep}.npy")
        if not os.path.exists(f):
            continue
        L = np.load(f)
        g = g.sort_values("epoch")
        sec_per_epoch = g.train_time.iloc[-1] / g.epoch.iloc[-1]
        fit_failed = bool(((failed.label == label) & (failed.n == n) & (failed.P_grid == P) & (failed.c == c)
                           & (failed.rep == rep)).any()) if len(failed) else False
        for cap in CAPS:
            Lc = L[:cap]
            for name, rule in R.items():
                if cap > grid_epochs or (name.startswith("fixed_") and int(name.split("_")[1]) > cap):
                    continue
                e = rule(Lc)
                stop = e if e is not None else min(cap, len(L))
                at = g[g.epoch <= stop]
                if at.empty:
                    continue
                last = at.iloc[-1]
                rows.append({**dict(zip(KEY, key)), "rule": name, "cap": cap, "stop": stop,
                             "censored": e is None, "failed": fit_failed, "score_epoch": int(last.epoch),
                             "train_time": stop * sec_per_epoch, **{s: last[s] for s in SCORES if s in last}})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(d, "stopping_sweep.csv"), index=False)
    print(f"{len(out)} rows -> {os.path.join(d, 'stopping_sweep.csv')}")


if __name__ == "__main__":
    main()
