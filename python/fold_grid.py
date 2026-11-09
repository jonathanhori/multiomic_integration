"""
Fixed grid of optimiser settings on the CV folds of the full-length Optuna studies,
to choose one final VIPER-Shared setting (November 2026).

Each setting (controlled_optimizer_grid.STAGE_FG by default: eta x t x minibatch
fraction over the region chosen in sim/exploration/optuna_full.qmd, low-variance guide
start) is trained on every fold of each tuning dataset exactly as a full-length Optuna
trial (optuna_controlled.run_fold with opts["chunk"]: same data, folds, seeds, guide
start; 1000 epochs, no early stopping), scored every --early-chunk epochs until
--early-until and every --chunk epochs after (validation MSE, covariance error,
diagnostic exact risk, optimisation time). Stopping rules are replayed afterwards on
the saved losses. The selection criterion is CV outcome MSE (differences under 0.01
negligible, faster preferred); times for the final choice come from the solo timing
benchmark (time_settings.py), since the times here are taken with several fits sharing
the machine.

Tuning datasets (the Optuna studies' data and folds):
    full_n500_c1       controlled:500,3000,1,11
    full_n500_c10      controlled:500,3000,10,11
    full_n200_P600_c3  controlled:200,600,3,4

Outputs, under sim/results/integration_shared_controlled/optuna/fold_grid/:
    <study>/<label>_curves.csv           per-checkpoint scores of the label's 3 folds
    <study>/losses/<label>_fold<f>.npy   per-epoch training loss
A (study, label) job is skipped when its curves file exists (resumable); a fold that
fails is logged and left out, so a job's file can have fewer than 3 folds.

usage: OMP_NUM_THREADS=2 python fold_grid.py [--labels ...] [--workers 4]
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

from optuna_controlled import OUT_DIR

OUT = os.path.join(OUT_DIR, "fold_grid")
STUDIES = {"full_n500_c1": "controlled:500,3000,1,11",
           "full_n500_c10": "controlled:500,3000,10,11",
           "full_n200_P600_c3": "controlled:200,600,3,4"}


def run_job(study, label, args):
    from controlled_optimizer_grid import CONFIGS
    from optuna_controlled import load_training, run_fold
    from run_methods import make_cv_folds
    from data_utils import zero_variance_col_filter
    d = os.path.join(OUT, study)
    os.makedirs(os.path.join(d, "losses"), exist_ok=True)
    out = os.path.join(d, f"{label}_curves.csv")
    if os.path.exists(out):
        return f"{study} {label}: done before"
    sim = load_training(STUDIES[study])
    filters = [zero_variance_col_filter(X) for X in sim["X_l"]]          # as in optuna_controlled.py
    X_raw = [X[:, f] for X, f in zip(sim["X_l"], filters)]
    y_raw = sim["y"].reshape(-1)
    folds = make_cv_folds(X_raw[0].shape[0], n_folds=3, seed=42)
    c = CONFIGS[label]
    opts = {"min_epochs": 0, "max_epochs": args.max_epochs, "predict": "closed_form", "cov": "implied",
            "init_scale_Lambda": c["init_scale_Lambda"], "init_scale_Z": c["init_scale_Z"],
            "chunk": args.chunk, "early_chunk": args.early_chunk, "early_until": args.early_until}
    cfg = {"eta": c["eta"], "t": c["t"], "minibatch_fraction": c["minibatch_fraction"],
           "convergence_criterion": "ma_patience"}
    curves, msg = [], []
    for i, (tr, va) in enumerate(folds):
        try:
            r = run_fold(sim, X_raw, y_raw, filters, tr, va, cfg, opts, seed=42 + i)
        except Exception as e:
            msg.append(f"fold {i} failed: {repr(e)[:100]}")
            continue
        np.save(os.path.join(d, "losses", f"{label}_fold{i}.npy"), r.pop("loss"))
        curves.append(pd.DataFrame(r.pop("curve")).assign(label=label, fold=i, **{k: c[k] for k in
                                                                                  ("eta", "t", "minibatch_fraction")}))
        msg.append(f"fold {i} mse {r['outcome_mse']:.4f} {r['train_time']:.0f}s")
    pd.concat(curves).to_csv(out, index=False) if curves else open(out + ".failed", "w").write("\n".join(msg))
    return f"{study} {label}: " + "; ".join(msg)


def main():
    from controlled_optimizer_grid import STAGE_FG
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=STAGE_FG)
    ap.add_argument("--studies", nargs="+", default=list(STUDIES))
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--early-chunk", type=int, default=10)
    ap.add_argument("--early-until", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    # slowest jobs (n 500, small minibatch) first so the pool ends with short jobs
    from controlled_optimizer_grid import CONFIGS
    jobs = sorted([(s, l) for s in args.studies for l in args.labels],
                  key=lambda j: (j[0] == "full_n200_P600_c3", CONFIGS[j[1]]["minibatch_fraction"]))
    print(f"{len(jobs)} jobs, {args.workers} workers, start {time.ctime()}", flush=True)
    from joblib import Parallel, delayed
    for m in Parallel(n_jobs=args.workers, backend="loky", return_as="generator_unordered")(
            delayed(run_job)(s, l, args) for s, l in jobs):
        print(time.strftime("%H:%M:%S"), m, flush=True)
    print(f"ALL_DONE {time.ctime()}", flush=True)


if __name__ == "__main__":
    main()
