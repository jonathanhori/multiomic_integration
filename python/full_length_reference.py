"""
Reference settings on the folds of a full-length Optuna study (optuna_controlled.py
--full-length), so they can be compared with its trials on exactly the same data,
CV folds, seeds, guide start and epoch budget.

Each reference setting is trained like a full-length trial (run_fold with
opts["chunk"]: 1000 epochs, no early stopping, scored every 50 epochs). Outputs, next
to the study:
    <study>_ref_curves.csv              per-checkpoint scores, with column "label"
    <study>_losses/ref_<label>_fold<f>.npy  per-epoch training loss
The losses use the study's loss folder with a "ref_" prefix, so
optuna_stopping_replay.py (which reads trial<i>_fold<f>.npy) does not pick them up.

Reference settings (low-variance guide start, as in the studies):
    current      eta 0.25, t 0.1,   minibatch 0.15   (sim_run_shared_parallel defaults)
    long_eta25   eta 0.25, t 0.025, minibatch 0.051  (validated in settings_validation.qmd)

usage: python full_length_reference.py --study full_n500_c1 --data controlled:500,3000,1,11
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(1, "/Users/jonathanhori/multiomic_integration/python")
from optuna_controlled import OUT_DIR, load_training, run_fold

REFERENCES = {
    "current": {"eta": 0.25, "t": 0.1, "minibatch_fraction": 0.15},
    "long_eta25": {"eta": 0.25, "t": 0.024970737145052723, "minibatch_fraction": 0.05063981628665742},
}


def main():
    from run_methods import make_cv_folds
    from data_utils import zero_variance_col_filter
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", nargs="+", default=list(REFERENCES))
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=50)
    args = ap.parse_args()

    sim = load_training(args.data)
    filters = [zero_variance_col_filter(X) for X in sim["X_l"]]          # as in optuna_controlled.py
    X_raw = [X[:, f] for X, f in zip(sim["X_l"], filters)]
    y_raw = sim["y"].reshape(-1)
    folds = make_cv_folds(X_raw[0].shape[0], n_folds=args.folds, seed=args.seed)
    opts = {"min_epochs": 0, "max_epochs": args.max_epochs, "predict": "closed_form", "cov": "implied",
            "init_scale_Lambda": 0.01, "init_scale_Z": 0.1, "chunk": args.chunk}
    loss_dir = os.path.join(OUT_DIR, f"{args.study}_losses")
    os.makedirs(loss_dir, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{args.study}_ref_curves.csv")
    for label in args.labels:
        cfg = {**REFERENCES[label], "convergence_criterion": "ma_patience"}
        for i, (tr, va) in enumerate(folds):
            try:
                r = run_fold(sim, X_raw, y_raw, filters, tr, va, cfg, opts, seed=args.seed + i)
            except Exception as e:
                print(f"{label} fold {i} failed: {repr(e)[:120]}", flush=True)
                continue
            np.save(os.path.join(loss_dir, f"ref_{label}_fold{i}.npy"), r.pop("loss"))
            cur = pd.DataFrame(r.pop("curve")).assign(label=label, fold=i, **REFERENCES[label])
            cur.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            print(f"{label} fold {i}: mse {r['outcome_mse']:.4f} time {r['train_time']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
