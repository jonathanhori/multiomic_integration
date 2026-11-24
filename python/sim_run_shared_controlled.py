"""
SupMultiviewShared (VIPER-Shared) on the controlled-strength datasets
(weak-factor study, Phases 2-3).

Data: R/generate_controlled_data.R (generator documented in
docs/controlled_strength_data.md)
    sim/data/multi_view_shared_controlled/k.5/{train,test}/n{n}_P{P}_c{c}/
        sim_data_controlled_n{n}_P{P}_c{c}_rep{rep}.rds

Each replicate is trained and evaluated with sim_run_shared_parallel._run_rep,
i.e. exactly the misaligned study's pipeline and training settings (MGP prior,
SVD initialisation, AdagradRMSProp, elbo_snr stopping, local SVI for the
pipeline's test metrics), with K_fit = K + K_DELTA = 15 and outcome_weight alpha.
The saved param stores are scored afterwards by controlled_scoring.py
(closed-form prediction, exact risk, kappa2_hat, g_hat, E_q[tau]).

Outputs: sim/results/integration_shared_controlled/{models,metrics}/
    shared_ctrl_n{n}_P{P}_c{c}_alpha{alpha}_rep{rep}[ .pth | _paramstore.pth | .csv | _cov.csv ]

Grid: one task per (n, P, c, alpha); --task-id (1-based) runs one task, for
cluster arrays; reps of a task run in parallel (up to MAX_WORKERS).

usage: python sim_run_shared_controlled.py [--task-id I] [--reps 1 2] [--alphas 1]
       [--only n,P,c] [--list]
"""
import argparse
import itertools
import os
from pathlib import Path

import sim_run_shared_parallel as base

DATA_ROOT = "~/multiomic_integration/sim/data/multi_view_shared_controlled/k.5/"
OUT_ROOT = os.path.expanduser("~/multiomic_integration/sim/results/integration_shared_controlled")

N_ARRAY = (100, 200, 500)
P_ARRAY = (600, 3000)
C_ARRAY = (0.3, 0.6, 1, 1.5, 2, 3, 10)
ALPHAS = (1,)            # outcome_weight; Phase 2 uses alpha = 1 (Phase 3: --alphas)
REPS = 10
K_DELTA = 10             # K_fit = 5 + 10 = 15
L = 3
MAX_WORKERS = 4


def cond_name(n, P, c):
    return f"n{n}_P{P}_c{c:g}"


def grid(alphas):
    return list(itertools.product(N_ARRAY, P_ARRAY, C_ARRAY, alphas))


def rep_args_for(n, P, c, alpha, rep, train_config, fdict, tdict):
    cond = cond_name(n, P, c)
    stem = f"shared_ctrl_{cond}_alpha{alpha:g}_rep{rep}"
    return {
        "rep": rep, "n": n, "p": P // L, "snr_x": float("nan"), "snr_y": 2,
        "sparsity": 0.25, "k_delta": K_DELTA, "align_label": None,
        "outcome_weight": float(alpha),
        "device": str(base.DEVICE), "cond_str": cond,
        "train_sim_data_path": DATA_ROOT + "train/", "test_sim_data_path": DATA_ROOT + "test/",
        "filename_dict": fdict, "test_filename_dict": tdict,
        "rep_stem": f"sim_data_controlled_{cond}_rep{rep}",
        "model_out_filename": os.path.join(OUT_ROOT, "models", stem),
        "metric_out_filename": os.path.join(OUT_ROOT, "metrics", stem),
        "training_split": False, "training_size": 0.8, "random_seed": 123,
        "n_posterior_samples": base.N_POSTERIOR_SAMPLES,
        "min_epochs_local": base.MIN_EPOCHS_LOCAL,
        "train_config": train_config, "svd_init": base.SVD_INIT,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--reps", nargs="*", type=int, default=list(range(1, REPS + 1)))
    ap.add_argument("--alphas", nargs="*", type=float, default=list(ALPHAS))
    ap.add_argument("--only", default=None, help="n,P,c: run one cell")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--list", action="store_true", help="print the task grid and exit")
    args = ap.parse_args()

    tasks = grid(args.alphas)
    if args.list:
        for i, t in enumerate(tasks, 1):
            print(i, *t)
        return
    if args.only:
        n, P, c = (float(v) for v in args.only.split(","))
        tasks = [t for t in tasks if (t[0], t[1], t[2]) == (int(n), int(P), c)]
    if args.task_id is not None:
        tasks = [tasks[args.task_id - 1]]

    for sub in ("models", "metrics"):
        Path(OUT_ROOT, sub).mkdir(parents=True, exist_ok=True)
    train_config = {
        "eta": base.ETA, "delta": base.DELTA, "t": base.T, "num_particles": base.NUM_PARTICLES,
        "minibatch_fraction": base.MINIBATCH_FRACTION, "min_epochs": base.MIN_EPOCHS,
        "max_epochs": base.NUM_EPOCHS, "convergence_criterion": base.CONVERGENCE_CRITERION,
        "window": base.WINDOW, "snr_threshold": base.SNR_THRESHOLD,
    }
    fdict = base._build_filename_dict(DATA_ROOT + "train/")
    tdict = base._build_filename_dict(DATA_ROOT + "test/")

    from joblib import Parallel, delayed
    for n, P, c, alpha in tasks:
        print(f"\n=== {cond_name(n, P, c)}  alpha={alpha:g} ===", flush=True)
        jobs = [rep_args_for(n, P, c, alpha, r, train_config, fdict, tdict) for r in args.reps]
        res = Parallel(n_jobs=min(args.workers, len(jobs)), backend="loky")(
            delayed(base._run_rep_worker)(a) for a in jobs)
        for r in res:
            if r and "_error" in r:
                print(f"  rep {r['rep']}: FAILED\n{r['_error']}", flush=True)


if __name__ == "__main__":
    main()
