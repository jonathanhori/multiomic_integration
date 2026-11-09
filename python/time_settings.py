"""
Dedicated runtime benchmark for VIPER-Shared optimiser settings.

Wall times recorded during grids and Optuna studies are taken with several fits
sharing the machine (and, before, with the machine sometimes asleep), so they are only
approximate. Here each setting is timed alone: one fit at a time, a fixed number of
threads, nothing else running (the launcher wraps it in `caffeinate -i` so the Mac does
not sleep). Run it when no other fits are running.

The cost of an epoch depends on the data size (n, P) and the minibatch fraction (number
of optimiser steps per epoch), not on the signal strength or the replicate, so one
dataset per (n, P) is used (c = 1, rep 1). For each setting x (n, P) x repeat, the
model is trained with SVD initialisation for --epochs epochs with no early stopping
(min_epochs = max_epochs, as in the full-length runs), recording:
    svd_time         seconds for the SVD initialisation
    sec_per_epoch    SVI seconds / epochs (the guide/model steps only)
    sec_first_half, sec_second_half   SVI seconds of each half of the epochs, to check
                     that the cost per epoch is constant over training
    time_to_E        svd_time + E x sec_per_epoch for E in 200, 500, 1000
Settings: controlled_optimizer_grid.CONFIGS by label (default: STAGE_VAL).

With --mofa, MOFA2 (controlled_scoring.fit_mofa: 15 factors, as in the validation grid)
is timed on the same datasets, interleaved with the VIPER settings, with the same
thread count; its time_to_* columns hold the MOFA fit time (the R fit_mofa call only,
as recorded in the grid), so VIPER can be scored at the time MOFA takes.

Output: sim/results/integration_shared_controlled/timing/<name>.csv (appended per run,
resumable by setting x n x P x repeat).

usage: OMP_NUM_THREADS=2 python time_settings.py --name val [--epochs 100] [--repeats 3]
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

OUT = os.path.expanduser("~/multiomic_integration/sim/results/integration_shared_controlled/timing")
SIZES = [(n, P) for n in (100, 200, 500) for P in (600, 3000)]
COLS = ["label", "n", "P", "repeat", "epochs", "threads", "eta", "t", "minibatch_fraction", "svd_time",
        "sec_per_epoch", "sec_first_half", "sec_second_half", "minibatch_size", "steps_per_epoch",
        "time_to_200", "time_to_500", "time_to_1000", "mofa_time"]   # fixed, so appended rows line up


def time_one(n, P, cfg, epochs, seed):
    import pyro
    import multiview_shared  # noqa: F401  (import before seeding)
    import sim_run_shared_parallel as base
    from run_methods import process_data, _build_model, _build_handler, _minibatch_size, compute_svd_init
    from controlled_viper_diagnosis import load_dataset

    tr, te = load_dataset(n, P, 1.0, 1)
    train_subset, _, dp = process_data(tr, te, model_type="shared")
    model_config = {"k": int(tr["K"]) + 10, "dense_model": False, "outcome_weight": 1.0,
                    "init_scale_Lambda": cfg.get("init_scale_Lambda", 1.0),
                    "init_scale_Z": cfg.get("init_scale_Z", 1.0), "init_param": cfg.get("init_param", 1.0)}
    train_config = {"eta": cfg["eta"], "delta": cfg.get("delta", base.DELTA), "t": cfg["t"],
                    "num_particles": cfg["num_particles"], "minibatch_fraction": cfg["minibatch_fraction"],
                    "min_epochs": epochs, "max_epochs": epochs, "convergence_criterion": "ma_patience"}
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    t0 = time.perf_counter()
    X = list(train_subset.tensors[1:-1])
    Z0, L0 = compute_svd_init(X, model_config["k"])
    svd_time = time.perf_counter() - t0
    model = _build_model("shared", model_config, "cpu", Z0, L0)
    handler = _build_handler("train", model, train_config, "shared", "adagrad", "cpu")
    half = epochs // 2
    secs = []
    for e in (half, epochs - half):
        t1 = time.perf_counter()
        handler.do_inference(train_dataset=train_subset, min_epochs=e, epochs=e, minibatch_flag=True,
                             minibatch_size=_minibatch_size(train_config, len(train_subset)),
                             variational_diff_func=np.mean, verbose=False, convergence_criterion="ma_patience")
        secs.append(time.perf_counter() - t1)
    pyro.clear_param_store()
    spe = sum(secs) / epochs
    return {"svd_time": svd_time, "sec_per_epoch": spe, "sec_first_half": secs[0], "sec_second_half": secs[1],
            "minibatch_size": _minibatch_size(train_config, len(train_subset)),
            "steps_per_epoch": len(train_subset) // _minibatch_size(train_config, len(train_subset)),
            **{f"time_to_{E}": svd_time + E * spe for E in (200, 500, 1000)}}


def time_mofa(n, P, threads, seed):
    import tempfile
    from run_methods import process_data
    from controlled_viper_diagnosis import load_dataset
    from controlled_scoring import fit_mofa
    tr, te = load_dataset(n, P, 1.0, 1)
    _, _, dp = process_data(tr, te, model_type="shared")
    with tempfile.TemporaryDirectory() as d:
        rt = fit_mofa(dp, os.path.join(d, "mofa.npz"), k_fit=15, threads=threads, seed=123 + seed)
    return {"svd_time": 0.0, "sec_per_epoch": float("nan"), **{f"time_to_{E}": rt for E in (200, 500, 1000)},
            "mofa_time": rt}


def main():
    from controlled_optimizer_grid import CONFIGS, STAGE_VAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--configs", nargs="+", default=STAGE_VAL)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--sizes", nargs="+", default=[f"{n},{P}" for n, P in SIZES])
    ap.add_argument("--mofa", action="store_true", help="also time MOFA2 on the same datasets")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"{args.name}.csv")
    done = set()
    if os.path.exists(out):
        prev = pd.read_csv(out)
        done = set(zip(prev.label, prev.n, prev.P, prev.repeat))
    threads = os.environ.get("OMP_NUM_THREADS", "unset")
    # repeats outermost and settings interleaved, so slow drift in machine state affects all settings alike
    for r in range(args.repeats):
        for size in args.sizes:
            n, P = map(int, size.split(","))
            for label in args.configs + (["mofa"] if args.mofa else []):
                if (label, n, P, r) in done:
                    continue
                if label == "mofa":
                    nt = int(threads) if threads.isdigit() else 2
                    row = {"label": "mofa", "n": n, "P": P, "repeat": r, "epochs": 0, "threads": threads,
                           **time_mofa(n, P, nt, seed=r)}
                    pd.DataFrame([row]).reindex(columns=COLS).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
                    print(f"mofa n{n} P{P} rep{r}: {row['mofa_time']:.1f}s", flush=True)
                    continue
                try:
                    timed = time_one(n, P, CONFIGS[label], args.epochs, seed=r)
                except Exception as e:   # unstable setting (e.g. Gamma concentration error): not recorded
                    print(f"{label} n{n} P{P} rep{r}: failed {repr(e)[:80]}", flush=True)
                    continue
                row = {"label": label, "n": n, "P": P, "repeat": r, "epochs": args.epochs, "threads": threads,
                       **{k: CONFIGS[label][k] for k in ("eta", "t", "minibatch_fraction")}, **timed}
                pd.DataFrame([row]).reindex(columns=COLS).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
                print(f"{label} n{n} P{P} rep{r}: {row['sec_per_epoch']:.3f} s/epoch, "
                      f"1000 epochs {row['time_to_1000']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
