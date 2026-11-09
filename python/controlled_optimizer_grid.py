"""
AdagradRMSProp hyperparameter grid for VIPER-Shared on the controlled-strength data.

Pyro's AdagradRMSProp updates each parameter by
    theta <- theta - eta * k^(-1/2 + delta) * g / (1 + sqrt(s_k)),
    s_k = t * g^2 + (1 - t) * s_{k-1}    (s_1 = g^2),
with k the parameter's step count (about 1 / minibatch_fraction steps per epoch).
eta scales the step, delta sets how fast it decays (delta = 0: k^(-1/2);
delta -> 0.5: no decay), t is the weight of the newest squared gradient, and the
"1 +" in the denominator makes parameters with small gradients (|g| << 1) take
plain SGD steps of size eta * k^(-1/2 + delta).

Each configuration is run with controlled_training_length.run_one (same data,
seeds, SVD initialisation, K_fit = 15, alpha = 1, no early stopping) for a fixed
epoch budget and scored every CHUNK epochs, so the result is risk vs epochs per
configuration, cell and replicate; replicates use the same seed across
configurations (paired comparisons).

Output: sim/results/integration_shared_controlled/training_length/grid_<name>/checkpoints.csv
(rows tagged with label, eta, delta, t, minibatch_fraction; resumes by
label x dataset) and loss_<label>_<dataset>.npy.

Stage "lv" (--stage lv --cov): the low-variance guide start crossed over eta, t and
minibatch fraction (STAGE_LV), with covariance error and optimisation time recorded at
every checkpoint; stopping rules are replayed afterwards by stopping_rule_sweep.py.

Stage "ft" (--stage ft --cov --test-mse --max-epochs 1000 --chunk 50): the fine-tuning
grid around the low-variance Optuna optimum (STAGE_FT), also recording the held-out
outcome MSE at every checkpoint, the selection criterion.

Stage "val" (--stage val --cov --test-mse --max-epochs 1000 --chunk 50 --reps 1 2 3):
the region chosen from the full-length Optuna studies against the current and
long_eta25 settings (STAGE_VAL), on the whole controlled grid (FULL_CELLS).

usage: python controlled_optimizer_grid.py --name stageA [--configs base eta4 ...]
       [--cells n,P,c ...] [--reps 1 2] [--max-epochs 600] [--workers 4]
"""
import argparse
import os
import sys
import time
import warnings

import pandas as pd

sys.path.insert(1, "/Users/jonathanhori/multiomic_integration/python")
warnings.filterwarnings("ignore")

from controlled_training_length import RESULTS, run_one

# TODO(user): grid values. Proposal after the smoke test (eta = 4 and eta = 1 with
# delta = 0.25 diverged at once in the Gamma shape parameters; delta = 0.45 never
# found the weak factor): moderate eta, step-count (minibatch) and noise (t, particles)
# variations, and per-parameter step scales (eta_loc for loc_* / scale_* parameters,
# eta for the shape and rate parameters). "base" is the setting used so far.
_B = dict(eta=0.25, delta=1e-16, t=0.1, minibatch_fraction=0.15, num_particles=1, eta_loc=None)
CONFIGS = {
    "base":       dict(_B),
    "eta0.5":     dict(_B, eta=0.5),
    "eta1":       dict(_B, eta=1.0),
    "mb0.05":     dict(_B, minibatch_fraction=0.05),
    "mb0.5":      dict(_B, minibatch_fraction=0.5),
    "t0.5":       dict(_B, t=0.5),
    # crossed with the above: full 2 x 2 x 2 over eta {0.25, 0.5} x t {0.1, 0.5} x minibatch {0.05, 0.15}
    "eta0.5_t0.5":         dict(_B, eta=0.5, t=0.5),
    "eta0.5_mb0.05":       dict(_B, eta=0.5, minibatch_fraction=0.05),
    "t0.5_mb0.05":         dict(_B, t=0.5, minibatch_fraction=0.05),
    "eta0.5_t0.5_mb0.05":  dict(_B, eta=0.5, t=0.5, minibatch_fraction=0.05),
    "loc_x4":     dict(_B, eta_loc=1.0),
    "loc_x16":    dict(_B, eta_loc=4.0),
    # stage-1 smoke test: diverged / stuck (kept for reference)
    "eta4":       dict(_B, eta=4.0),
    "eta1_d0.25": dict(_B, eta=1.0, delta=0.25),
    "d0.45":      dict(_B, delta=0.45),
}
# Low-variance guide start (init_scale_Lambda 0.01, init_scale_Z 0.1; Gamma guides at
# init_param 1), November 2026: eta x t x minibatch, a cross that contains the three
# settings validated so far exactly (t and minibatch "0.025" / "0.05" are the long
# setting's Optuna values): current = lv_eta0.25_t0.1_mb0.15, long_eta25 =
# lv_eta0.25_t0.025_mb0.05, long = lv_eta0.35_t0.025_mb0.05.
# TODO(user): grid values (eta 0.15 is the one step below anything tried so far).
_LV = dict(_B, init_scale_Lambda=0.01, init_scale_Z=0.1)
_T = {"0.025": 0.024970737145052723, "0.1": 0.1}
_MB = {"0.05": 0.05063981628665742, "0.15": 0.15}
for _eta in (0.15, 0.25, 0.35):
    for _t, _tv in _T.items():
        for _mb, _mbv in _MB.items():
            CONFIGS[f"lv_eta{_eta}_t{_t}_mb{_mb}"] = dict(_LV, eta=_eta, t=_tv, minibatch_fraction=_mbv)
STAGE_LV = [k for k in CONFIGS if k.startswith("lv_")]
# Fine-tuning grid after the low-variance Optuna search (sim/exploration/optuna_lowvar.qmd,
# 5 November 2026): best CV MSE at eta 0.25-0.32, t 0.03-0.04, minibatch 0.05-0.06, with
# t and minibatch best at the lower edge of the searched range, so the grid extends
# below it (t 0.015, minibatch 0.035), and covers the larger-minibatch region (0.10, 0.15)
# where small-t trials run to the cap came close in about half the time. Low-variance start; "0.025" / "0.05" are again the
# long setting's exact values, so ft_eta0.25_t0.025_mb0.05 is long_eta25's optimiser.
# Run with --test-mse --cov and no early stopping; stopping rules are replayed
# afterwards (stopping_rule_sweep.py), selecting by held-out outcome MSE.
# TODO(user): grid values.
_T_FT = {"0.015": 0.015, "0.025": 0.024970737145052723, "0.035": 0.035}
_MB_FT = {"0.035": 0.035, "0.05": 0.05063981628665742, "0.1": 0.1, "0.15": 0.15}
for _eta in (0.25, 0.3):
    for _t, _tv in _T_FT.items():
        for _mb, _mbv in _MB_FT.items():
            CONFIGS[f"ft_eta{_eta}_t{_t}_mb{_mb}"] = dict(_LV, eta=_eta, t=_tv, minibatch_fraction=_mbv)
STAGE_FT = [k for k in CONFIGS if k.startswith("ft_")]
# Validation of the region chosen from the full-length Optuna studies
# (sim/exploration/optuna_full.qmd, 6 November 2026): eta 0.22-0.27, t 0.015-0.025,
# minibatch about 0.06, plus minibatch 0.10. Three points of the eta x t region at each
# minibatch: the centre, the conservative corner (low eta, high t) and the aggressive
# corner (high eta, low t; nearest the unstable eta > 0.29, t < 0.025 corner), and the
# two earlier settings, all with the low-variance start. Run on the whole controlled
# grid with --test-mse --cov --max-epochs 1000 --chunk 50; selection by held-out MSE,
# differences under 0.01 negligible, faster preferred. TODO(user): grid values.
for _mb in (0.06, 0.10):
    for _eta, _t in ((0.25, 0.02), (0.22, 0.025), (0.27, 0.015)):
        CONFIGS[f"val_eta{_eta}_t{_t}_mb{_mb}"] = dict(_LV, eta=_eta, t=_t, minibatch_fraction=_mb)
CONFIGS["val_current"] = dict(_LV, eta=0.25, t=0.1, minibatch_fraction=0.15)
CONFIGS["val_long_eta25"] = dict(_LV, eta=0.25, t=_T["0.025"], minibatch_fraction=_MB["0.05"])
STAGE_VAL = [k for k in CONFIGS if k.startswith("val_")]
# Fine-tuning grid over the region chosen from the full-length Optuna studies
# (eta 0.22-0.27, t 0.015-0.025, minibatch 0.06-0.10), run on the three tuning datasets'
# CV folds by fold_grid.py (not on the controlled grid); the one setting chosen from it
# is then benchmarked on the whole controlled grid. TODO(user): grid values.
FG_ETA = (0.22, 0.25, 0.27)
FG_T = (0.015, 0.02, 0.025)
FG_MB = (0.06, 0.08, 0.10)
for _eta in FG_ETA:
    for _t in FG_T:
        for _mb in FG_MB:
            CONFIGS[f"fg_eta{_eta}_t{_t}_mb{_mb}"] = dict(_LV, eta=_eta, t=_t, minibatch_fraction=_mb)
STAGE_FG = [k for k in CONFIGS if k.startswith("fg_")]
STAGE_A = ["base", "eta0.5", "eta1", "mb0.05", "mb0.5", "t0.5", "loc_x4", "loc_x16"]
# TODO(user): cells and budget. Proposal: every (n, P) at c = 1 (weak) and c = 10 (strong).
DEFAULT_CELLS = [f"{n},{P},{c}" for n in (100, 200, 500) for P in (600, 3000) for c in (1, 10)]
# The full controlled grid (settings_validation.qmd), for --stage val.
FULL_CELLS = [f"{n},{P},{c}" for n in (100, 200, 500) for P in (600, 3000) for c in (0.3, 0.6, 1, 1.5, 2, 3, 10)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--configs", nargs="+", default=None, help="default: the --stage list")
    ap.add_argument("--stage", choices=["A", "lv", "ft", "val"], default="A")
    ap.add_argument("--cov", action="store_true",
                    help="also record covariance error and optimisation time at each checkpoint")
    ap.add_argument("--test-mse", action="store_true",
                    help="also record the held-out (test-set) outcome MSE at each checkpoint")
    ap.add_argument("--recovery", action="store_true",
                    help="also record factor recovery, cross-method factor comparison and 95%% coverage "
                         "at each checkpoint (checkpoint_metrics.py)")
    ap.add_argument("--mc-draws", type=int, default=200,
                    help="with --recovery: draws for the Monte Carlo outcome interval (0 = none)")
    ap.add_argument("--mc-every", type=int, default=100,
                    help="with --recovery: Monte Carlo interval every MC_EVERY epochs and at the end")
    ap.add_argument("--cells", nargs="+", default=None, help="default: FULL_CELLS for --stage val, else DEFAULT_CELLS")
    ap.add_argument("--reps", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--max-epochs", type=int, default=600)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--early-chunk", type=int, default=None,
                    help="score every EARLY_CHUNK epochs until --early-until, then every --chunk")
    ap.add_argument("--early-until", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.configs is None:
        args.configs = {"A": STAGE_A, "lv": STAGE_LV, "ft": STAGE_FT, "val": STAGE_VAL}[args.stage]
    if args.cells is None:
        args.cells = FULL_CELLS if args.stage == "val" else DEFAULT_CELLS

    out_dir = os.path.join(RESULTS, f"grid_{args.name}")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "checkpoints.csv")
    done = set()
    if os.path.exists(out):
        prev = pd.read_csv(out)
        done = set(zip(prev.label, prev.n, prev.P_grid, prev.c, prev.rep))
    jobs = []
    for label in args.configs:
        cfg = {"label": label, "chunk": args.chunk, "max_epochs": args.max_epochs, "alpha": 1.0,
               "cov": args.cov, "test_mse": args.test_mse,
               "recovery": args.recovery, "mc_draws": args.mc_draws, "mc_every": args.mc_every,
               "early_chunk": args.early_chunk, "early_until": args.early_until, **CONFIGS[label]}
        for cell in args.cells:
            n, P, c = cell.split(",")
            for r in args.reps:
                if (label, int(n), int(P), float(c), r) not in done:
                    jobs.append((int(n), int(P), float(c), r, cfg))
    # largest datasets first so the pool ends with short jobs
    jobs.sort(key=lambda j: -j[0] * j[1])
    print(f"{len(jobs)} runs ({len(args.configs)} configurations), {args.max_epochs} epochs each", flush=True)

    from joblib import Parallel, delayed
    t0 = time.time()
    batch = 2 * args.workers
    for i in range(0, len(jobs), batch):
        res = Parallel(n_jobs=args.workers, backend="loky")(
            delayed(run_one)(n, P, c, r, cfg, out_dir) for n, P, c, r, cfg in jobs[i:i + batch])
        new = pd.DataFrame([row for rows in res for row in rows])
        if os.path.exists(out):
            cols = list(pd.read_csv(out, nrows=0).columns)
            if [c for c in new.columns if c not in cols]:
                pd.concat([pd.read_csv(out), new], ignore_index=True).to_csv(out, index=False)
                print(f"{min(i + batch, len(jobs))}/{len(jobs)} runs, {time.time() - t0:.0f}s", flush=True)
                continue
            new = new.reindex(columns=cols)
        new.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        print(f"{min(i + batch, len(jobs))}/{len(jobs)} runs, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
