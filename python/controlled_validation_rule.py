"""
Validation-loss stopping rule for VIPER-Shared (Tran, Nguyen & Dao 2021,
arXiv:2103.01327, Alg. 9 "NAGVAC": stop when a held-out loss has not reached a new
minimum for P checks).

Loss-based rules cannot tell a weak outcome factor from a strong one: the ELBO
flattens long before a weak factor is learned. A held-out *outcome* loss tracks
prediction directly, so it may stop early when the outcome factor is strong and
keep training when it is weak, without knowing the regime in advance.

Per training dataset (controlled-strength data): a random VAL_FRACTION of the
training samples is held out (fixed seed), the rest is standardised and fitted
exactly as in controlled_training_length.py (SVD init, current optimiser settings,
no early stopping) for MAX_EPOCHS epochs. Every CHUNK epochs it records
    val_mse      held-out MSE of the standardised outcome (closed-form prediction)
    risk_rel     exact relative excess risk of VIPER's own prediction (truth)
plus the per-epoch training loss. The rule is replayed offline
(stopping_rules.val_patience): its stop, the risk at the stop and at the best
held-out checkpoint (the iterate the paper returns).

Output: sim/results/integration_shared_controlled/validation_rule/<label>/checkpoints.csv
        and loss_<dataset>.npy

usage: python controlled_validation_rule.py --cells 500,3000,1 500,3000,10 --reps 1 2
       [--val-fraction 0.2] [--max-epochs 1500] [--chunk 25] [--workers 4] [--label smoke]
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

OUT = os.path.expanduser("~/multiomic_integration/sim/results/integration_shared_controlled/validation_rule")
VAL_FRACTION = 0.2      # TODO(user): held-out share of the training samples


def subset_rows(sim, idx):
    """A copy of a simulated dataset restricted to the samples idx (true parameters kept)."""
    out = dict(sim)
    out["X_l"] = [X[idx] for X in sim["X_l"]]
    out["y"] = sim["y"][idx]
    out["Z"] = sim["Z"][idx]
    return out


def run_one(n, P, c, rep, cfg, out_dir):
    import pyro
    import multiview_shared  # noqa: F401  (import before seeding)
    import risk_metrics as rm
    import sim_run_shared_parallel as base
    from run_methods import process_data, train_model, _minibatch_size
    from controlled_viper_diagnosis import cond_name, load_dataset, viper_variants
    from viper_predict import closed_form_scores, guide_moments

    tr, _ = load_dataset(n, P, c, rep)
    gen = torch.Generator().manual_seed(1000 + rep)
    perm = torch.randperm(n, generator=gen)
    n_val = int(round(cfg["val_fraction"] * n))
    fit, val = subset_rows(tr, perm[n_val:]), subset_rows(tr, perm[:n_val])
    fit_subset, _, dp = process_data(fit, val, model_type="shared")   # val standardised with fit statistics
    theta = rm.true_params(tr)
    orc = rm.oracle(theta)
    rel = [k for k in range(theta["beta"].numel()) if float(theta["beta"][k]) != 0]
    y_fit = fit["y"].double().reshape(-1)
    ym, ysd = float(y_fit.mean()), float(y_fit.std())
    yv = dp["test_y_clean"].double().reshape(-1)

    chunk = cfg["chunk"]
    train_config = {"eta": base.ETA, "delta": base.DELTA, "t": base.T, "num_particles": 1,
                    "minibatch_fraction": base.MINIBATCH_FRACTION, "min_epochs": chunk, "max_epochs": chunk,
                    "convergence_criterion": "elbo_snr", "window": 50, "snr_threshold": 5.0}
    pyro.clear_param_store()
    pyro.set_rng_seed(rep)
    t0 = time.time()
    model, handler, _ = train_model({"k": int(tr["K"]) + 10, "dense_model": False}, train_config, fit_subset,
                                    None, model_type="shared", opt="adagrad", write=False, svd_init=True)
    key = {"n": n, "P_grid": P, "c": c, "rep": rep, "n_fit": n - n_val, "n_val": n_val,
           "g_over_sqrt_gamma": float(tr["g_true"].item()) / np.sqrt(P / n)}
    rows, epoch = [], chunk
    while True:
        t_score = time.time()
        store = pyro.get_param_store()
        cf = closed_form_scores(guide_moments(store, dp["L"]), dp["test_X_l_list_clean"])
        val_mse = float(((cf["yhat"] - yv) ** 2).mean())
        own = [r for r in viper_variants(store, dp, theta, orc, ym, ysd, rel) if r["variant"] == "own"][0]
        score_time = time.time() - t_score
        rows.append({**key, "epoch": epoch, "val_mse": val_mse, "risk_rel": own["risk_rel"],
                     "kappa2_y": own["kappa2_y"], "elapsed": time.time() - t0, "score_time": score_time})
        if epoch >= cfg["max_epochs"]:
            break
        handler.do_inference(train_dataset=fit_subset, min_epochs=chunk, epochs=chunk, minibatch_flag=True,
                             minibatch_size=_minibatch_size(train_config, len(fit_subset)),
                             variational_diff_func=np.mean, verbose=False,
                             convergence_criterion="elbo_snr", window=50, snr_threshold=5.0)
        epoch += chunk
    np.save(os.path.join(out_dir, f"loss_{cond_name(n, P, c)}_rep{rep}.npy"), np.asarray(model.loss_history, float))
    pyro.clear_param_store()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--reps", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--max-epochs", type=int, default=1500)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--label", default="smoke")
    args = ap.parse_args()
    cfg = {"val_fraction": args.val_fraction, "max_epochs": args.max_epochs, "chunk": args.chunk}
    out_dir = os.path.join(OUT, args.label)
    os.makedirs(out_dir, exist_ok=True)
    pd.Series(cfg).to_csv(os.path.join(out_dir, "config.csv"))
    jobs = [(int(n), int(P), float(c), r) for cell in args.cells for n, P, c in [cell.split(",")] for r in args.reps]
    from joblib import Parallel, delayed
    t0 = time.time()
    res = Parallel(n_jobs=args.workers, backend="loky")(delayed(run_one)(*j, cfg, out_dir) for j in jobs)
    pd.DataFrame([r for rows in res for r in rows]).to_csv(os.path.join(out_dir, "checkpoints.csv"), index=False)
    print(f"{len(jobs)} runs, {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
