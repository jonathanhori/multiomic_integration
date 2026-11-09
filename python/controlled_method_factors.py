"""
Factor recovery, coverage and cross-method factor comparison for the comparison
methods on the controlled-strength grid, with the same definitions as the VIPER
checkpoints (checkpoint_metrics.py), so the final benchmark can set every method side
by side.

Per dataset (all 42 cells, reps --reps):
    pca5, pca15, mofa (PCA / the Phase 2 MOFA2 fit + OLS on the scores; see
    checkpoint_metrics.reference_fits): test outcome MSE, 95% OLS prediction-interval
    coverage and width (cov95_y, width95_y), factor recovery (rec_*, incl. rec_span5_truth)
    and, for every pair of these methods, the cross-method measures
    (span5, outcongr, predcor) in long form.
Signal coverage is not defined for these methods (no posterior for the loadings);
ridge / elastic net have no factors and no intervals (their test MSE and support
recovery are in validation/grid_3reps/scores.csv).

Output: sim/results/integration_shared_controlled/method_factors/{methods.csv,pairs.csv}
(resumable by dataset).

usage: python controlled_method_factors.py [--reps 1 2 3]
"""
import argparse
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(1, "/Users/jonathanhori/multiomic_integration/python")
warnings.filterwarnings("ignore")

from controlled_scoring import C_ARRAY, N_ARRAY, P_ARRAY, RESULTS

OUT = os.path.join(RESULTS, "method_factors")


def score_dataset(n, P, c, rep):
    import risk_metrics as rm
    from checkpoint_metrics import compare, reference_fits, span_truth
    from controlled_viper_diagnosis import cond_name, load_dataset
    from run_methods import process_data
    from validate_settings import factor_recovery

    tr, te = load_dataset(n, P, c, rep)
    _, _, dp = process_data(tr, te, model_type="shared")
    theta = rm.true_params(tr)
    stem = f"{cond_name(n, P, c)}_rep{rep}"
    refs = reference_fits(dp, os.path.join(RESULTS, "mofa", f"mofa_ctrl_{stem}.npz"))
    y = dp["test_y_clean"].double().reshape(-1).numpy()
    key = {"n": n, "P_grid": P, "c": c, "rep": rep}
    rows, pairs = [], []
    for m, r in refs.items():
        # predictor coefficients on the standardised scale, for the support measures
        Wc = np.vstack(r["load"])
        if m.startswith("pca"):
            coef_x = Wc @ r["coef"]
        else:
            coef_x = Wc @ np.linalg.solve(Wc.T @ Wc, r["coef"])
        bounds = np.cumsum([0] + [w.shape[0] for w in r["load"]])
        coef_s = [coef_x[bounds[l]:bounds[l + 1]] for l in range(len(r["load"]))]
        rec = factor_recovery(r["load"], r["coef"], r["scores_tr"], coef_s, dp, theta)
        rows.append({**key, "method": m, "test_mse": float(np.mean((r["pred_te"] - y) ** 2)),
                     "cov95_y": float(np.mean((y >= r["lo"]) & (y <= r["hi"]))),
                     "width95_y": float(np.mean(r["hi"] - r["lo"])),
                     **{f"rec_{k}": v for k, v in rec.items()},
                     "rec_span5_truth": span_truth(r["load"], dp, theta)})
    for a, b in itertools.combinations(refs, 2):
        pairs.append({**key, "method_a": a, "method_b": b, **compare(refs[a]["summary"], refs[b]["summary"])})
    return rows, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    f_m, f_p = os.path.join(OUT, "methods.csv"), os.path.join(OUT, "pairs.csv")
    done = set()
    if os.path.exists(f_m):
        d = pd.read_csv(f_m)
        done = set(zip(d.n, d.P_grid, d.c, d.rep))
    for n in N_ARRAY:
        for P in P_ARRAY:
            for c in C_ARRAY:
                for rep in args.reps:
                    if (n, P, float(c), rep) in done:
                        continue
                    rows, pairs = score_dataset(n, P, c, rep)
                    for df, f in ((pd.DataFrame(rows), f_m), (pd.DataFrame(pairs), f_p)):
                        df.to_csv(f, mode="a", header=not os.path.exists(f), index=False)
            print(f"n{n} P{P} done", flush=True)


if __name__ == "__main__":
    main()
