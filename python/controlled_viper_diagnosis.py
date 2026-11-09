"""
Where does VIPER-Shared's excess risk come from? Coefficient diagnosis on the
saved Phase 2 fits (controlled-strength data, alpha = 1).

VIPER-Shared predicts y_hat = m_z(x)^T E_q[beta], where m_z(x) are the closed-form
test scores inferred from x alone (viper_predict.closed_form_scores). Its beta is
learned jointly with the training scores loc_Z, which the outcome likelihood also
informs. PCA + OLS instead regresses y on scores that are the same function of x
at training and test time. Three predictors share VIPER's loadings and noise
(hence the same score map m_z(x)) and differ only in the outcome coefficients:

    own          E_q[beta]                                      (the model's prediction)
    ols_xonly    OLS (with intercept) of training y on m_z(X_train): the x-only training
                 scores, the same map as at test time (the PCA + OLS analogue)
    ols_locZ     OLS (with intercept) of training y on loc_Z, the y-informed training scores

All three are linear in x: y_hat = c + sum_l x_l^T diag(w_l) M_l A_bar^{-1} coef, so
each has an exact risk (risk_metrics). Also recorded: training R^2 of y on loc_Z and
on m_z(X_train), and ||E_q[beta]|| vs ||OLS beta||.

Output: sim/results/integration_shared_controlled/viper_diagnosis.csv

usage: python controlled_viper_diagnosis.py [--workers 4] [--alpha 1]
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

DATA_ROOT = os.path.expanduser("~/multiomic_integration/sim/data/multi_view_shared_controlled/k.5")
RESULTS = os.path.expanduser("~/multiomic_integration/sim/results/integration_shared_controlled")
N_ARRAY = (100, 200, 500)
P_ARRAY = (600, 3000)
C_ARRAY = (0.3, 0.6, 1, 1.5, 2, 3, 10)


def cond_name(n, P, c):
    return f"n{n}_P{P}_c{c:g}"


def load_dataset(n, P, c, rep):
    from run_methods import load_rds
    cond = cond_name(n, P, c)
    fname = f"sim_data_controlled_{cond}_rep{rep}.rds"
    return (load_rds(f"{DATA_ROOT}/train/{cond}/{fname}"),
            load_rds(f"{DATA_ROOT}/test/{cond}/{fname}"))


def _ols(T, y):
    D = np.column_stack([np.ones(len(T)), T])
    coef, *_ = np.linalg.lstsq(D, y, rcond=None)
    r2 = 1 - ((y - D @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return coef[0], coef[1:], float(r2)


def viper_variants(store, dp, theta, orc, y_mean, y_sd, rel):
    """Exact-risk rows for the three coefficient variants of the VIPER fit in `store`
    (a param store or dict of constrained values with the guide's parameters)."""
    import risk_metrics as rm
    from constants import Params
    from viper_predict import closed_form_scores, guide_moments

    L = dp["L"]
    mom = guide_moments(store, L)
    w = [a / b for a, b in zip(mom["a_psi"], mom["b_psi"])]
    cf_tr = closed_form_scores(mom, dp["train_X_l_list_clean"])
    A_inv = torch.linalg.inv(cf_tr["A_bar"])
    to_x = lambda coef_z: [(M_l * w_l[:, None]) @ (A_inv @ torch.as_tensor(coef_z, dtype=torch.float64))
                           for M_l, w_l in zip(mom["M"], w)]
    y_tr = dp["train_y_clean"].double().reshape(-1).numpy()
    m_tr = cf_tr["m_z"].numpy()
    loc_Z = store[Params.loc_Z].detach().double().cpu().numpy()

    c_x, b_x, r2_x = _ols(m_tr, y_tr)
    c_z, b_z, r2_z = _ols(loc_Z, y_tr)
    variants = {"own": (mom["m_beta"].numpy(), 0.0), "ols_xonly": (b_x, c_x), "ols_locZ": (b_z, c_z)}
    k2 = rm.kappa2_hat(rm.raw_loadings(mom["M"], dp), theta)
    common = {"kappa2_y": k2["kappa2_y"], "kappa2_rel": float(min(k2["kappa2_k"][k] for k in rel)),
              "train_r2_locZ": r2_z, "train_r2_xonly": r2_x,
              "norm_beta_own": float(np.linalg.norm(mom["m_beta"].numpy())),
              "norm_beta_ols_xonly": float(np.linalg.norm(b_x)),
              "norm_beta_ols_locZ": float(np.linalg.norm(b_z))}
    rows = []
    for name, (coef_z, cs) in variants.items():
        r = rm.risk_summary(to_x(coef_z), dp, y_mean, y_sd, theta, orc, cs)
        rows.append({"variant": name, **r, **common})
    return rows


def score_one(n, P, c, rep, alpha):
    import pyro
    import risk_metrics as rm
    from run_methods import process_data

    stem = os.path.join(RESULTS, "models", f"shared_ctrl_{cond_name(n, P, c)}_alpha{alpha:g}_rep{rep}_paramstore.pth")
    if not os.path.exists(stem):
        return []
    tr, te = load_dataset(n, P, c, rep)
    _, _, dp = process_data(tr, te, model_type="shared")
    theta = rm.true_params(tr)
    orc = rm.oracle(theta)
    rel = [k for k in range(theta["beta"].numel()) if float(theta["beta"][k]) != 0]
    y_raw = tr["y"].double().reshape(-1)
    pyro.clear_param_store()
    pyro.get_param_store().set_state(torch.load(stem, weights_only=False, map_location="cpu"))
    rows = viper_variants(pyro.get_param_store(), dp, theta, orc, float(y_raw.mean()), float(y_raw.std()), rel)
    key = {"n": n, "P_grid": P, "c": c, "rep": rep, "alpha": alpha,
           "g_over_sqrt_gamma": float(tr["g_true"].item()) / np.sqrt(P / n)}
    return [{**key, **r} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=1)
    ap.add_argument("--out", default=os.path.join(RESULTS, "viper_diagnosis.csv"))
    args = ap.parse_args()
    from joblib import Parallel, delayed
    jobs = [(n, P, c, r) for n in N_ARRAY for P in P_ARRAY for c in C_ARRAY for r in range(1, 11)]
    t0 = time.time()
    res = Parallel(n_jobs=args.workers, backend="loky")(delayed(score_one)(*j, args.alpha) for j in jobs)
    out = pd.DataFrame([r for rows in res for r in rows])
    out.to_csv(args.out, index=False)
    print(f"{out[['n', 'P_grid', 'c', 'rep']].drop_duplicates().shape[0]} datasets, {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
