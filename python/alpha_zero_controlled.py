"""
Supervision vs sparsity for VIPER-Shared on the Optuna tuning folds.

On the weak dataset (controlled n = 500, P = 3000, c = 1, rep 11) VIPER's best CV
MSE (0.68) is below the single-factor theory for a supervised method (0.71), and the
elastic net reaches the oracle (0.52) from sparsity alone. To see how much of
VIPER's gain comes from the outcome term and how much from the MGP prior, each
setting is fitted with outcome_weight alpha = 0 (factors learned from X only; an
MGP factor model) and alpha = 1 (the supervised model), on exactly the folds of the
Optuna studies (optuna_controlled.py: zero-variance columns dropped, make_cv_folds
seed 42, fold-wise standardisation, seed 42 + fold, SVD init, K_fit = K + 10).

Training runs for a fixed MAX_EPOCHS budget (the settings below hit the 1500-epoch
cap in the Optuna studies), in chunks of CHUNK epochs; after each chunk the fit is
scored, so the results also show the budget each variant needs. Per checkpoint:
    val_mse_own    validation MSE with E_q[beta] (closed form; meaningless at alpha 0,
                   where beta stays at its prior; alpha 0 is fitted with weight ALPHA_ZERO)
    val_mse_ols    validation MSE with OLS (with intercept) of training y on the
                   x-only closed-form training scores m_z(X_train): the "factor model +
                   OLS" analogue of PCA + OLS and MOFA + OLS
    risk_rel_*     exact relative excess risk of those two predictors (risk_metrics)
    kappa2_y       alignment of the fitted loadings with the true outcome direction
    cov_frob       implied-covariance error (Lambda Sigma_Z Lambda^T, as in the studies)
    elapsed        training wall time so far (scoring excluded)
plus the per-epoch loss, so the ma_patience stop can be replayed offline.

Output: sim/results/integration_shared_controlled/alpha_zero/<label>/checkpoints.csv

usage: OMP_NUM_THREADS=2 python alpha_zero_controlled.py --configs 14 6 --alphas 0 1
           --folds 0 1 2 [--data controlled:500,3000,1,11] [--max-epochs 1500]
           [--chunk 100] [--workers 4] [--label full]
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

OUT = os.path.expanduser("~/multiomic_integration/sim/results/integration_shared_controlled/alpha_zero")
N_FOLDS, CV_SEED = 3, 42           # as in the Optuna studies
# Pyro's scale handler needs a positive weight: alpha = 0 is fitted with this weight
# on the outcome log-likelihood (negligible next to the views' ~1e6-scale term).
ALPHA_ZERO = 1e-6

# Settings from the Optuna rerun (c1 / anti / c10 studies), configuration number -> settings.
# The two best on c = 1 that completed all three folds there (configuration 33 diverged
# on folds 0 and 2, so its c = 1 MSE is one fold).
SETTINGS = {
    6: {"eta": 0.16409286730647915, "t": 0.021930485556643693, "minibatch_fraction": 0.05807932994623227},
    14: {"eta": 0.35033984911586863, "t": 0.024970737145052723, "minibatch_fraction": 0.05063981628665742},
}


def run_one(spec, config, alpha, fold, max_epochs, chunk, out_dir):
    import pyro
    import multiview_shared  # noqa: F401  (import before seeding)
    from torch.utils.data import TensorDataset
    import risk_metrics as rm
    from constants import Params
    from data_utils import normalize_tensor_by_col, implied_cov_errors, zero_variance_col_filter
    from run_methods import train_model, make_cv_folds, _minibatch_size
    from optuna_controlled import load_training, K_DELTA
    from controlled_viper_diagnosis import _ols
    from viper_predict import closed_form_scores, guide_moments

    sim = load_training(spec)
    filters = [zero_variance_col_filter(X) for X in sim["X_l"]]
    X_raw = [X[:, f] for X, f in zip(sim["X_l"], filters)]
    y_raw = sim["y"].reshape(-1)
    tr, va = make_cv_folds(X_raw[0].shape[0], n_folds=N_FOLDS, seed=CV_SEED)[fold]
    norms = [normalize_tensor_by_col(X[tr]) for X in X_raw]
    Xtr = [d["data_clean"] for d in norms]
    means, sds = [d["means"] for d in norms], [d["sds"] for d in norms]
    Xva = [normalize_tensor_by_col(X[va], m, s)["data_clean"] for X, m, s in zip(X_raw, means, sds)]
    ym, ysd = y_raw[tr].mean(), y_raw[tr].std()
    ytr, yva = (y_raw[tr] - ym) / ysd, (y_raw[va] - ym) / ysd
    L = len(X_raw)
    dp = {"SIM_decomp": {"SIM_Lambda_l_list": sim["Lambda_l"]}, "X_l_list_column_filters": filters,
          "X_l_sd_list": sds, "X_l_mean_list": means, "L": L}
    theta = rm.true_params(sim)
    orc = rm.oracle(theta)
    train_ds = TensorDataset(torch.arange(len(tr)), *Xtr, ytr)
    y_tr = ytr.double().numpy()
    y_va = yva.double().numpy()

    cfg = SETTINGS[config]
    train_config = {**cfg, "delta": 1e-16, "num_particles": 1, "min_epochs": chunk, "max_epochs": chunk,
                    "convergence_criterion": "elbo_snr", "window": 50, "snr_threshold": 5.0}
    pyro.clear_param_store()
    pyro.set_rng_seed(CV_SEED + fold)
    t0 = time.time()
    model, handler, _ = train_model({"k": int(sim["K"]) + K_DELTA, "dense_model": False,
                                     "outcome_weight": float(alpha) or ALPHA_ZERO}, train_config, train_ds, None,
                                    model_type="shared", opt="adagrad", write=False, svd_init=True)
    train_time = time.time() - t0
    key = {"data": spec, "config": config, "alpha": alpha, "outcome_weight": float(alpha) or ALPHA_ZERO,
           "fold": fold, **cfg}
    rows, epoch = [], chunk
    while True:
        store = pyro.get_param_store()
        mom = guide_moments(store, L)
        w = [a / b for a, b in zip(mom["a_psi"], mom["b_psi"])]
        cf_tr = closed_form_scores(mom, Xtr)
        cf_va = closed_form_scores(mom, Xva)
        A_inv = torch.linalg.inv(cf_tr["A_bar"])
        to_x = lambda cz: [(M_l * w_l[:, None]) @ (A_inv @ torch.as_tensor(cz, dtype=torch.float64))
                           for M_l, w_l in zip(mom["M"], w)]
        c_ols, b_ols, r2 = _ols(cf_tr["m_z"].numpy(), y_tr)
        yhat_own = cf_va["yhat"].numpy()
        yhat_ols = c_ols + cf_va["m_z"].numpy() @ b_ols
        r_own = rm.risk_summary(to_x(mom["m_beta"].numpy()), dp, float(ym), float(ysd), theta, orc, 0.0)
        r_ols = rm.risk_summary(to_x(b_ols), dp, float(ym), float(ysd), theta, orc, c_ols)
        Lam = [store[Params.loc_Lambda_l.format(l=l)].detach().double().squeeze() for l in range(L)]
        M = store[Params.loc_Z].detach().double()
        S = store[Params.scale_Z].detach().double()
        Sigma_Z = (M.T @ M + torch.diag((S ** 2).sum(0))) / M.shape[0]
        rows.append({**key, "epoch": epoch, "elapsed": train_time,
                     "val_mse_own": float(((yhat_own - y_va) ** 2).mean()),
                     "val_mse_ols": float(((yhat_ols - y_va) ** 2).mean()),
                     "risk_rel_own": r_own["risk_rel"], "risk_rel_ols": r_ols["risk_rel"],
                     "train_r2_ols": r2,
                     "kappa2_y": rm.kappa2_hat(rm.raw_loadings(mom["M"], dp), theta)["kappa2_y"],
                     "cov_frob": float(implied_cov_errors(Lam, Sigma_Z, dp).cov_frob.mean())})
        if epoch >= max_epochs:
            break
        t1 = time.time()
        handler.do_inference(train_dataset=train_ds, min_epochs=chunk, epochs=chunk, minibatch_flag=True,
                             minibatch_size=_minibatch_size(train_config, len(train_ds)),
                             variational_diff_func=np.mean, verbose=False,
                             convergence_criterion="elbo_snr", window=50, snr_threshold=5.0)
        train_time += time.time() - t1
        epoch += chunk
    np.save(os.path.join(out_dir, f"loss_cfg{config}_alpha{alpha:g}_fold{fold}.npy"),
            np.asarray(model.loss_history, float))
    pyro.clear_param_store()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="controlled:500,3000,1,11")
    ap.add_argument("--configs", nargs="+", type=int, default=[14, 6])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 1.0])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max-epochs", type=int, default=1500)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--label", default="smoke")
    args = ap.parse_args()
    out_dir = os.path.join(OUT, args.label)
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(args.data, c, a, f) for c in args.configs for a in args.alphas for f in args.folds]
    from joblib import Parallel, delayed
    t0 = time.time()
    res = Parallel(n_jobs=args.workers, backend="loky")(
        delayed(run_one)(*j, args.max_epochs, args.chunk, out_dir) for j in jobs)
    out = pd.DataFrame([r for rows in res for r in rows])
    out.to_csv(os.path.join(out_dir, "checkpoints.csv"), index=False)
    print(out[["config", "alpha", "fold", "epoch", "elapsed", "val_mse_own", "val_mse_ols",
               "kappa2_y", "cov_frob"]].round(4).to_string())
    print(f"{len(jobs)} runs, {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
