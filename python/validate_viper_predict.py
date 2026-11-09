"""
Check the closed-form VIPER-Shared test prediction (viper_predict.py) against
the pipeline's local SVI on saved misaligned fits.

For each dataset: load the saved global fit (k_delta = 10, K_fit = 15), run local
SVI on the test views with the settings of sim_run_shared_parallel.py, and
compare with the closed-form scores computed from the same param store:
    m_z     closed form vs loc_Z_pred
    sd_z    closed form (1/sqrt(A_bar_kk)) vs scale_Z_pred
    yhat    closed form vs loc_Z_pred @ loc_beta (plug-in), and vs the pipeline's
            posterior-mean y_pred (100 predictive draws, so it carries Monte Carlo noise)
Option 2 (Monte Carlo over q_g) is reported alongside.

Local SVI and the closed form optimise the same objective, the local ELBO for a
diagonal Gaussian q(z_i) = N(m_i, diag(s_i^2)) with the global factors of q fixed:
    ELBO_i = -1/2 (m_i^T A_bar m_i + sum_k A_bar_kk s_ik^2) + h_i^T m_i + sum_k log s_ik + const,
which the closed form maximises exactly. local_elbo_gap = mean_i (ELBO_i(closed
form) - ELBO_i(local SVI)) >= 0 measures how far local SVI is from that optimum.

Output: sim/results/integration_shared_misaligned/viper_predict_validation.csv

usage: python validate_viper_predict.py
"""
import os
import sys

import numpy as np
import pandas as pd
import pyro
import torch

sys.path.insert(1, "/Users/jonathanhori/multiomic_integration/python")
from constants import Params, Sites
from run_methods import load_model, load_rds, process_data, train_model_locally
from viper_predict import closed_form_scores, guide_moments, mc_scores
import sim_run_shared_parallel as cfg

DATA_ROOT = os.path.expanduser("~/multiomic_integration/sim/data/multi_view_shared_misaligned/k.5")
RESULTS = os.path.expanduser("~/multiomic_integration/sim/results/integration_shared_misaligned")
DATASETS = [  # (cond_dir, n, p, label, rep): small/large gamma, different alignments
    ("n100p1000_snr2.2_sparse0.25_align_anti_aligned", 100, 1000, "anti_aligned", 1),
    ("n500p100_snr2.2_sparse0.25_align_strong_misalignment", 500, 100, "strong_misalignment", 1),
    ("n1000p500_snr2.2_sparse0.25_align_mild_misalignment", 1000, 500, "mild_misalignment", 1),
]
K_DELTA = 10


def local_elbo(m, s, A, h):
    """Exact local ELBO per sample (up to a constant) for diagonal Gaussian q(z)."""
    return (-0.5 * (torch.einsum("ik,kl,il->i", m, A, m) + (s ** 2 * torch.diag(A)).sum(1))
            + (h * m).sum(1) + torch.log(s).sum(1))


def main():
    train_config = {"eta": cfg.ETA, "delta": cfg.DELTA, "t": cfg.T, "num_particles": cfg.NUM_PARTICLES,
                    "minibatch_fraction": cfg.MINIBATCH_FRACTION, "min_epochs": cfg.MIN_EPOCHS_LOCAL,
                    "max_epochs": cfg.NUM_EPOCHS, "convergence_criterion": cfg.CONVERGENCE_CRITERION,
                    "window": cfg.WINDOW, "snr_threshold": cfg.SNR_THRESHOLD}
    rows = []
    for cond, n, p, label, rep in DATASETS:
        tr = load_rds(f"{DATA_ROOT}/train/{cond}/sim_data_shared_rep{rep}.rds")
        te = load_rds(f"{DATA_ROOT}/test/{cond}/sim_data_shared_rep{rep}.rds")
        train_subset, test_subset, dp = process_data(tr, te, model_type="shared")
        L = dp["L"]
        stem = os.path.join(RESULTS, "models", f"shared_asym_n{n}p{p}_align{label}_deltak{K_DELTA}_rep{rep}")
        k_fit = int(tr["K"]) + K_DELTA

        pyro.set_rng_seed(rep)
        model, train_handler, _ = load_model(stem, {"k": k_fit, "dense_model": False}, len(train_subset),
                                             train_config, model_type="shared")
        moments = guide_moments(pyro.get_param_store(), L)      # before local SVI adds its params
        X_te = dp["test_X_l_list_clean"]
        cf = closed_form_scores(moments, X_te, cov="diag")
        mc = mc_scores(moments, X_te, n_draws=500, seed=rep)

        model, test_handler, _ = train_model_locally(model, train_config, test_subset,
                                                     model_type="shared", opt="adagrad")
        store = pyro.get_param_store()
        loc_Zp = store[Params.loc_Z_pred].detach().double()
        sd_Zp = store[Params.scale_Z_pred].detach().double()
        y_plugin = loc_Zp @ moments["m_beta"]
        post = test_handler.predict(X_te, cfg.N_POSTERIOR_SAMPLES, [Sites.y_pred])
        y_pipe = post[Sites.y_pred].double().mean(0).reshape(-1)
        y_te = dp["test_y_clean"].double().reshape(-1)

        h = cf["m_z"] @ cf["A_bar"]
        sd_cf = cf["cov_z"].diag().sqrt().expand_as(sd_Zp)
        gap = local_elbo(cf["m_z"], sd_cf, cf["A_bar"], h) - local_elbo(loc_Zp, sd_Zp, cf["A_bar"], h)
        mse = lambda a: float(((a - y_te) ** 2).mean())
        mx = lambda a, b: float((a - b).abs().max())
        # Columns that the MGP prior switched off have ~0 loadings and scores ~ N(0, 1);
        # report score differences over all columns and over the active ones.
        active = torch.diag(cf["A_bar"]) > 1.5
        rows.append({
            "dataset": cond, "rep": rep, "n_train": n, "n_test": len(y_te), "k_fit": k_fit,
            "n_active_cols": int(active.sum()),
            "max_abs_diff_m_z": mx(cf["m_z"], loc_Zp),
            "max_abs_diff_m_z_active": mx(cf["m_z"][:, active], loc_Zp[:, active]),
            "sd_of_m_z_active": float(cf["m_z"][:, active].std()),
            "max_abs_diff_sd_z": mx(sd_cf, sd_Zp),
            "local_elbo_gap": float(gap.mean()), "local_elbo_gap_min": float(gap.min()),
            "max_abs_diff_yhat_vs_plugin": mx(cf["yhat"], y_plugin),
            "max_abs_diff_yhat_vs_pipeline": mx(cf["yhat"], y_pipe),
            "max_abs_diff_yhat_mc_vs_cf": mx(mc["yhat"], cf["yhat"]),
            "sd_of_yhat": float(cf["yhat"].std()),
            "mse_closed_form": mse(cf["yhat"]), "mse_mc": mse(mc["yhat"]),
            "mse_local_svi_plugin": mse(y_plugin), "mse_pipeline": mse(y_pipe),
            "mean_pred_sd_closed_form": float(cf["pred_var"].sqrt().mean()),
            "mean_pred_sd_mc": float(mc["pred_var"].sqrt().mean()),
        })
        print(pd.Series(rows[-1]).to_string(), "\n", flush=True)
        pyro.clear_param_store()

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "viper_predict_validation.csv"), index=False)
    print(out.T.to_string())


if __name__ == "__main__":
    main()
