"""
Covariance reconstruction of the shared model on the misaligned datasets,
recomputed from the saved June 2026 fits with the corrected metric.

The June metric files scored covariance as Lambda Lambda^T, which assumes the
fitted scores have unit variance. This script uses the fitted model's implied
covariance Lambda Sigma_Z Lambda^T, with
    Sigma_Z = (1/n) sum_i E_q[z_i z_i^T] = (1/n) (M^T M + sum_i diag(s_i^2)),
M the variational means (loc_Z) and s_i the variational standard deviations
(scale_Z) of the training scores — the same quantity run_methods.evaluate_fitted_model
now uses. Parameters are read through pyro.param (constrained values).

Output: sim/results/integration_shared_misaligned/shared_cov.csv
        (one row per dataset: cov_within, cov_cross)

usage: python misaligned_shared_cov.py
"""
import os
import re
import sys
import warnings

import pandas as pd
import pyro
import torch

sys.path.insert(1, "/Users/jonathanhori/multiomic_integration/python")
from run_methods import load_rds, process_data
from data_utils import implied_cov_errors
from constants import Params

warnings.filterwarnings("ignore")

DATA_ROOT = os.path.expanduser("~/multiomic_integration/sim/data/multi_view_shared_misaligned/k.5")
RESULTS = os.path.expanduser("~/multiomic_integration/sim/results/integration_shared_misaligned")
COND_PATTERN = re.compile(r"n(\d+)p(\d+)_snr([\d.]+)\.([\d.]+)_sparse([\d.]+)_align_(.+)")
K_DELTA = 10


def main():
    rows = []
    for cond in sorted(os.listdir(f"{DATA_ROOT}/train")):
        m = COND_PATTERN.match(cond)
        if not m:
            continue
        n, p, label = int(m[1]), int(m[2]), m[6]
        for rep in range(1, 11):
            stem = os.path.join(RESULTS, "models",
                                f"shared_asym_n{n}p{p}_align{label}_deltak{K_DELTA}_rep{rep}_paramstore.pth")
            if not os.path.exists(stem):
                continue
            tr = load_rds(f"{DATA_ROOT}/train/{cond}/sim_data_shared_rep{rep}.rds")
            te = load_rds(f"{DATA_ROOT}/test/{cond}/sim_data_shared_rep{rep}.rds")
            _, _, dp = process_data(tr, te, model_type="shared")
            L = len(dp["train_X_l_list_clean"])

            pyro.clear_param_store()
            pyro.get_param_store().set_state(torch.load(stem, weights_only=False))
            Lam = [pyro.param(Params.loc_Lambda_l.format(l=l)).detach().double().squeeze() for l in range(L)]
            M = pyro.param(Params.loc_Z).detach().double()
            S = pyro.param(Params.scale_Z).detach().double()
            Sigma_Z = (M.T @ M + torch.diag((S ** 2).sum(0))) / M.shape[0]

            cov = implied_cov_errors(Lam, Sigma_Z, dp)
            rows.append({"n": n, "p": p, "align_label": label, "rep": rep,
                         "cov_within": float(cov.loc[cov["cov_type"] == "within", "cov_frob"].mean()),
                         "cov_cross": float(cov.loc[cov["cov_type"] == "cross", "cov_frob"].mean())})
        print(f"{cond}: done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "shared_cov.csv"), index=False)
    print(f"{len(rows)} datasets -> shared_cov.csv")


if __name__ == "__main__":
    main()
