"""
Covariance reconstruction for the global-eta study's simulation fits.

For each saved fit (sim/results/global_eta_study/<data>_rep<r>_k<K>_<opt>_fits.npz)
and each saved checkpoint ("ma" = the MA stop's checkpoint, 1000 = end of
training), the factor part of the implied covariance Lambda_l Sigma_Z Lambda_m^T
is compared with the true one on the standardised scale, for every pair of
views l <= m, with data_utils.implied_cov_errors (mean squared entrywise error,
the measure used elsewhere in the repo), and as a relative error: divided by
the mean squared entry of the true covariance (the error of predicting 0). Sigma_Z is the second moment of the
fitted score means (the score variances are not saved, so the posterior
variance of Z is left out; it is small next to the signal at n = 500).

TCGA has no true covariance, so only the simulations are scored.

Output: sim/results/global_eta_study/cov_error.csv
usage: python global_eta_cov.py
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/multiomic_integration")
sys.path.insert(0, f"{ROOT}/python")
OUT = f"{ROOT}/sim/results/global_eta_study"


def main():
    import torch
    from data_utils import implied_cov_errors, scores_second_moment
    from kmax_prior_check import get_data
    from run_methods import process_data

    pat = re.compile(r"(ctrl|mis)_rep(\d+)_k(\d+)_(.+)_fits\.npz")
    files = sorted(glob.glob(f"{OUT}/*_fits.npz"))
    dps, rows = {}, []
    for f in files:
        m = pat.match(os.path.basename(f))
        if not m:
            continue
        data, rep, k, opt = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        if (data, rep) not in dps:
            tr, te = get_data(data, rep)
            dps[(data, rep)] = process_data(tr, te, model_type="shared")[2]
        dp = dps[(data, rep)]
        p = [x.shape[1] for x in dp["train_X_l_list_clean"]]
        if (data, rep, "base") not in dps:     # error of an all-zero estimate = mean true^2
            dps[(data, rep, "base")] = implied_cov_errors([torch.zeros(q, 1) for q in p], None, dp).cov_frob.values
        d = np.load(f)
        for when in ("ma", "1000"):
            if f"{when}_W" not in d:
                continue
            W, Z = d[f"{when}_W"], d[f"{when}_Z"]
            Lam = np.split(W, np.cumsum(p)[:-1])
            err = implied_cov_errors([torch.as_tensor(x) for x in Lam], scores_second_moment(Z), dp)
            base = dps[(data, rep, "base")]
            for r, b in zip(err.itertuples(), base):
                rows.append({"data": data, "rep": rep, "K_max": k, "opt": opt,
                             "when": "MA stop" if when == "ma" else "epoch 1000",
                             "view_l": r.view_l, "view_m": r.view_m, "cov_type": r.cov_type,
                             "cov_frob": r.cov_frob, "cov_rel": r.cov_frob / b})
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/cov_error.csv", index=False)
    print(out.groupby(["data", "K_max", "opt", "when"]).cov_rel.mean().unstack("when").round(4).to_string())


if __name__ == "__main__":
    main()
