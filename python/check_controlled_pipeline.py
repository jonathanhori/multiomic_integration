"""
Check that controlled-strength data (R generate_factor_data_shared_controlled)
pass through the Python pipeline: load_rds -> process_data -> risk_metrics.

For r_out in {0, 0.25}: R writes a training replicate, a 100-sample test set and a
large (40 000-sample) test set to a temporary directory, together with R's
controlled_oracle R*. Python then checks
    - process_data keeps every feature (psi = 1, so no all-zero columns)
    - risk_metrics.true_params reads psi, sigma2_y and theta_out from the file
    - risk_metrics.oracle R* equals R's R* (float32 transfer, so to ~1e-5 relative)
    - the oracle's exact risk (risk_rel = 0) matches its empirical MSE on the large set
    - g_true from the file equals G[k*, k*] from risk_metrics
    - a fitted linear predictor (ridge) has exact risk equal to its large-set MSE

usage: python check_controlled_pipeline.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(1, "/Users/jonathanhori/multiomic_integration/python")
from run_methods import load_rds, process_data
import risk_metrics as rm

R_SCRIPT = r"""
source(file.path(Sys.getenv("HOME"), "multiomic_integration", "R", "sim_methods.R"))
args <- commandArgs(trailingOnly = TRUE)
out <- args[1]; r_out <- as.numeric(args[2])
set.seed(7)
tr <- generate_factor_data_shared_controlled(200, c(200, 200, 200), 3, r_out = r_out)
saveRDS(tr, file.path(out, "train.rds"))
saveRDS(generate_factor_data_shared_controlled_testing(100, tr), file.path(out, "test.rds"))
saveRDS(generate_factor_data_shared_controlled_testing(40000, tr), file.path(out, "big.rds"))
writeLines(format(controlled_oracle(tr)$R_star, digits = 15), file.path(out, "R_star.txt"))
"""


def ok(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok ", msg)


def main():
    from sklearn.linear_model import RidgeCV
    for r_out in (0.0, 0.25):
        print(f"r_out = {r_out}")
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "gen.R")
            open(script, "w").write(R_SCRIPT)
            subprocess.run(["Rscript", script, tmp, str(r_out)], check=True, stdout=subprocess.DEVNULL)
            tr, te, big = (load_rds(os.path.join(tmp, f)) for f in ("train.rds", "test.rds", "big.rds"))
            R_star_R = float(open(os.path.join(tmp, "R_star.txt")).read())

        _, _, dp = process_data(tr, te, model_type="shared")
        ok(all(bool(f.all()) for f in dp["X_l_list_column_filters"]), "no columns dropped")
        theta = rm.true_params(tr)
        ok(all(bool((p == 1).all()) for p in theta["psi"]), "psi read from file (all 1)")
        ok((theta["theta"] is None) == (r_out == 0), "theta_out read from file")
        orc = rm.oracle(theta)
        ok(abs(orc["R_star"] - R_star_R) / R_star_R < 1e-5, f"R* Python {orc['R_star']:.6f} = R {R_star_R:.6f}")
        k = int(tr["k_star"].item()) - 1
        ok(abs(float(orc["G"][k, k]) - float(tr["g_true"].item())) / float(tr["g_true"].item()) < 1e-5,
           "g_true = G[k*, k*]")

        Xb = torch.cat([x.double() for x in big["X_l"]], 1)
        yb = big["y"].double().reshape(-1)
        b_star = torch.cat(orc["b_star"])
        sq = (yb - Xb @ b_star) ** 2
        se = float(sq.std() / np.sqrt(len(sq)))
        ok(abs(float(sq.mean()) - orc["R_star"]) < 4 * se,
           f"oracle empirical MSE {float(sq.mean()):.4f} vs R* {orc['R_star']:.4f} (se {se:.4f})")

        X = torch.cat(dp["train_X_l_list_clean"], 1).double().numpy()
        y = dp["train_y_clean"].double().reshape(-1).numpy()
        fit = RidgeCV(alphas=np.logspace(-2, 5, 30), cv=5).fit(X, y)
        y_raw = tr["y"].double().reshape(-1)
        b, c = rm.to_raw_scale(fit.coef_, dp, y_raw.mean(), y_raw.std(), fit.intercept_)
        risk = rm.exact_risk(b, c, theta, orc)["risk"]
        sq = (yb - (c + Xb @ torch.cat(b))) ** 2
        se = float(sq.std() / np.sqrt(len(sq)))
        ok(abs(float(sq.mean()) - risk) < 4 * se,
           f"ridge exact risk {risk:.4f} vs empirical {float(sq.mean()):.4f} (se {se:.4f})")
    print("all checks passed")


if __name__ == "__main__":
    main()
