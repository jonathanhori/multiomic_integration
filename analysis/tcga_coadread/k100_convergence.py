"""
Is the shared model trained long enough with K = 100 factors?

In k_sweep.py every shared fit stopped at about 200 epochs (min_epochs, chosen by
convergence_check.py at K = 20), and with K = 100 its test R² and test
reconstruction trail PCA's. This refits K = 100 with fixed training lengths (the
stopping rule switched off by min_epochs = max_epochs) on two splits and records,
for each length:

    final training loss per sample (mean of the last 20 epochs)
    test R², 95% interval coverage (log(1 + TMB) scale), test RSE of the views
    test R² with beta refitted by OLS on the training scores
    how much the fitted signal Z Lambda^T changed from the previous length

Splits: 42 (the notebook's split) and 2 (where K = 100 did worse than K = 50).
Other settings as in compare_methods.py / k_sweep.py.

Outputs (results/k_sweep/)
    convergence_k100.csv            one row per split and training length
    convergence_k100_loss_s<seed>_e<epochs>.csv   loss per epoch

usage: python k100_convergence.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pyro
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, HERE)
import compare_methods as cm
from run_methods import train_model, train_model_locally
from data_utils import summarise_post_samples
from constants import Sites, Params

warnings.filterwarnings("ignore")

OUT    = os.path.join(cm.RESULTS, "k_sweep")
K      = 100
SEEDS  = [42, 2]
EPOCHS = [200, 400, 800]


def main():
    out_csv = os.path.join(OUT, "convergence_k100.csv")
    rows = pd.read_csv(out_csv).to_dict("records") if os.path.exists(out_csv) else []
    done = {(r["seed"], r["epochs"]) for r in rows}
    for seed in SEEDS:
        data, train_subset, test_subset, dp = cm.load_split(seed)
        y_mean, y_sd = float(data["y"].mean()), float(data["y"].std())
        y_true = data["y_test"].reshape(-1)
        y_std_train = dp["train_y_clean"].reshape(-1)
        previous_signal = None
        for epochs in EPOCHS:
            if (seed, epochs) in done:
                continue
            pyro.clear_param_store()
            pyro.set_rng_seed(seed)
            config = {**cm.SHARED_TRAIN_CONFIG, "min_epochs": epochs, "max_epochs": epochs}
            fm, _, gs = train_model({**cm.SHARED_MODEL_CONFIG, "k": K}, config, train_subset,
                                    model_type="shared", write=False, svd_init=True)
            Z = pyro.param(Params.loc_Z).detach()
            Lam = [pyro.param(Params.loc_Lambda_l.format(l=l)).detach().squeeze() for l in range(len(cm.VIEWS))]
            signal = torch.cat([Z @ L.T for L in Lam], 1)
            change = (float(((signal - previous_signal) ** 2).sum() / (previous_signal ** 2).sum())
                      if previous_signal is not None else np.nan)
            previous_signal = signal
            beta_ols = torch.linalg.lstsq(Z, y_std_train.unsqueeze(1)).solution.squeeze(1)

            fm, teh, _ = train_model_locally(fm, cm.SHARED_TRAIN_CONFIG, test_subset, model_type="shared")
            post = teh.predict(dp["test_X_l_list_clean"], cm.N_POSTERIOR_SAMPLES, [Sites.y_pred, Sites.Z_pred])
            pred = summarise_post_samples(post[Sites.y_pred].squeeze().cpu() * y_sd + y_mean)
            Z_test = post[Sites.Z_pred].mean(0).squeeze().cpu()
            r2 = lambda p: float(1 - ((p - y_true) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())
            loss = np.array(fm.loss_history) / len(train_subset)
            pd.DataFrame({"epoch": np.arange(1, len(loss) + 1), "loss_per_sample": loss}).to_csv(
                os.path.join(OUT, f"convergence_k100_loss_s{seed}_e{epochs}.csv"), index=False)
            rows.append({
                "seed": seed, "epochs": epochs,
                "final_loss_per_sample": float(loss[-20:].mean()),
                "test_r2": r2(pred["mean"]),
                "test_r2_ols_beta": r2(Z_test @ beta_ols * y_sd + y_mean),
                "coverage_95": float(((y_true >= pred["q2.5"]) & (y_true <= pred["q97.5"])).float().mean()),
                "test_rse": cm.rse(dp["test_X_l_list_clean"], [Z_test @ L.T for L in Lam]),
                "signal_change_vs_previous": change,
                "train_seconds": gs["inference_time"],
            })
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"seed {seed} epochs {epochs}: " + ", ".join(
                f"{k}={v:.3f}" for k, v in rows[-1].items() if k not in ("seed", "epochs")), flush=True)


if __name__ == "__main__":
    main()
