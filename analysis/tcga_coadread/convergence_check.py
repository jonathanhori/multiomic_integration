"""
Is the shared model trained long enough on TCGA COAD/READ?

Fits the model from tcga_coadread.qmd (same split, genes, settings and seed)
with fixed training lengths — the stopping rule is switched off by setting
min_epochs = max_epochs — and records, for each length:

    final training loss per sample (mean of the last 20 epochs)
    test R², MSE, 95% interval coverage and width (log(1 + TMB) scale)
    number of factors explaining > 1% of the data
    how much the fitted signal changed from the previous length
        (relative squared difference of Z Lambda^T on the training samples)

Outputs: results/convergence_check.csv, results/convergence_loss_<epochs>.csv

usage: python convergence_check.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pyro
import torch

sys.path.insert(1, os.path.expanduser("~/multiomic_integration/python"))
from coadread_data import load_coadread, make_view_list
from run_methods import process_data, train_model, train_model_locally
from data_utils import summarise_post_samples
from constants import Sites, Params

warnings.filterwarnings("ignore")

HERE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
EPOCHS  = [100, 200, 400, 800, 1600]
SEED    = 42
VIEWS   = ["expr", "meth_tss200"]
K       = 20
MODEL_CONFIG = {"k": K, "dense_model": False, "outcome_weight": 1.0}
BASE_TRAIN_CONFIG = {
    "eta": 0.25, "delta": 1e-16, "t": 0.1, "num_particles": 1,
    "minibatch_fraction": 0.15, "convergence_criterion": "elbo_snr",
    "window": 50, "snr_threshold": 2,
}
# Local inference keeps the notebook's settings (it starts from the frozen fit).
LOCAL_CONFIG = {**BASE_TRAIN_CONFIG, "min_epochs": 100, "max_epochs": 1000}


def main():
    data = load_coadread(top_rna_genes=2000, top_meth_genes=1000, log1p_tmb=True,
                         train_frac=0.8, random_seed=SEED, preprocess_on_train_only=True)
    real = lambda split, y: {"X_l": make_view_list(data, VIEWS, split=split), "y": y,
                             "K": torch.tensor(K), "Z": None}
    train_subset, test_subset, dp = process_data(
        real("train", data["y"]), real("test", data["y_test"]), model_type="shared")
    y_mean, y_sd = float(data["y"].mean()), float(data["y"].std())
    y_true = data["y_test"].reshape(-1)
    n_train = len(train_subset)

    rows, previous_signal = [], None
    for epochs in EPOCHS:
        pyro.clear_param_store()
        pyro.set_rng_seed(SEED)
        config = {**BASE_TRAIN_CONFIG, "min_epochs": epochs, "max_epochs": epochs}
        fm, _, gs = train_model(MODEL_CONFIG, config, train_subset, model_type="shared",
                                write=False, svd_init=True)
        Z = pyro.param(Params.loc_Z).detach()
        Lam = [pyro.param(Params.loc_Lambda_l.format(l=l)).detach().squeeze() for l in range(len(VIEWS))]
        signal = torch.cat([Z @ L.T for L in Lam], 1)
        ss = sum(np.sum(L.numpy() ** 2, 0) * np.sum(Z.numpy() ** 2, 0) for L in Lam)
        total = sum(float((x ** 2).sum()) for x in dp["train_X_l_list_clean"])
        n_active = int((ss / total > 0.01).sum())
        change = (float(((signal - previous_signal) ** 2).sum() / (previous_signal ** 2).sum())
                  if previous_signal is not None else np.nan)
        previous_signal = signal

        fm, teh, ls = train_model_locally(fm, LOCAL_CONFIG, test_subset, model_type="shared")
        post = teh.predict(dp["test_X_l_list_clean"], 300, [Sites.y_pred])
        pred = summarise_post_samples(post[Sites.y_pred].squeeze().cpu() * y_sd + y_mean)
        resid = pred["mean"] - y_true
        loss = np.array(fm.loss_history) / n_train
        pd.DataFrame({"epoch": np.arange(1, len(loss) + 1), "loss_per_sample": loss}).to_csv(
            os.path.join(RESULTS, f"convergence_loss_{epochs}.csv"), index=False)
        rows.append({
            "epochs": epochs,
            "final_loss_per_sample": float(loss[-20:].mean()),
            "test_r2": float(1 - (resid ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()),
            "test_mse": float((resid ** 2).mean()),
            "coverage_95": float(((y_true >= pred["q2.5"]) & (y_true <= pred["q97.5"])).float().mean()),
            "width_95": float((pred["q97.5"] - pred["q2.5"]).mean()),
            "active_factors": n_active,
            "signal_change_vs_previous": change,
            "train_seconds": gs["inference_time"],
        })
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "convergence_check.csv"), index=False)
        print(rows[-1], flush=True)


if __name__ == "__main__":
    main()
