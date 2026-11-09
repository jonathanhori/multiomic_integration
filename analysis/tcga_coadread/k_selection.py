"""
How many factors does the shared model actually use as K_max grows?

The MGP prior's column shrinkage tau_lk = prod_{h<=k} delta_lh should switch off
factors the data do not need, so the number of factors in use should level off
as K_max increases. This refits the shared model with K_max = 20, 50, 100 on
two splits, each for a fixed 800 epochs (k100_convergence.py: 200 epochs is not
enough at large K; the stopping rule is switched off), and saves the full
parameter store so the column shrinkage can be read:

    E_q[tau_lk] = prod_{h<=k} a_lh / b_lh      (viper_predict.expected_tau)

Splits: 42 (the notebook's split) and 2. Other settings as in compare_methods.py.

Outputs (results/k_sweep/selection/)
    selection_fits.csv                  one row per split and K_max: test R², coverage,
                                        test RSE, final loss
    seed<s>_k<K>_paramstore.pt          Pyro parameter store after training
    seed<s>_k<K>_factors.npz            training/test scores, loadings and posterior sd,
                                        beta, E[tau] per view

usage: python k_selection.py
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
from viper_predict import expected_tau

warnings.filterwarnings("ignore")
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))   # shares the machine with other studies

OUT    = os.path.join(cm.RESULTS, "k_sweep", "selection")
SEEDS  = [42, 2]
K_GRID = [20, 50, 100]
EPOCHS = 800


def main():
    os.makedirs(OUT, exist_ok=True)
    out_csv = os.path.join(OUT, "selection_fits.csv")
    rows = pd.read_csv(out_csv).to_dict("records") if os.path.exists(out_csv) else []
    done = {(r["seed"], r["k"]) for r in rows}
    for seed in SEEDS:
        data, train_subset, test_subset, dp = cm.load_split(seed)
        y_mean, y_sd = float(data["y"].mean()), float(data["y"].std())
        y_true = data["y_test"].reshape(-1)
        L = len(cm.VIEWS)
        for k in K_GRID:
            if (seed, k) in done:
                continue
            pyro.clear_param_store()
            pyro.set_rng_seed(seed)
            config = {**cm.SHARED_TRAIN_CONFIG, "min_epochs": EPOCHS, "max_epochs": EPOCHS}
            try:
                fm, _, gs = train_model({**cm.SHARED_MODEL_CONFIG, "k": k}, config, train_subset,
                                        model_type="shared", write=False, svd_init=True)
            except ValueError as e:          # NaN parameters: record the failure and move on
                print(f"seed {seed} K={k}: FAILED: {str(e).splitlines()[0][:120]}", flush=True)
                rows.append({"seed": seed, "k": k, "failed": True})
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                continue
            store = pyro.get_param_store()
            store.save(os.path.join(OUT, f"seed{seed}_k{k}_paramstore.pt"))
            tau = [t.numpy() for t in expected_tau(store, L)]
            Lam = [pyro.param(Params.loc_Lambda_l.format(l=l)).detach().squeeze() for l in range(L)]
            Lam_sd = [pyro.param(Params.scale_Lambda_l.format(l=l)).detach().squeeze() for l in range(L)]
            Z_train = pyro.param(Params.loc_Z).detach()
            beta = pyro.param(Params.loc_beta).detach().reshape(-1)

            fm, teh, _ = train_model_locally(fm, cm.SHARED_TRAIN_CONFIG, test_subset, model_type="shared")
            post = teh.predict(dp["test_X_l_list_clean"], cm.N_POSTERIOR_SAMPLES, [Sites.y_pred, Sites.Z_pred])
            pred = summarise_post_samples(post[Sites.y_pred].squeeze().cpu() * y_sd + y_mean)
            Z_test = post[Sites.Z_pred].mean(0).squeeze().cpu()
            np.savez(os.path.join(OUT, f"seed{seed}_k{k}_factors.npz"),
                     Z_train=Z_train.numpy(), Z_test=Z_test.numpy(), beta=beta.numpy(),
                     **{f"loadings_view{l}": Lam[l].numpy() for l in range(L)},
                     **{f"scale_view{l}": Lam_sd[l].numpy() for l in range(L)},
                     **{f"tau_view{l}": tau[l] for l in range(L)})
            loss = np.array(fm.loss_history) / len(train_subset)
            rows.append({
                "seed": seed, "k": k,
                "test_r2": float(1 - ((pred["mean"] - y_true) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()),
                "coverage_95": float(((y_true >= pred["q2.5"]) & (y_true <= pred["q97.5"])).float().mean()),
                "test_rse": cm.rse(dp["test_X_l_list_clean"], [Z_test @ Lv.T for Lv in Lam]),
                "median_loss_last50": float(np.median(loss[-50:])),
                "train_seconds": gs["inference_time"],
            })
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"seed {seed} K={k}: " + ", ".join(f"{a}={b:.3f}" for a, b in rows[-1].items()
                                                     if a not in ("seed", "k")), flush=True)


if __name__ == "__main__":
    main()
