"""
Validation of the candidate VIPER-Shared optimiser settings on the controlled-strength
grid (held-out test sets), with the comparison methods and factor recovery.

Settings (from the Optuna rerun, sim/exploration/optuna_rerun.qmd):
    current   eta 0.25, t 0.1,   minibatch 0.15, elbo_snr (window 50, threshold 5),
              min 200 / max 500 epochs (sim_run_shared_parallel.py, unchanged)
    default   Optuna configuration 9:  eta 0.211, t 0.042, minibatch 0.166,
              ma_patience (window 50, patience 20), min 200 / max 1500 epochs
    long      Optuna configuration 14: eta 0.350, t 0.025, minibatch 0.051,
              ma_patience (window 114, patience 157), min 0 / max 1500 epochs
    long_mb15 long with at least 15 samples per minibatch (run_methods min_minibatch_size)
    long_eta25 long with eta 0.25 (the current setting's eta, which never diverged)
All: MGP prior, K_fit = K + 10 = 15, outcome_weight 1, SVD initialisation, delta 1e-16,
one particle, AdagradRMSProp; seed = rep (multiview_shared imported before seeding).

Per dataset (R/generate_controlled_data.R; docs/controlled_strength_data.md):
  1. train each setting on the training set; save the param store to
     results/.../validation/models/<setting>_<cond>_rep<rep>_paramstore.pth (reused
     when present) with its wall time and epochs;
  2. misaligned_exact_risk.score_methods on the test set: oracle, viper_<setting>
     (closed form) and viper_<setting>_mc, pca5_ols, pca15_ols, mofa (the Phase 2 MOFA
     fit, results/.../mofa/), ridge, optionally elastic_net: exact risk, test outcome
     MSE, kappa2_hat, g_hat;
  3. factor recovery (factor_recovery below) for the factor methods (VIPER settings,
     PCA5, PCA15, MOFA) and support recovery of the predictor for every method.

Factor recovery, on the raw feature scale (all original features; noise variance is 1
for every feature, so raw = whitened up to the standardisation):
    congr_rel           max over fitted columns of |Tucker congruence| with the true
                        relevant loading column (1 = one fitted factor is exactly it)
    congr_nuis_mean     the same, averaged over the 4 nuisance factors
    kappa2_rel / kappa2_nuis_mean  squared cosine of each true column with the span of
                        the fitted loadings (risk_metrics.kappa2_hat)
    out_factor_share    share of the fitted outcome signal carried by the single fitted
                        factor with the largest contribution |coef_j| sd(score_j); 1 means
                        the prediction runs through one factor
    out_factor_congr    |congruence| of that factor's loadings with the true relevant column
    out_factor_auc      AUROC of that factor's |loadings| for the true support S
    out_factor_prec_s   share of S among that factor's s largest |loadings|
    out_factor_strength VIPER only: that factor's fitted strength sum_j lambda_j^2 on the raw
                        scale (noise variance 1, scores ~ N(0, 1)), to set against g_true; PCA /
                        MOFA loadings have an arbitrary scale. (Counting "active" columns by
                        strength does not work: spurious columns fitting small loadings on
                        thousands of noise features reach strength 10-60, above a weak true
                        factor's; E_q[tau] per column is in Etau_view<l>.)
    coef_auc, coef_prec_s   the same support measures for the predictor coefficients b
                            (every method, incl. ridge / elastic net / oracle)

Output: results/.../validation/<label>/scores.csv (one row per dataset x method)
        and training.csv (time, epochs, failures per dataset x setting).

usage: OMP_NUM_THREADS=2 python validate_settings.py --cells 500,3000,1 --reps 1 2
           [--settings current default long] [--with-enet] [--workers 4] [--label smoke]
       python validate_settings.py --all-cells --reps 1 2 3 ...
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

RESULTS = os.path.expanduser("~/multiomic_integration/sim/results/integration_shared_controlled")
OUT = os.path.join(RESULTS, "validation")
N_ARRAY = (100, 200, 500)
P_ARRAY = (600, 3000)
C_ARRAY = (0.3, 0.6, 1, 1.5, 2, 3, 10)
K_DELTA = 10

SETTINGS = {
    "current": {"eta": 0.25, "t": 0.1, "minibatch_fraction": 0.15,
                "convergence_criterion": "elbo_snr", "window": 50, "snr_threshold": 5.0,
                "min_epochs": 200, "max_epochs": 500},
    "default": {"eta": 0.2113705944064572, "t": 0.0420167205437253, "minibatch_fraction": 0.16559149440361903,
                "convergence_criterion": "ma_patience", "ma_window": 50, "ma_patience": 20,
                "min_epochs": 200, "max_epochs": 1500},
    "long":    {"eta": 0.35033984911586863, "t": 0.024970737145052723,
                "minibatch_fraction": 0.05063981628665742,
                "convergence_criterion": "ma_patience", "ma_window": 114, "ma_patience": 157,
                "min_epochs": 0, "max_epochs": 1500},
    # long with a floor of 15 samples per minibatch, the smallest the current setting
    # uses (n = 100); the fraction alone gives 5 at n = 100 and 10 at n = 200, where
    # "long" diverged. Identical to "long" at n = 500 (25 samples).
    "long_mb15": {"eta": 0.35033984911586863, "t": 0.024970737145052723,
                  "minibatch_fraction": 0.05063981628665742, "min_minibatch_size": 15,
                  "convergence_criterion": "ma_patience", "ma_window": 114, "ma_patience": 157,
                  "min_epochs": 0, "max_epochs": 1500},
    # long with eta lowered to 0.25, the current setting's value (no failures in the
    # validation grid); no minibatch floor. long diverged within the first seconds at
    # P = 3000 (Gamma concentration leaving its domain), with and without the floor.
    "long_eta25": {"eta": 0.25, "t": 0.024970737145052723,
                   "minibatch_fraction": 0.05063981628665742,
                   "convergence_criterion": "ma_patience", "ma_window": 114, "ma_patience": 157,
                   "min_epochs": 0, "max_epochs": 1500},
}


def cond_name(n, P, c):
    return f"n{n}_P{P}_c{c:g}"


def train_setting(tr, te, setting, path, seed):
    """Train one setting (or reuse the saved fit); returns time, epochs, status."""
    import pyro
    import multiview_shared  # noqa: F401  (import before seeding)
    from run_methods import process_data, train_model

    if os.path.exists(path):
        return {"status": "reused"}
    train_subset, _, _ = process_data(tr, te, model_type="shared")
    cfg = {**SETTINGS[setting], "delta": 1e-16, "num_particles": 1}
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    t0 = time.time()
    try:
        model, _, _ = train_model({"k": int(tr["K"]) + K_DELTA, "dense_model": False}, cfg, train_subset,
                                  None, model_type="shared", opt="adagrad", write=False, svd_init=True)
    except Exception as e:                       # e.g. a diverging optimiser step
        return {"status": f"failed: {repr(e)[:100]}", "train_time": time.time() - t0}
    out = {"status": "ok", "train_time": time.time() - t0, "epochs": model.total_epochs}
    torch.save(pyro.get_param_store().get_state(), path)
    pyro.clear_param_store()
    return out


def _congr(A, b):
    """|Tucker congruence| of each column of A with the vector b."""
    na, nb = A.norm(dim=0), b.norm()
    return (A.T @ b).abs() / (na * nb).clamp(min=1e-300)


def _auc(score, truth):
    """AUROC of score for the boolean truth (ties counted half)."""
    from scipy.stats import rankdata
    pos = int(truth.sum()); neg = truth.size - pos
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[truth].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _prec_s(score, truth):
    s = int(truth.sum())
    return float(truth[np.argsort(-score)[:s]].mean()) if s else float("nan")


def factor_recovery(load_s, score_coef, scores_tr, coef_s, dp, theta, count_factors=False):
    """Factor- and support-recovery measures (see module docstring).
    load_s: standardised-scale loadings (list per view, kept columns x K_fit) or None;
    score_coef: outcome coefficient per fitted factor; scores_tr: training scores (n x K_fit);
    coef_s: predictor coefficients on the standardised scale (list per view)."""
    import risk_metrics as rm
    k_star = [k for k in range(theta["beta"].numel()) if float(theta["beta"][k]) != 0][0]
    Lam = torch.cat([Lm.double() for Lm in theta["Lambda"]], 0)
    truth = (Lam[:, k_star] != 0).numpy()
    # standardised-scale coefficients -> raw scale (b_s / sd) over all features, dropped ones 0
    b_raw = np.concatenate([np.zeros(f.numel()) for f in dp["X_l_list_column_filters"]])
    off = 0
    for b, f, sd in zip(coef_s, dp["X_l_list_column_filters"], dp["X_l_sd_list"]):
        idx = off + f.nonzero().reshape(-1).numpy()
        b_raw[idx] = np.asarray(b, float).reshape(-1) / sd.double().reshape(-1).numpy()
        off += f.numel()
    out = {"coef_auc": _auc(np.abs(b_raw), truth), "coef_prec_s": _prec_s(np.abs(b_raw), truth)}
    if load_s is None:
        return out
    F = torch.cat(rm.raw_loadings(load_s, dp), 0)
    k2 = rm.kappa2_hat(rm.raw_loadings(load_s, dp), theta)
    nuis = [k for k in range(Lam.shape[1]) if k != k_star]
    contrib = np.abs(np.asarray(score_coef, float).reshape(-1)) * np.asarray(scores_tr).std(0)
    j = int(np.argmax(contrib))
    norms = F.pow(2).sum(0)
    out.update({
        "congr_rel": float(_congr(F, Lam[:, k_star]).max()),
        "congr_nuis_mean": float(np.mean([float(_congr(F, Lam[:, k]).max()) for k in nuis])),
        "kappa2_rel": float(k2["kappa2_k"][k_star]),
        "kappa2_nuis_mean": float(k2["kappa2_k"][nuis].mean()),
        "out_factor_share": float(contrib[j] ** 2 / max((contrib ** 2).sum(), 1e-300)),
        "out_factor_congr": float(_congr(F[:, [j]], Lam[:, k_star])[0]),
        "out_factor_auc": _auc(F[:, j].abs().numpy(), truth),
        "out_factor_prec_s": _prec_s(F[:, j].abs().numpy(), truth),
    })
    if count_factors:
        out["out_factor_strength"] = float(norms[j])
    return out


def recovery_rows(tr, te, fits, mofa_npz, rows):
    """Add factor_recovery columns to the score_methods rows of one dataset."""
    from sklearn.decomposition import PCA
    from run_methods import process_data
    from viper_predict import closed_form_scores, load_guide_moments
    import risk_metrics as rm

    _, _, dp = process_data(tr, te, model_type="shared")
    L = dp["L"]
    theta = rm.true_params(tr)
    Xtr_s = dp["train_X_l_list_clean"]
    X_tr = torch.cat(Xtr_s, 1).double().numpy()
    y_tr = dp["train_y_clean"].double().reshape(-1).numpy()
    sizes = [x.shape[1] for x in Xtr_s]
    split = lambda A: list(torch.split(torch.as_tensor(np.asarray(A), dtype=torch.float64), sizes))
    by_method = {r["method"]: r for r in rows}

    for label, path in fits.items():
        if label not in by_method:
            continue
        mom = load_guide_moments(path, L)
        cf = closed_form_scores(mom, Xtr_s)
        by_method[label].update(factor_recovery(mom["M"], mom["m_beta"].numpy(), cf["m_z"].numpy(),
                                                closed_form_scores(mom, dp["test_X_l_list_clean"])["coef"],
                                                dp, theta, count_factors=True))
    for kc in (5, 15):
        pca = PCA(n_components=kc, random_state=0).fit(X_tr)
        T = pca.transform(X_tr)
        coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(T)), T]), y_tr, rcond=None)
        V = pca.components_.T
        by_method[f"pca{kc}_ols"].update(factor_recovery(split(V), coef[1:], T, split(V @ coef[1:]), dp, theta))
    if mofa_npz is not None and os.path.exists(mofa_npz) and "mofa" in by_method:
        d = np.load(mofa_npz)
        W = [d[f"weights_view{l}"] for l in range(L)]
        Wc = np.vstack(W)
        coef, *_ = np.linalg.lstsq(d["Z_train"], y_tr, rcond=None)
        by_method["mofa"].update(factor_recovery(W, coef, d["Z_train"], split(Wc @ np.linalg.solve(Wc.T @ Wc, coef)),
                                                 dp, theta))
    # coefficient-only methods: support recovery of b from their exact-risk coefficients is
    # not returned by score_methods, so ridge / elastic net / oracle are refitted cheaply here
    from sklearn.linear_model import ElasticNetCV, RidgeCV
    from misaligned_baselines import RIDGE_ALPHAS, ENET_L1_RATIOS
    if "ridge" in by_method:
        b = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_tr, y_tr).coef_
        by_method["ridge"].update(factor_recovery(None, None, None, split(b), dp, theta))
    if "elastic_net" in by_method:
        b = ElasticNetCV(l1_ratio=ENET_L1_RATIOS, n_alphas=50, cv=5, max_iter=5000, n_jobs=1).fit(X_tr, y_tr).coef_
        by_method["elastic_net"].update(factor_recovery(None, None, None, split(b), dp, theta))
    return rows


def run_dataset(n, P, c, rep, settings, with_enet, out_dir):
    from run_methods import load_rds
    from misaligned_exact_risk import score_methods
    from controlled_scoring import DATA_ROOT

    cond = cond_name(n, P, c)
    fname = f"sim_data_controlled_{cond}_rep{rep}.rds"
    tr = load_rds(f"{DATA_ROOT}/train/{cond}/{fname}")
    te = load_rds(f"{DATA_ROOT}/test/{cond}/{fname}")
    mdir = os.path.join(OUT, "models")
    os.makedirs(mdir, exist_ok=True)
    key = {"n": n, "P_grid": P, "c": c, "rep": rep, "g_target": float(tr["g_target"].item()),
           "g_true": float(tr["g_true"].item()), "s": int(tr["s"].item()),
           "g_over_sqrt_gamma": float(tr["g_true"].item()) / np.sqrt(P / n)}
    fits, train_rows = {}, []
    for s in settings:
        path = os.path.join(mdir, f"{s}_{cond}_rep{rep}_paramstore.pth")
        info = train_setting(tr, te, s, path, seed=rep)
        train_rows.append({**key, "setting": s, **info})
        fits[f"viper_{s}"] = path
    mofa_npz = os.path.join(RESULTS, "mofa", f"mofa_ctrl_{cond}_rep{rep}.npz")
    rows = score_methods(tr, te, key, fits, mofa_npz, with_enet=with_enet, seed=rep)
    rows = recovery_rows(tr, te, fits, mofa_npz, rows)
    tr_info = {r["setting"]: r for r in train_rows}
    for r in rows:
        s = r["method"].replace("viper_", "").replace("_mc", "")
        if r["method"].startswith("viper_") and s in tr_info:
            r.update(train_time=tr_info[s].get("train_time"), epochs=tr_info[s].get("epochs"))
    return rows, train_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=None, help="n,P,c ...")
    ap.add_argument("--all-cells", action="store_true")
    ap.add_argument("--reps", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS))
    ap.add_argument("--with-enet", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--label", default="smoke")
    args = ap.parse_args()
    cells = ([(n, P, c) for n in N_ARRAY for P in P_ARRAY for c in C_ARRAY] if args.all_cells else
             [(int(n), int(P), float(c)) for n, P, c in (x.split(",") for x in args.cells)])
    out_dir = os.path.join(OUT, args.label)
    os.makedirs(out_dir, exist_ok=True)
    score_csv, train_csv = os.path.join(out_dir, "scores.csv"), os.path.join(out_dir, "training.csv")
    done = set()
    if os.path.exists(score_csv):
        prev = pd.read_csv(score_csv)
        done = set(zip(prev.n, prev.P_grid, prev.c, prev.rep))
    # largest datasets first, so the slowest jobs do not finish last
    jobs = sorted([(n, P, c, r) for n, P, c in cells for r in args.reps if (n, P, c, r) not in done],
                  key=lambda j: -j[0] * j[1])
    print(f"{len(jobs)} datasets", flush=True)
    from joblib import Parallel, delayed
    t0 = time.time()
    batch = 2 * args.workers
    for i in range(0, len(jobs), batch):
        res = Parallel(n_jobs=args.workers, backend="loky")(
            delayed(run_dataset)(*j, args.settings, args.with_enet, out_dir) for j in jobs[i:i + batch])
        for rows, trows in res:
            for f, d in ((score_csv, rows), (train_csv, trows)):
                pd.DataFrame(d).to_csv(f, mode="a", header=not os.path.exists(f), index=False)
        print(f"{min(i + batch, len(jobs))}/{len(jobs)} datasets, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
