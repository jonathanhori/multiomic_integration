"""
Multi-factor controlled-strength study, scenario B (docs/viper_shared_extensions_design.md §1):
a strong (g_1 = 10 sqrt(gamma)) and a weak (g_2 = c sqrt(gamma)) outcome-relevant factor,
the weak factor carrying a share omega_2 of the outcome signal.
Data: R/generate_multifactor_data.R -> sim/data/multi_view_shared_multifactor/B/.

VIPER-Shared is fitted exactly as in controlled_grid_default.py: package defaults from
run_methods.with_defaults (train_model with an empty train_config), MGP prior, k = 15,
pyro seed = rep. Nothing here sets an optimiser or model setting of its own.

Per dataset:
  1. fit VIPER (param store saved to results/.../models/); stop epoch, fit time, number
     of factors with > 1 % of training variance, implied-covariance error;
  2. MOFA2 (controlled_scoring.fit_mofa, 15 factors; saved to results/.../mofa/);
  3. misaligned_exact_risk.score_methods: oracle, viper_default (+ _mc), pca5_ols,
     pca15_ols, mofa, ridge, elastic_net (test outcome MSE, exact risk [diagnostic],
     kappa2, g_hat, E_q[tau]);
  4. viper_scores_ols: OLS (with intercept) of y on VIPER's closed-form training scores,
     applied to the closed-form test scores. Same scores as viper_default, own coefficients,
     so viper_default vs viper_scores_ols separates the beta step from direction recovery;
  5. per relevant factor: kappa2 of the strong and of the weak factor direction for every
     factor method (kappa2_strong, kappa2_weak; kappa2_rel_mean = subspace kappa2, the
     relevant directions being orthogonal after whitening), realised g_true, omega;
  6. outcome UQ on the test set (viper_default closed-form Gaussian predictive, MOFA OLS
     prediction interval) and implied-covariance error for the factor methods.

No-regression check (--baseline): the same pipeline on the existing single-factor
controlled datasets (reps 1-3, the c values of scenario B), compared cell by cell with
grid_default/scores.csv (controlled_grid_default.py). With unchanged defaults the VIPER
rows must match exactly; MOFA reuses the saved fits there.

Output (one CSV per cell, so cluster array tasks never write the same file):
  sim/results/integration_shared_multifactor/B/scores/<cond>.csv
  sim/results/integration_shared_multifactor/baseline/scores/<cond>.csv
Resumable: reps already in a cell's CSV are skipped.

usage (from python/):
  OMP_NUM_THREADS=2 python sim_run_shared_multifactor.py --list                 # cells / task ids
  OMP_NUM_THREADS=2 python sim_run_shared_multifactor.py --task-id 7 --generate # one cell (SGE)
  OMP_NUM_THREADS=2 python sim_run_shared_multifactor.py --cells all --workers 4
  OMP_NUM_THREADS=2 python sim_run_shared_multifactor.py --baseline --cells all --workers 4
  python sim_run_shared_multifactor.py --baseline-compare
"""
import argparse, glob, os, sys, time, warnings
import numpy as np, pandas as pd
sys.path.insert(1, os.path.expanduser("~/multiomic_integration/python"))
warnings.filterwarnings("ignore")

ROOT = os.path.expanduser("~/multiomic_integration")
DATA_B = os.path.join(ROOT, "sim", "data", "multi_view_shared_multifactor", "B")
OUT_ROOT = os.path.join(ROOT, "sim", "results", "integration_shared_multifactor")
CTRL_RESULTS = os.path.join(ROOT, "sim", "results", "integration_shared_controlled")

N_ARRAY = (100, 200, 500)
P_ARRAY = (600, 3000)
C_ARRAY = (0.6, 1, 2, 3, 10)
W_ARRAY = (0.25, 0.5, 0.75)
K_FIT = 15
REPS_B = tuple(range(1, 11))
REPS_BASELINE = (1, 2, 3)


def cond_b(n, P, c, w):
    return f"n{n}_P{P}_c{c:g}_w{w:g}"


def cells(baseline):
    if baseline:
        return [(n, P, c, None) for n in N_ARRAY for P in P_ARRAY for c in C_ARRAY]
    return [(n, P, c, w) for n in N_ARRAY for P in P_ARRAY for w in W_ARRAY for c in C_ARRAY]


def load(n, P, c, w, rep):
    from run_methods import load_rds
    if w is None:
        from controlled_viper_diagnosis import load_dataset
        return load_dataset(n, P, c, rep)
    cond = cond_b(n, P, c, w)
    f = f"sim_data_mfB_{cond}_rep{rep}.rds"
    return load_rds(os.path.join(DATA_B, "train", cond, f)), load_rds(os.path.join(DATA_B, "test", cond, f))


def uq(y, mu, var, df=None):
    from scipy import stats
    sd = np.sqrt(var); q = stats.t.ppf(0.975, df) if df else stats.norm.ppf(0.975)
    lo, hi = mu - q * sd, mu + q * sd
    return {"cov95": float(np.mean((y >= lo) & (y <= hi))), "width95": float(np.mean(hi - lo)),
            "iscore95": float(np.mean((hi - lo) + 40 * (lo - y) * (y < lo) + 40 * (y - hi) * (y > hi)))}


def one(n, P, c, w, rep):
    """All rows for one dataset (w None: single-factor baseline dataset)."""
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import torch, pyro
    torch.set_num_threads(2)
    import multiview_shared  # noqa: F401
    from run_methods import process_data, train_model
    from misaligned_exact_risk import score_methods
    from data_utils import implied_cov_errors
    from viper_predict import load_guide_moments, closed_form_scores
    from controlled_scoring import fit_mofa, cond_name
    from sklearn.decomposition import PCA
    from constants import Params
    import risk_metrics as rm

    baseline = w is None
    cond = cond_name(n, P, c) if baseline else cond_b(n, P, c, w)
    out_dir = os.path.join(OUT_ROOT, "baseline" if baseline else "B")
    models = os.path.join(out_dir, "models"); os.makedirs(models, exist_ok=True)
    tr, te = load(n, P, c, w, rep)
    sub, _, dp = process_data(tr, te, model_type="shared")
    L = dp["L"]

    # 1. VIPER, package defaults (as controlled_grid_default.one)
    stem = os.path.join(models, f"shared_default_{cond}_rep{rep}")
    pyro.clear_param_store(); pyro.set_rng_seed(rep)
    t0 = time.time()
    model, handler, _ = train_model({"k": K_FIT, "dense_model": False}, {}, sub, None,
                                    model_type="shared", write=False)
    fit_time = time.time() - t0
    store = pyro.get_param_store()
    store.save(stem + "_paramstore.pth")
    hist = handler.avg_tracker.history
    Zm = store[Params.loc_Z].detach().double(); Sz = store[Params.scale_Z].detach().double()
    Lam = [store[Params.loc_Lambda_l.format(l=l)].detach().double().squeeze() for l in range(L)]
    Xtr = [x.double() for x in dp["train_X_l_list_clean"]]
    pve = sum((Zm ** 2).mean(0) * (M ** 2).sum(0) for M in Lam) * Zm.shape[0] / sum(float((x ** 2).sum()) for x in Xtr)
    Sigma_Z = (Zm.T @ Zm + torch.diag((Sz ** 2).sum(0))) / Zm.shape[0]
    extra_viper = {"stop_epoch": model.total_epochs, "fit_time": fit_time,
                   "stat_Lambda_final": hist[-1][1]["Lambda"] if hist else np.nan,
                   "stat_beta_final": hist[-1][1]["beta"] if hist else np.nan,
                   "n_factors_pve1": int((pve > 0.01).sum()),
                   "cov_frob": float(implied_cov_errors(Lam, Sigma_Z, dp).cov_frob.mean())}
    pyro.clear_param_store()

    # 2. MOFA (baseline: the saved single-factor fits)
    if baseline:
        mofa_npz = os.path.join(CTRL_RESULTS, "mofa", f"mofa_ctrl_{cond}_rep{rep}.npz")
    else:
        os.makedirs(os.path.join(out_dir, "mofa"), exist_ok=True)
        mofa_npz = os.path.join(out_dir, "mofa", f"mofa_{cond}_rep{rep}.npz")
    if not os.path.exists(mofa_npz):
        fit_mofa(dp, mofa_npz)

    g_true = np.atleast_1d(tr["g_true"].double().numpy())
    key = {"n": n, "P_grid": P, "c": c, "rep": rep,
           "omega_weak": np.nan if baseline else w,
           "g_target": float(np.atleast_1d(tr["g_target"].double().numpy())[-1]),
           "g_true": float(g_true[-1]),                       # weak factor (the only one in baseline)
           "g_true_strong": np.nan if baseline else float(g_true[0]),
           "s": int(np.atleast_1d(tr["s"].numpy())[-1])}

    # 3. Shared scoring of all linear methods
    rows = score_methods(tr, te, key, {"viper_default": stem + "_paramstore.pth"},
                         mofa_npz, with_enet=True, seed=rep)

    theta = rm.true_params(tr); orc = rm.oracle(theta)
    rel = [k for k in range(theta["beta"].numel()) if float(theta["beta"][k]) != 0]
    k_strong, k_weak = (rel[0], rel[-1])            # baseline: one relevant factor, both the same
    y_raw = tr["y"].double().reshape(-1); y_mean, y_sd = float(y_raw.mean()), float(y_raw.std())
    y_tr = dp["train_y_clean"].double().reshape(-1).numpy(); y_te = dp["test_y_clean"].double().reshape(-1).numpy()
    X_te = [x.double() for x in dp["test_X_l_list_clean"]]

    # 4. VIPER scores + OLS
    mom = load_guide_moments(stem + "_paramstore.pth", L)
    cf_tr = closed_form_scores(mom, Xtr); cf_te = closed_form_scores(mom, X_te)
    t0 = time.time()
    D = np.column_stack([np.ones(n), cf_tr["m_z"].numpy()])
    b, *_ = np.linalg.lstsq(D, y_tr, rcond=None)
    wts = [a_ / b_ for a_, b_ in zip(mom["a_psi"], mom["b_psi"])]
    A_inv = torch.linalg.inv(cf_tr["A_bar"])
    coef_x = [(M_l * w_l[:, None]) @ (A_inv @ torch.as_tensor(b[1:], dtype=torch.float64))
              for M_l, w_l in zip(mom["M"], wts)]
    pred = b[0] + cf_te["m_z"].numpy() @ b[1:]
    base = {k: v for k, v in rows[0].items() if k in key or k in ("P", "R_star", "var_y", "g_y", "gamma", "sqrt_gamma")}
    rows.append({**base, "method": "viper_scores_ols",
                 **rm.risk_summary(coef_x, dp, y_mean, y_sd, theta, orc, float(b[0])),
                 "outcome_mse": float(np.mean((pred - y_te) ** 2)), "runtime": time.time() - t0})

    # 5. Per-factor kappa2 for the factor methods
    Xc = torch.cat(Xtr, 1).numpy(); sizes = [x.shape[1] for x in Xtr]
    split = lambda A: list(torch.split(torch.as_tensor(A, dtype=torch.float64), sizes))
    d = np.load(mofa_npz); W_mofa = [d[f"weights_view{l}"] for l in range(L)]
    loads = {"viper_default": mom["M"], "viper_default_mc": mom["M"], "viper_scores_ols": mom["M"],
             "mofa": [torch.as_tensor(x) for x in W_mofa]}
    pcas = {}
    for kc in (5, 15):
        pcas[kc] = PCA(n_components=kc, random_state=0).fit(Xc)
        loads[f"pca{kc}_ols"] = split(pcas[kc].components_.T)
    lam_s = [Lm[f] / sd.double().reshape(-1)[:, None] for Lm, f, sd in
             zip(theta["Lambda"], dp["X_l_list_column_filters"], dp["X_l_sd_list"])]
    loads["oracle"] = lam_s
    k2 = {m: rm.kappa2_hat(rm.raw_loadings(ld, dp), theta)["kappa2_k"] for m, ld in loads.items()}

    # 6. UQ and covariance error
    u_v = uq(y_te, cf_te["yhat"].numpy(), cf_te["pred_var"].numpy())
    Z, Zt = d["Z_train"], d["Z_test"]
    cm, *_ = np.linalg.lstsq(Z, y_tr, rcond=None); kz = Z.shape[1]
    s2 = float(((y_tr - Z @ cm) ** 2).sum() / (n - kz)); Gm = np.linalg.inv(Z.T @ Z)
    u_m = uq(y_te, Zt @ cm, s2 * (1 + np.einsum("ij,jk,ik->i", Zt, Gm, Zt)), n - kz)
    cov = {"mofa": float(implied_cov_errors([torch.as_tensor(x) for x in W_mofa], torch.as_tensor(Z.T @ Z / n), dp).cov_frob.mean())}
    for kc, pca in pcas.items():
        s2n = (Xc.var(0, ddof=1).sum() - pca.explained_variance_.sum()) / (Xc.shape[1] - kc)
        Wp = pca.components_.T * np.sqrt(np.clip(pca.explained_variance_ - s2n, 0, None))
        cov[f"pca{kc}_ols"] = float(implied_cov_errors(split(Wp), None, dp).cov_frob.mean())

    for r in rows:
        m = r["method"]
        if m in k2:
            r.update(kappa2_strong=float(k2[m][k_strong]), kappa2_weak=float(k2[m][k_weak]),
                     kappa2_rel_mean=float(k2[m][rel].mean()))
        if m == "viper_default":
            r.update(extra_viper); r.update(u_v); r["runtime"] = fit_time
        elif m == "mofa":
            r.update(u_m); r["cov_frob"] = cov["mofa"]; r["runtime"] = float(d["runtime"])
        elif m in cov:
            r["cov_frob"] = cov[m]
    return rows


def generate_cell(n, P, c, w, reps):
    """Generate the cell's scenario-B datasets with R/generate_multifactor_data.R (skips existing)."""
    import subprocess
    subprocess.run(["Rscript", os.path.join(ROOT, "R", "generate_multifactor_data.R"),
                    "--only", f"{n},{P},{c:g},{w:g}", "--reps", ",".join(map(str, reps))],
                   check=True, stdout=subprocess.DEVNULL)


def run_cell(n, P, c, w, reps, generate=False):
    from controlled_scoring import cond_name
    baseline = w is None
    cond = cond_name(n, P, c) if baseline else cond_b(n, P, c, w)
    if generate and not baseline:
        generate_cell(n, P, c, w, reps)
    sdir = os.path.join(OUT_ROOT, "baseline" if baseline else "B", "scores"); os.makedirs(sdir, exist_ok=True)
    out = os.path.join(sdir, f"{cond}.csv")
    done = set(pd.read_csv(out).rep) if os.path.exists(out) else set()
    for rep in reps:
        if rep in done:
            continue
        t0 = time.time()
        new = pd.DataFrame(one(n, P, c, w, rep))
        if os.path.exists(out):
            new = pd.concat([pd.read_csv(out), new], ignore_index=True)
        new.to_csv(out, index=False)
        print(f"{cond} rep {rep}: {time.time() - t0:.0f}s", flush=True)


def baseline_compare():
    """Per (n, P, c) cell: new baseline run vs grid_default/scores.csv, reps 1-3."""
    new = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT_ROOT, "baseline", "scores", "*.csv"))])
    old = pd.read_csv(os.path.join(CTRL_RESULTS, "grid_default", "scores.csv"))
    keys = ["n", "P_grid", "c", "rep", "method"]
    m = new.merge(old, on=keys, suffixes=("_new", "_old"))
    m = m[m.method.isin(["viper_default", "pca5_ols", "pca15_ols", "mofa", "ridge", "elastic_net", "oracle"])]
    m["d_mse"] = m.outcome_mse_new - m.outcome_mse_old
    m["d_risk_rel"] = m.risk_rel_new - m.risk_rel_old
    m["exact"] = m.d_mse.abs() < 1e-12
    print(f"{len(m)} matched rows ({m[['n', 'P_grid', 'c', 'rep']].drop_duplicates().shape[0]} datasets)")
    print("\nrows identical (outcome MSE to 1e-12), by method:")
    print(m.groupby("method").exact.agg(["sum", "count"]).to_string())
    v = m[m.method == "viper_default"]
    print("\nviper_default per cell (mean over reps; old vs new outcome MSE, stop epoch):")
    t = v.groupby(["n", "P_grid", "c"]).agg(mse_old=("outcome_mse_old", "mean"), mse_new=("outcome_mse_new", "mean"),
                                            max_abs_d=("d_mse", lambda x: x.abs().max()),
                                            stop_old=("stop_epoch_old", "mean"), stop_new=("stop_epoch_new", "mean"))
    print(t.round(4).to_string())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="single-factor controlled datasets, reps 1-3")
    ap.add_argument("--baseline-compare", action="store_true")
    ap.add_argument("--task-id", type=int, help="1-based cell index (SGE_TASK_ID)")
    ap.add_argument("--cells", default=None, help="'all' to run every cell")
    ap.add_argument("--reps", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--generate", action="store_true", help="generate missing scenario-B data first (R)")
    a = ap.parse_args()
    if a.baseline_compare:
        baseline_compare(); return
    cl = cells(a.baseline)
    reps = a.reps or (REPS_BASELINE if a.baseline else REPS_B)
    if a.list:
        for i, x in enumerate(cl, 1):
            print(i, x)
        return
    if a.task_id is not None:
        run_cell(*cl[a.task_id - 1], reps, a.generate); return
    if a.cells != "all":
        ap.error("give --task-id N or --cells all")
    cl = sorted(cl, key=lambda x: -x[0] * x[1])          # largest first
    from joblib import Parallel, delayed
    Parallel(n_jobs=a.workers, backend="loky")(delayed(run_cell)(*x, reps, a.generate) for x in cl)
    print("ALL_DONE", time.ctime(), flush=True)


if __name__ == "__main__":
    main()
