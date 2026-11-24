"""
Is one fit enough? Compare the factors of a single VIPER-Shared fit (package defaults,
one 80/20 split, as a one-off real-data analysis would be run) with the consensus
factors of the 5-fold CV fits (factor_annotation.py, --method shared --tag _default).

Stage "fit": VIPER-Shared with the package defaults on the seed-42 split of
compare_methods.py (the split of tcga_coadread.qmd), K_max = 20, 50 and 100, test
scores by local inference at the package defaults. Outputs results/single_fit/seed42/
k<K>/shared_factors.npz (compare_methods.save_shared layout), genes_view<l>.txt,
train_samples.txt, test_samples.txt, metrics.csv.

Stage "compare": every single fit is processed exactly as a CV fold is for the
consensus (factor_annotation.load_fold: varimax rotation of both views jointly, scores
rescaled to SD 1, factors with PVE >= 0.5% kept). For each consensus factor CF:
  - best |r| of its loadings with any single-fit factor (genes both selected), and the
    one-to-one (Hungarian) match; "recovered" when the matched |r| >= 0.8, the cut
    that defines the consensus clusters;
  - reference: for each CV fold that has the CF, |r| of that fold's factor with the
    consensus of the OTHER folds (leave-one-fold-out), i.e. how well one fit agrees with
    a consensus it did not contribute to; and the CF's mean |r| between folds;
  - beta per SD of the matched single-fit factor vs the CF's mean.
Single-fit factors that match no CF (|r| < 0.8) are listed with PVE and beta.

NOTE: the seed-42 split's training samples are exactly CV fold 4's, so the single fit
is fold 4 refitted with another random seed. Fold 4 is therefore left out of every
comparison of the single fit with the CV fits, and the single fit vs fold 4 measures
the effect of the random seed alone.

Stage "reproduce" (no reference to the K_max = 20 consensus): for every single-fit
factor (PVE >= 0.5%), |r| with the best factor of the fold-4 fit (same data, other
random seed) and the number of folds 0-3 at the SAME K_max that contain a factor with
|r| >= 0.8 ("reproducible": >= 3 of 4), together with
quantities available from the single fit alone: PVE, |beta| per SD, and the loading
signal-to-noise ratio ||w_j||^2 / sum of posterior variances of w_j (rotated with the
varimax rotation, mean-field). The question: can the single fit tell which of its
factors would reproduce?

Stage "annotate" (K_max = ANNOT_K): every single-fit factor with PVE >= 0.5% annotated
like the consensus factors (factor_annotation.associations on the training and test
scores of all 530 samples, factor_annotation.gsea on the loadings; sign set so the
largest expression loadings are positive), tagged with reproducibility (stage
"reproduce") and its closest consensus factor; and the factors' role in prediction:
test MSE of the plug-in prediction Z beta (log(1 + TMB) scale) using all columns,
dropping the non-reproducible factors, dropping each factor in turn, and using only
the reproducible factors. Outputs results/factors/associations_single42_k<K>.csv,
gsea_single42_k<K>.csv, results/single_fit/seed42/factors_k<K>.csv,
prediction_k<K>.csv.

Stage "cov" (K_max = ANNOT_K): covariance reconstruction when columns are dropped.
Model-implied covariance of the standardised views Sigma = W C W^T + diag(E[psi]),
with W the rotated loadings (scaled to SD-1 scores), C the covariance of the rotated
training scores (restricted to the kept columns) and E[psi] the posterior mean noise
variances. Compared with the sample covariance of the 106 test samples (standardised
with training statistics; out of sample) and of the 424 training samples, as the
relative Frobenius error ||Sigma - S||_F / ||S||_F, over the whole matrix and over the
off-diagonal only (gene-gene covariances). Column sets: all, drop non-reproducible,
reproducible only, drop PVE < 0.5%, noise only, and keep PVE >= t for a grid of t.
Output results/single_fit/seed42/cov_k<K>.csv.

Stage "rescore": test MSE / coverage of each single fit on the log(1 + TMB) scale
(re-runs local inference from the saved state; the saved factors are not changed).

Outputs: results/single_fit/seed42/cf_match_k<K>.csv, unmatched_k<K>.csv,
reference_loo.csv, summary.csv, reproduce.csv, metrics.csv per fit
usage: OMP_NUM_THREADS=2 python single_fit_vs_consensus.py --stage fit --k 20
       python single_fit_vs_consensus.py --stage compare
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, HERE)
import factor_annotation as fa  # noqa: E402  (also puts python/ on the path)

SEED = 42
SAME_DATA_FOLD = 4
ANNOT_K = 50          # CV fold whose training samples equal the seed-42 split's
OUT = os.path.join(HERE, "results", "single_fit", f"seed{SEED}")
TAG = "_default"


def fit(k):
    import warnings
    warnings.filterwarnings("ignore")
    import torch, pyro
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    import compare_methods as cm
    from k_sweep import n_active
    from run_methods import train_model
    d = os.path.join(OUT, f"k{k}")
    os.makedirs(d, exist_ok=True)
    data, train_subset, test_subset, dp = cm.load_split(SEED)
    for name, ids in (("train_samples.txt", data["sample_ids_train"]), ("test_samples.txt", data["sample_ids_test"])):
        open(os.path.join(d, name), "w").write("\n".join(ids))
    for l, v in enumerate(cm.VIEWS):
        key = {"expr": "gene_names_expr", "meth_tss200": "gene_names_tss200"}[v]
        open(os.path.join(d, f"genes_view{l}.txt"), "w").write("\n".join(data[key]))
    pyro.clear_param_store()
    pyro.set_rng_seed(SEED)
    t0 = time.time()
    fm, _, _ = train_model({"k": k, "dense_model": False, "outcome_weight": 1.0}, {}, train_subset,
                           model_type="shared", write=False)
    secs = time.time() - t0
    import copy
    state = copy.deepcopy(pyro.get_param_store().get_state())
    t1 = time.time()
    r = {**cm._score_shared(fm, test_subset, dp, local_config={}), "epoch": fm.total_epochs, "state": state}
    local_secs = time.time() - t1
    cm.save_shared(d, "shared", r)
    m, lo, hi = (np.asarray(x, float) for x in r["pred"])
    y_mean, y_sd = float(data["y"].mean()), float(data["y"].std())
    m, lo, hi = (v * y_sd + y_mean for v in (m, lo, hi))           # log(1 + TMB) scale
    y = np.asarray(data["y_test"]).reshape(-1)
    pd.DataFrame([{"k": k, "epochs": fm.total_epochs, "train_seconds": secs, "score_seconds": local_secs,
                   "mse": float(np.mean((m - y) ** 2)), "coverage_95": float(np.mean((y >= lo) & (y <= hi))),
                   "active_factors": n_active(r["Z_train"], r["Lambda"])}]).to_csv(os.path.join(d, "metrics.csv"), index=False)
    print(f"K={k}: {fm.total_epochs} epochs, train {secs:.0f}s, scoring {local_secs:.0f}s", flush=True)


def load_single(k):
    """A single fit processed exactly as factor_annotation.load_fold processes a CV fold,
    plus the rotated loadings' signal-to-noise ratio."""
    d = os.path.join(OUT, f"k{k}")
    z = np.load(os.path.join(d, "shared_factors.npz"))
    W = [z[f"loadings_view{l}"] for l in range(fa.L)]
    rot, R = fa._rotate_loadings(W)
    Ztr = z["Z_train"] @ R
    sd = Ztr.std(0)
    var = sum(((z[f"scale_view{l}"].reshape(W[l].shape) ** 2) @ (R ** 2)).sum(0) for l in range(fa.L)) * sd ** 2
    rot = [w * sd for w in rot]
    Ztr = Ztr / sd
    beta = (R.T @ z["beta"]) * sd
    genes = [fa.lines(os.path.join(d, f"genes_view{l}.txt")) for l in range(fa.L)]
    P = sum(len(g) for g in genes)
    pve = (Ztr ** 2).sum(0) * sum((w ** 2).sum(0) for w in rot) / (Ztr.shape[0] * P)
    snr = sum((w ** 2).sum(0) for w in rot) / var
    return {"W": rot, "beta": beta, "genes": genes, "pve": pve, "snr": snr}


def reproduce():
    rows = []
    for k in (20, 50, 100):
        s = load_single(k)
        keep = np.where(s["pve"] > fa.MIN_PVE)[0]
        folds = {f: fa.load_fold(f, k, "shared") for f in range(fa.N_FOLDS)}
        fs = {f: [fa.as_series(folds[f], j) for j in np.where(folds[f]["pve"] > fa.MIN_PVE / 2)[0]] for f in folds}
        for j in keep:
            x = fa.as_series(s, j)
            best = {f: max(abs(corr(x, y)) for y in fs[f]) for f in folds}
            other = [best[f] for f in folds if f != SAME_DATA_FOLD]
            rows.append({"K_max": k, "factor": f"F{j + 1}", "PVE": s["pve"][j], "|beta| per SD": abs(s["beta"][j]),
                         "loading SNR": s["snr"][j], "same data, other seed |r|": best[SAME_DATA_FOLD],
                         "folds 0-3 found (|r| >= 0.8)": int(sum(b >= fa.R_CUT for b in other)),
                         "median |r| folds 0-3": float(np.median(other))})
    R = pd.DataFrame(rows)
    R["reproducible"] = R["folds 0-3 found (|r| >= 0.8)"] >= 3
    R.to_csv(os.path.join(OUT, "reproduce.csv"), index=False)
    pd.set_option("display.width", 250)
    for k, g in R.groupby("K_max"):
        print(f"\n=== K_max = {k}: {g.reproducible.sum()}/{len(g)} single-fit factors reproducible")
        print(g.sort_values("PVE", ascending=False).drop(columns="K_max").round(3).to_string(index=False))


def rotated_single(k):
    """Rotated, SD-1 training and test scores, loadings and beta of the single fit
    (as load_single), with the sign convention of the consensus factors."""
    d = os.path.join(OUT, f"k{k}")
    z = np.load(os.path.join(d, "shared_factors.npz"))
    W = [z[f"loadings_view{l}"] for l in range(fa.L)]
    rot, R = fa._rotate_loadings(W)
    Ztr, Zte = z["Z_train"] @ R, z["Z_test"] @ R
    sd = Ztr.std(0)
    rot = [w * sd for w in rot]
    Ztr, Zte = Ztr / sd, Zte / sd
    beta = (R.T @ z["beta"]) * sd
    sgn = np.where(rot[0].max(0) >= -rot[0].min(0), 1.0, -1.0)
    return {"W": [w * sgn for w in rot], "Ztr": Ztr * sgn, "Zte": Zte * sgn, "beta": beta * sgn,
            "beta_raw": z["beta"], "Ztr_raw": z["Z_train"], "Zte_raw": z["Z_test"],
            "genes": [fa.lines(os.path.join(d, f"genes_view{l}.txt")) for l in range(fa.L)],
            "train": fa.lines(os.path.join(d, "train_samples.txt")), "test": fa.lines(os.path.join(d, "test_samples.txt"))}


def annotate(k=ANNOT_K, stages=("table", "assoc", "gsea")):
    import compare_methods as cm
    s = load_single(k)
    r = rotated_single(k)
    keep = np.where(s["pve"] > fa.MIN_PVE)[0]
    names = [f"F{j + 1}" for j in keep]
    rep_ = pd.read_csv(os.path.join(OUT, "reproduce.csv"))
    rep_ = rep_[rep_.K_max == k].set_index("factor")
    cons = pd.read_csv(f"{fa.OUT}/consensus_loadings{TAG}.csv", index_col=0)
    tab = []
    for j, n in zip(keep, names):
        x = fa.as_series(s, j)
        cc = {c: abs(corr(cons[c], x)) for c in cons}
        best = max(cc, key=cc.get)
        tab.append({"factor": n, "PVE": s["pve"][j], "beta per SD": r["beta"][j],
                    "reproducible": bool(rep_.loc[n, "reproducible"]),
                    "folds 0-3 found": int(rep_.loc[n, "folds 0-3 found (|r| >= 0.8)"]),
                    "same data, other seed |r|": rep_.loc[n, "same data, other seed |r|"],
                    "closest consensus factor": best, "|r| with it": cc[best]})
    T = pd.DataFrame(tab).sort_values("PVE", ascending=False)
    T.to_csv(os.path.join(OUT, f"factors_k{k}.csv"), index=False)

    # prediction: plug-in Z beta (rotation-invariant over all columns), log(1 + TMB) scale
    data = cm.load_coadread(top_rna_genes=cm.N_RNA, top_meth_genes=cm.N_METH, log1p_tmb=True,
                            train_frac=0.8, random_seed=SEED, preprocess_on_train_only=True)
    y_mean, y_sd = float(data["y"].mean()), float(data["y"].std())     # as cv_k_sweep.py
    yte, ytr = data["y_test"].reshape(-1).numpy(), data["y"].reshape(-1).numpy()
    nonrep = [j for j, n in zip(keep, names) if not T.set_index("factor").loc[n, "reproducible"]]
    repr_ = [j for j in keep if j not in nonrep]
    small = [j for j in range(r["Ztr"].shape[1]) if j not in keep]

    def mse(cols):
        cols = list(cols)
        p = r["Zte"][:, cols] @ r["beta"][cols] * y_sd + y_mean
        q = r["Ztr"][:, cols] @ r["beta"][cols] * y_sd + y_mean
        return float(np.mean((p - yte) ** 2)), float(1 - np.mean((q - ytr) ** 2) / np.var(ytr))
    allc = range(r["Ztr"].shape[1])
    rows = [{"predictor": "all columns", **dict(zip(("test MSE", "train R2"), mse(allc)))},
            {"predictor": "drop the non-reproducible factors", **dict(zip(("test MSE", "train R2"), mse([j for j in allc if j not in nonrep])))},
            {"predictor": "reproducible factors only", **dict(zip(("test MSE", "train R2"), mse(repr_)))},
            {"predictor": "drop columns with PVE < 0.5%", **dict(zip(("test MSE", "train R2"), mse(keep)))},
            {"predictor": "training mean", "test MSE": float(np.mean((ytr.mean() - yte) ** 2)), "train R2": 0.0}]
    for j, n in zip(keep, names):
        rows.append({"predictor": f"drop {n}", **dict(zip(("test MSE", "train R2"), mse([i for i in allc if i != j])))})
    Pdf = pd.DataFrame(rows)
    Pdf.to_csv(os.path.join(OUT, f"prediction_k{k}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False)); print(Pdf.round(4).to_string(index=False), flush=True)

    load = pd.DataFrame({n: pd.concat([pd.Series(r["W"][l][:, j], index=[f"{fa.VIEWS[l]}:{g}" for g in r["genes"][l]])
                                       for l in range(fa.L)]).loc[lambda x: ~x.index.duplicated()]
                         for j, n in zip(keep, names)})
    scores = pd.DataFrame(np.vstack([r["Ztr"][:, keep], r["Zte"][:, keep]]), index=r["train"] + r["test"], columns=names)
    tag = f"_single{SEED}_k{k}"
    load.to_csv(os.path.join(OUT, f"loadings_k{k}.csv")); scores.to_csv(os.path.join(OUT, f"scores_k{k}.csv"))
    if "assoc" in stages:
        fa.associations(scores, fa.sample_annotation(), tag=tag)
        print("associations done", flush=True)
    if "gsea" in stages:
        fa.gsea(load, tag=tag)
        print("gsea done", flush=True)


def cov_error(k=ANNOT_K):
    import torch
    import compare_methods as cm
    from constants import Params
    s = load_single(k)
    r = rotated_single(k)
    data, train_subset, test_subset, dp = cm.load_split(SEED)
    Xte = np.hstack([x.double().numpy() for x in dp["test_X_l_list_clean"]])
    Xtr = np.hstack([x.double().numpy() for x in dp["train_X_l_list_clean"]])
    W = np.vstack(r["W"])
    assert W.shape[0] == Xte.shape[1], (W.shape, Xte.shape)
    st = torch.load(os.path.join(OUT, f"k{k}", "shared_paramstore.pt"), weights_only=False)["params"]
    psi = np.concatenate([(st[Params.b_psi_l.format(l=l)] / (st[Params.a_psi_l.format(l=l)] - 1)).detach().double().numpy().reshape(-1)
                          for l in range(fa.L)])
    C = np.cov(r["Ztr"], rowvar=False)
    S_te, S_tr = np.cov(Xte, rowvar=False), np.cov(Xtr, rowvar=False)
    off = ~np.eye(W.shape[0], dtype=bool)
    trust = pd.read_csv(os.path.join(OUT, f"factors_k{k}.csv"))
    name = {f"F{j + 1}": j for j in range(W.shape[1])}
    nonrep = [name[f] for f in trust.loc[~trust.reproducible, "factor"]]
    repr_ = [name[f] for f in trust.loc[trust.reproducible, "factor"]]
    keep = list(np.where(s["pve"] > fa.MIN_PVE)[0])
    allc = list(range(W.shape[1]))

    def err(cols):
        cols = list(cols)
        Sig = W[:, cols] @ C[np.ix_(cols, cols)] @ W[:, cols].T + np.diag(psi)
        out = {}
        for lab, S in (("test", S_te), ("train", S_tr)):
            out[f"{lab} rel. error"] = float(np.linalg.norm(Sig - S) / np.linalg.norm(S))
            out[f"{lab} rel. error, off-diagonal"] = float(np.linalg.norm((Sig - S)[off]) / np.linalg.norm(S[off]))
        return {"columns": len(cols), **out}
    rows = [{"set": "all columns", **err(allc)},
            {"set": "drop the non-reproducible factors", **err([j for j in allc if j not in nonrep])},
            {"set": "reproducible factors only", **err(repr_)},
            {"set": "drop columns with PVE < 0.5%", **err(keep)},
            {"set": "noise only (no factors)", **err([])}]
    for t in (0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05):
        rows.append({"set": f"keep PVE >= {t:.2%}", **err(np.where(s["pve"] >= t)[0])})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, f"cov_k{k}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(R.round(4).to_string(index=False))


def rescore():
    import warnings
    warnings.filterwarnings("ignore")
    import torch, pyro
    import compare_methods as cm
    from run_methods import _build_model
    data, train_subset, test_subset, dp = cm.load_split(SEED)
    y_mean, y_sd = float(data["y"].mean()), float(data["y"].std())
    y = np.asarray(data["y_test"]).reshape(-1)
    for k in (20, 50, 100):
        d = os.path.join(OUT, f"k{k}")
        pyro.clear_param_store()
        pyro.get_param_store().set_state(torch.load(os.path.join(d, "shared_paramstore.pt"), weights_only=False))
        fm = _build_model("shared", {"k": k, "dense_model": False, "outcome_weight": 1.0,
                                     "init_scale_Lambda": 0.01, "init_scale_Z": 0.1})
        fm.n, fm.params = len(train_subset), pyro.get_param_store()
        pyro.set_rng_seed(SEED)
        r = cm._score_shared(fm, test_subset, dp, local_config={})
        m, lo, hi = (np.asarray(v, float) * y_sd + y_mean for v in r["pred"])
        met = pd.read_csv(os.path.join(d, "metrics.csv"))
        met["mse"], met["coverage_95"] = float(np.mean((m - y) ** 2)), float(np.mean((y >= lo) & (y <= hi)))
        met["width_95"] = float(np.mean(hi - lo))
        met.to_csv(os.path.join(d, "metrics.csv"), index=False)
        print(met.round(3).to_string(index=False), flush=True)


def corr(a, b):
    x, y = a.align(b, join="inner")
    ok = x.notna() & y.notna()
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def reference_loo(summ):
    """|r| of each CV fold's member factor with the consensus of the other folds."""
    fits = {f: fa.load_fold(f, fa.K, "shared") for f in range(fa.N_FOLDS)}
    rows = []
    for _, row in summ.iterrows():
        mem = [(int(m.split(":")[0][4:]), int(m.split(":F")[1]) - 1) for m in row.members.split(";")]
        S = {f: fa.as_series(fits[f], j) for f, j in mem}
        anchor = mem[0][0]
        sgn = {f: np.sign(corr(S[f], S[anchor])) for f, _ in mem}
        for f, _ in mem:
            others = pd.concat([S[g] * sgn[g] for g, _ in mem if g != f], axis=1).mean(axis=1)
            rows.append({"factor": row.factor, "fold": f, "abs_r_loo": abs(corr(S[f], others))})
    return pd.DataFrame(rows)


def compare():
    from scipy.optimize import linear_sum_assignment
    summ = pd.read_csv(f"{fa.OUT}/consensus_summary{TAG}.csv")
    cons = pd.read_csv(f"{fa.OUT}/consensus_loadings{TAG}.csv", index_col=0)
    ref = reference_loo(summ)
    ref.to_csv(os.path.join(OUT, "reference_loo.csv"), index=False)
    refs = ref.groupby("factor").abs_r_loo.agg(["min", "median"]).rename(
        columns={"min": "CV fold vs rest: min |r|", "median": "CV fold vs rest: median |r|"})
    out = []
    for k in sorted(int(p[1:]) for p in os.listdir(OUT) if p.startswith("k") and
                    os.path.exists(os.path.join(OUT, p, "shared_factors.npz"))):
        s = load_single(k)
        keep = np.where(s["pve"] > fa.MIN_PVE)[0]
        S = {j: fa.as_series(s, j) for j in keep}
        C = np.array([[corr(cons[c], S[j]) for j in keep] for c in cons.columns])
        A = np.abs(C)
        ri, ci = linear_sum_assignment(-A)
        hung = dict(zip(ri, ci))
        rows = []
        for i, c in enumerate(cons.columns):
            jb = int(np.argmax(A[i]))
            jm = hung.get(i)
            r_m = A[i, jm] if jm is not None else np.nan
            sgn = np.sign(C[i, jm]) if jm is not None else np.nan
            rows.append({"factor": c, "best |r|": A[i, jb], "matched factor": f"F{keep[jm] + 1}" if jm is not None else "",
                         "matched |r|": r_m, "recovered": bool(r_m >= fa.R_CUT),
                         "beta per SD (single fit)": s["beta"][keep[jm]] * sgn if jm is not None else np.nan,
                         "PVE (single fit)": s["pve"][keep[jm]] if jm is not None else np.nan})
        M = pd.DataFrame(rows).set_index("factor").join(
            summ.set_index("factor")[["folds", "mean |r| between folds", "PVE (mean)", "beta (mean, per SD)"]]).join(refs)
        M.to_csv(os.path.join(OUT, f"cf_match_k{k}.csv"))
        matched = {hung[i] for i in hung if A[i, hung[i]] >= fa.R_CUT}
        um = pd.DataFrame([{"factor": f"F{keep[j] + 1}", "PVE": s["pve"][keep[j]], "beta per SD": s["beta"][keep[j]],
                            "best |r| with any CF": A[:, j].max(), "closest CF": cons.columns[int(np.argmax(A[:, j]))]}
                           for j in range(len(keep)) if j not in matched]).sort_values("PVE", ascending=False)
        um.to_csv(os.path.join(OUT, f"unmatched_k{k}.csv"), index=False)
        met = pd.read_csv(os.path.join(OUT, f"k{k}", "metrics.csv")).iloc[0]
        out.append({"K_max": k, "epochs": int(met.epochs), "test MSE": met.mse, "coverage": met.coverage_95,
                    "factors (PVE > 0.5%)": len(keep), "CFs recovered (|r| >= 0.8)": int(M.recovered.sum()),
                    "of the 14 in all 5 folds": int(M[M.folds == 5].recovered.sum()),
                    "median matched |r|": M["matched |r|"].median(),
                    "unmatched single-fit factors": len(um),
                    "unmatched PVE total": um.PVE.sum() if len(um) else 0.0})
        pd.set_option("display.width", 250)
        print(f"\n=== K_max = {k}")
        print(M.round(3).to_string())
        print(um.round(3).to_string(index=False))
    S = pd.DataFrame(out)
    S.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print(); print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("fit", "compare", "reproduce", "rescore", "annotate", "predict", "cov"), required=True)
    ap.add_argument("--k", type=int, default=20)
    a = ap.parse_args()
    {"fit": lambda: fit(a.k), "compare": compare, "reproduce": reproduce, "rescore": rescore,
     "annotate": lambda: annotate(), "predict": lambda: annotate(stages=("table",)), "cov": cov_error}[a.stage]()
