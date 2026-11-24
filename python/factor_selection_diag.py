"""
Per-factor diagnostics for judging which fitted factors to analyse (November 2026;
docs/low_pve_predictive_factors.md). Called by sim_run_controlled_study.py for every
dataset; everything a selection rule might use is saved, together with the truth, so
rules can be compared offline without refitting.

For the main VIPER fit, MOFA and PCA (15 components), per column, as fitted and after
the multiview varimax rotation (VIPER and MOFA; factor_interpretation._rotate_loadings):

  structure      PVE (share of the training views' variance), effective features
                 (sum w^2)^2 / sum w^4 as a share of P, share of the squared loadings
                 in each view, strength g = sum_j w_j^2 E[1/psi_j] (loadings per SD of
                 the training scores; VIPER only: E[1/psi] from the guide, others 1)
                 and g / sqrt(P/n), E_q[tau] per view (VIPER, as fitted only)
  outcome, train beta per SD of the training scores and its sd (VIPER: mean-field
                 posterior, rotated R^T diag(s2) R; MOFA / PCA: OLS on all columns),
                 z = beta / sd
  x-only (VIPER) x-only agreement corr(training score, E[z | x_train]) and x-only t
                 (OLS of y_train on all x-only training scores)
  held-out       test scores from the test views only (VIPER: closed form "cf" and the
                 local-inference posterior means "loc"; MOFA / PCA: their projection):
                 partial t (test y minus the other columns' plug-in prediction,
                 regressed on the column; signed: > 0 = same sign as the fitted beta),
                 marginal t (test y on the column alone, signed the same way), test MSE
                 of the plug-in prediction without the column
  truth          from the whitened raw-scale loading vector f (risk_metrics
                 ._whitened_stack): share of ||f||^2 on each relevant factor's direction
                 (rel_share_<r>, r = 0 the first / strong relevant factor) and on the
                 nuisance span; kind = relevant_<r> / nuisance (share > 0.5) else
                 mixed / spurious; |corr| of the training scores with each true score
                 column (true_corr_k<k>)

Extra VIPER fits (same package defaults and seed), for rules that need refitting:
  permuted_y   the training outcome permuted (null calibration: every beta is noise);
               the same per-column table (closed-form held-out t; test y unpermuted)
  sub<i>       a fit on a random 80 % of the training samples (i = 0, 1, ...); per main
               column, the best |r| of its rotated loading vector with the subsample
               fit's rotated columns (reproducibility as in the TCGA CV, r >= 0.8)

Saved arrays (<stem>.npz): for each fit (main, permuted_y, sub<i>) the rotation R, the
loadings (P x K, standardised features, as fitted), training score means and sds,
beta mean and variance, E[1/psi], E_q[tau]; for the main fit also the x-only training
scores, closed-form and local test score means (and local sds), plus the training
indices of each subsample and the permutation. With the saved param store and the data
this reconstructs every quantity above.
"""
import numpy as np, pandas as pd

R_CUT = 0.8


def _t_partial(Zte, b, y, j):
    """Signed t of column j's test score for test y minus the other columns' prediction."""
    resid = y - (Zte @ b - Zte[:, j] * b[j])
    x = Zte[:, j] - Zte[:, j].mean()
    if (x ** 2).sum() < 1e-12:
        return 0.0
    r = resid - resid.mean()
    bj = (x @ r) / (x @ x)
    e = r - bj * x
    t = bj / np.sqrt((e @ e) / (len(y) - 2) / (x @ x))
    return float(t * (np.sign(b[j]) if b[j] != 0 else 1.0))


def _t_marginal(Zte, b, y, j):
    x = Zte[:, j] - Zte[:, j].mean()
    if (x ** 2).sum() < 1e-12:
        return 0.0
    r = y - y.mean()
    bj = (x @ r) / (x @ x)
    e = r - bj * x
    t = bj / np.sqrt((e @ e) / (len(y) - 2) / (x @ x))
    return float(t * (np.sign(b[j]) if b[j] != 0 else 1.0))


def _ols_t(T, y):
    D = np.c_[np.ones(len(y)), T]
    coef, *_ = np.linalg.lstsq(D, y, rcond=None)
    res = y - D @ coef
    dof = max(len(y) - D.shape[1], 1)
    cov = (res @ res / dof) * np.linalg.pinv(D.T @ D)
    se = np.sqrt(np.maximum(np.diag(cov)[1:], 1e-300))
    return coef[1:], se


def _truth(theta, dp, W_list):
    """Shares of each column's whitened raw loading on each relevant direction and on the
    nuisance span."""
    import torch
    import risk_metrics as rm
    F = rm._whitened_stack(theta, rm.raw_loadings([torch.as_tensor(w) for w in W_list], dp)).numpy()
    Lw = rm._whitened_stack(theta, theta["Lambda"]).numpy()
    beta = theta["beta"].double().numpy().reshape(-1)
    rel = [k for k in range(len(beta)) if beta[k] != 0]
    nuis = [k for k in range(len(beta)) if beta[k] == 0]
    nf = np.maximum((F ** 2).sum(0), 1e-300)
    out = {}
    for r, k in enumerate(rel):
        u = Lw[:, k] / np.linalg.norm(Lw[:, k])
        out[f"rel_share_{r}"] = (u @ F) ** 2 / nf
    Qn, _ = np.linalg.qr(Lw[:, nuis])
    out["nuis_share"] = ((Qn.T @ F) ** 2).sum(0) / nf
    kind = np.array(["mixed / spurious"] * F.shape[1], dtype=object)
    kind[out["nuis_share"] > 0.5] = "nuisance"
    for r in range(len(rel)):
        kind[out[f"rel_share_{r}"] > 0.5] = f"relevant_{r}"
    out["kind"] = kind
    return out


def column_table(W_list, Ztr, Zte_sets, b, bsd, y_tr, y_te, tot, theta, dp, Z_true_tr,
                 inv_psi=None, tau=None, Zx=None):
    """Per-column table. W_list: per-view loadings (standardised features, per unit of
    the score as given in Ztr); Ztr training scores; Zte_sets {label: test scores};
    b, bsd beta per unit score and its sd; returns a DataFrame."""
    n, K = Ztr.shape
    sd = Ztr.std(0)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    W = np.vstack(W_list)
    P = W.shape[0]
    Ws = W * sd                                                  # per SD of the score
    ipsi = np.ones(P) if inv_psi is None else inv_psi
    g = ((Ws ** 2) * ipsi[:, None]).sum(0)
    eff = (Ws ** 2).sum(0) ** 2 / np.maximum((Ws ** 4).sum(0), 1e-300) / P
    pve = (Ztr ** 2).sum(0) * (W ** 2).sum(0) / tot
    sizes = np.cumsum([0] + [w.shape[0] for w in W_list])
    vshare = np.stack([(W[sizes[l]:sizes[l + 1]] ** 2).sum(0) for l in range(len(W_list))], 1)
    vshare = vshare / np.maximum(vshare.sum(1, keepdims=True), 1e-300)
    d = {"column": np.arange(K), "PVE": pve, "beta_per_sd": b * sd, "beta_sd_per_sd": bsd * sd,
         "z": b / np.maximum(bsd, 1e-300), "strength_g": g, "g_ratio": g / np.sqrt(P / n),
         "eff_share": eff, **{f"view_share_{l}": vshare[:, l] for l in range(len(W_list))}}
    if tau is not None:
        for l, t in enumerate(tau):
            d[f"tau_view{l}"] = np.asarray(t)
    if Zx is not None:
        d["xonly_agree"] = np.array([np.corrcoef(Ztr[:, j], Zx[:, j])[0, 1] if sd[j] > 1e-12 else np.nan
                                     for j in range(K)])
        cx, sx = _ols_t(Zx, y_tr)
        d["xonly_t"] = cx / sx * np.sign(np.where(b != 0, b, 1))
    for lab, Zte in Zte_sets.items():
        d[f"heldout_t_{lab}"] = [_t_partial(Zte, b, y_te, j) for j in range(K)]
        d[f"heldout_tmarg_{lab}"] = [_t_marginal(Zte, b, y_te, j) for j in range(K)]
        base = float(np.mean((Zte @ b - y_te) ** 2))
        d[f"test_mse_all_{lab}"] = base
        d[f"test_mse_drop_{lab}"] = [float(np.mean((Zte @ b - Zte[:, j] * b[j] - y_te) ** 2)) for j in range(K)]
    tr = _truth(theta, dp, W_list)
    d.update({k: v for k, v in tr.items()})
    Zt = np.asarray(Z_true_tr)
    for k in range(Zt.shape[1]):
        d[f"true_corr_k{k}"] = [abs(np.corrcoef(Ztr[:, j], Zt[:, k])[0, 1]) if sd[j] > 1e-12 else 0.0
                                for j in range(K)]
    return pd.DataFrame(d)


def viper_parts(store, L, Xtr, Xte):
    """Loadings, scores and beta moments of the VIPER fit in the param store."""
    import torch
    from viper_predict import guide_moments, closed_form_scores, expected_tau
    from constants import Params
    mom = guide_moments(store, L)
    Z = store[Params.loc_Z].detach().double().numpy()
    Zs = store[Params.scale_Z].detach().double().numpy()
    Lam = [m.double().numpy() for m in mom["M"]]
    Lam = [m[:, None] if m.ndim == 1 else m for m in Lam]
    ipsi = np.concatenate([(a / b).double().numpy().reshape(-1) for a, b in zip(mom["a_psi"], mom["b_psi"])])
    tau = [t.numpy() for t in expected_tau(store, L)]
    cf_tr = closed_form_scores(mom, Xtr)["m_z"].double().numpy()
    cf_te = closed_form_scores(mom, Xte)["m_z"].double().numpy()
    return {"Z": Z, "Zs": Zs, "Lam": Lam, "m_beta": mom["m_beta"].double().numpy(),
            "s2_beta": mom["s2_beta"].double().numpy(), "inv_psi": ipsi, "tau": tau,
            "Zx": cf_tr, "Zte_cf": cf_te}


def viper_tables(v, y_tr, y_te, tot, theta, dp, Z_true_tr, Zte_loc=None):
    """As-fitted and varimax tables for one VIPER fit (dict from viper_parts)."""
    from factor_interpretation import _rotate_loadings
    out = []
    K = v["Z"].shape[1]
    for basis in ("as fitted", "varimax"):
        if basis == "as fitted":
            Wl, R = v["Lam"], np.eye(K)
        else:
            Wl, R = _rotate_loadings(v["Lam"]); Wl = [np.asarray(w) for w in Wl]
        b = R.T @ v["m_beta"]
        bsd = np.sqrt(np.diag(R.T @ np.diag(v["s2_beta"]) @ R))
        sets = {"cf": v["Zte_cf"] @ R}
        if Zte_loc is not None:
            sets["loc"] = Zte_loc @ R
        t = column_table(Wl, v["Z"] @ R, sets, b, bsd, y_tr, y_te, tot, theta, dp, Z_true_tr,
                         inv_psi=v["inv_psi"], tau=v["tau"] if basis == "as fitted" else None, Zx=v["Zx"] @ R)
        out.append((basis, t, R, Wl))
    return out


def _fit(train_sub, k, seed):
    import pyro
    from run_methods import train_model
    pyro.clear_param_store(); pyro.set_rng_seed(seed)
    model, _, _ = train_model({"k": k, "dense_model": False}, {}, train_sub, None, model_type="shared", write=False)
    return model, pyro.get_param_store()


def run(tr, dp, train_sub, main_paramstore, m_loc, s_loc, mofa_npz, k_fit, seed, out_stem,
        n_perm=1, n_sub=2, sub_frac=0.8):
    """All per-factor tables for one dataset; writes <out_stem>.csv and <out_stem>.npz."""
    import torch, pyro
    from torch.utils.data import TensorDataset
    from sklearn.decomposition import PCA
    from factor_interpretation import _rotate_loadings
    import risk_metrics as rm
    L = dp["L"]
    Xtr = [x.double() for x in dp["train_X_l_list_clean"]]
    Xte = [x.double() for x in dp["test_X_l_list_clean"]]
    y_tr = dp["train_y_clean"].double().reshape(-1).numpy()
    y_te = dp["test_y_clean"].double().reshape(-1).numpy()
    tot = sum(float((x ** 2).sum()) for x in Xtr)
    theta = rm.true_params(tr)
    Z_true = tr["Z"].double().numpy()
    arrays, tables = {}, []

    pyro.clear_param_store(); pyro.get_param_store().load(main_paramstore)
    v = viper_parts(pyro.get_param_store(), L, Xtr, Xte)
    main_rot = None
    for basis, t, R, Wl in viper_tables(v, y_tr, y_te, tot, theta, dp, Z_true, Zte_loc=m_loc):
        tables.append(t.assign(method="viper", fit="main", basis=basis))
        if basis == "varimax":
            main_rot = (np.vstack(Wl), R)
    arrays.update({"main_R": main_rot[1], "main_Lambda": np.vstack(v["Lam"]), "main_Z": v["Z"], "main_Zs": v["Zs"],
                   "main_m_beta": v["m_beta"], "main_s2_beta": v["s2_beta"], "main_inv_psi": v["inv_psi"],
                   "main_tau": np.stack(v["tau"]), "main_Zx": v["Zx"], "main_Zte_cf": v["Zte_cf"],
                   "main_Zte_loc": m_loc, "main_Zte_loc_sd": s_loc})

    # MOFA and PCA15
    if mofa_npz is not None:
        d = np.load(mofa_npz)
        Wm = [d[f"weights_view{l}"] for l in range(L)]
        for basis in ("as fitted", "varimax"):
            Wl, R = (Wm, np.eye(Wm[0].shape[1])) if basis == "as fitted" else _rotate_loadings(Wm)
            Wl = [np.asarray(w) for w in Wl]
            Ztr, Zte = d["Z_train"] @ R, d["Z_test"] @ R
            b, se = _ols_t(Ztr, y_tr)
            tables.append(column_table(Wl, Ztr, {"proj": Zte}, b, se, y_tr, y_te, tot, theta, dp, Z_true)
                          .assign(method="mofa", fit="main", basis=basis))
    A = torch.cat(Xtr, 1).numpy(); At = torch.cat(Xte, 1).numpy(); sizes = [x.shape[1] for x in Xtr]
    pca = PCA(n_components=15, random_state=0).fit(A)
    sc_tr, sc_te = pca.transform(A), pca.transform(At)
    b, se = _ols_t(sc_tr, y_tr)
    Wp = np.split(pca.components_.T, np.cumsum(sizes)[:-1])
    tables.append(column_table(Wp, sc_tr, {"proj": sc_te}, b, se, y_tr, y_te, tot, theta, dp, Z_true)
                  .assign(method="pca15", fit="main", basis="as fitted"))

    idx, *Xs, y = train_sub.tensors
    n = len(y)
    rng = np.random.default_rng(1000 + seed)
    # permuted-y fits
    for i in range(n_perm):
        perm = rng.permutation(n)
        sub = TensorDataset(idx, *Xs, y[torch.as_tensor(perm)])
        _, store = _fit(sub, k_fit, seed)
        vp = viper_parts(store, L, Xtr, Xte)
        for basis, t, R, Wl in viper_tables(vp, y_tr[perm], y_te, tot, theta, dp, Z_true):
            tables.append(t.assign(method="viper", fit=f"permuted_y{i}", basis=basis))
        arrays.update({f"permuted_y{i}_perm": perm, f"permuted_y{i}_R": R, f"permuted_y{i}_Lambda": np.vstack(vp["Lam"]),
                       f"permuted_y{i}_Z": vp["Z"], f"permuted_y{i}_m_beta": vp["m_beta"],
                       f"permuted_y{i}_s2_beta": vp["s2_beta"], f"permuted_y{i}_tau": np.stack(vp["tau"])})
    # subsample fits: reproducibility of the main rotated columns
    Wmain = main_rot[0]
    for i in range(n_sub):
        keep = np.sort(rng.choice(n, int(round(sub_frac * n)), replace=False))
        kt = torch.as_tensor(keep)
        sub = TensorDataset(torch.arange(len(keep), dtype=idx.dtype), *[x[kt] for x in Xs], y[kt])
        _, store = _fit(sub, k_fit, seed)
        vs = viper_parts(store, L, Xtr, Xte)
        Ws, Rs = _rotate_loadings(vs["Lam"]); Ws = np.vstack([np.asarray(w) for w in Ws])
        C = np.corrcoef(Wmain.T, Ws.T)[:Wmain.shape[1], Wmain.shape[1]:]
        best = np.nanmax(np.abs(C), 1)
        main_tab = tables[1]                                     # viper, main, varimax
        main_tab[f"sub{i}_best_r"] = best
        arrays.update({f"sub{i}_keep": keep, f"sub{i}_R": Rs, f"sub{i}_Lambda": np.vstack(vs["Lam"]),
                       f"sub{i}_m_beta": vs["m_beta"], f"sub{i}_s2_beta": vs["s2_beta"], f"sub{i}_tau": np.stack(vs["tau"])})
    if n_sub:
        rs = tables[1][[f"sub{i}_best_r" for i in range(n_sub)]].to_numpy()
        tables[1]["sub_found"] = (rs >= R_CUT).sum(1)
    pyro.clear_param_store()
    T = pd.concat(tables, ignore_index=True)
    T.to_csv(out_stem + ".csv", index=False)
    np.savez_compressed(out_stem + ".npz", **arrays)
    return T
