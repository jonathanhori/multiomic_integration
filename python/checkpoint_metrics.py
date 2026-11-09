"""
Factor recovery, cross-method factor comparison and credible-interval coverage for
VIPER-Shared fits on the controlled-strength data, scored at training checkpoints
(opt-in: controlled_optimizer_grid.py --recovery, which sets cfg["recovery"] for
controlled_training_length.run_one). Also used by controlled_method_factors.py for the
comparison methods, so every method is scored with the same definitions.

Scales. Loadings are compared on the raw feature scale over all original features
(risk_metrics.raw_loadings; dropped columns are zero), where every feature has noise
variance 1. Outcome and signal intervals are on the standardised scale the model is
fitted on (training means and sds).

Factor recovery (validate_settings.factor_recovery, columns prefixed rec_):
    rec_congr_rel, rec_congr_nuis_mean, rec_kappa2_rel, rec_kappa2_nuis_mean,
    rec_out_factor_share, rec_out_factor_congr, rec_out_factor_auc, rec_out_factor_prec_s,
    rec_out_factor_strength (VIPER only), rec_coef_auc, rec_coef_prec_s
    rec_span5_truth   mean of the 5 largest squared cosines of the principal angles between
                      the span of the fitted loadings and the span of the true loadings
                      (K = 5): 1 = the fitted factors span the true factor space

Cross-method comparison of the factors (VIPER against each comparison method m; the
comparison methods against each other in controlled_method_factors.py):
    xm_<m>_span5      the same principal-angle measure between the two methods' loading
                      spans (top 5 of min(K_a, K_b) angles)
    xm_<m>_outcongr   |Tucker congruence| between the two methods' outcome factors (the
                      fitted factor carrying most of the prediction: largest
                      |coef_j| sd(score_j), as rec_out_factor_*)
    xm_<m>_predcor    correlation of the two methods' test-set outcome predictions
Comparison methods (fitted once per dataset): pca5 / pca15 (PCA of the stacked
standardised training views, OLS of y on the scores) and mofa (the Phase 2 MOFA2 fit,
results/.../mofa/<stem>.npz, OLS of y on its scores, as in validate_settings).

Coverage (95%):
    cov95_y_cf, width95_y_cf   test outcome, closed-form predictive (viper_predict
                               closed_form_scores, cov="diag"): yhat +- 1.96 sd
    cov95_y_mc, width95_y_mc   the same from the Monte Carlo predictive (mc_scores) mean
                               and variance, computed only when mc_draws > 0
    cov95_signal, width95_signal  entrywise coverage of the true training signal
                               Z Lambda_l^T (standardised scale, as data_utils.scale_sim_struct)
                               by the guide's marginal intervals. Under the mean-field guide
                               each entry sum_k z_ik lambda_jk has mean m_z m_lambda^T and
                               variance m_z^2 s_lambda + s_z m_lambda^2 + s_z s_lambda
                               (elementwise, summed over k); the interval uses a normal
                               approximation to it.
TODO(user): the normal approximation for the signal intervals (exact sampling is
possible but needs n x P x draws memory per fit), and the 5 in the span measures
(the true number of factors) are design choices.
"""
import os

import numpy as np
import torch

Z95 = 1.959963984540054


def _span_overlap(A, B, k=5):
    """Mean of the k largest squared cosines of the principal angles between span(A) and span(B)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    A, B = A[:, np.linalg.norm(A, axis=0) > 0], B[:, np.linalg.norm(B, axis=0) > 0]
    if A.shape[1] == 0 or B.shape[1] == 0:
        return float("nan")
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.mean(np.sort(s)[::-1][:min(k, len(s))] ** 2))


def _congr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(abs(a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))


def _out_factor(coef, scores):
    contrib = np.abs(np.asarray(coef, float).reshape(-1)) * np.asarray(scores, float).std(0)
    return int(np.argmax(contrib))


def _raw(load_s, dp):
    import risk_metrics as rm
    return torch.cat(rm.raw_loadings(load_s, dp), 0).numpy()


def factor_summary(load_s, coef, scores_tr, pred_te, dp):
    """What the cross-method comparison needs from one factor method."""
    F = _raw(load_s, dp)
    j = _out_factor(coef, scores_tr)
    return {"F": F, "out": F[:, j], "pred": np.asarray(pred_te, float).reshape(-1)}


def compare(a, b):
    """Cross-method measures between two factor_summary dicts."""
    return {"span5": _span_overlap(a["F"], b["F"]), "outcongr": _congr(a["out"], b["out"]),
            "predcor": float(np.corrcoef(a["pred"], b["pred"])[0, 1])}


def reference_fits(dp, mofa_npz):
    """The comparison methods' factors for one dataset: name -> dict with load (standardised
    loadings per view), coef, scores_tr, pred_te (test predictions), lo / hi (95% OLS
    prediction intervals), summary (factor_summary)."""
    from sklearn.decomposition import PCA
    from misaligned_baselines import ols_prediction_interval
    Xtr_l, Xte_l = dp["train_X_l_list_clean"], dp["test_X_l_list_clean"]
    X_tr = torch.cat(Xtr_l, 1).double().numpy()
    X_te = torch.cat(Xte_l, 1).double().numpy()
    y_tr = dp["train_y_clean"].double().reshape(-1).numpy()
    bounds = np.cumsum([0] + [x.shape[1] for x in Xtr_l])
    split = lambda A: [A[bounds[l]:bounds[l + 1]] for l in range(len(Xtr_l))]
    refs = {}
    for kc in (5, 15):
        pca = PCA(n_components=kc, random_state=0).fit(X_tr)
        T, Tte = pca.transform(X_tr), pca.transform(X_te)
        coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(T)), T]), y_tr, rcond=None)
        pred, lo, hi = ols_prediction_interval(T, y_tr, Tte)
        refs[f"pca{kc}"] = {"load": split(pca.components_.T), "coef": coef[1:], "scores_tr": T,
                            "pred_te": pred, "lo": lo, "hi": hi}
    if mofa_npz is not None and os.path.exists(mofa_npz):
        d = np.load(mofa_npz)
        W = [d[f"weights_view{l}"] for l in range(dp["L"])]
        coef, *_ = np.linalg.lstsq(d["Z_train"], y_tr, rcond=None)      # as validate_settings
        pred, lo, hi = ols_prediction_interval(d["Z_train"], y_tr, d["Z_test"])
        refs["mofa"] = {"load": W, "coef": coef, "scores_tr": d["Z_train"], "pred_te": pred, "lo": lo, "hi": hi}
    for r in refs.values():
        r["summary"] = factor_summary(r["load"], r["coef"], r["scores_tr"], r["pred_te"], dp)
    return refs


def span_truth(load_s, dp, theta):
    Lam = torch.cat([Lm.double() for Lm in theta["Lambda"]], 0).numpy()
    return _span_overlap(_raw(load_s, dp), Lam)


def viper_metrics(store, dp, theta, refs, mc_draws=0):
    """All measures above for the VIPER fit in `store` (the Pyro param store)."""
    from constants import Params
    from validate_settings import factor_recovery
    from viper_predict import closed_form_scores, guide_moments, mc_scores
    from data_utils import scale_sim_struct

    L = dp["L"]
    mom = guide_moments(store, L)
    cf_tr = closed_form_scores(mom, dp["train_X_l_list_clean"])
    cf_te = closed_form_scores(mom, dp["test_X_l_list_clean"])
    y = dp["test_y_clean"].double().reshape(-1).numpy()
    out = {}
    yhat, sd = cf_te["yhat"].numpy(), cf_te["pred_var"].clamp(min=0).sqrt().numpy()
    out["cov95_y_cf"] = float(np.mean(np.abs(y - yhat) <= Z95 * sd))
    out["width95_y_cf"] = float(np.mean(2 * Z95 * sd))
    if mc_draws:
        mc = mc_scores(mom, dp["test_X_l_list_clean"], n_draws=mc_draws, seed=0)
        ym, sm = mc["yhat"].numpy(), mc["pred_var"].clamp(min=0).sqrt().numpy()
        out["cov95_y_mc"] = float(np.mean(np.abs(y - ym) <= Z95 * sm))
        out["width95_y_mc"] = float(np.mean(2 * Z95 * sm))

    # entrywise coverage of the true training signal Z Lambda_l^T
    mZ = store[Params.loc_Z].detach().double().cpu()
    sZ = store[Params.scale_Z].detach().double().cpu() ** 2
    sim = dp["train_SIM_decomp"]
    truth = scale_sim_struct(
        [sim["SIM_Z"].double() @ sim["SIM_Lambda_l_list"][l][dp["X_l_list_column_filters"][l]].double().T
         for l in range(L)], [m.double() for m in dp["X_l_mean_list"]], [s.double() for s in dp["X_l_sd_list"]])
    hit, wid, cnt = 0.0, 0.0, 0
    for l in range(L):
        M, S = mom["M"][l], mom["S"][l]
        mean = mZ @ M.T
        var = (mZ ** 2) @ S.T + sZ @ (M ** 2).T + sZ @ S.T
        half = Z95 * var.sqrt()
        hit += float(((truth[l] - mean).abs() <= half).sum())
        wid += float((2 * half).sum())
        cnt += mean.numel()
    out["cov95_signal"], out["width95_signal"] = hit / cnt, wid / cnt

    rec = factor_recovery(mom["M"], mom["m_beta"].numpy(), cf_tr["m_z"].numpy(), cf_te["coef"], dp, theta,
                          count_factors=True)
    out.update({f"rec_{k}": v for k, v in rec.items()})
    out["rec_span5_truth"] = span_truth(mom["M"], dp, theta)
    me = factor_summary(mom["M"], mom["m_beta"].numpy(), cf_tr["m_z"].numpy(), yhat, dp)
    for m, r in refs.items():
        out.update({f"xm_{m}_{k}": v for k, v in compare(me, r["summary"]).items()})
    return out
