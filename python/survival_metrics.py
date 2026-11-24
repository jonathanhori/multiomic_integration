"""
Survival metrics and baselines for the right-censored simulation study
(docs/viper_shared_extensions_design.md §4; sim_run_survival.py).

Metrics on the observed (censored) test data, all from sksurv:
    uno_c          Uno's IPCW concordance, truncated at the 80th percentile of the
                   observed test follow-up (robust to covariate-dependent censoring)
    harrell_c      Harrell's concordance
    ibs            integrated Brier score (IPCW) over the evaluation times
    mean_auc       mean time-dependent (cumulative / dynamic) AUC over the evaluation times
Evaluation times: the 10th, 20th, ..., 80th percentiles of the observed test event times,
kept inside the training and test follow-up range. For the IPCW metrics (Uno's C, IBS,
AUC) test follow-up beyond the largest training time is censored administratively there
(sksurv needs it; Harrell's C uses the full test data).

Survival curves of log-normal AFT predictions: S(t | x) = 1 - Phi((log t - m) / s).

Baselines:
    censored_normal_mle   log-normal AFT (Tobit) fit of a design matrix by maximum
                          likelihood (scipy L-BFGS, analytic gradient)
    cox_fit               Cox PH on a small design (sksurv CoxPHSurvivalAnalysis, light ridge)
    coxnet_cv             elastic-net Cox on the stacked views (sksurv
                          CoxnetSurvivalAnalysis), penalty chosen by 5-fold CV on Harrell's C
"""
import numpy as np
from scipy import optimize, stats


def surv_array(time, event):
    from sksurv.util import Surv
    return Surv.from_arrays(event=np.asarray(event).astype(bool), time=np.asarray(time, dtype=float))


def eval_times(train_time, test_time, test_event):
    t_ev = np.asarray(test_time)[np.asarray(test_event) == 1]
    if t_ev.size < 5:
        return np.array([])
    q = np.quantile(t_ev, np.arange(0.1, 0.81, 0.1))
    hi = min(np.max(train_time), np.max(test_time))
    q = q[(q > np.min(test_time)) & (q < hi)]
    return np.unique(q)


def aft_survival(m_log, s_log, times):
    """n x len(times) survival probabilities of log-normal predictions."""
    m_log = np.asarray(m_log).reshape(-1, 1); s_log = np.asarray(s_log).reshape(-1, 1)
    return stats.norm.sf((np.log(np.asarray(times)).reshape(1, -1) - m_log) / s_log)


def survival_scores(train_time, train_event, test_time, test_event, risk, surv=None, times=None):
    """risk: higher = shorter survival. surv: n_test x len(times) survival probabilities."""
    from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw,
                                cumulative_dynamic_auc, integrated_brier_score)
    s_tr = surv_array(train_time, train_event)
    out = {"uno_c": np.nan, "harrell_c": np.nan, "ibs": np.nan, "mean_auc": np.nan}
    risk = np.asarray(risk, dtype=float)
    try:
        s_full = surv_array(test_time, test_event)
        out["harrell_c"] = float(concordance_index_censored(s_full["event"], s_full["time"], risk)[0])
    except Exception:
        pass
    # IPCW metrics need every test time below the largest training follow-up time:
    # censor the test follow-up administratively just below it (a valid censoring).
    t_max = float(np.max(train_time)) * (1 - 1e-6)
    tt = np.asarray(test_time, float); ee = np.asarray(test_event).astype(bool)
    s_te = surv_array(np.minimum(tt, t_max), ee & (tt < t_max))
    try:
        tau = float(np.quantile(test_time, 0.8))
        out["uno_c"] = float(concordance_index_ipcw(s_tr, s_te, risk, tau=tau)[0])
    except Exception:
        pass
    if times is None:
        times = eval_times(train_time, test_time, test_event)
    times = np.asarray(times)[np.asarray(times) < min(t_max, float(s_te["time"].max()))]
    if len(times) >= 2:
        try:
            out["mean_auc"] = float(cumulative_dynamic_auc(s_tr, s_te, risk, times)[1])
        except Exception:
            pass
        if surv is not None:
            try:
                out["ibs"] = float(integrated_brier_score(s_tr, s_te, surv, times))
            except Exception:
                pass
    return out


def censored_normal_mle(D, y, event, ridge=1e-6):
    """Log-normal AFT (Tobit) fit: y = D coef + sigma e, e ~ N(0, 1), right-censored
    where event == 0. D includes the intercept column. Returns (coef, sigma)."""
    D = np.asarray(D, float); y = np.asarray(y, float); ev = np.asarray(event).astype(bool)
    coef0, *_ = np.linalg.lstsq(D, y, rcond=None)
    x0 = np.r_[coef0, np.log(max(np.std(y - D @ coef0), 1e-3))]

    def nll(par):
        b, ls = par[:-1], par[-1]
        s = np.exp(ls)
        r = (y - D @ b) / s
        lsf = stats.norm.logsf(r[~ev])
        f = -(stats.norm.logpdf(r[ev]).sum() - ls * ev.sum() + lsf.sum()) + 0.5 * ridge * (b[1:] @ b[1:])
        # d log phi(r) / dr = -r (events); d log sf(r) / dr = -phi(r) / sf(r) (censored)
        dll_dr = np.empty_like(r)
        dll_dr[ev] = -r[ev]
        dll_dr[~ev] = -np.exp(stats.norm.logpdf(r[~ev]) - lsf)
        # dr / db = -D / s ;  dr / d log s = -r
        gb = D.T @ dll_dr / s
        gb[1:] += ridge * b[1:]
        gls = (dll_dr * r).sum() + ev.sum()
        return f, np.r_[gb, gls]

    res = optimize.minimize(nll, x0, jac=True, method="L-BFGS-B")
    return res.x[:-1], float(np.exp(res.x[-1]))


def cox_fit(T_tr, time, event, alpha=1e-2):
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    return CoxPHSurvivalAnalysis(alpha=alpha).fit(np.asarray(T_tr, float), surv_array(time, event))


def coxnet_cv(X, time, event, l1_ratio=0.5, n_alphas=30, folds=5, seed=0):
    """Elastic-net Cox; penalty by K-fold CV on Harrell's C. Returns (model, alpha)."""
    from sklearn.model_selection import StratifiedKFold
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    X = np.asarray(X, float); y = surv_array(time, event)
    full = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, n_alphas=n_alphas, alpha_min_ratio=0.05,
                                  fit_baseline_model=True, max_iter=10000).fit(X, y)
    alphas = full.alphas_
    scores = np.zeros(len(alphas)); counts = np.zeros(len(alphas))
    ev = np.asarray(event).astype(int)
    k = min(folds, max(2, int(ev.sum()) // 2), int((1 - ev).sum()) if (1 - ev).sum() >= 2 else folds)
    skf = StratifiedKFold(n_splits=max(2, k), shuffle=True, random_state=seed)
    for tr_i, va_i in skf.split(X, ev):
        try:
            m = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=alphas, max_iter=10000).fit(X[tr_i], y[tr_i])
        except Exception:
            continue
        for j, a in enumerate(alphas):
            try:
                r = m.predict(X[va_i], alpha=a)
                scores[j] += concordance_index_censored(y[va_i]["event"], y[va_i]["time"], r)[0]; counts[j] += 1
            except Exception:
                pass
    mean = np.where(counts > 0, scores / np.maximum(counts, 1), -np.inf)
    return full, float(alphas[int(np.argmax(mean))])
