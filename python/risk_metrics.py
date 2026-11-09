"""
Exact prediction risk, direction recovery and factor-strength diagnostics.

All quantities use one replicate's true parameters Theta0 on the raw
(unstandardised) scale; see docs/evaluation_metrics.md ("Exact risk and factor
strength") for the definitions and how to read them.

Theta0 (``true_params``)
    Lambda_l (p_l x K), psi_l (p_l), beta (K), sigma_y^2, true means 0.
    psi_l is read from the data ("psi" field, controlled-strength generator) or,
    for the earlier generators, reconstructed as ||lambda_j||^2 / snr_x,l, which is
    how generate_factor_data_shared_asymmetric draws the noise. sigma_y^2 is read
    from "sigma2_y" if present, otherwise beta^T beta / snr_y (the earlier generators).
    Features with psi_j = 0 and lambda_j = 0 are constant zero; they carry no
    information and are left out of every sum.
    theta_l (outside-factor coefficients, controlled-strength generator with
    r_out > 0): y = z^T beta + sum_l theta_l^T eps_l + e. Read from "theta_out";
    zero (absent) for the earlier generators.

Oracle (``oracle``)
    G = sum_l Lambda_l^T Psi_l^{-1} Lambda_l,  A = I + G,  g_k = eig(G)
    b*_l = Psi_l^{-1} Lambda_l A^{-1} beta,   R* = sigma_y^2 + beta^T A^{-1} beta
    With an outside-factor term, Cov(x, y) = Lambda beta + Psi theta and (Woodbury)
    b* = Psi^{-1} Lambda beta + theta - Psi^{-1} Lambda A^{-1} (G beta + Lambda^T theta),
    R* = Var(y) - Cov(x, y)^T b*, Var(y) = sigma_y^2 + beta^T beta + theta^T Psi theta;
    both reduce to the expressions above when theta = 0.

Exact risk of a linear predictor yhat = c + sum_l b_l^T x_l (``exact_risk``)
    R(b, c) = R* + (b - b*)^T Sigma_x (b - b*) + c^2,
    Sigma_x = Lambda Lambda^T + Psi (stacked), evaluated as ||Lambda^T d||^2 + sum_j psi_j d_j^2.
    A method fitted on standardised data, xs = D (x - mu_hat), ys = (y - ybar)/sd_y,
    yhat_s = cs + bs^T xs, has raw coefficients b = sd_y D bs and
    c = ybar + sd_y cs - b^T mu_hat (``to_raw_scale``).

Direction recovery (``kappa2_hat``)
    u = Psi0^{-1/2} Lambda0 beta (stacked; for one relevant factor this is
    +- Psi0^{-1/2} lambda_rel up to scale), F = Psi0^{-1/2} L_hat_raw,
    kappa2_hat = ||P_col(F) u||^2 / ||u||^2. Rotation invariant. Also returned per
    true factor k (u_k = Psi0^{-1/2} lambda_k).

Strength estimate from data (``g_hat``)
    Whiten the training views with a fitted noise variance, take the sample
    eigenvalue mu of the component that overlaps most with the true whitened
    outcome direction, and invert the spiked-covariance map:
        g_hat = ((mu - 1 - gamma) + sqrt((mu - 1 - gamma)^2 - 4 gamma)) / 2   if mu > (1 + sqrt(gamma))^2
    else "below threshold" (g_hat = NaN). gamma = P / n.

Single-relevant-factor theory curves (``theory_*``)
    kappa2_PCA = (1 - gamma/g^2) / (1 + gamma/g) if g > sqrt(gamma) else 0
    kappa2_OL  = g / (g + gamma)
    R(kappa2)  = sigma_y^2 + beta^2 / (1 + g kappa2),  H = R(kappa2_PCA) - R(kappa2_OL)
"""
import numpy as np
import torch

_D = torch.float64


def _as(x):
    return torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x).to(_D).reshape(-1) \
        if x is not None else None


# ---------------------------------------------------------------------------
# True parameters and oracle
# ---------------------------------------------------------------------------

def true_params(sim):
    """Theta0 from a simulated training dataset (dict from run_methods.load_rds)."""
    Lam = [L.to(_D) for L in sim["Lambda_l"]]
    L = len(Lam)
    if sim.get("psi") is not None:
        psi_src = sim["psi"]
        psi = [p.to(_D).reshape(-1) for p in (psi_src if isinstance(psi_src, list) else [psi_src])]
    else:
        snr_x = sim["snr_x"].to(_D).reshape(-1)
        snr_x = snr_x.expand(L) if snr_x.numel() == 1 else snr_x
        psi = [(Lam[l] ** 2).sum(1) / snr_x[l] for l in range(L)]
    beta = sim["beta"].to(_D).reshape(-1)
    if sim.get("sigma2_y") is not None:
        sigma2_y = float(sim["sigma2_y"].reshape(-1)[0])
    else:
        snr_y = float(sim["snr_y"].reshape(-1)[0])
        bb = float(beta @ beta)
        sigma2_y = bb / snr_y if bb > 0 else 1.0
    theta = None
    if sim.get("theta_out") is not None:
        th = sim["theta_out"]
        theta = [t.to(_D).reshape(-1) for t in (th if isinstance(th, list) else [th])]
        if all(float(t.abs().sum()) == 0 for t in theta):
            theta = None
    # informative features: nonzero noise or nonzero loading
    active = [(p > 0) | ((Lm ** 2).sum(1) > 0) for p, Lm in zip(psi, Lam)]
    if any(bool(((p == 0) & a).any()) for p, a in zip(psi, active)):
        raise ValueError("a feature has signal but zero noise; Sigma_x / Psi^-1 undefined")
    return {"Lambda": Lam, "psi": psi, "beta": beta, "sigma2_y": sigma2_y, "active": active,
            "theta": theta}


def oracle(theta):
    """G, its eigenvalues, A = I + G, oracle coefficients b*_l and risk R*."""
    Lam, psi, beta, act = theta["Lambda"], theta["psi"], theta["beta"], theta["active"]
    K = beta.numel()
    G = torch.zeros(K, K, dtype=_D)
    for Lm, p, a in zip(Lam, psi, act):
        G += Lm[a].T @ (Lm[a] / p[a][:, None])
    A = torch.eye(K, dtype=_D) + G
    Ainv_beta = torch.linalg.solve(A, beta)
    b_star = []
    for Lm, p, a in zip(Lam, psi, act):
        b = torch.zeros(Lm.shape[0], dtype=_D)
        b[a] = (Lm[a] / p[a][:, None]) @ Ainv_beta
        b_star.append(b)
    R_star = theta["sigma2_y"] + float(beta @ Ainv_beta)
    var_y = theta["sigma2_y"] + float(beta @ beta)
    if theta.get("theta") is not None:
        th = theta["theta"]
        Lt_th = sum(Lm[a].T @ t[a] for Lm, t, a in zip(Lam, th, act))
        corr = torch.linalg.solve(A, G @ beta + Lt_th)
        b_star, cov_xy_b = [], 0.0
        for Lm, p, t, a in zip(Lam, psi, th, act):
            b = torch.zeros(Lm.shape[0], dtype=_D)
            PinvL = Lm[a] / p[a][:, None]
            b[a] = PinvL @ beta + t[a] - PinvL @ corr
            cov_xy_b += float((Lm[a] @ beta + p[a] * t[a]) @ b[a])
            b_star.append(b)
        var_y += float(sum((p[a] * t[a] ** 2).sum() for p, t, a in zip(psi, th, act)))
        R_star = var_y - cov_xy_b
    g_y = float(beta @ G @ beta / (beta @ beta)) if float(beta @ beta) > 0 else float("nan")
    return {"G": G, "g": torch.linalg.eigvalsh(G), "A": A, "b_star": b_star,
            "R_star": R_star, "var_y": var_y, "g_y": g_y}


# ---------------------------------------------------------------------------
# Exact risk
# ---------------------------------------------------------------------------

def to_raw_scale(coef_s, dp, y_mean, y_sd, intercept_s=0.0):
    """Standardised-scale linear predictor -> raw-scale (b_l list over all original
    features, c). coef_s: list of per-view coefficient vectors on the kept columns
    (dp["X_l_list_column_filters"]) or one stacked vector."""
    filters = dp["X_l_list_column_filters"]
    sizes = [int(f.sum()) for f in filters]
    if not isinstance(coef_s, (list, tuple)):
        v = _as(coef_s)
        coef_s = list(torch.split(v, sizes))
    b_raw, c = [], float(y_mean) + float(y_sd) * float(intercept_s)
    for bs, f, mu, sd in zip(coef_s, filters, dp["X_l_mean_list"], dp["X_l_sd_list"]):
        b = torch.zeros(f.numel(), dtype=_D)
        b_kept = float(y_sd) * _as(bs) / sd.to(_D).reshape(-1)
        b[f] = b_kept
        c -= float(b_kept @ mu.to(_D).reshape(-1))
        b_raw.append(b)
    return b_raw, c


def exact_risk(b_raw, c, theta, orc):
    """R(b, c) for raw-scale coefficients b_raw (list over views) and intercept c."""
    K = theta["beta"].numel()
    LtD = torch.zeros(K, dtype=_D)
    noise = 0.0
    for b, bs, Lm, p, a in zip(b_raw, orc["b_star"], theta["Lambda"], theta["psi"], theta["active"]):
        d = (b.to(_D) - bs)[a]
        LtD += Lm[a].T @ d
        noise += float((p[a] * d ** 2).sum())
    excess = float(LtD @ LtD) + noise
    R = orc["R_star"] + excess + float(c) ** 2
    return {"risk": R, "risk_excess": excess, "risk_intercept": float(c) ** 2,
            "risk_rel": (R - orc["R_star"]) / orc["R_star"]}


def risk_summary(coef_s, dp, y_mean, y_sd, theta, orc, intercept_s=0.0):
    """Convenience: exact-risk columns for a standardised-scale linear predictor."""
    b, c = to_raw_scale(coef_s, dp, y_mean, y_sd, intercept_s)
    out = exact_risk(b, c, theta, orc)
    out["risk_over_var_y"] = out["risk"] / orc["var_y"]
    return out


# ---------------------------------------------------------------------------
# Direction recovery and strength estimate
# ---------------------------------------------------------------------------

def _whitened_stack(theta, cols):
    """Stack Psi0^{-1/2} * rows over active features, for per-view (p_l x m) matrices."""
    return torch.cat([c.to(_D)[a] / p[a].sqrt()[:, None]
                      for c, p, a in zip(cols, theta["psi"], theta["active"])], 0)


def kappa2_hat(loadings_raw, theta, rel_tol=1e-10):
    """Squared cosine between the true whitened outcome direction and the span of
    the whitened fitted loadings.

    loadings_raw : list of (p_l x K_fit) raw-scale fitted loadings over *all*
        original features (rows of dropped columns zero); see raw_loadings().
    Returns {"kappa2_y": ..., "kappa2_k": tensor over true factors}.
    """
    F = _whitened_stack(theta, loadings_raw)
    U, s, _ = torch.linalg.svd(F, full_matrices=False)
    Q = U[:, s > rel_tol * s.max()] if s.numel() and s.max() > 0 else U[:, :0]
    Lw = _whitened_stack(theta, theta["Lambda"])                 # whitened true loadings (P x K)
    u_y = Lw @ theta["beta"]
    proj = lambda u: float((Q.T @ u).pow(2).sum() / u.pow(2).sum()) if float(u.pow(2).sum()) > 0 else float("nan")
    return {"kappa2_y": proj(u_y),
            "kappa2_k": torch.tensor([proj(Lw[:, k]) for k in range(Lw.shape[1])], dtype=_D),
            "span_rank": int(Q.shape[1])}


def raw_loadings(load_s, dp):
    """Standardised-scale fitted loadings (list of kept-column p_l x K) ->
    raw scale over all original features: row j multiplied by sd_j, dropped rows zero."""
    out = []
    for W, f, sd in zip(load_s, dp["X_l_list_column_filters"], dp["X_l_sd_list"]):
        W = torch.as_tensor(np.asarray(W) if not isinstance(W, torch.Tensor) else W).to(_D)
        full = torch.zeros(f.numel(), W.shape[1], dtype=_D)
        full[f] = W * sd.to(_D).reshape(-1)[:, None]
        out.append(full)
    return out


def g_from_mu(mu, gamma):
    """Invert the spiked-covariance eigenvalue map; NaN when below the BBP threshold."""
    if mu <= (1 + np.sqrt(gamma)) ** 2:
        return float("nan")
    a = mu - 1 - gamma
    return float((a + np.sqrt(max(a * a - 4 * gamma, 0.0))) / 2)


def g_hat(X_s_list, noise_var_s, theta, dp):
    """Strength of the outcome-relevant component estimated from the training views.

    X_s_list    : standardised training views (kept columns), column means 0.
    noise_var_s : fitted noise variances on the standardised scale, a list of
                  per-view vectors (kept columns) or a scalar (isotropic, e.g. PPCA).
    The component is the sample eigenvector of the whitened covariance with the
    largest squared overlap with the true outcome direction expressed in the same
    whitened coordinates (an oracle choice, used only as a diagnostic).
    """
    X = torch.cat([x.to(_D) for x in X_s_list], 1)
    n, P = X.shape
    if np.isscalar(noise_var_s) or (isinstance(noise_var_s, torch.Tensor) and noise_var_s.numel() == 1):
        w = torch.full((P,), 1.0 / float(noise_var_s), dtype=_D)
    else:
        w = 1.0 / torch.cat([_as(v) for v in noise_var_s])
    Xw = X * w.sqrt()
    _, s, Vt = torch.linalg.svd(Xw / np.sqrt(n), full_matrices=False)
    mu_all = s ** 2
    # true outcome direction on the standardised kept columns, then whitened by the fitted noise
    t = torch.cat([(Lm @ theta["beta"])[f] / sd.to(_D).reshape(-1)
                   for Lm, f, sd in zip(theta["Lambda"], dp["X_l_list_column_filters"], dp["X_l_sd_list"])])
    t = t * w.sqrt()
    overlap = (Vt @ t) ** 2 / (t @ t)
    i = int(torch.argmax(overlap))
    gamma = P / n
    mu = float(mu_all[i])
    return {"g_hat": g_from_mu(mu, gamma), "g_hat_mu": mu, "g_hat_overlap": float(overlap[i]),
            "g_hat_rank": i + 1, "g_hat_above_threshold": mu > (1 + np.sqrt(gamma)) ** 2,
            "gamma": gamma}


# ---------------------------------------------------------------------------
# Single-relevant-factor theory curves
# ---------------------------------------------------------------------------

def theory_kappa2_pca(g, gamma):
    g = np.asarray(g, float)
    return np.where(g > np.sqrt(gamma), (1 - gamma / g ** 2) / (1 + gamma / g), 0.0)


def theory_kappa2_ol(g, gamma):
    g = np.asarray(g, float)
    return g / (g + gamma)


def theory_risk(kappa2, g, beta, sigma2_y):
    return sigma2_y + np.asarray(beta, float) ** 2 / (1 + np.asarray(g, float) * kappa2)


def theory_headroom(g, gamma, beta, sigma2_y):
    return (theory_risk(theory_kappa2_pca(g, gamma), g, beta, sigma2_y)
            - theory_risk(theory_kappa2_ol(g, gamma), g, beta, sigma2_y))
