"""
Closed-form test-time prediction for SupMultiviewShared (VIPER-Shared).

Replaces local SVI at test time: the global variational parameters are read
from the trained param store and each test sample's scores are computed in
closed form from its views alone (the test outcome is never used).

Notation (standardised scale, the scale the model is fitted on)
---------------------------------------------------------------
For view l, from the guide (guide_mgp / guide_dense):
    M_l = loc_Lambda_l                  (p_l x K)  posterior mean of the loadings
    S_l = scale_Lambda_l^2              (p_l x K)  posterior variances of the loadings
    w_l = a_psi_l / b_psi_l             (p_l)      E_q[1 / psi_l]
q(psi_lj) is InverseGamma(a, b) with concentration a and rate b (Pyro's
parameterisation), so 1/psi_lj ~ Gamma(a, rate b) and E_q[1/psi_lj] = a / b.
Also m_beta = loc_beta, S_beta = diag(scale_beta^2), and
E_q[sigma_y^2] = b_sigma_y / (a_sigma_y - 1).

Option 1, mean field (``closed_form_scores``)
    A_bar = I + sum_l [ M_l^T diag(w_l) M_l + diag(sum_j w_lj S_l[j, :]) ]
    m_z   = A_bar^{-1} sum_l M_l^T diag(w_l) x_l
    cov_z = A_bar^{-1}                 (cov="full")
            diag(1 / A_bar_kk)         (cov="diag", matches the diagonal guide)
    yhat  = m_z^T m_beta
    Var(y | x) = E_q[sigma_y^2] + tr(S_beta cov_z) + m_beta^T cov_z m_beta + m_z^T S_beta m_z

A_bar and m_z are the optimum of the local ELBO for a Gaussian q(z) with the
global factors of q held fixed, which is what local SVI maximises by
stochastic gradients; cov="diag" is the optimum within the diagonal family that
the guide uses. yhat is linear in x: yhat = sum_l bs_l^T x_l with
    bs_l = diag(w_l) M_l A_bar^{-1} m_beta.

Option 2, Monte Carlo over q_g (``mc_scores``)
    Draw (Lambda, psi, beta, sigma_y^2) ~ q. Per draw s,
        A_s = I + sum_l Lambda_l^T Psi_l^{-1} Lambda_l,  h_s = sum_l Lambda_l^T Psi_l^{-1} x_l,
        z | x, draw s ~ N(A_s^{-1} h_s, A_s^{-1}).
    The predictive for z is the mixture over draws: its mean is the average of
    A_s^{-1} h_s, its covariance mean_s(A_s^{-1}) + Cov_s(A_s^{-1} h_s). For y,
        E[y | x]   = mean_s beta_s^T A_s^{-1} h_s,
        Var(y | x) = Var_s(beta_s^T A_s^{-1} h_s) + mean_s(beta_s^T A_s^{-1} beta_s + sigma_y,s^2).
    E[y | x] is again linear in x, with bs_l = mean_s Psi_l,s^{-1} Lambda_l,s A_s^{-1} beta_s.

Both predictors return the coefficients bs_l (standardised scale), so the
exact risk of the predictor can be computed (risk_metrics.exact_risk).

usage
    moments = guide_moments(store, L)            # store: param store or dict of constrained values
    out = closed_form_scores(moments, X_test_list, cov="diag")
    out2 = mc_scores(moments, X_test_list, n_draws=500, seed=0)
"""
import torch

from constants import Params


def guide_moments(store, L, dtype=torch.float64):
    """Moments of the trained guide needed for prediction.

    store : the Pyro param store (after set_state) or a dict of *constrained*
        parameter values. Read the store through pyro.param / store[name], not
        from the raw saved state, so positive parameters are on their natural scale.
    """
    get = lambda name: store[name].detach().to(dtype).cpu()
    a_y, b_y = get(Params.a_sigma_y), get(Params.b_sigma_y)
    return {
        "M": [get(Params.loc_Lambda_l.format(l=l)).squeeze() for l in range(L)],
        "S": [get(Params.scale_Lambda_l.format(l=l)).squeeze() ** 2 for l in range(L)],
        "a_psi": [get(Params.a_psi_l.format(l=l)) for l in range(L)],
        "b_psi": [get(Params.b_psi_l.format(l=l)) for l in range(L)],
        "m_beta": get(Params.loc_beta).reshape(-1),
        "s2_beta": get(Params.scale_beta).reshape(-1) ** 2,
        "a_sigma_y": a_y,
        "b_sigma_y": b_y,
        # E_q[sigma_y^2] exists only for a > 1
        "E_sigma2_y": b_y / (a_y - 1) if float(a_y) > 1 else torch.tensor(float("nan"), dtype=dtype),
    }


def load_guide_moments(paramstore_path, L):
    """Read a saved <stem>_paramstore.pth into the param store and return guide_moments."""
    import pyro
    pyro.clear_param_store()
    pyro.get_param_store().set_state(torch.load(paramstore_path, weights_only=False, map_location="cpu"))
    return guide_moments(pyro.get_param_store(), L)


def closed_form_scores(moments, X_list, cov="diag"):
    """Option 1: mean-field closed-form test scores and outcome prediction.

    X_list : list of (n, p_l) standardised test views (training statistics).
    Returns dict with
        m_z (n x K), cov_z (K x K, the same for every sample), A_bar (K x K),
        yhat (n), pred_var (n), coef (list of p_l vectors bs_l, yhat = sum_l X_l bs_l).
    """
    M, S, m_beta = moments["M"], moments["S"], moments["m_beta"]
    w = [a / b for a, b in zip(moments["a_psi"], moments["b_psi"])]       # E_q[1/psi]
    K = M[0].shape[1]
    A_bar = torch.eye(K, dtype=M[0].dtype)
    for M_l, S_l, w_l in zip(M, S, w):
        A_bar = A_bar + (M_l * w_l[:, None]).T @ M_l + torch.diag((w_l[:, None] * S_l).sum(0))
    A_inv = torch.linalg.inv(A_bar)
    h = sum(X_l.to(M_l.dtype) @ (M_l * w_l[:, None]) for X_l, M_l, w_l in zip(X_list, M, w))
    m_z = h @ A_inv                                    # A_bar symmetric
    if cov == "full":
        cov_z = A_inv
    elif cov == "diag":
        cov_z = torch.diag(1.0 / torch.diag(A_bar))
    else:
        raise ValueError(f"cov must be 'full' or 'diag', got {cov!r}")

    S_beta = torch.diag(moments["s2_beta"])
    yhat = m_z @ m_beta
    pred_var = (moments["E_sigma2_y"] + torch.trace(S_beta @ cov_z) + m_beta @ cov_z @ m_beta
                + ((m_z ** 2) * moments["s2_beta"]).sum(1))
    coef = [(M_l * w_l[:, None]) @ (A_inv @ m_beta) for M_l, w_l in zip(M, w)]
    return {"m_z": m_z, "cov_z": cov_z, "A_bar": A_bar, "yhat": yhat,
            "pred_var": pred_var, "coef": coef}


def mc_scores(moments, X_list, n_draws=500, seed=0):
    """Option 2: Monte Carlo over the global factors of q (exact z | x per draw).

    Returns dict with
        m_z (n x K) mixture mean, cov_z (n x K x K) mixture covariance per sample,
        yhat (n), pred_var (n), coef (list of p_l vectors: E[y | x] = sum_l X_l bs_l).
    """
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return _mc_scores(moments, X_list, n_draws)


def _mc_scores(moments, X_list, n_draws):
    Gamma = torch.distributions.Gamma
    M, S = moments["M"], moments["S"]
    dtype = M[0].dtype
    K = M[0].shape[1]
    X_list = [X_l.to(dtype) for X_l in X_list]
    n = X_list[0].shape[0]

    sum_m = torch.zeros(n, K, dtype=dtype)
    sum_mm = torch.zeros(n, K, K, dtype=dtype)
    sum_Ainv = torch.zeros(K, K, dtype=dtype)
    sum_y = torch.zeros(n, dtype=dtype)
    sum_yy = torch.zeros(n, dtype=dtype)
    sum_cond_var = 0.0
    sum_coef = [torch.zeros(M_l.shape[0], dtype=dtype) for M_l in M]

    for _ in range(n_draws):
        Lam = [M_l + S_l.sqrt() * torch.randn(M_l.shape, dtype=dtype) for M_l, S_l in zip(M, S)]
        # psi ~ InverseGamma(a, b)  <=>  1/psi ~ Gamma(a, rate b)
        prec = [Gamma(a, b).sample() for a, b in zip(moments["a_psi"], moments["b_psi"])]
        beta = moments["m_beta"] + moments["s2_beta"].sqrt() * torch.randn(K, dtype=dtype)
        sigma2_y = 1.0 / Gamma(moments["a_sigma_y"], moments["b_sigma_y"]).sample()

        B = [L_l * p_l[:, None] for L_l, p_l in zip(Lam, prec)]          # Psi^-1 Lambda
        A_s = torch.eye(K, dtype=dtype) + sum(L_l.T @ B_l for L_l, B_l in zip(Lam, B))
        A_inv = torch.linalg.inv(A_s)
        m_s = sum(X_l @ B_l for X_l, B_l in zip(X_list, B)) @ A_inv
        y_s = m_s @ beta

        sum_m += m_s
        sum_mm += m_s[:, :, None] * m_s[:, None, :]
        sum_Ainv += A_inv
        sum_y += y_s
        sum_yy += y_s ** 2
        sum_cond_var += float(beta @ A_inv @ beta + sigma2_y)
        v = A_inv @ beta
        for j, B_l in enumerate(B):
            sum_coef[j] += B_l @ v

    m_z = sum_m / n_draws
    cov_z = sum_Ainv / n_draws + sum_mm / n_draws - m_z[:, :, None] * m_z[:, None, :]
    yhat = sum_y / n_draws
    pred_var = (sum_yy / n_draws - yhat ** 2) + sum_cond_var / n_draws
    return {"m_z": m_z, "cov_z": cov_z, "yhat": yhat, "pred_var": pred_var,
            "coef": [c / n_draws for c in sum_coef]}


def expected_tau(store, L, dtype=torch.float64):
    """E_q[tau_lk] per view and column for the MGP prior (pruning check).

    tau_lk = prod_{h <= k} delta_lh with independent q(delta_lh) = Gamma(a, rate b),
    so E_q[tau_lk] = prod_{h <= k} a_lh / b_lh. Returns a list of K-vectors.
    """
    out = []
    for l in range(L):
        a = store[Params.a_delta_lambda_l.format(l=l)].detach().to(dtype).cpu()
        b = store[Params.b_delta_lambda_l.format(l=l)].detach().to(dtype).cpu()
        out.append(torch.cumprod(a / b, 0))
    return out
