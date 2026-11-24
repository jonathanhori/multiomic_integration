"""
Multiview Varimax and MatchAlign postprocessing for JAFAR-style factor models.

This module implements:
  1. regular_varimax     – single-view Kaiser Varimax via power-iteration SVD
  2. multiview_varimax   – the JAFAR multiview generalisation
  3. match_align         – sign/column-permutation alignment across posterior samples
  4. multiview_match_align – full postprocessing pipeline (Varimax then MatchAlign)

Mathematical background is documented inline.  See also:
  Anceschi et al. (2025) "Bayesian Joint Additive Factor Models for Multiview Learning"
  Poworoznek et al. (2025) "Efficiently Resolving Rotational Ambiguity in Bayesian Matrix
      Sampling with Matching", Bayesian Analysis.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# 1.  Single-view Varimax
# ---------------------------------------------------------------------------

def regular_varimax(
    L: np.ndarray,
    normalize: bool = False,
    eps: float = 1e-5,
    max_iter: int = 1000,
) -> dict:
    """
    Kaiser Varimax rotation for a single loading matrix.

    Parameters
    ----------
    L : (p, K) ndarray
        Loading matrix whose columns correspond to latent factors.
    normalize : bool
        If True, row-normalise L before optimisation (Kaiser normalisation)
        and un-normalise afterward.
    eps : float
        Convergence tolerance on the objective.
    max_iter : int
        Maximum number of power-iteration sweeps.

    Returns
    -------
    dict with keys
        "loadings" : (p, K) rotated loading matrix  L @ R
        "rotmat"   : (K, K) orthogonal rotation matrix R
        "obj"      : list of objective values per iteration

    Mathematical background
    -----------------------
    Varimax seeks an orthogonal rotation  R ∈ O(K)  that maximises the sum
    of within-column variances of squared loadings:

        V(R) = Σ_k  [ (1/p) Σ_j (LR)_{jk}^4
                      - ((1/p) Σ_j (LR)_{jk}^2)^2 ]

    Equivalently, using the identity  Var(u²) = E[u⁴] - (E[u²])² and
    absorbing the constant 1/p, this equals

        V(R) = (1/p) Σ_{jk} (LR)_{jk}^4
               - (1/p²) Σ_k  (Σ_j (LR)_{jk}^2)²

    Kaiser (1958) showed that the gradient of V with respect to R is

        ∇_R V  ∝  L^T [ (LR)³  −  (LR) diag( col-mean of (LR)² ) ]

    (where (LR)³ means element-wise cube and col-mean is taken over rows).

    The closed-form update from the stationary condition ∇_R V = λR is
    obtained by taking the SVD of the gradient matrix Q = L^T [·] and
    setting  R ← U V^T.  This is a projected-gradient step onto O(K) that
    monotonically increases V.
    """
    p, K = L.shape
    if K < 2:
        return {"loadings": L.copy(), "rotmat": np.eye(K), "obj": []}

    sc = None
    if normalize:
        sc = np.sqrt((L ** 2).sum(axis=1, keepdims=True))
        sc = np.where(sc == 0, 1.0, sc)
        L = L / sc

    R = np.eye(K)
    obj_vals = []

    for _ in range(max_iter):
        LR = L @ R          # (p, K)
        LR2 = LR ** 2       # (p, K)

        # Gradient of V w.r.t. R (dropping the irrelevant factor of 4):
        #
        #   ∂V/∂R  ∝  L^T [ w∘(LR)^3  -  w∘(LR) ∘ S ]
        #
        # where w_j = 1/p for all j (single-view) and
        # S_{jh} = (1/p) Σ_{j'} (LR)_{j'h}^2  (same for all j in this case).
        #
        # Setting w = 1/p everywhere, this simplifies to:
        #   Q = (L/p)^T [ (LR)^3  -  LR * col-mean((LR)^2) ]
        col_sum_LR2 = LR2.sum(axis=0)                    # (K,)
        Q = (L / p).T @ (LR2 * LR - LR * col_sum_LR2 / p)

        U, s, Vt = np.linalg.svd(Q, full_matrices=False)
        R = U @ Vt

        # Objective: tr(Q^T R) = Σ singular values when R = U V^T
        val = s.sum()
        obj_vals.append(val)

        if len(obj_vals) >= 2 and obj_vals[-1] < obj_vals[-2] * (1.0 + eps):
            break
    else:
        warnings.warn("regular_varimax: max_iter reached without convergence.")

    LR = L @ R
    if normalize and sc is not None:
        LR = LR * sc

    return {"loadings": LR, "rotmat": R, "obj": obj_vals}


# ---------------------------------------------------------------------------
# 2.  Multiview Varimax
# ---------------------------------------------------------------------------

def multiview_varimax(
    L: np.ndarray,
    p_views: list[int],
    normalize: bool = False,
    eps: float = 1e-5,
    max_iter: int = 1000,
) -> dict:
    """
    JAFAR multiview Varimax rotation.

    Parameters
    ----------
    L : (Σ p_l, K) ndarray
        Vertically stacked loading matrices for all M views:
            L = [ Λ^(1); Λ^(2); …; Λ^(M) ]
        where  Λ^(l)  has shape  (p_l, K).
    p_views : list of int, length M
        Number of features in each view.  Must sum to  L.shape[0].
    normalize, eps, max_iter : same as regular_varimax.

    Returns
    -------
    dict with keys "loadings", "rotmat", "obj"  (same conventions as above).

    Mathematical background
    -----------------------
    The multiview objective replaces the single normalisation 1/p with a
    per-view normalisation 1/p_l.  Concretely the objective is:

        V_mv(R) = Σ_l Σ_k
                  [ (1/p_l) Σ_{j∈view l} (LR)_{jk}^4
                    - ((1/p_l) Σ_{j∈view l} (LR)_{jk}^2)^2 ]

    Writing w_j = 1/p_{l(j)}  for the weight of feature j (where l(j) is
    the view that j belongs to), the gradient is

        ∂V_mv/∂R  ∝  (wL)^T [ (LR)^3  -  w*(LR) * S_l ]

    where  S_l  is a matrix that, for each row j in view l, holds the
    column-sums  Σ_{j'∈l} (LR)_{j'k}^2  (i.e. the per-view analog of the
    global col-sum used in regular Varimax).

    Concretely:
        S_l[j, k]  =  (1/p_l) Σ_{j'∈view l} (LR)_{j'k}^²,   j ∈ view l

    This is implemented by:
      1. computing  LR² = (LR)²,
      2. for each view l, summing LR² over the rows belonging to l,
      3. broadcasting that sum back to every row of that view,
      4. building the matrix  colSumsLR2  of shape (Σp_l, K)  by stacking
         those per-view sums.

    The resulting gradient matrix  Q = (w*L)^T [LR^3 - (w*LR)*colSumsLR2]
    has the same shape (K, K) as in regular Varimax, and the same SVD-based
    projected-gradient step applies:  R ← U V^T.

    Why does per-view normalisation matter?
    Views with many more features would otherwise dominate the objective,
    making the rotation driven entirely by the largest view.  Dividing by
    p_l gives each view equal influence, matching the scientific goal of
    balanced multi-omic integration.
    """
    M = len(p_views)
    assert sum(p_views) == L.shape[0], "sum(p_views) must equal L.shape[0]"
    p_tot, K = L.shape

    if K < 2:
        return {"loadings": L.copy(), "rotmat": np.eye(K), "obj": []}

    # Build row-weight vector w[j] = 1 / p_{l(j)}
    w = np.repeat(1.0 / np.array(p_views), p_views)   # (Σp_l,)

    # Row index ranges for each view
    ends = np.cumsum(p_views)
    starts = np.concatenate([[0], ends[:-1]])

    sc = None
    if normalize:
        sc = np.sqrt((L ** 2).sum(axis=1, keepdims=True))
        sc = np.where(sc == 0, 1.0, sc)
        L = L / sc

    R = np.eye(K)
    obj_vals = []

    for _ in range(max_iter):
        LR = L @ R          # (Σp_l, K)
        LR2 = LR ** 2       # (Σp_l, K)

        # Per-view column sums of LR², broadcast back to each row of that view.
        # colSumsLR2[j, k] = (1/p_l) Σ_{j' in view l} LR2[j', k],  j in view l
        colSumsLR2 = np.empty_like(LR2)
        for l in range(M):
            s, e = starts[l], ends[l]
            view_sum = LR2[s:e, :].sum(axis=0)          # (K,)
            colSumsLR2[s:e, :] = view_sum / p_views[l]  # broadcast over rows

        # Gradient of Eq.(3) from Anceschi et al. (2025) w.r.t. R, dropping
        # the irrelevant positive scalar factor of 4:
        #
        #   Q  =  Lambda^T [ w o (LambdaR)^3  -  w o (LambdaR) o S ]
        #
        # where w_j = 1/p_{l(j)}  and
        #   S_{jh} = (1/p_{l(j)}) sum_{j' in l(j)} (LambdaR)_{j'h}^2
        #          = colSumsLR2[j, h]
        #
        # First term:  d/dR  sum_{m,j,h} (1/p_m)(LambdaR)_{jh}^4
        #            = Lambda^T [w o (LambdaR)^3]
        # Second term: d/dR  sum_{m,h} c_{mh}^2,
        #              c_{mh} = (1/p_m) sum_{j in m}(LambdaR)_{jh}^2
        #            = Lambda^T [w o (LambdaR) o S]
        # Full gradient: Q = Lambda^T [w o (LambdaR)^3 - w o (LambdaR) o S]
        #                  = (wL)^T [(LR)^3 - (wLR) o S]
        wL = w[:, np.newaxis] * L      # (Σp_l, K)
        wLR = w[:, np.newaxis] * LR    # (Σp_l, K)
        Q = wL.T @ (LR * LR2 - wLR * colSumsLR2)

        U, s_vals, Vt = np.linalg.svd(Q, full_matrices=False)
        R = U @ Vt

        val = s_vals.sum()
        obj_vals.append(val)

        if len(obj_vals) >= 2 and obj_vals[-1] < obj_vals[-2] * (1.0 + eps):
            break
    else:
        warnings.warn("multiview_varimax: max_iter reached without convergence.")

    LR = L @ R
    if normalize and sc is not None:
        LR = LR * sc

    return {"loadings": LR, "rotmat": R, "obj": obj_vals}


# ---------------------------------------------------------------------------
# 3.  MatchAlign helpers
# ---------------------------------------------------------------------------

def _sign_and_permute(A: np.ndarray, pivot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the best column permutation + sign flip that aligns A with pivot.

    Parameters
    ----------
    A : (p, K) loading matrix to be aligned.
    pivot : (p, K) reference loading matrix.

    Returns
    -------
    perm   : (K,) integer array – column permutation
    signs  : (K,) array of ±1  – sign flips

    Algorithm  (msfOUT / aplr in the R code)
    -----------------------------------------
    We want to find a signed permutation matrix  P  (K×K, one ±1 per
    row/column) minimising  ‖A P − pivot‖_F².

    Expanding:  ‖AP − pivot‖_F² = ‖A‖_F² + ‖pivot‖_F² − 2 tr(pivot^T A P).

    Minimising the Frobenius norm is therefore equivalent to maximising
    tr(pivot^T A P) = Σ_k (pivot^T A)_{σ(k),k} * sign_k,
    where P encodes permutation σ and signs.

    For a fixed permutation σ, the optimal sign for column k is
        sign_k = sign( (pivot^T A)_{σ(k), k} )
    giving contribution  |(pivot^T A)_{σ(k), k}|.

    So the problem reduces to maximising  Σ_k |C_{σ(k), k}|  over
    permutations σ, where  C = |pivot^T A|.  This is a linear assignment
    problem solved by the Hungarian algorithm in O(K³).
    """
    K = A.shape[1]
    # C[i, k] = |<pivot_col_i, A_col_k>|
    C = np.abs(pivot.T @ A)     # (K, K)
    # Hungarian algorithm: maximise trace; scipy minimises, so negate
    row_ind, col_ind = linear_sum_assignment(-C)
    # row_ind is always [0,1,...,K-1]; col_ind is the permutation of A's columns
    perm = col_ind
    # Optimal signs for the chosen permutation
    signs = np.sign(np.diag(pivot.T @ A[:, perm]))
    signs = np.where(signs == 0, 1.0, signs)
    return perm, signs


def _apply_alignment(A: np.ndarray, perm: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Apply column permutation and sign flips to matrix A."""
    return A[:, perm] * signs[np.newaxis, :]


# ---------------------------------------------------------------------------
# 4.  Full multiview MatchAlign pipeline
# ---------------------------------------------------------------------------

def multiview_alignment(
    lambda_samples: list[np.ndarray],
    eta_samples: Optional[list[np.ndarray]] = None,
    theta_samples: Optional[list[np.ndarray]] = None,
    p_views: Optional[list[int]] = None,
    normalize: bool = False,
    eps: float = 1e-5,
    max_iter: int = 1000,
) -> dict:
    """
    Combined Varimax + MatchAlign postprocessing across posterior samples.

    Parameters
    ----------
    lambda_samples : list of T  (Σp_l, K) arrays
        One stacked loading matrix per posterior sample.
    eta_samples : list of T  (n, K) arrays, optional
        Factor score matrices to be co-rotated with the loadings.
    theta_samples : list of T  (K,) or (K, ?) arrays, optional
        Outcome coefficient vectors/matrices to be co-rotated.
    p_views : list of int, optional
        Feature counts per view.  If None, regular (single-view) Varimax is used.
    normalize, eps, max_iter : passed to varimax routine.

    Returns
    -------
    dict with keys:
        "lambda" : list of T rotated+aligned loading matrices
        "eta"    : list of T rotated+aligned factor scores (if provided)
        "theta"  : list of T rotated+aligned outcome coefficients (if provided)

    Algorithm overview
    ------------------
    Step 1 – Varimax rotation (per sample)
        For each posterior sample t, solve for the orthogonal rotation R_t
        that maximises the (multiview) Varimax criterion applied to Λ_t.
        Also rotate η_t ← η_t R_t  and  θ_t ← θ_t R_t.

        After this step each sample has been individually ortho-rotated to
        its own most-sparse representation, but columns can still be in
        different orders or have opposite signs across samples.

    Step 2 – Choose pivot
        Select a pivot sample  Λ_piv  as the sample whose spectral norm is
        closest to the median spectral norm across all samples.  This is a
        robust representative sample.

    Step 3 – MatchAlign (per sample)
        For each sample t, find the signed permutation P_t ∈ {±1}^K × S_K
        that minimises  ‖Λ_t P_t − Λ_piv‖_F  (equivalently, maximises
        alignment with the pivot).  Apply P_t consistently to Λ_t, η_t, θ_t.

        After this step all samples share the same column ordering and sign
        conventions, so that posterior summaries (means, credible intervals)
        are meaningful.

    Why this two-step approach?
        Varimax makes each sample's loadings as interpretable as possible by
        inducing column-wise sparsity.  MatchAlign then removes the remaining
        discrete label-switching ambiguity across samples.  Neither step alone
        is sufficient: Varimax without MatchAlign leaves sign/permutation
        inconsistency; MatchAlign without Varimax leaves the rotational
        ambiguity unresolved within each sample (loadings still dense).
    """
    T = len(lambda_samples)

    # ------------------------------------------------------------------
    # Step 1: Varimax per sample
    # ------------------------------------------------------------------
    rotated_loads = []
    rotated_eta   = [] if eta_samples is not None else None
    rotated_theta = [] if theta_samples is not None else None

    for t, Lam in enumerate(lambda_samples):
        if p_views is not None:
            res = multiview_varimax(Lam, p_views=p_views,
                                    normalize=normalize, eps=eps,
                                    max_iter=max_iter)
        else:
            res = regular_varimax(Lam, normalize=normalize, eps=eps,
                                  max_iter=max_iter)

        R = res["rotmat"]           # (K, K) orthogonal
        rotated_loads.append(res["loadings"])

        if eta_samples is not None:
            rotated_eta.append(eta_samples[t] @ R)      # (n, K)

        if theta_samples is not None:
            th = theta_samples[t]
            # theta may be (K,) or (K, ?) – rotate along first axis
            if th.ndim == 1:
                rotated_theta.append(th @ R)             # (K,) via dot
            else:
                rotated_theta.append(th @ R)             # (K, ?) generalisation

    # ------------------------------------------------------------------
    # Step 2: Choose pivot
    # ------------------------------------------------------------------
    spec_norms = np.array([np.linalg.norm(L, ord=2) for L in rotated_loads])
    median_norm = np.median(spec_norms)
    pivot_idx = int(np.argmin(np.abs(spec_norms - median_norm)))
    pivot = rotated_loads[pivot_idx]

    # ------------------------------------------------------------------
    # Step 3: MatchAlign per sample
    # ------------------------------------------------------------------
    aligned_loads = []
    aligned_eta   = [] if rotated_eta is not None else None
    aligned_theta = [] if rotated_theta is not None else None

    for t in range(T):
        perm, signs = _sign_and_permute(rotated_loads[t], pivot)

        aligned_loads.append(_apply_alignment(rotated_loads[t], perm, signs))

        if rotated_eta is not None:
            aligned_eta.append(_apply_alignment(rotated_eta[t], perm, signs))

        if rotated_theta is not None:
            th = rotated_theta[t]
            if th.ndim == 1:
                aligned_theta.append(th[perm] * signs)
            else:
                aligned_theta.append(th[perm, :] * signs[:, np.newaxis])

    out = {"lambda": aligned_loads}
    if aligned_eta is not None:
        out["eta"] = aligned_eta
    if aligned_theta is not None:
        out["theta"] = aligned_theta
    return out


# ---------------------------------------------------------------------------
# 5.  Top-level convenience wrapper (mirrors multiviewMatchAlign in R)
# ---------------------------------------------------------------------------

def multiview_match_align(
    Lambda_m_samples: list[list[np.ndarray]],   # [view][sample] -> (K, p_l) array  (R convention: K rows)
    eta_samples: Optional[list[np.ndarray]] = None,
    theta_samples: Optional[list[np.ndarray]] = None,
    Gamma_m_samples: Optional[list[list[np.ndarray]]] = None,
    phi_m_samples: Optional[list[list[np.ndarray]]] = None,
    theta_m_samples: Optional[list[list[np.ndarray]]] = None,
    p_views: Optional[list[int]] = None,
) -> dict:
    """
    High-level wrapper matching the R function ``multiviewMatchAlign``.

    Conventions
    -----------
    Following Python/numpy convention (observations in rows), all loading
    matrices are expected as  (p_l, K) arrays.

    Parameters
    ----------
    Lambda_m_samples : list (length M) of list (length T) of (p_l, K) arrays
        Posterior samples of shared loading matrices, one list per view.
    eta_samples : list of T  (n, K) arrays, optional
        Posterior samples of shared factor scores.
    theta_samples : list of T  (K,) or (1, K) arrays, optional
        Posterior samples of shared outcome regression coefficients.
    Gamma_m_samples : list (M) of list (T) of (p_l, K_l) arrays, optional
        View-specific loading matrix samples.
    phi_m_samples : list (M) of list (T) of (n, K_l) arrays, optional
        View-specific factor score samples.
    theta_m_samples : list (M) of list (T) of (K_l,) arrays, optional
        View-specific outcome coefficient samples.
    p_views : list of int, optional
        Feature counts per view (auto-inferred from Lambda_m_samples if None).

    Returns
    -------
    dict with keys "Lambda_m", "eta", "theta" (shared),
    and optionally "Gamma_m", "phi_m", "theta_m" (view-specific).
    """
    M = len(Lambda_m_samples)
    T = len(Lambda_m_samples[0])

    if p_views is None:
        p_views = [Lambda_m_samples[m][0].shape[0] for m in range(M)]

    # Stack shared loadings: for each sample t, stack over views -> (Σp_l, K)
    stacked_samples = [
        np.vstack([Lambda_m_samples[m][t] for m in range(M)])
        for t in range(T)
    ]

    theta_list = None
    if theta_samples is not None:
        theta_list = [
            (th.ravel() if th.ndim > 1 else th) for th in theta_samples
        ]

    result_shared = multiview_alignment(
        stacked_samples,
        eta_samples=eta_samples,
        theta_samples=theta_list,
        p_views=p_views,
    )

    # Unstack shared loadings back to per-view
    ends = np.cumsum(p_views)
    starts = np.concatenate([[0], ends[:-1]])

    Lambda_m_out = [
        [result_shared["lambda"][t][starts[m]:ends[m], :] for t in range(T)]
        for m in range(M)
    ]

    out = {
        "Lambda_m": Lambda_m_out,
        "eta":      result_shared.get("eta"),
        "theta":    result_shared.get("theta"),
    }

    # View-specific postprocessing (regular Varimax + MatchAlign per view)
    if Gamma_m_samples is not None:
        Gamma_m_out  = []
        phi_m_out    = []
        theta_m_out  = [] if theta_m_samples is not None else None

        for m in range(M):
            th_m = theta_m_samples[m] if theta_m_samples is not None else None
            res_m = multiview_alignment(
                Gamma_m_samples[m],
                eta_samples=phi_m_samples[m] if phi_m_samples is not None else None,
                theta_samples=th_m,
                p_views=None,   # single-view Varimax for view-specific part
            )
            Gamma_m_out.append(res_m["lambda"])
            phi_m_out.append(res_m.get("eta"))
            if theta_m_out is not None:
                theta_m_out.append(res_m.get("theta"))

        out["Gamma_m"] = Gamma_m_out
        out["phi_m"]   = phi_m_out
        if theta_m_out is not None:
            out["theta_m"] = theta_m_out

    return out


# ---------------------------------------------------------------------------
# Minimal smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    n, K = 50, 4
    p_views = [20, 30, 25]
    P = sum(p_views)
    T = 10   # number of posterior samples

    # Simulate T posterior samples of loading matrices + factor scores
    lambda_samples = [rng.standard_normal((P, K)) for _ in range(T)]
    eta_samples    = [rng.standard_normal((n, K)) for _ in range(T)]
    theta_samples  = [rng.standard_normal(K)       for _ in range(T)]

    # -- multiview Varimax on a single sample --
    res = multiview_varimax(lambda_samples[0], p_views=p_views)
    print("Varimax obj final:", res["obj"][-1])
    print("Rotmat orthogonality check (should be ~I):\n",
          np.round(res["rotmat"] @ res["rotmat"].T, 4))

    # -- Full MatchAlign pipeline --
    aligned = multiview_alignment(
        lambda_samples,
        eta_samples=eta_samples,
        theta_samples=theta_samples,
        p_views=p_views,
    )
    print("\nAligned lambda[0] shape:", aligned["lambda"][0].shape)
    print("Aligned eta[0]    shape:", aligned["eta"][0].shape)
    print("Aligned theta[0]  shape:", aligned["theta"][0].shape)
    print("Smoke test passed.")
