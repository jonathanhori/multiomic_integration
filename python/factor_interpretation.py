"""
factor_interpretation.py

Reusable helpers for interpreting latent factors from MatrixDecomp and
SupMultiviewShared models: proportion of variance explained (PVE), MOFA-style
R² per factor, loading heatmaps, score scatter plots, and feature tables.

All functions accept numpy arrays or torch Tensors and work for both single-
view (pass a one-element list) and multiview (pass a list per view) inputs.
"""
from __future__ import annotations

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Union

from multiview_varimax import multiview_varimax, regular_varimax


# ── internal conversion helpers ───────────────────────────────────────────────

def _np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _t(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(np.asarray(x), dtype=torch.float32)


def _rotate_loadings(Lambda_list: List) -> tuple:
    """
    Apply multiview varimax to a list of per-view loading matrices.

    For a single view, regular_varimax is used (equivalent objective).
    For multiple views, all loading matrices are stacked, rotated jointly with
    multiview_varimax (which weights each view equally regardless of feature
    count), then unstacked back to per-view.

    Returns
    -------
    rotated_list : list of (p_l, K) rotated loading matrices
    R            : (K, K) shared orthogonal rotation matrix
    """
    arrays = [_np(L) for L in Lambda_list]

    if len(arrays) == 1:
        res = regular_varimax(arrays[0])
        return [res["loadings"]], res["rotmat"]

    p_views   = [L.shape[0] for L in arrays]
    L_stacked = np.vstack(arrays)                         # (Σp_l, K)
    res       = multiview_varimax(L_stacked, p_views=p_views)

    ends   = np.cumsum(p_views)
    starts = np.concatenate([[0], ends[:-1]])
    rotated_list = [res["loadings"][starts[l]:ends[l], :] for l in range(len(arrays))]

    return rotated_list, res["rotmat"]


# ── Proportion of Variance Explained ─────────────────────────────────────────

def compute_pve(Lambda_list: List) -> dict:
    """
    Proportion of variance explained per factor, per view and combined.

    For each view l:
        pve_l[k] = sum_i Lambda_l[i,k]^2 / sum_{i,k} Lambda_l[i,k]^2

    Combined: treat all views as one concatenated loading matrix.

    Parameters
    ----------
    Lambda_list : list of (p_l, K) array-like — factor loadings per view

    Returns
    -------
    dict with keys:
        "per_view"  : list of (K,) numpy arrays — per-view PVE
        "combined"  : (K,) numpy array — combined PVE across all views
    """
    arrays = [_np(L) for L in Lambda_list]
    ssl_per_view = [np.sum(L ** 2, axis=0) for L in arrays]  # list of (K,)

    per_view_pve = [s / s.sum() for s in ssl_per_view]
    combined_ssl = np.sum(ssl_per_view, axis=0)               # (K,)
    combined_pve = combined_ssl / combined_ssl.sum()

    return {"per_view": per_view_pve, "combined": combined_pve}


# ── R² per factor ─────────────────────────────────────────────────────────────

def compute_r2_per_factor(
    X_list: List,
    Z: Union[np.ndarray, torch.Tensor],
    Lambda_list: List,
) -> dict:
    """
    MOFA-style R² per factor for each view and combined.

    For factor k in view l:
        signal_k = z_k ⊗ λ_lk^T  (rank-1 approximation)
        R²_l[k]  = 1 - ||X_l - signal_k||_F² / ||X_l||_F²

    X_list should contain mean-centred data (as produced by process_data_*).

    Parameters
    ----------
    X_list      : list of (n, p_l) array-like — observed data (mean-centred)
    Z           : (n, K) array-like — posterior mean factor scores
    Lambda_list : list of (p_l, K) array-like — factor loadings per view

    Returns
    -------
    dict with keys:
        "per_view"  : list of (K,) numpy arrays — per-view R²
        "combined"  : (K,) numpy array — combined R² across all views
    """
    Z_t = _t(Z)
    rss_list, tss_list, per_view_r2 = [], [], []

    for X, Lambda in zip(X_list, Lambda_list):
        X_t, L_t = _t(X), _t(Lambda)
        TSS = (X_t ** 2).sum()

        # signal: (n, p, K) — rank-1 contribution of each factor
        signal   = Z_t.unsqueeze(1) * L_t.unsqueeze(0)  # (n, p, K)
        residual = X_t.unsqueeze(2) - signal              # (n, p, K)
        RSS      = (residual ** 2).sum(dim=(0, 1))        # (K,)

        tss_list.append(TSS)
        rss_list.append(RSS)
        per_view_r2.append((1.0 - RSS / TSS).detach().numpy())

    combined_r2 = (
        1.0 - torch.stack(rss_list).sum(0) / torch.stack(tss_list).sum()
    ).detach().numpy()

    return {"per_view": per_view_r2, "combined": combined_r2}


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_pve(
    Lambda_list: List,
    view_names: Optional[List[str]] = None,
    rotate: bool = False,
    title: Optional[str] = None,
) -> dict:
    """
    Bar chart of PVE per factor with per-view and combined panels.

    Parameters
    ----------
    Lambda_list : list of (p_l, K) array-like
    view_names  : optional view labels; defaults to "View 0", "View 1", …
    rotate      : if True, apply varimax before computing PVE
    title       : figure title

    Returns
    -------
    pve dict — see compute_pve
    """
    arrays = [_np(L) for L in Lambda_list]
    if rotate:
        arrays, _ = _rotate_loadings(arrays)

    pve = compute_pve(arrays)
    L_views = len(arrays)
    K = len(pve["combined"])
    factors = np.arange(1, K + 1)

    if view_names is None:
        view_names = [f"View {l}" for l in range(L_views)]

    panel_data = [(view_names[l], pve["per_view"][l]) for l in range(L_views)]
    if L_views > 1:
        panel_data.append(("Combined", pve["combined"]))

    n_panels = len(panel_data)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), squeeze=False)

    for ax, (name, pv) in zip(axes.flatten(), panel_data):
        ax.bar(factors, pv)
        ax.set_xlabel("Factor")
        ax.set_ylabel("PVE")
        ax.set_title(name)

    rot_note = " (varimax-rotated)" if rotate else ""
    fig.suptitle(title or f"Proportion of Variance Explained per Factor{rot_note}")
    plt.tight_layout()
    plt.show()
    return pve


def plot_r2(
    X_list: List,
    Z: Union[np.ndarray, torch.Tensor],
    Lambda_list: List,
    view_names: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> dict:
    """
    Bar chart of MOFA-style R² per factor with per-view and combined panels.

    Parameters
    ----------
    X_list      : list of (n, p_l) array-like (mean-centred)
    Z           : (n, K) array-like — factor scores
    Lambda_list : list of (p_l, K) array-like
    view_names  : optional view labels
    title       : figure title

    Returns
    -------
    r2 dict — see compute_r2_per_factor
    """
    r2 = compute_r2_per_factor(X_list, Z, Lambda_list)
    L_views = len(X_list)
    K = len(r2["combined"])
    factors = np.arange(1, K + 1)

    if view_names is None:
        view_names = [f"View {l}" for l in range(L_views)]

    panel_data = [(view_names[l], r2["per_view"][l]) for l in range(L_views)]
    if L_views > 1:
        panel_data.append(("Combined", r2["combined"]))

    n_panels = len(panel_data)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), squeeze=False)

    for ax, (name, rv) in zip(axes.flatten(), panel_data):
        ax.bar(factors, rv)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("Factor")
        ax.set_ylabel("R²")
        ax.set_title(name)

    fig.suptitle(title or "R² per Factor (MOFA-style)")
    plt.tight_layout()
    plt.show()
    return r2


def plot_factor_loadings(
    Lambda_list: List,
    view_names: Optional[List[str]] = None,
    rotate: bool = True,
    n_factors: Optional[int] = None,
    title: Optional[str] = None,
) -> List[np.ndarray]:
    """
    Heatmap of factor loadings, one panel per view.

    Parameters
    ----------
    Lambda_list : list of (p_l, K) array-like
    view_names  : optional view labels
    rotate      : if True, apply varimax rotation first
    n_factors   : if set, display only the first n_factors columns
    title       : figure title

    Returns
    -------
    list of (rotated) loading matrices as numpy arrays
    """
    arrays = [_np(L) for L in Lambda_list]
    L_views = len(arrays)

    if view_names is None:
        view_names = [f"View {l}" for l in range(L_views)]

    if rotate:
        rotated_list, _ = _rotate_loadings(arrays)
    else:
        rotated_list = [L.copy() for L in arrays]

    if n_factors is not None:
        rotated_list = [L[:, :n_factors] for L in rotated_list]

    fig, axes = plt.subplots(1, L_views, figsize=(6 * L_views, 6), squeeze=False)
    rot_suffix = " (varimax)" if rotate else ""

    for ax, name, L_rot in zip(axes.flatten(), view_names, rotated_list):
        sns.heatmap(L_rot, cmap="coolwarm", center=0, ax=ax,
                    xticklabels=True, yticklabels=False)
        ax.set_title(f"{name} loadings{rot_suffix}")
        ax.set_xlabel("Factor")
        ax.set_ylabel("Feature")

    fig.suptitle(title or f"Factor Loadings{rot_suffix}")
    plt.tight_layout()
    plt.show()
    return rotated_list


def plot_factor_scores_2d(
    Z: Union[np.ndarray, torch.Tensor],
    y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    top_factors: Optional[List[int]] = None,
    n_factors: int = 5,
    y_label: str = "outcome",
    title: Optional[str] = None,
) -> None:
    """
    Pairwise scatterplot matrix of factor scores, optionally coloured by outcome y.
    Diagonal panels show per-factor histograms; off-diagonal panels show scatter plots.

    Parameters
    ----------
    Z           : (n, K) array-like — factor scores
    y           : optional (n,) array-like for colouring off-diagonal scatter points
    top_factors : ordered factor indices to include; defaults to the first n_factors
    n_factors   : number of factors to include in the matrix (ignored if top_factors given)
    y_label     : colour-bar label when y is supplied
    title       : figure title
    """
    Z_np = _np(Z)
    K = Z_np.shape[1]

    if top_factors is None:
        top_factors = list(range(min(K, n_factors)))
    else:
        top_factors = list(top_factors)[:n_factors]

    m = len(top_factors)
    if m < 2:
        return

    y_np = _np(y) if y is not None else None
    vmin = float(y_np.min()) if y_np is not None else None
    vmax = float(y_np.max()) if y_np is not None else None

    fig, axes = plt.subplots(m, m, figsize=(2.5 * m, 2.5 * m), squeeze=False)

    sc_ref = None
    for row in range(m):
        for col in range(m):
            ax = axes[row, col]
            fi = top_factors[row]
            fj = top_factors[col]
            if row == col:
                ax.hist(Z_np[:, fi], bins=20, color="steelblue", alpha=0.7,
                        edgecolor="none")
                ax.set_yticks([])
            else:
                if y_np is not None:
                    sc = ax.scatter(Z_np[:, fj], Z_np[:, fi],
                                    c=y_np, cmap="viridis", s=10, alpha=0.6,
                                    vmin=vmin, vmax=vmax)
                    sc_ref = sc
                else:
                    ax.scatter(Z_np[:, fj], Z_np[:, fi], s=10, alpha=0.6)

            if row == m - 1:
                ax.set_xlabel(f"F{fj + 1}")
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(f"F{fi + 1}")
            else:
                ax.set_yticklabels([])

    if y_np is not None and sc_ref is not None:
        fig.colorbar(sc_ref, ax=axes, label=y_label, shrink=0.6, pad=0.02)

    fig.suptitle(title or "Factor score scatterplot matrix", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_factor_outcome(
    Z: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    top_factors: Optional[List[int]] = None,
    n_factors_show: int = 6,
    outcome_label: str = "outcome",
    title: Optional[str] = None,
) -> None:
    """
    For each factor, plot outcome values along samples ranked by factor score.

    This reveals whether the score-outcome relationship is monotone or
    non-linear. A rolling-mean trend line is overlaid.

    Parameters
    ----------
    Z              : (n, K) array-like — factor scores
    y              : (n,) array-like — outcome values
    top_factors    : ordered factor indices; defaults to 0 … n_factors_show
    n_factors_show : number of factors to plot
    outcome_label  : y-axis label
    title          : figure title
    """
    Z_np, y_np = _np(Z), _np(y).squeeze()
    K = Z_np.shape[1]

    if top_factors is None:
        top_factors = list(range(K))
    show = top_factors[:n_factors_show]

    ncols = min(3, len(show))
    nrows = (len(show) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for plot_i, f_idx in enumerate(show):
        ax = axes_flat[plot_i]
        order = np.argsort(Z_np[:, f_idx])
        sc = ax.scatter(
            np.arange(len(order)), y_np[order],
            c=Z_np[order, f_idx], cmap="coolwarm", s=14, alpha=0.7,
        )
        plt.colorbar(sc, ax=ax, label="score")
        window = max(1, len(order) // 10)
        smoothed = (
            pd.Series(y_np[order])
            .rolling(window, center=True, min_periods=1)
            .mean()
        )
        ax.plot(np.arange(len(order)), smoothed, color="black", lw=1.5)
        ax.set_xlabel(f"Sample rank by Factor {f_idx + 1}")
        ax.set_ylabel(outcome_label)
        ax.set_title(f"Factor {f_idx + 1}")

    for ax in axes_flat[len(show):]:
        ax.set_visible(False)

    fig.suptitle(title or f"{outcome_label} vs. ranked factor score")
    plt.tight_layout()
    plt.show()


# ── Factor summary metrics ────────────────────────────────────────────────────

def _factor_metrics_from_arrays(
    X_list_np: List[np.ndarray],
    Z_np: np.ndarray,
    Lambda_np_list: List[np.ndarray],
    beta_np: Optional[np.ndarray] = None,
    y_np: Optional[np.ndarray] = None,
    beta_lower_np: Optional[np.ndarray] = None,
    beta_upper_np: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute scalar and per-factor variance-explained metrics.

    Parameters
    ----------
    X_list_np       : list of (n, p_l) float32 arrays (mean-centred)
    Z_np            : (n, K) float32 array — factor scores
    Lambda_np_list  : list of (p_l, K) float32 arrays — factor loadings
    beta_np         : (K,) array — outcome regression coefficients (optional)
    y_np            : (n,) array — outcome values (optional; needed for
                      beta_r2_y and total_outcome_r2)
    beta_lower_np   : (K,) array — lower bound of 95% CI for each β_k
    beta_upper_np   : (K,) array — upper bound of 95% CI for each β_k

    Returns
    -------
    dict with keys:
        total_model_r2    : float  — (TSS - RSS) / TSS
        total_signal_r2   : float  — Σ_k ESS_k / TSS
        per_view_model_r2 : list of float  — per-view R²
        total_outcome_r2  : float or None — 1 - ||y - Zβ||² / Var(y); None if
                            beta or y not supplied
        per_factor        : DataFrame sorted by pve_data desc; cols:
                            factor, ess, pve_factor, pve_model, pve_data,
                            r2_mofa, beta, beta_lower, beta_upper,
                            beta_pve, beta_r2_y
                            (beta cols are NaN when beta_np not supplied;
                             beta_lower/upper are NaN when not supplied)
        cumulative_pve_data : (K,) array  — cumsum of pve_data sorted desc
    """
    K = Z_np.shape[1]

    TSS_list, RSS_list = [], []
    for X, L in zip(X_list_np, Lambda_np_list):
        TSS_list.append(np.sum(X ** 2))
        resid = X - Z_np @ L.T
        RSS_list.append(np.sum(resid ** 2))

    TSS = sum(TSS_list)
    RSS = sum(RSS_list)

    total_model_r2 = float(1.0 - RSS / TSS) if TSS > 0 else 0.0
    per_view_r2 = [
        float(1.0 - RSS_list[l] / TSS_list[l]) if TSS_list[l] > 0 else 0.0
        for l in range(len(X_list_np))
    ]

    # ESS_k = Σ_l ||z_k||² ||λ_{lk}||²
    z_sq = np.sum(Z_np ** 2, axis=0)        # (K,)
    ESS = np.zeros(K)
    for L in Lambda_np_list:
        ESS += z_sq * np.sum(L ** 2, axis=0)

    total_signal = ESS.sum()
    total_signal_r2 = float(total_signal / TSS) if TSS > 0 else 0.0

    model_explained = TSS - RSS
    pve_factor = ESS / total_signal if total_signal > 0 else np.zeros(K)
    pve_model  = ESS / model_explained if model_explained > 0 else np.zeros(K)
    pve_data   = ESS / TSS if TSS > 0 else np.zeros(K)

    r2_mofa = compute_r2_per_factor(X_list_np, Z_np, Lambda_np_list)["combined"]

    sort_order = np.argsort(ESS)[::-1]
    cumulative_pve_data = np.cumsum(pve_data[sort_order])

    # ── outcome importance (optional) ─────────────────────────────────────────
    # beta_ess_k = β_k² ||z_k||²  — rank-1 signal energy of factor k in ŷ.
    # Exact decomposition of ||ŷ||² when Z has orthogonal columns.
    beta_col    = np.full(K, np.nan)
    beta_lower  = np.full(K, np.nan)
    beta_upper  = np.full(K, np.nan)
    beta_pve    = np.full(K, np.nan)
    beta_r2_y   = np.full(K, np.nan)
    total_outcome_r2 = None

    if beta_np is not None:
        b = np.asarray(beta_np, dtype=np.float32).ravel()  # (K,)
        beta_col = b
        beta_ess = b ** 2 * z_sq                            # (K,)
        beta_total = beta_ess.sum()
        beta_pve = beta_ess / beta_total if beta_total > 0 else np.zeros(K)

        if beta_lower_np is not None:
            beta_lower = np.asarray(beta_lower_np, dtype=np.float32).ravel()
            beta_upper = np.asarray(beta_upper_np, dtype=np.float32).ravel()

        if y_np is not None:
            y_c   = np.asarray(y_np, dtype=np.float32).ravel()
            y_c   = y_c - y_c.mean()                        # centre y
            TSS_y = float(np.sum(y_c ** 2))
            y_hat = Z_np @ b
            y_hat = y_hat - y_hat.mean()
            RSS_y = float(np.sum((y_c - y_hat) ** 2))
            total_outcome_r2 = float(1.0 - RSS_y / TSS_y) if TSS_y > 0 else 0.0
            beta_r2_y = beta_ess / TSS_y if TSS_y > 0 else np.zeros(K)

    per_factor = (
        pd.DataFrame({
            "factor":     np.arange(1, K + 1).astype(int),
            "ess":        ESS,
            "pve_factor": pve_factor,
            "pve_model":  pve_model,
            "pve_data":   pve_data,
            "r2_mofa":    r2_mofa,
            "beta":       beta_col,
            "beta_lower": beta_lower,
            "beta_upper": beta_upper,
            "beta_pve":   beta_pve,
            "beta_r2_y":  beta_r2_y,
        })
        .sort_values("pve_data", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "total_model_r2":      total_model_r2,
        "total_signal_r2":     total_signal_r2,
        "per_view_model_r2":   per_view_r2,
        "total_outcome_r2":    total_outcome_r2,
        "per_factor":          per_factor,
        "cumulative_pve_data": cumulative_pve_data,
    }


def _print_factor_summary(
    label: str,
    metrics: dict,
    view_names: Optional[List[str]] = None,
    n_show: Optional[int] = None,
) -> None:
    """Print a formatted factor summary table."""
    df = metrics["per_factor"]
    has_beta = not df["beta_pve"].isna().all()
    has_ci   = has_beta and not df["beta_lower"].isna().all()
    has_r2y  = not df["beta_r2_y"].isna().all()

    sep_width = 64
    if has_beta: sep_width += 24
    if has_ci:   sep_width += 20
    if has_r2y:  sep_width += 10
    sep = "=" * sep_width
    print(f"\n{sep}")
    print(f"  Factor Summary — {label}")
    print(sep)
    print(f"  Total model R²          : {metrics['total_model_r2']:.4f}")
    print(f"  Total signal / TSS      : {metrics['total_signal_r2']:.4f}")
    for l, r2 in enumerate(metrics["per_view_model_r2"]):
        name = view_names[l] if view_names else f"View {l}"
        print(f"  R² [{name:<20s}]  : {r2:.4f}")
    if metrics.get("total_outcome_r2") is not None:
        print(f"  Total outcome R²        : {metrics['total_outcome_r2']:.4f}")

    if n_show is not None:
        df = df.head(n_show)

    print(f"\n  Per-factor breakdown (sorted by data PVE):")
    header = (
        f"  {'Factor':>6}  {'ESS':>10}  {'PVE(factor)':>11}  "
        f"{'PVE(model)':>10}  {'PVE(data)':>9}  {'R²(MOFA)':>8}"
    )
    row_fmt = (
        "  {factor:>6}  {ess:>10.2f}  {pve_factor:>11.4f}  "
        "{pve_model:>10.4f}  {pve_data:>9.4f}  {r2_mofa:>8.4f}"
    )
    if has_beta:
        header  += f"  {'β':>8}  {'β-PVE':>7}"
        row_fmt += "  {beta:>8.4f}  {beta_pve:>7.4f}"
    if has_ci:
        header  += f"  {'95% CI':>18}  {'':>1}"
        row_fmt += "  [{beta_lower:>7.4f},{beta_upper:>7.4f}]  {sig:>1}"
    if has_r2y:
        header  += f"  {'β-R²(y)':>8}"
        row_fmt += "  {beta_r2_y:>8.4f}"

    print(header)
    print("  " + "-" * (sep_width - 2))
    for _, row in df.iterrows():
        d = row.to_dict()
        d["factor"] = int(d["factor"])
        if has_ci:
            d["sig"] = "*" if (d["beta_lower"] > 0 or d["beta_upper"] < 0) else " "
        print(row_fmt.format(**d))

    cum = metrics["cumulative_pve_data"]
    cum_str = "  ".join(f"{v:.3f}" for v in cum)
    print(f"\n  Cumulative PVE (data): [{cum_str}]")
    if has_ci:
        print(f"  * 95% CI excludes zero")
    print(sep)


def _plot_cumulative_pve(pre: dict, post: dict) -> None:
    """Side-by-side chart: per-factor PVE bars and cumulative PVE lines."""
    K = len(pre["cumulative_pve_data"])
    ranks = np.arange(1, K + 1)

    pre_sorted  = pre["per_factor"]["pve_data"].values
    post_sorted = post["per_factor"]["pve_data"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    w = 0.35
    ax.bar(ranks - w / 2, pre_sorted,  w, label="Unrotated", alpha=0.8)
    ax.bar(ranks + w / 2, post_sorted, w, label="Varimax",   alpha=0.8)
    ax.set_xlabel("Factor rank (by PVE)")
    ax.set_ylabel("PVE relative to sample variance")
    ax.set_title("Per-factor variance explained")
    ax.legend()

    ax = axes[1]
    ax.plot(ranks, pre["cumulative_pve_data"],  marker="o", label="Unrotated")
    ax.plot(ranks, post["cumulative_pve_data"], marker="s", label="Varimax")
    ax.set_xlabel("Number of factors (by PVE)")
    ax.set_ylabel("Cumulative PVE")
    ax.set_title("Cumulative variance explained")
    ax.set_ylim(0, None)
    ax.legend()

    plt.suptitle("Factor variance explained — Unrotated vs Varimax")
    plt.tight_layout()
    plt.show()


def summarize_factors(
    X_list: List,
    Z: Union[np.ndarray, torch.Tensor],
    Lambda_list: List,
    beta: Optional[Union[np.ndarray, torch.Tensor]] = None,
    y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    beta_samples: Optional[Union[np.ndarray, torch.Tensor]] = None,
    view_names: Optional[List[str]] = None,
    plot: bool = True,
    verbose: bool = True,
    n_show: Optional[int] = None,
) -> dict:
    """
    Postprocessing summary for MatrixDecomp or SupMultiviewShared models.

    Computes variance-explained metrics before and after varimax rotation
    (single-view: regular varimax; multiview: joint multiview varimax).

    Parameters
    ----------
    X_list        : list of (n, p_l) array-like — mean-centred observed data
    Z             : (n, K) array-like — posterior mean factor scores
    Lambda_list   : list of (p_l, K) array-like — factor loadings per view
    beta          : (K,) array-like — posterior mean regression coefficients
                    (optional). When supplied, beta_pve and beta_r2_y columns
                    are added to per_factor. After rotation, beta is rotated
                    as R^T @ beta.
    y             : (n,) array-like — outcome values (optional; needed for
                    beta_r2_y and total_outcome_r2)
    beta_samples  : (S, K) array-like — posterior draws of β (optional).
                    When supplied, 95% credible intervals are computed as the
                    2.5th / 97.5th sample quantiles and stored as beta_lower /
                    beta_upper in per_factor.  For the post-rotation summary,
                    each draw is rotated first (samples @ R) before taking
                    quantiles, so the CIs correctly reflect the rotated
                    posterior.
    view_names    : optional view labels; defaults to "View 0", …
    plot          : if True, display PVE bar charts, R² bar charts, and
                    cumulative PVE comparison
    verbose       : if True, print a formatted summary table
    n_show        : rows to print in per-factor table (None = all)

    Returns
    -------
    dict with keys:
        "pre"  : metrics dict before rotation  (see _factor_metrics_from_arrays)
        "post" : metrics dict after rotation
        "R"    : (K, K) numpy array — orthogonal varimax rotation matrix

    Metrics dict keys (both pre and post):
        total_model_r2     — (TSS - RSS) / TSS
        total_signal_r2    — Σ_k ESS_k / TSS (≈ total_model_r2 when Z orthogonal)
        per_view_model_r2  — list of per-view R²
        total_outcome_r2   — 1 - ||y - Zβ||² / Var(y); None if beta/y not given
        per_factor         — DataFrame sorted by pve_data desc; columns:
                             factor, ess, pve_factor, pve_model, pve_data,
                             r2_mofa, beta, beta_lower, beta_upper,
                             beta_pve, beta_r2_y
                             (beta columns are NaN when beta not supplied;
                              beta_lower/upper are NaN when beta_samples not given)
        cumulative_pve_data — cumsum of pve_data sorted desc
    """
    X_np_list = [_np(X) for X in X_list]
    Z_np      = _np(Z)
    L_np_list = [_np(L) for L in Lambda_list]
    beta_np   = _np(beta).ravel() if beta is not None else None
    y_np      = _np(y).ravel()    if y    is not None else None

    if beta_samples is not None:
        samp_np = _np(beta_samples)           # (S, K)
        if samp_np.ndim == 1:
            samp_np = samp_np[np.newaxis, :]  # handle single-sample edge case
        pre_lower = np.quantile(samp_np, 0.025, axis=0)
        pre_upper = np.quantile(samp_np, 0.975, axis=0)
    else:
        samp_np = pre_lower = pre_upper = None

    if view_names is None:
        view_names = [f"View {l}" for l in range(len(X_list))]

    pre = _factor_metrics_from_arrays(X_np_list, Z_np, L_np_list,
                                      beta_np, y_np, pre_lower, pre_upper)

    L_rot_list, R = _rotate_loadings(L_np_list)
    Z_rot    = Z_np @ R
    beta_rot = R.T @ beta_np if beta_np is not None else None

    if samp_np is not None:
        samp_rot  = samp_np @ R               # (S, K) — rotate each draw
        post_lower = np.quantile(samp_rot, 0.025, axis=0)
        post_upper = np.quantile(samp_rot, 0.975, axis=0)
    else:
        post_lower = post_upper = None

    post = _factor_metrics_from_arrays(X_np_list, Z_rot, L_rot_list,
                                       beta_rot, y_np, post_lower, post_upper)

    if verbose:
        _print_factor_summary("Unrotated",       pre,  view_names, n_show)
        _print_factor_summary("Varimax-rotated", post, view_names, n_show)

    if plot:
        plot_pve(L_np_list,  view_names=view_names, rotate=False,
                 title="PVE per factor (unrotated)")
        plot_pve(L_rot_list, view_names=view_names, rotate=False,
                 title="PVE per factor (varimax-rotated)")
        plot_r2(X_np_list, Z_np,  L_np_list,  view_names=view_names,
                title="R² per factor (unrotated)")
        plot_r2(X_np_list, Z_rot, L_rot_list, view_names=view_names,
                title="R² per factor (varimax-rotated)")
        _plot_cumulative_pve(pre, post)

    return {"pre": pre, "post": post, "R": R}


def tabulate_top_features(
    Lambda_list: List,
    feature_names_list: Optional[List] = None,
    n_top_factors: int = 8,
    n_top_features: int = 10,
    rotate: bool = True,
) -> List[pd.DataFrame]:
    """
    Wide table of top features per factor by |loading|, one table per view.

    Parameters
    ----------
    Lambda_list         : list of (p_l, K) array-like
    feature_names_list  : list of feature-name arrays, one per view; defaults
                          to numeric indices ("f0", "f1", …)
    n_top_factors       : number of top factors to annotate (ranked by SSL)
    n_top_features      : number of features per factor column
    rotate              : if True, apply varimax rotation first

    Returns
    -------
    list of wide DataFrames (one per view), rows = rank, columns = factor
    """
    arrays = [_np(L) for L in Lambda_list]

    if rotate:
        rotated_arrays, _ = _rotate_loadings(arrays)
    else:
        rotated_arrays = arrays

    results = []

    for l, L in enumerate(rotated_arrays):

        fnames = (
            np.asarray(feature_names_list[l])
            if feature_names_list is not None
            else np.array([f"f{i}" for i in range(L.shape[0])])
        )

        factor_var = np.sum(L ** 2, axis=0)
        total_var  = factor_var.sum()
        top_f      = np.argsort(factor_var)[::-1][:n_top_factors]

        rows = []
        for f_idx in top_f:
            pct_var    = factor_var[f_idx] / total_var * 100
            sorted_idx = np.argsort(np.abs(L[:, f_idx]))[::-1][:n_top_features]
            for rank, feat_i in enumerate(sorted_idx):
                rows.append({
                    "Factor":   int(f_idx + 1),
                    "Pct_var":  round(pct_var, 2),
                    "Rank":     rank + 1,
                    "Feature":  fnames[feat_i],
                    "Loading":  round(float(L[feat_i, f_idx]), 4),
                })

        df   = pd.DataFrame(rows)
        wide = df.pivot_table(
            index="Rank", columns="Factor", values="Feature", aggfunc="first"
        )
        wide.columns = [
            f"F{c} ({df[df.Factor == c]['Pct_var'].iloc[0]:.1f}%)"
            for c in wide.columns
        ]
        results.append(wide)

    return results
