"""
PCGSE: Principal Component Gene Set Enrichment Analysis
Implements the approach from Frost et al. (2015) for testing gene sets
against factor loadings, compatible with single-view (MatrixDecomp) and
multi-view (SupMultiviewDecomp / SupMultiviewShared) factor models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def convert_symbols_to_ensembl(
    symbols: List[str],
    species: str = "human",
    fields: str = "ensembl.gene",
) -> Dict[str, str]:
    """
    Convert gene symbols to Ensembl gene IDs using the MyGene.info API.

    Parameters
    ----------
    symbols : list of gene symbol strings (e.g. ['TP53', 'BRCA1'])
    species : MyGene species filter (default 'human')
    fields  : MyGene field to retrieve (default 'ensembl.gene')

    Returns
    -------
    dict mapping symbol → Ensembl ID.
    Symbols with no match or ambiguous hits are omitted.
    Call with verbose=True in querymany to see unmapped genes.
    """
    try:
        import mygene
    except ImportError as e:
        raise ImportError("mygene is required: pip install mygene") from e

    mg = mygene.MyGeneInfo()
    result = mg.querymany(
        symbols,
        scopes="symbol",
        fields=fields,
        species=species,
        as_dataframe=True,
        verbose=False,
    )

    mapping: Dict[str, str] = {}
    if result is None or result.empty:
        return mapping

    # result index = queried symbol; 'ensembl.gene' may be a str or a list
    for sym, row in result.iterrows():
        if "notfound" in row and row["notfound"] is True:
            continue
        val = row.get("ensembl.gene", None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        # When multiple Ensembl IDs map to one symbol, take the first
        if isinstance(val, list):
            ensg = val[0].get("gene", None) if isinstance(val[0], dict) else str(val[0])
        elif isinstance(val, dict):
            ensg = val.get("gene", None)
        else:
            ensg = str(val)
        if ensg:
            mapping[str(sym)] = ensg

    return mapping


def binary_matrix_to_gene_sets(
    df: pd.DataFrame,
    min_genes: int = 1,
) -> Dict[str, List[str]]:
    """
    Convert a binary gene-set membership DataFrame to the dict format
    expected by run_pcgse().

    The DataFrame should have gene sets as rows and gene IDs as columns,
    with non-zero / True values indicating membership (as produced by
    pyreadr.read_r() on an R gene-set matrix such as reactomeGS).

    Parameters
    ----------
    df        : DataFrame (gene_sets × genes), values truthy = member
    min_genes : drop sets with fewer members than this after conversion

    Returns
    -------
    dict {set_name: [gene_id, ...]}
    """
    gene_sets: Dict[str, List[str]] = {}
    cols = np.asarray(df.columns.tolist())
    for set_name, row in df.iterrows():
        members = cols[row.values.astype(bool)].tolist()
        if len(members) >= min_genes:
            gene_sets[str(set_name)] = members
    return gene_sets


def load_gmt(path: str | Path) -> Dict[str, List[str]]:
    """Load gene sets from an MSigDB .gmt file."""
    gene_sets: Dict[str, List[str]] = {}
    with open(path, "r") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = [g for g in parts[2:] if g]  # parts[1] is description URL
            if genes:
                gene_sets[name] = genes
    return gene_sets


def _estimate_mean_corr(X: Optional[np.ndarray], gene_mask: np.ndarray) -> float:
    """
    Estimate average pairwise Pearson correlation among genes in a set via
    Fisher z-transform. Returns 0 when X is None or fewer than 2 genes in set.
    X: (n_samples, n_genes); gene_mask: boolean (n_genes,)
    """
    if X is None:
        return 0.0
    idx = np.where(gene_mask)[0]
    if len(idx) < 2:
        return 0.0
    corr = np.corrcoef(X[:, idx], rowvar=False)
    upper = corr[np.triu_indices(corr.shape[0], k=1)]
    upper = np.clip(upper, -0.9999, 0.9999)
    rho_bar = float(np.tanh(np.mean(np.arctanh(upper))))
    return rho_bar


def _pcgse_mean_diff(
    loadings: np.ndarray,
    gene_mask: np.ndarray,
    rho_bar: float,
) -> Tuple[float, float, float, float]:
    """
    Correlation-adjusted mean-difference statistic (Frost 2015, Eq. 4).

    S = (z̄_set - z̄_comp) / (σ_p * sqrt(VIF/m + 1/(p-m)))
    VIF = 1 + (m-1) * ρ̄

    Returns (statistic, p_value, mean_z_set, mean_z_complement).
    """
    p = len(loadings)
    m = int(gene_mask.sum())
    if m == 0 or m == p:
        return np.nan, np.nan, np.nan, np.nan

    z_set = loadings[gene_mask]
    z_comp = loadings[~gene_mask]
    mean_set = float(z_set.mean())
    mean_comp = float(z_comp.mean())

    sigma_p = float(loadings.std(ddof=1))
    if sigma_p < 1e-12:
        return 0.0, 1.0, mean_set, mean_comp

    vif = 1.0 + (m - 1) * max(rho_bar, 0.0)
    se = sigma_p * np.sqrt(vif / m + 1.0 / (p - m))
    stat = (mean_set - mean_comp) / se
    # Two-sided t-test; df = p - 2 following CAMERA convention
    pval = float(2 * stats.t.sf(abs(stat), df=p - 2))
    return float(stat), pval, mean_set, mean_comp


def _pcgse_rank_sum(
    loadings: np.ndarray,
    gene_mask: np.ndarray,
    rho_bar: float,
) -> Tuple[float, float, float, float]:
    """
    Correlation-adjusted Wilcoxon rank-sum statistic (Frost 2015, Eq. 5).

    σ²_VIF,W = m(p-m)/(2π) * [arcsin(1) + (p-m-1)*arcsin(0.5)
                               + (m-1)*arcsin(ρ/2)
                               + (m-1)(p-m-1)*arcsin((ρ+1)/2)]

    Returns (statistic, p_value, mean_z_set, mean_z_complement).
    """
    p = len(loadings)
    m = int(gene_mask.sum())
    if m == 0 or m == p:
        return np.nan, np.nan, np.nan, np.nan

    mean_set = float(loadings[gene_mask].mean())
    mean_comp = float(loadings[~gene_mask].mean())

    ranks = stats.rankdata(loadings)
    W = float(ranks[gene_mask].sum())
    mu_W = m * (p + 1) / 2.0

    rho = max(rho_bar, 0.0)
    var_W = (m * (p - m) / (2 * np.pi)) * (
        np.arcsin(1.0)
        + (p - m - 1) * np.arcsin(0.5)
        + (m - 1) * np.arcsin(rho / 2.0)
        + (m - 1) * (p - m - 1) * np.arcsin((rho + 1.0) / 2.0)
    )
    if var_W <= 0:
        var_W = m * (p - m) * (p + 1) / 12.0  # fallback to standard variance

    stat = (W - mu_W) / np.sqrt(var_W)
    pval = float(2 * stats.norm.sf(abs(stat)))
    return float(stat), pval, mean_set, mean_comp


def run_pcgse(
    Lambda_list: List[np.ndarray],
    gene_sets: Dict[str, List[str]],
    feature_names_list: List[List[str]],
    X_list: Optional[List[np.ndarray]] = None,
    method: str = "mean_diff",
    min_genes: int = 5,
    max_genes: int = 500,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run PCGSE gene set enrichment for every (view, factor, gene_set) triple.

    Parameters
    ----------
    Lambda_list : list of (p_l, k) arrays
        Factor loadings per view. Pass a single-element list for single-view models.
    gene_sets : dict {name: [gene, ...]}
        Gene set definitions (e.g. from load_gmt()).
    feature_names_list : list of lists
        Gene / feature names for each view, matching the rows of each Lambda.
    X_list : list of (n, p_l) arrays or None
        Data matrices used to estimate within-set correlation ρ̄.
        Pass None to skip correlation adjustment (ρ̄ = 0).
    method : {'mean_diff', 'rank_sum'}
        Test statistic to use.
    min_genes : int
        Minimum overlap between gene set and view features to run a test.
    max_genes : int
        Maximum gene-set size (genes overlapping this view) to run a test.
    fdr_alpha : float
        Significance threshold after BH correction.

    Returns
    -------
    pd.DataFrame with columns:
        view, factor, gene_set, n_genes_set, n_genes_used,
        statistic, p_value, adj_p_value, significant,
        direction, mean_z_set, mean_z_complement
    """
    if method not in ("mean_diff", "rank_sum"):
        raise ValueError(f"method must be 'mean_diff' or 'rank_sum', got {method!r}")

    test_fn = _pcgse_mean_diff if method == "mean_diff" else _pcgse_rank_sum
    rows = []

    for view_idx, (Lambda, feat_names) in enumerate(zip(Lambda_list, feature_names_list)):
        Lambda = np.asarray(Lambda, dtype=float)  # (p, k)
        p, k = Lambda.shape
        feat_index = {g: i for i, g in enumerate(feat_names)}
        feat_set = set(feat_names)
        X = X_list[view_idx] if X_list is not None else None

        for factor_idx in range(k):
            loadings = Lambda[:, factor_idx]

            for gs_name, gs_genes in gene_sets.items():
                overlap = [g for g in gs_genes if g in feat_set]
                n_used = len(overlap)
                if n_used < min_genes or n_used > max_genes:
                    continue

                gene_mask = np.zeros(p, dtype=bool)
                for g in overlap:
                    gene_mask[feat_index[g]] = True

                rho_bar = _estimate_mean_corr(X, gene_mask)
                stat, pval, mean_set, mean_comp = test_fn(loadings, gene_mask, rho_bar)

                rows.append({
                    "view": view_idx,
                    "factor": factor_idx,
                    "gene_set": gs_name,
                    "n_genes_set": len(gs_genes),
                    "n_genes_used": n_used,
                    "statistic": stat,
                    "p_value": pval,
                    "adj_p_value": np.nan,
                    "significant": False,
                    "direction": np.nan,
                    "mean_z_set": mean_set,
                    "mean_z_complement": mean_comp,
                })

    _COLS = [
        "view", "factor", "gene_set", "n_genes_set", "n_genes_used",
        "statistic", "p_value", "adj_p_value", "significant",
        "direction", "mean_z_set", "mean_z_complement",
    ]
    if not rows:
        return pd.DataFrame(columns=_COLS)

    df = pd.DataFrame(rows)

    # BH correction within each (view, factor) group
    adj_pvals = np.full(len(df), np.nan)
    for (v, f), grp in df.groupby(["view", "factor"]):
        idx = grp.index.to_numpy()
        pvals = grp["p_value"].to_numpy()
        valid = ~np.isnan(pvals)
        if valid.sum() == 0:
            continue
        _, adj, _, _ = multipletests(pvals[valid], method="fdr_bh", alpha=fdr_alpha)
        adj_pvals[idx[valid]] = adj

    df["adj_p_value"] = adj_pvals
    df["significant"] = df["adj_p_value"] < fdr_alpha
    df["direction"] = np.sign(df["statistic"])
    return df[_COLS].reset_index(drop=True)


def plot_gsea(
    df: pd.DataFrame,
    view: int = 0,
    top_n_sets: int = 30,
    fdr_threshold: float = 0.05,
    factor_labels: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
):
    """
    Heatmap of gene sets × factors coloured by sign(stat) × −log10(adj_p).
    Significant cells (adj_p < fdr_threshold) are outlined in black.

    Parameters
    ----------
    df : output of run_pcgse()
    view : which view index to plot
    top_n_sets : keep the top gene sets ranked by minimum adj_p across factors
    fdr_threshold : significance threshold for cell outlines
    factor_labels : optional list of factor label strings (length = k)
    figsize : (width, height) in inches; auto-computed if None

    Returns
    -------
    (fig, ax) matplotlib objects
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import TwoSlopeNorm

    sub = df[df["view"] == view].copy()
    if sub.empty:
        raise ValueError(f"No PCGSE results for view {view}")

    factors = sorted(sub["factor"].unique())
    k = len(factors)

    best_p = sub.groupby("gene_set")["adj_p_value"].min().nsmallest(top_n_sets)
    top_sets = best_p.index.tolist()
    sub = sub[sub["gene_set"].isin(top_sets)]

    pivot = sub.pivot_table(
        index="gene_set", columns="factor",
        values=["statistic", "adj_p_value"], aggfunc="first",
    )
    stat_mat = pivot["statistic"].reindex(index=top_sets, columns=factors, fill_value=0.0)
    adjp_mat = pivot["adj_p_value"].reindex(index=top_sets, columns=factors, fill_value=1.0)

    signed_logp = np.sign(stat_mat.values) * (-np.log10(np.clip(adjp_mat.values, 1e-300, 1.0)))

    n_sets = len(top_sets)
    if figsize is None:
        figsize = (max(8, k * 0.7 + 2), max(6, n_sets * 0.35 + 2))

    fig, ax = plt.subplots(figsize=figsize)
    vmax = max(float(np.nanmax(np.abs(signed_logp))), 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(signed_logp, aspect="auto", cmap="RdBu_r", norm=norm)

    sig = adjp_mat.values < fdr_threshold
    for r in range(n_sets):
        for c in range(k):
            if sig[r, c]:
                rect = mpatches.FancyBboxPatch(
                    (c - 0.45, r - 0.45), 0.9, 0.9,
                    linewidth=1.5, edgecolor="black", facecolor="none",
                    boxstyle="square,pad=0",
                )
                ax.add_patch(rect)

    ax.set_xticks(range(k))
    if factor_labels is not None:
        ax.set_xticklabels(factor_labels, rotation=45, ha="right")
    else:
        ax.set_xticklabels([f"Factor {f + 1}" for f in factors], rotation=45, ha="right")

    ax.set_yticks(range(n_sets))
    ax.set_yticklabels(top_sets, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("sign(stat) × −log₁₀(adj p)", fontsize=9)

    ax.set_title(f"PCGSE — view {view} (top {n_sets} gene sets by adj p)", fontsize=11)
    fig.tight_layout()
    return fig, ax
