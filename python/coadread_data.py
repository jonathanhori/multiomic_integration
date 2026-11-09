"""
Data loading and preprocessing for the TCGA COAD/READ PanCan Atlas 2018 dataset.

Three views available:
  - Gene expression (RNA-seq V2 RSEM, log2(x+1) normalised)
  - TSS200 methylation (HM27/HM450 probes within 200 bp of TSS, beta values)
  - Gene-body methylation (HM27/HM450 Body probes, beta values)

Each methylation view is aggregated to gene level by averaging beta values across
probes that map unambiguously to one gene in the selected region.

Gene overlaps (approximate, single-gene probes only):
  RNA ∩ TSS200 : ~3 143 genes
  RNA ∩ Body   : ~3 032 genes
  All three    : ~670 genes   (use restrict_to_common_genes=True)

Samples are aligned to the 589-sample intersection of RNA, methylation, and
clinical data that have complete TMB values.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data" / "tcga" / "coadread_tcga_pan_can_atlas_2018"
)
_CLINICAL_FILE    = _DEFAULT_DATA_DIR / "data_clinical_sample.txt"
_RNA_FILE         = _DEFAULT_DATA_DIR / "data_mrna_seq_v2_rsem.txt"
_METH_FILE        = _DEFAULT_DATA_DIR / "data_methylation_hm27_hm450_merged.txt"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_clinical(path: Path) -> pd.DataFrame:
    """Return clinical sample table with SAMPLE_ID and TMB_NONSYNONYMOUS."""
    df = pd.read_csv(path, sep="\t", skiprows=4)
    return df[["SAMPLE_ID", "TMB_NONSYNONYMOUS"]].dropna()


def _load_rna(
    path: Path,
    samples: List[str],
    impute_missing: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load RNA-seq RSEM matrix, restrict to requested samples.
    Returns (samples × genes DataFrame, gene_names list).
    Values are log2(RSEM+1) as provided by cBioPortal PanCan Atlas.

    Parameters
    ----------
    impute_missing : if True, fill remaining NaN values with column means.
                     If False, NaNs are left in place (will cause errors in
                     normalisation / SVD unless handled downstream).
    """
    raw = pd.read_csv(path, sep="\t", index_col="Hugo_Symbol")
    raw = raw.drop(columns=["Entrez_Gene_Id"], errors="ignore")
    raw = raw.loc[~raw.index.isna()]                    # drop unannotated rows
    raw = raw.loc[raw.index.astype(str) != "nan"]
    # Transpose: rows = samples, cols = genes
    rna = raw[raw.columns.intersection(samples)].T
    rna = rna.reindex(samples).dropna(how="all")
    if impute_missing:
        rna = rna.fillna(rna.mean(axis=0))
    gene_names = rna.columns.tolist()
    return rna, gene_names


def _aggregate_methylation(
    path: Path,
    region: str,
    samples: List[str],
    impute_missing: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load HM450 methylation data and aggregate to gene level for one region.

    Only probes that:
      - map to exactly one gene (no semicolons in NAME)
      - have DESCRIPTION containing `region` (e.g. 'TSS200' or 'Body')
      - have non-missing NAME

    are included. Beta values are averaged across probes per (gene, sample).

    Returns (samples × genes DataFrame, gene_names list).

    Parameters
    ----------
    impute_missing : if True, fill remaining NaN values (genes with no probe
                     coverage for a given sample) with column means.
    """
    raw = pd.read_csv(path, sep="\t")

    # Keep only unambiguously single-gene probes in the requested region
    mask = (
        raw["NAME"].notna()
        & ~raw["NAME"].str.contains(";", na=True)
        & raw["DESCRIPTION"].str.contains(region, na=False)
    )
    raw = raw.loc[mask]

    # Sample columns only
    sample_cols = [c for c in raw.columns if c in set(samples)]
    data = raw[["NAME"] + sample_cols].copy()

    # Average beta values per gene across probes
    agg = data.groupby("NAME")[sample_cols].mean()  # genes × samples

    # Transpose and reindex to requested sample order
    meth = agg.T.reindex(samples)
    if impute_missing:
        meth = meth.fillna(meth.mean(axis=0))
    gene_names = meth.columns.tolist()
    return meth, gene_names


def _variance_filter(
    df: pd.DataFrame,
    n_genes: Optional[int],
) -> pd.DataFrame:
    """Keep the top n_genes by variance across samples. No-op if n_genes is None."""
    if n_genes is None or n_genes >= df.shape[1]:
        return df
    var = df.var(axis=0)
    top = var.nlargest(n_genes).index
    return df[top]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_coadread(
    data_dir: Optional[str | Path] = None,
    top_rna_genes: Optional[int] = None,
    top_meth_genes: Optional[int] = None,
    restrict_to_common_genes: bool = False,
    impute_missing: bool = True,
    log1p_tmb: bool = True,
    train_frac: float = 0.8,
    random_seed: int = 42,
    preprocess_on_train_only: bool = False,
) -> Dict:
    """
    Load and align TCGA COAD/READ gene expression, TSS200 methylation, and
    gene-body methylation data for supervised multiview modelling.

    Parameters
    ----------
    data_dir : path to the coadread_tcga_pan_can_atlas_2018 directory.
               Defaults to the project-relative path.
    top_rna_genes : if set, keep only the top N most-variable RNA genes
                    (applied after sample alignment, before train/test split)
    top_meth_genes : same for each methylation view independently
    restrict_to_common_genes : if True, restrict all three views to the
                                intersection of their gene sets so every view
                                covers the same genes (~670 genes).
                                Default False — each view uses its own maximal
                                gene set (RNA∩TSS200 ~3 143; RNA∩Body ~3 032).
    impute_missing : if True (default), fill NaN values in each view with
                     column means computed over all aligned samples.
                     Set to False to leave NaNs in place, e.g. when you want
                     to apply a different imputation strategy downstream.
    log1p_tmb : apply log1p transform to TMB_NONSYNONYMOUS (recommended;
                TMB is right-skewed)
    train_frac : fraction of samples used for training
    random_seed : random seed for train/test split
    preprocess_on_train_only : if True, choose the most variable genes and
                     compute the imputation means from the training samples
                     only, so test samples do not influence preprocessing.
                     Default False reproduces the original behaviour (both
                     computed over all aligned samples).

    Returns
    -------
    dict with keys:
      X_expr          torch.Tensor (n_train, p_rna)
      X_meth_tss200   torch.Tensor (n_train, p_tss200)
      X_meth_body     torch.Tensor (n_train, p_body)
      y               torch.Tensor (n_train,)
      X_expr_test         ...
      X_meth_tss200_test  ...
      X_meth_body_test    ...
      y_test              ...
      gene_names_expr       list[str]
      gene_names_tss200     list[str]
      gene_names_body       list[str]
      sample_ids_train  list[str]
      sample_ids_test   list[str]
      overlap_summary   dict   — gene counts per view and intersection
    """
    base = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    rna_path  = base / "data_mrna_seq_v2_rsem.txt"
    meth_path = base / "data_methylation_hm27_hm450_merged.txt"
    clin_path = base / "data_clinical_sample.txt"

    # ------------------------------------------------------------------
    # 1. Clinical: TMB and sample list
    # ------------------------------------------------------------------
    clin = _load_clinical(clin_path)
    clin_samples = set(clin["SAMPLE_ID"])

    # ------------------------------------------------------------------
    # 2. Identify overlapping samples across all three data sources
    # ------------------------------------------------------------------
    rna_raw = pd.read_csv(rna_path, sep="\t", nrows=1)
    rna_samples = set(rna_raw.columns[2:])  # skip Hugo_Symbol, Entrez_Gene_Id

    meth_raw = pd.read_csv(meth_path, sep="\t", nrows=1)
    meth_samples = set(meth_raw.columns[4:])  # skip annotation cols

    common_samples = sorted(rna_samples & meth_samples & clin_samples)
    if len(common_samples) == 0:
        raise ValueError("No samples found in common across RNA, methylation, and clinical data.")

    # ------------------------------------------------------------------
    # 3. Load each view
    # ------------------------------------------------------------------
    # With preprocess_on_train_only, imputation waits until the split is known.
    impute_now = impute_missing and not preprocess_on_train_only
    rna_df, rna_genes = _load_rna(rna_path, common_samples, impute_missing=impute_now)
    tss200_df, tss200_genes = _aggregate_methylation(meth_path, "TSS200", common_samples, impute_missing=impute_now)
    body_df, body_genes = _aggregate_methylation(meth_path, "Body", common_samples, impute_missing=impute_now)

    # ------------------------------------------------------------------
    # 4. Optionally restrict to three-way gene intersection
    # ------------------------------------------------------------------
    rna_set    = set(rna_genes)
    tss200_set = set(tss200_genes)
    body_set   = set(body_genes)

    overlap_summary = {
        "n_rna_genes": len(rna_set),
        "n_tss200_genes": len(tss200_set),
        "n_body_genes": len(body_set),
        "n_rna_tss200": len(rna_set & tss200_set),
        "n_rna_body": len(rna_set & body_set),
        "n_all_three": len(rna_set & tss200_set & body_set),
    }

    if restrict_to_common_genes:
        common_genes = sorted(rna_set & tss200_set & body_set)
        rna_df    = rna_df[common_genes]
        tss200_df = tss200_df[common_genes]
        body_df   = body_df[common_genes]
        rna_genes = tss200_genes = body_genes = common_genes
    else:
        # Each view restricted to its own RNA intersection
        rna_tss200 = sorted(rna_set & tss200_set)
        rna_body   = sorted(rna_set & body_set)
        tss200_df  = tss200_df[rna_tss200]
        body_df    = body_df[rna_body]
        tss200_genes = rna_tss200
        body_genes   = rna_body
        # RNA keeps its full gene set (or filtered below)

    # ------------------------------------------------------------------
    # 5. Train / test split indices (positions in the aligned sample order)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(random_seed)
    n = len(common_samples)
    idx = rng.permutation(n)
    n_train = int(np.floor(train_frac * n))
    train_idx = np.sort(idx[:n_train])
    test_idx  = np.sort(idx[n_train:])

    # ------------------------------------------------------------------
    # 6. Imputation and variance filtering (per view independently)
    # ------------------------------------------------------------------
    if preprocess_on_train_only:
        def _prep(df, n_genes):
            train = df.iloc[train_idx]
            if impute_missing:
                df = df.fillna(train.mean(axis=0))
                train = df.iloc[train_idx]
            if n_genes is None or n_genes >= df.shape[1]:
                return df
            return df[train.var(axis=0).nlargest(n_genes).index]
        rna_df    = _prep(rna_df,    top_rna_genes)
        tss200_df = _prep(tss200_df, top_meth_genes)
        body_df   = _prep(body_df,   top_meth_genes)
    else:
        rna_df    = _variance_filter(rna_df,    top_rna_genes)
        tss200_df = _variance_filter(tss200_df, top_meth_genes)
        body_df   = _variance_filter(body_df,   top_meth_genes)

    rna_genes    = rna_df.columns.tolist()
    tss200_genes = tss200_df.columns.tolist()
    body_genes   = body_df.columns.tolist()

    # ------------------------------------------------------------------
    # 7. Align TMB to sample order
    # ------------------------------------------------------------------
    tmb = clin.set_index("SAMPLE_ID")["TMB_NONSYNONYMOUS"].reindex(rna_df.index)
    y_vals = np.log1p(tmb.values.astype(np.float32)) if log1p_tmb else tmb.values.astype(np.float32)

    def _t(df: pd.DataFrame, rows) -> torch.Tensor:
        return torch.tensor(df.iloc[rows].values.astype(np.float32))

    def _ty(arr: np.ndarray, rows) -> torch.Tensor:
        return torch.tensor(arr[rows])

    sample_ids = list(rna_df.index)

    return {
        # Training tensors
        "X_expr":          _t(rna_df,    train_idx),
        "X_meth_tss200":   _t(tss200_df, train_idx),
        "X_meth_body":     _t(body_df,   train_idx),
        "y":               _ty(y_vals,   train_idx),
        # Test tensors
        "X_expr_test":         _t(rna_df,    test_idx),
        "X_meth_tss200_test":  _t(tss200_df, test_idx),
        "X_meth_body_test":    _t(body_df,   test_idx),
        "y_test":              _ty(y_vals,   test_idx),
        # Gene name lists (per view)
        "gene_names_expr":    rna_genes,
        "gene_names_tss200":  tss200_genes,
        "gene_names_body":    body_genes,
        # Sample IDs
        "sample_ids_train": [sample_ids[i] for i in train_idx],
        "sample_ids_test":  [sample_ids[i] for i in test_idx],
        # Diagnostics
        "overlap_summary": overlap_summary,
    }


def make_view_list(
    data: Dict,
    views: List[str],
    split: str = "train",
) -> List[torch.Tensor]:
    """
    Build the X_list argument for SupMultiviewDecomp / SupMultiviewShared.

    Parameters
    ----------
    data   : dict returned by load_coadread()
    views  : list of view names, e.g. ['expr', 'meth_tss200', 'meth_body']
    split  : 'train' or 'test'

    Returns
    -------
    List of tensors in the same order as `views`.
    """
    suffix = "_test" if split == "test" else ""
    key_map = {
        "expr":       f"X_expr{suffix}",
        "meth_tss200": f"X_meth_tss200{suffix}",
        "meth_body":   f"X_meth_body{suffix}",
    }
    available = {"expr", "meth_tss200", "meth_body"}
    for v in views:
        if v not in available:
            raise ValueError(f"Unknown view '{v}'. Choose from {available}.")
    return [data[key_map[v]] for v in views]


def make_feature_names(data: Dict, views: List[str]) -> List[List[str]]:
    """Return the feature_names_list for run_pcgse() matching the given views."""
    key_map = {
        "expr":        "gene_names_expr",
        "meth_tss200": "gene_names_tss200",
        "meth_body":   "gene_names_body",
    }
    return [data[key_map[v]] for v in views]
