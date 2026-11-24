"""
Do the global-eta study's TCGA fits recover the 12 annotated consensus factors?

The consensus factors CF1-CF12 (results/factors/consensus_loadings.csv, from
factor_annotation.py) were built from the CV pipeline's shared-model fits at
K = 20 and annotated in tcga_coadread_factors.qmd. For each study fit (fold,
K_max, approach, checkpoint) the loadings are varimax-rotated as in
global_eta_study.qmd section 4 and each factor with > 1% PVE is correlated
with every consensus factor over the shared features. Reported per approach,
K_max and checkpoint (mean over the 5 folds):
  - CFs recovered: consensus factors with a fitted factor at |r| >= 0.8
  - mean best |r| over the 12 CFs
  - unmatched factors: active fitted factors with no CF at |r| >= 0.5

Output: results/global_eta_study/consensus_match.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser("~/multiomic_integration/python"))
from factor_interpretation import _rotate_loadings

DIR = f"{HERE}/results/global_eta_study"
CV = f"{HERE}/results/cv"
VIEWS = ["expression", "methylation"]
OPTS = ["default", "gslow0.3", "warm5", "dfreeze25", "warm5_qprior", "warm5_qflat", "dense"]


def fit_series(fold, k, opt, when):
    f = f"{DIR}/tcga_rep{fold}_k{k}_{opt}_fits.npz"
    if not os.path.exists(f):
        return None
    d = np.load(f)
    if f"{when}_W" not in d:
        return None
    W, Z = d[f"{when}_W"], d[f"{when}_Z"]
    genes = [open(f"{CV}/fold{fold}/genes_view{l}.txt").read().split("\n") for l in range(2)]
    p0 = len(genes[0])
    rot, R = _rotate_loadings([W[:p0], W[p0:]])
    Zr = Z @ R
    pve = (Zr ** 2).sum(0) * sum((r ** 2).sum(0) for r in rot) / (Zr.shape[0] * W.shape[0])
    idx = [f"{VIEWS[l]}:{g}" for l in range(2) for g in genes[l]]
    df = pd.DataFrame(np.vstack(rot), index=idx)
    df = df[~df.index.duplicated(keep="first")]
    return df.loc[:, pve > 0.01]


def main():
    cons = pd.read_csv(f"{HERE}/results/factors/consensus_loadings.csv", index_col=0)
    cons = cons[~cons.index.duplicated(keep="first")].fillna(0.0)   # CF8 / CF12 lack 15 / 19 of 3182 features
    rows = []
    for opt in OPTS:
        for k in (20, 50, 100):
            for when in ("ma", "1000"):
                for fold in range(5):
                    fs = fit_series(fold, k, opt, when)
                    if fs is None:
                        continue
                    common = fs.index.intersection(cons.index)
                    A, B = fs.loc[common].values, cons.loc[common].values
                    A = (A - A.mean(0)) / A.std(0)
                    B = (B - B.mean(0)) / B.std(0)
                    C = np.abs(A.T @ B) / len(common)          # fitted x consensus
                    rows.append({"approach": opt, "K_max": k, "when": "MA stop" if when == "ma" else "epoch 1000",
                                 "fold": fold, "active factors": A.shape[1],
                                 "CFs recovered (|r| >= 0.8)": int((C.max(0) >= 0.8).sum()),
                                 "mean best |r| over CFs": C.max(0).mean(),
                                 "unmatched factors (|r| < 0.5)": int((C.max(1) < 0.5).sum())})
    out = pd.DataFrame(rows)
    out.to_csv(f"{DIR}/consensus_match.csv", index=False)
    summ = out.groupby(["approach", "K_max", "when"]).mean(numeric_only=True).drop(columns="fold").round(2)
    print(summ.to_string())


if __name__ == "__main__":
    main()
