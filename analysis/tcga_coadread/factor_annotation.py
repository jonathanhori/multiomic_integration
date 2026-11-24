"""
Consensus factors and their annotation (read by tcga_coadread_factors.qmd).

Stage 1, consensus factors (K_max = 20 CV fits, results/cv/fold<f>/k20/):
  - each fold's loadings are varimax-rotated (both views jointly) and every
    rotated factor is rescaled so its training scores have SD 1 (loadings are
    then "change in the standardised feature per SD of the factor");
  - factors explaining at least 0.5% of the training data are pooled over the
    5 folds and clustered on |corr| of their loadings over the genes both folds
    selected (average linkage, cut at |r| = 0.8; at most one factor per fold
    per cluster);
  - a cluster seen in at least 4 of 5 folds is a consensus factor. Its gene
    loadings are the mean of the sign-aligned fold loadings; its out-of-fold
    scores come from each fold's held-out samples (so every sample's score is
    from a fit that did not see it; samples whose fold lacks the factor are
    left missing);
  - the sign is set so the factor's largest loadings are positive on the
    expression view.

Stage 2, sample annotation: clinical, molecular (MSI, subtype, aneuploidy,
mutations in BRAF / KRAS / APC / TP53 / POLE, CMS marker scores, hypoxia),
technical (tissue source site).

Stage 3, gene-set enrichment: preranked GSEA (gseapy) of each consensus
factor's expression loadings, and of minus its promoter-methylation loadings
(promoter methylation usually silences the gene, so genes whose promoters
gain methylation are ranked like genes that lose expression), against MSigDB
Hallmark, Reactome, KEGG (legacy), C8 cell-type signatures and the CMS marker
genes. The ranked universe is the genes the model saw (about 2,000 expression
genes, 1,000 methylation genes), not the genome; sets need >= 10 of them.

Outputs: results/factors/*.csv
usage: python factor_annotation.py [--stage consensus annot gsea] [--method shared_ma] [--tag ""]
       --method shared --tag _default: the same analysis on the default-setting fits
       (cv_k_sweep.py method "shared"); every output file gets the suffix <tag>
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = os.path.expanduser("~/multiomic_integration")
sys.path.insert(0, f"{ROOT}/python")
from factor_interpretation import _rotate_loadings  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CV = f"{HERE}/results/cv"
OUT = f"{HERE}/results/factors"
TCGA = f"{ROOT}/data/tcga/coadread_tcga_pan_can_atlas_2018"
GS = f"{ROOT}/data/genesets"
K, N_FOLDS, L = 20, 5, 2
VIEWS = ["expression", "methylation"]
MIN_PVE, R_CUT, MIN_FOLDS = 0.005, 0.8, 4


def lines(path):
    return open(path).read().split("\n")


def load_fold(f, k=K, method="shared_ma"):
    d = np.load(f"{CV}/fold{f}/k{k}/{method}_factors.npz")
    W = [d[f"loadings_view{l}"] for l in range(L)]
    rot, R = _rotate_loadings(W)
    Ztr, Zte = d["Z_train"] @ R, d["Z_test"] @ R
    sd = Ztr.std(0)
    rot = [w * sd for w in rot]
    Ztr, Zte = Ztr / sd, Zte / sd
    beta = (R.T @ d["beta"]) * sd
    genes = [lines(f"{CV}/fold{f}/genes_view{l}.txt") for l in range(L)]
    P = sum(len(g) for g in genes)
    pve = (Ztr ** 2).sum(0) * sum((w ** 2).sum(0) for w in rot) / (Ztr.shape[0] * P)
    return {"W": rot, "Ztr": Ztr, "Zte": Zte, "beta": beta, "genes": genes, "pve": pve,
            "train": lines(f"{CV}/fold{f}/train_samples.txt"), "test": lines(f"{CV}/fold{f}/test_samples.txt")}


def as_series(fit, j):
    """Loadings of factor j as one Series indexed by 'view:gene'."""
    s = pd.concat([pd.Series(fit["W"][l][:, j], index=[f"{VIEWS[l]}:{g}" for g in fit["genes"][l]])
                   for l in range(L)])
    return s[~s.index.duplicated(keep="first")]   # a few gene symbols repeat (aliases / NA)


def consensus(k=K, method="shared_ma", tag=""):
    fits = {f: load_fold(f, k, method) for f in range(N_FOLDS)}
    items = [(f, j) for f in fits for j in np.where(fits[f]["pve"] > MIN_PVE)[0]]
    S = {it: as_series(fits[it[0]], it[1]) for it in items}
    n = len(items)
    C = np.eye(n)
    for a, b in itertools.combinations(range(n), 2):
        if items[a][0] == items[b][0]:
            C[a, b] = C[b, a] = 0.0          # same fold: never merge
            continue
        x, y = S[items[a]].align(S[items[b]], join="inner")
        C[a, b] = C[b, a] = np.corrcoef(x, y)[0, 1]
    D = 1 - np.abs(C)
    np.fill_diagonal(D, 0)
    cl = fcluster(linkage(squareform(D, checks=False), "average"), t=1 - R_CUT, criterion="distance")

    rows, load, scores = [], {}, {}
    for c in np.unique(cl):
        mem = [i for i in range(n) if cl[i] == c]
        # at most one factor per fold: keep the one most correlated with the rest
        best = {}
        for i in mem:
            f = items[i][0]
            score = np.mean([abs(C[i, m]) for m in mem if m != i]) if len(mem) > 1 else 0
            if f not in best or score > best[f][1]:
                best[f] = (i, score)
        mem = [v[0] for v in best.values()]
        if len(mem) < MIN_FOLDS:
            continue
        anchor = max(mem, key=lambda i: fits[items[i][0]]["pve"][items[i][1]])
        sign = {i: np.sign(C[i, anchor]) if i != anchor else 1.0 for i in mem}
        lw = pd.concat([S[items[i]] * sign[i] for i in mem], axis=1).mean(axis=1)
        expr = lw[lw.index.str.startswith("expression:")]
        s0 = 1.0 if expr.max() >= -expr.min() else -1.0
        lw *= s0
        name = f"CF{len(rows) + 1}"
        load[name] = lw
        sc = {}
        for i in mem:
            f, j = items[i]
            sc.update(dict(zip(fits[f]["test"], fits[f]["Zte"][:, j] * sign[i] * s0)))
        scores[name] = pd.Series(sc)
        pairs = [abs(C[a, b]) for a, b in itertools.combinations(mem, 2)]
        rows.append({"factor": name, "folds": len(mem), "mean |r| between folds": np.mean(pairs),
                     "PVE (mean)": np.mean([fits[items[i][0]]["pve"][items[i][1]] for i in mem]),
                     "beta (mean, per SD)": np.mean([fits[items[i][0]]["beta"][items[i][1]] * sign[i] * s0 for i in mem]),
                     "members": ";".join(f"fold{items[i][0]}:F{items[i][1] + 1}" for i in mem)})
    summ = pd.DataFrame(rows).sort_values("PVE (mean)", ascending=False)
    order = {old: f"CF{i + 1}" for i, old in enumerate(summ["factor"])}
    summ["factor"] = summ["factor"].map(order)
    load = pd.DataFrame(load).rename(columns=order)[list(order.values())]
    scores = pd.DataFrame(scores).rename(columns=order)[list(order.values())]
    os.makedirs(OUT, exist_ok=True)
    summ.to_csv(f"{OUT}/consensus_summary{tag}.csv", index=False)
    load.to_csv(f"{OUT}/consensus_loadings{tag}.csv")
    scores.to_csv(f"{OUT}/consensus_scores_oof{tag}.csv")
    # how many pooled factors were left out
    print(f"{method} K={k}: {n} pooled factors, {len(summ)} consensus factors "
          f"covering {summ['folds'].sum()} of them")
    print(summ.round(3).to_string(index=False))
    return summ, load, scores


# ── Stage 2: sample annotation ──────────────────────────────────────────────

RIGHT = {"C18.0", "C18.1", "C18.2", "C18.3", "C18.4"}     # caecum .. transverse colon
LEFT = {"C18.5", "C18.6", "C18.7"}                        # splenic flexure .. sigmoid


def sample_annotation():
    s = pd.read_csv(f"{TCGA}/data_clinical_sample.txt", sep="\t", skiprows=4).set_index("SAMPLE_ID")
    p = pd.read_csv(f"{TCGA}/data_clinical_patient.txt", sep="\t", skiprows=4).set_index("PATIENT_ID")
    a = s[["PATIENT_ID", "MSI_SCORE_MANTIS", "ANEUPLOIDY_SCORE", "TMB_NONSYNONYMOUS", "TISSUE_SOURCE_SITE"]].join(
        p[["SUBTYPE", "AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "ICD_O_3_SITE"]], on="PATIENT_ID")
    a["log(1+TMB)"] = np.log1p(a["TMB_NONSYNONYMOUS"])
    a["MSI status"] = np.where(a["MSI_SCORE_MANTIS"] > 0.4, "MSI-H",
                               np.where(a["MSI_SCORE_MANTIS"].notna(), "MSS", None))
    site = a["ICD_O_3_SITE"].astype(str).str[:5]
    a["side"] = np.where(site.isin(RIGHT), "right colon", np.where(site.isin(LEFT), "left colon",
                         np.where(site.str.startswith(("C19", "C20")), "rectum", None)))
    st = a["AJCC_PATHOLOGIC_TUMOR_STAGE"].astype(str).str.extract(r"STAGE (IV|III|II|I)")[0]
    a["stage"] = st
    a["subtype"] = a["SUBTYPE"].str.replace("^(COAD|READ)_", "", regex=True)
    hyp = pd.read_csv(f"{TCGA}/data_clinical_supp_hypoxia.txt", sep="\t", skiprows=5).set_index("PATIENT_ID")
    a["hypoxia (Buffa)"] = a["PATIENT_ID"].map(hyp.iloc[:, 0])

    mut = pd.read_csv(f"{TCGA}/data_mutations.txt", sep="\t", comment="#", low_memory=False,
                      usecols=["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification", "HGVSp_Short"])
    silent = {"Silent", "Intron", "3'UTR", "5'UTR", "3'Flank", "5'Flank", "RNA", "IGR"}
    mut = mut[~mut["Variant_Classification"].isin(silent)]
    for g in ["APC", "TP53", "KRAS", "PIK3CA", "POLE"]:
        a[f"{g} mut"] = a.index.isin(mut.loc[mut["Hugo_Symbol"] == g, "Tumor_Sample_Barcode"])
    a["BRAF V600E"] = a.index.isin(mut.loc[(mut["Hugo_Symbol"] == "BRAF") & (mut["HGVSp_Short"] == "p.V600E"),
                                           "Tumor_Sample_Barcode"])
    # mutation calls exist only for sequenced samples
    seq = set(mut["Tumor_Sample_Barcode"])
    for c in [c for c in a if c.endswith(" mut") or c == "BRAF V600E"]:
        a[c] = a[c].astype(float)
        a.loc[~a.index.isin(seq), c] = np.nan

    # CMS marker scores: mean z-score (log2 RSEM, whole cohort) of each CMS
    # class's template genes (CMScaller, Eide et al. 2017); "CMS (approx.)" is
    # the highest-scoring class. An approximation of CMScaller's nearest
    # template prediction, for annotation only.
    ex = pd.read_csv(f"{TCGA}/data_mrna_seq_v2_rsem.txt", sep="\t").dropna(subset=["Hugo_Symbol"])
    ex = ex.drop(columns="Entrez_Gene_Id").groupby("Hugo_Symbol").mean()
    ex = np.log2(ex + 1)
    zx = ex.sub(ex.mean(1), axis=0).div(ex.std(1) + 1e-8, axis=0)
    tmpl = pd.read_csv(f"{GS}/CMScaller_templates_CMS.csv")
    for c, g in tmpl.groupby("class"):
        genes = [x for x in g["symbol"] if x in zx.index]
        a[f"{c} score"] = zx.loc[genes].mean().reindex(a.index)
    cms = a[[f"CMS{i} score" for i in range(1, 5)]]
    a["CMS (approx.)"] = cms.idxmax(axis=1).str.replace(" score", "").where(cms.notna().all(axis=1))
    os.makedirs(OUT, exist_ok=True)
    a.to_csv(f"{OUT}/sample_annotation.csv")
    return a


def associations(scores, a, tag=""):
    from scipy.stats import spearmanr, kruskal
    cont = ["log(1+TMB)", "MSI_SCORE_MANTIS", "ANEUPLOIDY_SCORE", "AGE", "hypoxia (Buffa)",
            "CMS1 score", "CMS2 score", "CMS3 score", "CMS4 score"]
    cat = ["MSI status", "subtype", "CMS (approx.)", "side", "stage", "SEX", "TISSUE_SOURCE_SITE",
           "BRAF V600E", "KRAS mut", "APC mut", "TP53 mut", "PIK3CA mut", "POLE mut"]
    rows = []
    for f in scores:
        x = scores[f].dropna()
        d = a.reindex(x.index)
        for c in cont:
            ok = d[c].notna()
            r, pv = spearmanr(x[ok], d.loc[ok, c])
            rows.append({"factor": f, "variable": c, "type": "Spearman rho", "effect": r, "p": pv, "n": int(ok.sum())})
        for c in cat:
            ok = d[c].notna()
            g = d.loc[ok, c].astype(str)
            keep = g.map(g.value_counts()) >= 5           # groups with at least 5 samples
            xs, g = x[ok][keep], g[keep]
            if g.nunique() < 2:
                continue
            groups = [xs[g == v] for v in g.unique()]
            pv = kruskal(*groups).pvalue
            eta2 = sum(len(v) * (v.mean() - xs.mean()) ** 2 for v in groups) / ((xs - xs.mean()) ** 2).sum()
            rows.append({"factor": f, "variable": c, "type": "eta^2", "effect": eta2, "p": pv, "n": len(xs)})
    r = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    r["q (BH)"] = multipletests(r["p"], method="fdr_bh")[1]
    r.to_csv(f"{OUT}/associations{tag}.csv", index=False)
    return r


# ── Stage 3: gene-set enrichment ────────────────────────────────────────────

COLLECTIONS = {"Hallmark": "h.all.v2024.1.Hs.symbols.gmt",
               "Reactome": "c2.cp.reactome.v2024.1.Hs.symbols.gmt",
               "KEGG": "c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt",
               "C8 cell type": "c8.all.v2024.1.Hs.symbols.gmt"}


def read_gmt(path):
    out = {}
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        out[parts[0]] = parts[2:]
    return out


def gene_sets():
    sets = {name: read_gmt(f"{GS}/{fn}") for name, fn in COLLECTIONS.items()}
    # C8: keep colon / intestine and immune / stromal cell-type signatures
    keep = ("COLON", "INTESTIN", "GUT", "IMMUNE", "T_CELL", "B_CELL", "MACROPHAGE", "FIBROBLAST",
            "ENDOTHELIAL", "MONOCYTE", "NK_CELL", "PLASMA", "MYELOID", "DENDRITIC", "MAST")
    sets["C8 cell type"] = {k: v for k, v in sets["C8 cell type"].items()
                            if k.startswith(("BUSSLAPP", "GAO_LARGE_INTESTINE", "GAO_SMALL_INTESTINE",
                                             "TRAVAGLINI", "HAY_BONE", "AIZARANI"))
                            or any(s in k for s in keep)}
    tmpl = pd.read_csv(f"{GS}/CMScaller_templates_CMS.csv")
    sets["CMS markers"] = {f"{c}_MARKERS": list(g["symbol"]) for c, g in tmpl.groupby("class")}
    return sets


def gsea(load, tag="", perms=2000, seed=7):
    import gseapy as gp
    sets = gene_sets()
    rows = []
    for f in load:
        for view, sgn in (("expression", 1.0), ("methylation", -1.0)):
            lw = load[f][load[f].index.str.startswith(view + ":")].dropna()
            rnk = pd.Series(sgn * lw.values, index=lw.index.str.split(":").str[1])
            rnk = rnk[~rnk.index.duplicated()].sort_values(ascending=False)
            for coll, gs in sets.items():
                res = gp.prerank(rnk=rnk, gene_sets=gs, min_size=10, max_size=500, permutation_num=perms,
                                 seed=seed, threads=4, outdir=None, verbose=False, no_plot=True)
                r = res.res2d.copy()
                r["factor"], r["view"], r["collection"] = f, view, coll
                rows.append(r)
        print("gsea", f, flush=True)
    r = pd.concat(rows)
    r = r.rename(columns={"Term": "gene set", "NES": "NES", "FDR q-val": "FDR", "NOM p-val": "p",
                          "Lead_genes": "leading genes", "Tag %": "tag"})
    r.to_csv(f"{OUT}/gsea{tag}.csv", index=False)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="*", default=["consensus"])
    ap.add_argument("--method", default="shared_ma")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if "consensus" in a.stage:
        consensus(method=a.method, tag=a.tag)
    load = pd.read_csv(f"{OUT}/consensus_loadings{a.tag}.csv", index_col=0)
    scores = pd.read_csv(f"{OUT}/consensus_scores_oof{a.tag}.csv", index_col=0)
    if "annot" in a.stage:
        ann = sample_annotation()
        r = associations(scores, ann, tag=a.tag)
        top = r[r["q (BH)"] < 0.01].sort_values(["factor", "p"])
        print(top.groupby("factor").head(5).round(4).to_string(index=False))
    if "gsea" in a.stage:
        gsea(load, tag=a.tag)
