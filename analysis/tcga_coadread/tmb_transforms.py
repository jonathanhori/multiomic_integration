"""
Which transformation of TMB is closest to normal?

The model's outcome is Gaussian (y ~ N(Z beta, sigma_y^2)); currently y = log(1 + TMB).
Hypermutated tumours are underpredicted on that scale. This compares power transforms
of TMB_NONSYNONYMOUS on all 530 aligned samples, in two ways:

  1. Marginal shape: histogram (coloured by MSI status, MANTIS score > 0.4) and normal
     Q-Q plot, with skewness, excess kurtosis and the Shapiro-Wilk W.
  2. Residual shape (what the Gaussian likelihood actually assumes): residuals of a
     5-fold cross-validated ridge regression on the same two views used by the model
     (2000 most variable expression genes, 1000 TSS200 methylation genes). Reported:
     residual skewness / kurtosis / W, the mean standardised residual of MSI-H tumours,
     and R² of the back-transformed predictions on the common log(1 + TMB) scale.

There are 6 samples with TMB = 0, so plain log and Box-Cox need an offset. Box-Cox is
fitted to 1 + TMB (for x >= 0 this is the same family as Yeo-Johnson), so lambda = 0 is
the current log(1 + TMB) and lambda = 1 is untransformed.

Outputs
    figures/tmb_transforms.png           histograms and Q-Q plots per transform
    figures/tmb_boxcox_profile.png       Box-Cox profile log-likelihood in lambda
    results/tmb_transforms.csv           shape statistics per transform

usage: python tmb_transforms.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, HERE)
import compare_methods as cm
from coadread_data import load_coadread, _CLINICAL_FILE

FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"


def load_all():
    """Raw TMB, the two model views and MSI-H status for all aligned samples."""
    d = load_coadread(top_rna_genes=cm.N_RNA, top_meth_genes=cm.N_METH, log1p_tmb=False,
                      train_frac=0.8, random_seed=42, preprocess_on_train_only=False)
    y = np.concatenate([d["y"].numpy(), d["y_test"].numpy()]).astype(float)
    X = np.hstack([np.vstack([d[f"X_{v}"].numpy(), d[f"X_{v}_test"].numpy()]) for v in cm.VIEWS])
    ids = list(d["sample_ids_train"]) + list(d["sample_ids_test"])
    clin = pd.read_csv(_CLINICAL_FILE, sep="\t", skiprows=4).set_index("SAMPLE_ID")
    msi = clin.loc[ids, "MSI_SCORE_MANTIS"].to_numpy(dtype=float) > 0.4
    return y, X, msi


def transforms(y):
    _, lam = stats.boxcox(1 + y)
    T = {
        "untransformed": (lambda x: x, lambda t: t),
        "square root": (np.sqrt, lambda t: np.clip(t, 0, None) ** 2),
        "cube root": (np.cbrt, lambda t: np.clip(t, 0, None) ** 3),
        "log(1 + TMB)  [current]": (np.log1p, np.expm1),
        "log(0.5 + TMB)": (lambda x: np.log(0.5 + x), lambda t: np.exp(t) - 0.5),
        f"Box-Cox on 1 + TMB, λ̂ = {lam:.2f}": (
            lambda x: ((1 + x) ** lam - 1) / lam,
            lambda t: np.clip(lam * t + 1, 1e-12, None) ** (1 / lam) - 1),
    }
    return T, lam


def cv_ridge(X, t, seed=0):
    pred = np.empty_like(t)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(1, 5, 25)))
        pred[te] = m.fit(X[tr], t[tr]).predict(X[te])
    return pred


def main():
    y, X, msi = load_all()
    T, lam = transforms(y)
    y_log = np.log1p(y)
    rows, cv = [], {}
    for name, (f, finv) in T.items():
        t = f(y)
        p = cv_ridge(X, t)
        r = t - p
        rs = (r - r.mean()) / r.std()
        p_log = np.log1p(np.clip(finv(p), 0, None))
        rows.append({
            "transform": name,
            "skewness": stats.skew(t), "excess_kurtosis": stats.kurtosis(t),
            "shapiro_W": stats.shapiro(t).statistic,
            "resid_skewness": stats.skew(r), "resid_excess_kurtosis": stats.kurtosis(r),
            "resid_shapiro_W": stats.shapiro(r).statistic,
            "resid_mean_sd_msi_h": rs[msi].mean(), "resid_mean_sd_other": rs[~msi].mean(),
            "ridge_cv_r2_log1p_scale": 1 - ((y_log - p_log) ** 2).sum() / ((y_log - y_log.mean()) ** 2).sum(),
        })
        cv[name] = (t, r)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(RES, "tmb_transforms.csv"), index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(tab.to_string(index=False))
    print(f"n = {len(y)}, MSI-H = {msi.sum()}, lambda_hat = {lam:.3f}")

    # --- histograms and Q-Q plots ---
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})
    n = len(T)
    fig, ax = plt.subplots(3, n, figsize=(3.0 * n, 8.2), gridspec_kw={"height_ratios": [1.1, 1, 1]})
    for j, (name, (t, r)) in enumerate(cv.items()):
        row = tab.iloc[j]
        bins = np.histogram_bin_edges(t, 35)
        ax[0, j].hist([t[~msi], t[msi]], bins=bins, stacked=True, color=[BLUE, ORANGE],
                      edgecolor="white", linewidth=0.6, label=["MSS / MSI-L", "MSI-H"])
        ax[0, j].set_title(name, fontsize=9.5, color=INK)
        ax[0, j].text(0.98, 0.97, f"skew {row.skewness:.2f}\nkurt {row.excess_kurtosis:.2f}\nW {row.shapiro_W:.3f}",
                      transform=ax[0, j].transAxes, ha="right", va="top", fontsize=8, color=MUTED)
        for i, (vals, lab, key) in enumerate([(t, "marginal", ""), (r, "CV ridge residual", "resid_")], start=1):
            z = (vals - vals.mean()) / vals.std()
            q = stats.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
            order = np.argsort(z)
            ax[i, j].plot([-3.5, 3.5], [-3.5, 3.5], color=MUTED, lw=1, ls="--")
            ax[i, j].scatter(q, z[order], s=9, c=np.where(msi[order], ORANGE, BLUE), linewidths=0)
            ax[i, j].set_xlim(-3.5, 3.5)
            ax[i, j].text(0.03, 0.97, f"{lab}\nskew {row[key + 'skewness']:.2f}\nW {row[key + 'shapiro_W']:.3f}",
                          transform=ax[i, j].transAxes, ha="left", va="top", fontsize=8, color=MUTED)
            if j == 0:
                ax[i, j].set_ylabel("standardised value" if i == 1 else "standardised residual")
            if i == 2:
                ax[i, j].set_xlabel("normal quantile")
    ax[0, 0].set_ylabel("samples")
    ax[0, 0].legend(frameon=False, loc="center right", fontsize=8)
    fig.suptitle("TMB under power transforms: distribution (top), normal Q-Q (middle), "
                 "Q-Q of 5-fold CV ridge residuals (bottom); n = 530", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "tmb_transforms.png"), dpi=150)

    # --- Box-Cox profile likelihood ---
    grid = np.linspace(-1, 1, 201)
    ll = np.array([stats.boxcox_llf(l, 1 + y) for l in grid])
    ci = grid[ll >= ll.max() - stats.chi2.ppf(0.95, 1) / 2]
    fig, a = plt.subplots(figsize=(6, 3.4))
    a.plot(grid, ll, color=BLUE, lw=2)
    a.axvspan(ci.min(), ci.max(), color=BLUE, alpha=0.12, lw=0)
    for l, lab in [(0, "log(1 + TMB)"), (1 / 3, "cube root"), (0.5, "square root"), (lam, f"λ̂ = {lam:.2f}")]:
        a.axvline(l, color=MUTED if l != lam else INK, lw=1, ls=":" if l != lam else "-")
        a.text(l, ll.min(), " " + lab, rotation=90, va="bottom", fontsize=8, color=MUTED)
    a.set_xlabel("Box-Cox λ (on 1 + TMB)"); a.set_ylabel("profile log-likelihood")
    a.set_title(f"Box-Cox profile likelihood, 95% interval [{ci.min():.2f}, {ci.max():.2f}]",
                fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "tmb_boxcox_profile.png"), dpi=150)


if __name__ == "__main__":
    main()
