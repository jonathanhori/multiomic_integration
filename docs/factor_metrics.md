# Factor Metrics Reference

This document describes every metric produced by `summarize_factors` in
`python/factor_interpretation.py`. Metrics are computed twice — once for the
unrotated posterior-mean loadings and scores, and once after joint varimax
rotation — and returned as `result["pre"]` and `result["post"]`.

---

## Model and notation

The fitted factor model reconstructs each data view $l$ as

$$X_l \approx Z \Lambda_l^\top, \qquad X_l \in \mathbb{R}^{n \times p_l}$$

where $Z \in \mathbb{R}^{n \times K}$ are the shared factor scores and
$\Lambda_l \in \mathbb{R}^{p_l \times K}$ are the per-view loadings.

When a supervised outcome is present:

$$y \approx Z \beta, \qquad y \in \mathbb{R}^n,\; \beta \in \mathbb{R}^K$$

All metrics use **posterior-mean** values of $Z$, $\Lambda_l$, and $\beta$
(i.e. the variational location parameters). The data $X_l$ should be
**mean-centred** before calling `summarize_factors`, matching the normalisation
applied during training. The outcome $y$ is centred internally when computing
$R^2_y$.

---

## Scalar metrics

### `total_model_r2`

$$R^2_\text{model} = 1 - \frac{\text{RSS}}{\text{TSS}}$$

where

$$\text{TSS} = \sum_l \|X_l\|_F^2, \qquad \text{RSS} = \sum_l \|X_l - Z\Lambda_l^\top\|_F^2$$

**Interpretation.** The fraction of total feature variance captured by the
factor structure. This is the single most important summary of reconstruction
quality and is **invariant to orthogonal rotation** of $(Z, \Lambda_l)$.

**Use.** Compare across models (different $K$, different priors) or conditions.
Values above 0.7 typically indicate a well-fitting model given the noise level.

---

### `total_signal_r2`

$$R^2_\text{signal} = \frac{\sum_k \text{ESS}_k}{\text{TSS}}$$

**Interpretation.** An ESS-based approximation of `total_model_r2`. Equals
`total_model_r2` exactly when $Z$ has mutually orthogonal columns
($Z^\top Z$ diagonal). The gap $|R^2_\text{signal} - R^2_\text{model}|$
quantifies how much cross-factor signal overlap (or cancellation) is present.

**Use.** Check whether the ESS-based per-factor attribution is trustworthy.
A gap larger than a few percent (pre-rotation) suggests the QR orthogonality
projection did not converge fully or the model is not well-identified.
Post-rotation the gap is expected to widen slightly because varimax does not
preserve column orthogonality of $Z$.

---

### `per_view_model_r2`

$$R^2_l = 1 - \frac{\|X_l - Z\Lambda_l^\top\|_F^2}{\|X_l\|_F^2}$$

**Interpretation.** Per-view reconstruction quality. Views with lower $R^2_l$
are either noisier, require more factors, or are less well explained by the
shared factor structure.

---

### `total_outcome_r2`

$$R^2_y = 1 - \frac{\|(y - \bar{y}) - (Z\beta - \overline{Z\beta})\|^2}{\|y - \bar{y}\|^2}$$

Only computed when both `beta` and `y` are passed to `summarize_factors`.
Both outcome and prediction are centred before computing $R^2$ to avoid
conflating intercept bias with variance explained.

**Interpretation.** How much outcome variance the factor model explains via
its regression on the scores. **Invariant to rotation** because $Z\beta =
(ZR)(R^\top\beta)$.

---

## Per-factor metrics (`per_factor` DataFrame)

The DataFrame is **sorted by `pve_data` descending** (most important factors
first). All ESS-based metrics share the same orthogonality caveat described
above.

---

### `ess` — Energy of Signal

$$\text{ESS}_k = \sum_l \|z_k\|^2 \|\lambda_{lk}\|^2$$

where $z_k \in \mathbb{R}^n$ is the $k$-th column of $Z$ and $\lambda_{lk}
\in \mathbb{R}^{p_l}$ is the $k$-th column of $\Lambda_l$.

**Derivation.** The rank-1 contribution of factor $k$ to view $l$ is the
outer product $z_k \lambda_{lk}^\top$, which has Frobenius norm squared
$\|z_k\|^2 \|\lambda_{lk}\|^2$. Summing over views gives the total signal
energy attributable to factor $k$ in isolation.

**When is it exact?** If $Z^\top Z$ is diagonal, then

$$\sum_k \text{ESS}_k = \left\|\sum_k z_k \lambda_{lk}^\top\right\|_F^2 = \|\hat{X}_l\|_F^2$$

because all cross terms $z_k^\top z_{k'} = 0$. The sum then equals
$\text{TSS} - \text{RSS}$.

**Comparison with SSL.** The sum-of-squared-loadings (SSL) metric used by
standard factor analysis is

$$\text{SSL}_k = \sum_l \|\lambda_{lk}\|^2$$

This is equivalent to ESS with $\|z_k\|^2 = 1$ for all $k$. SSL gives
incorrect relative importance when score columns have unequal norms —
a common outcome of variational inference. ESS accounts for the actual
posterior score magnitudes.

---

### `pve_factor` — PVE relative to total factor signal

$$\text{PVE}_\text{factor}[k] = \frac{\text{ESS}_k}{\sum_{k'} \text{ESS}_{k'}}$$

**Interpretation.** Factor $k$'s share of the total signal explained by all
factors combined. Sums to 1 by construction.

**Use.** Rank factors by their relative importance within the model. Use this
when you want to understand the internal structure of the factorisation,
regardless of total model fit.

---

### `pve_model` — PVE relative to model-explained variance

$$\text{PVE}_\text{model}[k] = \frac{\text{ESS}_k}{\text{TSS} - \text{RSS}}$$

**Interpretation.** Factor $k$'s ESS as a fraction of the variance the model
actually explains. Does **not** sum to 1 in general (only when the ESS
decomposition is exact and $\text{ESS}$ sums to $\text{TSS} - \text{RSS}$).

**Use.** Indicates whether a factor accounts for more or less than its "fair
share" of what the model explains. Values that sum noticeably away from 1
signal non-orthogonal scores or cross-factor interference.

---

### `pve_data` — PVE relative to sample variance

$$\text{PVE}_\text{data}[k] = \frac{\text{ESS}_k}{\text{TSS}}$$

**Interpretation.** Factor $k$'s signal energy as a fraction of the total
feature variance in the data. The most interpretable per-factor metric for
comparing factors across models or datasets.

**Use.** The primary ranking metric — the DataFrame is sorted by this column.
Also drives the cumulative PVE curve, which shows how many factors are needed
to explain a target fraction of the data.

---

### `r2_mofa` — MOFA-style R² per factor

$$R^2_\text{MOFA}[k] = 1 - \frac{\sum_l \|X_l - z_k \lambda_{lk}^\top\|_F^2}{\text{TSS}}$$

**Interpretation.** The fraction of variance remaining after removing only
the rank-1 signal of factor $k$ from the full data. Provides a complementary
view to `pve_data`: while ESS measures what a factor adds, $R^2_\text{MOFA}$
measures what it removes.

**Note.** $R^2_\text{MOFA}$ can be negative for a small or noisy factor
(removing one factor's contribution can make the residual slightly worse if
the factor partially cancels noise). Negative values indicate the factor
captures less signal than noise.

---

### `beta` — Outcome regression coefficient

The posterior-mean regression weight $\beta_k$ (passed in as `beta` argument).
After varimax rotation, this is $\tilde{\beta}_k = (R^\top \beta)_k$, which
preserves the prediction $Z\beta = \tilde{Z}\tilde{\beta}$.

**Interpretation.** Direction and magnitude of the outcome effect of each
factor. Large $|\beta_k|$ means factor $k$ is a strong predictor of the
outcome in the unrotated basis. After rotation factors may redistribute
predictive weight.

---

### `beta_pve` — Share of outcome prediction per factor

$$\text{beta\_pve}[k] = \frac{\beta_k^2 \|z_k\|^2}{\sum_{k'} \beta_{k'}^2 \|z_{k'}\|^2}$$

**Derivation.** The variance of $z_k \beta_k$ (the rank-1 contribution of
factor $k$ to $\hat{y} = Z\beta$) is $\beta_k^2 \|z_k\|^2 / n$. Normalising
across factors gives each factor's share of the total predicted outcome
variance. Sums to 1 by construction. Exact when $Z$ has orthogonal columns.

**Interpretation.** Which factors drive the outcome prediction. Combines
loading magnitude (via $\|z_k\|^2$ which embeds $\|\lambda_{lk}\|^2$ through
the model structure) with regression strength ($\beta_k^2$).

**Use.** Use alongside `pve_data` to identify factors that explain data
variance **and** predict the outcome (high in both), factors that explain
data variance but are outcome-neutral (high `pve_data`, low `beta_pve`), and
factors that are outcome-predictive but explain little data variance (low
`pve_data`, high `beta_pve`).

---

### `beta_r2_y` — Per-factor outcome R²

$$R^2_{y,k} = \frac{\beta_k^2 \|z_k\|^2}{\|y - \bar{y}\|^2}$$

Only computed when `y` is also supplied.

**Interpretation.** Factor $k$'s ESS-based contribution expressed as a
fraction of the centred outcome variance $\|y - \bar{y}\|^2$ (not the
model-explained outcome variance). Useful for comparing the absolute outcome
importance of a factor across datasets.

**Note.** $\sum_k R^2_{y,k}$ approximates `total_outcome_r2`, with the gap
growing when $Z$ is not orthogonal (same caveat as `total_signal_r2` vs
`total_model_r2`).

---

## `cumulative_pve_data`

$$\text{cum\_pve}[j] = \sum_{k=1}^{j} \text{PVE}_\text{data}[\pi(k)]$$

where $\pi$ is the permutation that sorts factors by $\text{ESS}$ descending
(i.e. the same order as the `per_factor` DataFrame rows).

**Interpretation.** How much of the total data variance is captured by the
top $j$ factors. Use to choose a final $K$: the elbow of this curve is the
point of diminishing returns.

---

## Rotation behaviour

All metrics are computed twice: **unrotated** (raw variational posterior) and
**varimax-rotated**. The rotation $R \in \mathbb{R}^{K \times K}$ is an
orthogonal matrix found by maximising the varimax criterion applied jointly
to all views' stacked loadings (see `python/multiview_varimax.py`).

| Metric | Invariant to rotation? |
|--------|------------------------|
| `total_model_r2` | Yes — reconstruction is $ZR(ΛR)^\top = ZΛ^\top$ |
| `total_outcome_r2` | Yes — prediction is $ZR \cdot R^\top\beta = Z\beta$ |
| `total_signal_r2` | No — Z column norms change |
| `per_view_model_r2` | Yes |
| `ess`, `pve_*`, `r2_mofa` | No — per-factor attribution changes |
| `beta`, `beta_pve`, `beta_r2_y` | No — per-factor attribution changes |

**Why rotate?** Varimax produces a sparser loading structure (each feature
loads on fewer factors), making factor interpretation easier. Because the
reconstruction is invariant, rotation is purely interpretive. Prefer the
**post-rotation** metrics for biological or feature-level interpretation;
use **pre-rotation** metrics to compare model fit across conditions.
