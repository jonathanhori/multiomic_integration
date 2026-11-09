# Evaluation metrics: how they are measured and how to read them

This is the reference for every metric reported by the simulation studies and
the pipeline notebooks. Notation follows `docs/main(1).tex`: views
$l = 1, \dots, L$ with data $X^{(l)}$ ($n \times p_l$), shared scores $Z$, joint
loadings $\Lambda^{(l)}$, shared signal $J^{(l)} = Z\Lambda^{(l)\top}$, noise
variances $\Psi^{(l)}$, outcome $y = Z\beta + e$, and $P = \sum_l p_l$.

Code: `run_methods.evaluate_fitted_model` (Python models), `R/mofa_run.R::eval_mofa_shared`
(MOFA), `python/misaligned_baselines.py` (oracles and simple baselines),
`python/risk_metrics.py` (exact risk, direction recovery, strength estimates).

## Conventions that apply to every metric

- **Scale.** Models are fitted to standardised data: each feature and the outcome
  are centred and divided by their *training* standard deviation (test data use
  the training statistics). Simulated truths are put on the same scale before
  comparison (`scale_sim_struct`, `scale_sim_cov`). All metrics are on this
  standardised scale unless stated otherwise.
- **Training vs. test.** Structure and covariance metrics compare the fit to the
  *training* data with the simulated truth. Test metrics use *local inference*:
  the global parameters are frozen and each test sample's scores are inferred
  **from its views only** — the test outcome is never used.
- **Point estimates.** Unless stated, a metric uses posterior means (for MOFA,
  its point estimates).
- **Rows and aggregation.** Metric tables have one row per view; outcome columns
  repeat on every row. Study summaries average per-view metrics over views, then
  over replicates and (n, p) cells. "Win rate" is the share of datasets where one
  method beats the other on the same dataset.

## Structure recovery (training set)

| Column | Models | Definition | Read as |
|---|---|---|---|
| `joint_rse` | shared, MOFA, VIPER | $\lVert \hat J^{(l)} - J^{(l)}\rVert_F^2 / \lVert J^{(l)}\rVert_F^2$ per view | share of the true shared signal's energy **not** recovered. 0 is perfect; 0.02 means 98% recovered |
| `structure_rse` | single view | the same for the single view's signal $J = Z\Lambda^\top$ | as above |
| individual / view-specific RSE | VIPER | the same for $A^{(l)} = \Phi^{(l)}\Gamma^{(l)\top}$ | as above, for the view-specific signal |
| `structure_95p_cov` | single view, VIPER | share of true signal entries inside their 95% posterior interval | target 0.95. Mean-field VI tends to understate uncertainty, so values below 0.95 are expected |

How $\hat J$ is formed differs slightly: shared model and MOFA use (posterior
mean of $Z$) × (posterior mean of $\Lambda^{(l)}$); single view and VIPER use the
posterior mean of the signal samples. The relative squared error is **squared**
(as in the methods document), so it is roughly the square of an "unsquared"
relative error.

## Covariance reconstruction

| Column | Models | Definition |
|---|---|---|
| `cov_frob` (with `cov_type` within / cross), or `cov_within` / `cov_cross` | shared model, MOFA, PCA | $\lVert \hat\Lambda^{(l)}\hat\Sigma_Z\hat\Lambda^{(m)\top} - D_l^{-1}\Lambda^{(l)}\Lambda^{(m)\top}D_m^{-1}\rVert_F^2 / (p_l p_m)$ for each pair of views $l \le m$ |
| `cov_norm_frob` | single view | the same with $l = m = 1$ |

$D_l$ holds view $l$'s training standard deviations. This is the **mean squared
error per covariance entry** of the factor part of the covariance (noise
variances are not included). Within-view blocks ($l = m$) check each view's own
correlation structure; cross-view blocks ($l \ne m$) check dependence between
views, which only shared factors can produce. The metric does not change when
the factors are rotated. Lower is better; 0.01 means a typical entry is off by
about 0.1.

**Why $\hat\Sigma_Z$.** The model assumes $\mathbf z_i \sim N(0, I)$, so its
covariance would be $\Lambda\Lambda^\top$. But a fitted factor model only
identifies the signal $Z\Lambda^\top$: replacing $\Lambda$ by $\Lambda A$ and $Z$
by $Z A^{-\top}$ fits equally well, and the prior keeps $A$ near the identity only
loosely. Shrinkage priors on the loadings pull $\hat\Lambda$ smaller and the scores
grow to compensate (MOFA's fitted factor variances reach 3–5 at $p = 1000$). The
covariance the fit actually represents on the training data uses the fitted
scores' second moment

$$
\hat\Sigma_Z = \frac{1}{n}\sum_{i=1}^n E_q[\mathbf z_i\mathbf z_i^\top]
             = \frac{1}{n}\Big(\hat M^\top\hat M + \sum_{i=1}^n \hat S_i\Big),
$$

with $\hat M$ the posterior means of the training scores and $\hat S_i$ the
posterior covariance of sample $i$'s scores.

| Method | $\hat\Sigma_Z$ |
|---|---|
| shared model, single view (pipeline) | averaged over posterior draws of $Z$ (includes $\hat S_i$) |
| shared model (saved June fits, `misaligned_shared_cov.py`) | `loc_Z` and `scale_Z`: $\hat S_i = \operatorname{diag}(s_i^2)$ |
| MOFA | expected factors only ($\hat S_i$ is not exposed by the R API); the omitted term is about $1/(1+\ell)$ per factor, small at the strengths simulated |
| probabilistic PCA | identity: PCA's likelihood integrates $z$ out, so $W = V_K(\Lambda_K - \sigma^2 I)^{1/2}$ is scaled to unit-variance factors, and the formula above gives exactly $I$ at the maximum-likelihood fit |

*History:* results before October 2026 used $\hat\Lambda\hat\Lambda^\top$ (no
$\hat\Sigma_Z$), which inflated MOFA's error increasingly with $p$. The
cross-validation helpers (`run_cv_grid*`) still use $\hat\Lambda\hat\Lambda^\top$;
they are used only for tuning.

*Differences from the methods document:* the document writes
$\lVert\cdot\rVert_F/(p_lp_m)$ (not squared), omits $\hat\Sigma_Z$, and describes
rescaling by $\Psi^{-1/2}$; the code squares the norm, includes $\hat\Sigma_Z$, and
rescales by the total training standard deviation $D$.

## Test-set reconstruction

| Column | Definition | Floor |
|---|---|---|
| `test_rse` | $\lVert X^{(l)}_{new} - \hat Z_{new}\hat\Lambda^{(l)\top}\rVert_F^2 / \lVert X^{(l)}_{new}\rVert_F^2$, against the **observed** test data | about $1/(1+\mathrm{SNR}_X)$ (0.33 at SNR 2) |

Because the observed data include noise, even a perfect model leaves the noise
share unexplained, so values near $1/(1+\mathrm{SNR}_X)$ are the best possible.
This checks that the frozen loadings generalise and that local inference finds
the test scores.

## Outcome prediction (test set)

| Column | Definition |
|---|---|
| `outcome_mse` | $\frac{1}{n_{test}}\sum_i (\hat y_i - y_i)^2$ on the standardised outcome; $\hat y_i$ = posterior predictive mean (Python models) or OLS prediction from the factors (MOFA) |
| `outcome_95p_coverage` (`outcome_95p_cov` for single view) | share of test outcomes inside their 95% interval |
| `outcome_95p_width` | mean width of those intervals |

**Reference values for `outcome_mse`.** With $\mathrm{SNR}_y = \beta^\top\beta/\sigma_y^2$:

- $1/(1+\mathrm{SNR}_y)$ — the outcome noise share (0.333 at SNR 2). This is the
  MSE when each test sample's *true* scores are known; no method beats it in
  expectation.
- **Best achievable from X** (`oracle_from_x`) — the MSE of
  $\beta^\top E[z \mid x]$ using the true $\Lambda, \Psi, \beta$. Every method has
  to infer the test scores from the views, so this is the realistic floor; it is
  above $1/(1+\mathrm{SNR}_y)$ by an amount that depends on how strongly the
  outcome-relevant factors show up in X (see factor strength below).
- 1.0 — predicting the mean (no information).

**Intervals.** The Python models' intervals are 2.5% / 97.5% quantiles of
posterior predictive draws, so they include the outcome noise and uncertainty
in the test scores and in $\beta$. MOFA's come from the OLS prediction interval
of $y$ on MOFA's test factors, which treats those factors as known exactly, so
they tend to be narrower. With outcome noise alone the width is about
$2 \times 1.96 \times \sqrt{1/(1+\mathrm{SNR}_y)}$ = 2.26 at SNR 2. Read coverage
and width together: the better interval reaches 0.95 coverage with less width.

## Run time

| Column | Meaning |
|---|---|
| `runtime` | seconds for global training (Python: model construction and SVI, excluding SVD initialisation; MOFA: model fit) |
| `predict_runtime` | seconds for local inference on the test set |
| `svd_time` | seconds for the SVD initialisation (0 if not used) |

## Diagnostics: oracles and factor strength

From `python/misaligned_baselines.py` and `factor_interpretation.factor_strength`.

- `oracle_known_z`, `oracle_from_x` — the two outcome floors above.
- `pca5_ols`, `pca15_ols` — PCA of the stacked standardised views (5 or 15
  components), then OLS of $y$ on the training scores: the simplest unsupervised
  two-stage method.
- `ridge` — ridge regression of $y$ on the stacked views (no factors), penalty
  chosen by 5-fold CV on the training set.
- **Factor strength.** $M = \sum_l \Lambda^{(l)\top}\Psi^{(l)-1}\Lambda^{(l)}$ (a
  $K \times K$ matrix). Its eigenvalues are the strengths of the factor
  directions; for a single factor $k$ the strength is
  $\ell_k = \sum_l\sum_j (\Lambda^{(l)}_{jk})^2/\psi^{(l)}_j$, its signal summed over
  all $P$ features relative to the noise. The strength along the outcome
  direction is $\ell_y = \beta^\top M\beta/\beta^\top\beta$. Two thresholds:
  - $\ell > \sqrt{P/n}$: the direction stands out in the sample covariance, so an
    unsupervised method (PCA, MOFA) can find it without the outcome
    (spiked-covariance detection threshold: Baik, Ben Arous & Péché, 2005;
    Paul, 2007). This is an asymptotic guide for an idealised method, not an
    exact boundary for a particular algorithm.
  - $\ell \gtrsim 1$: a new sample's score along the direction can be inferred
    from its views (posterior variance about $1/(1+\ell)$).

  Supervision can only improve prediction over an unsupervised method when the
  outcome-relevant direction is between the two ($1 \lesssim \ell < \sqrt{P/n}$),
  which requires $P > n$. For a fitted model, pass the scores' second moment
  (`Z_second_moment`) so strength is measured on the scale of the fitted signal;
  read the fitted parameters through `pyro.param` (constrained values), not from
  the raw saved state.

## Exact risk and factor strength (raw scale, true parameters)

Code: `python/risk_metrics.py`; misaligned re-scoring in
`python/misaligned_exact_risk.py` → `sim/results/integration_shared_misaligned/exact_risk.csv`.
Unlike the metrics above, these use one replicate's true parameters $\Theta_0$
on the **raw (unstandardised)** scale and involve no test-set noise.

**Definitions.** $G = \sum_l \Lambda_l^\top\Psi_l^{-1}\Lambda_l$ ($K\times K$),
factor strengths $g_k = \operatorname{eig}(G)$, $A = I + G$. Oracle coefficients
$b^*_l = \Psi_l^{-1}\Lambda_l A^{-1}\beta$ (the population regression of $y$ on
$x$), oracle risk $R^* = \sigma_y^2 + \beta^\top A^{-1}\beta$. For any linear
predictor $\hat y = c + \sum_l b_l^\top x_l$ ($x$ has true mean 0),

$$R(b,c) = R^* + (b-b^*)^\top\Sigma_x(b-b^*) + c^2,\qquad
\Sigma_x = \Lambda\Lambda^\top + \Psi \text{ (stacked)}.$$

Methods are fitted on standardised data ($x_s = D(x-\hat\mu)$,
$y_s = (y-\bar y)/\mathrm{sd}_y$, $\hat y_s = c_s + b_s^\top x_s$); on the raw scale
$b = \mathrm{sd}_y D b_s$ and $c = \bar y + \mathrm{sd}_y c_s - b^\top\hat\mu$.

**True $\Psi$.** Read from the data (`psi`, controlled-strength generator) or, for
the earlier generators, reconstructed as $\psi_j = \lVert\lambda_j\rVert^2/\mathrm{SNR}_{x,l}$,
which is how they draw the noise. $\sigma_y^2$ is `sigma2_y` if stored, else
$\beta^\top\beta/\mathrm{SNR}_y$. All-zero features (no loading, no noise) are left out.

| Column | Definition | Read as |
|---|---|---|
| `risk` | $R(b,c)$ | expected squared error on a new raw-scale sample |
| `risk_rel` | $(R - R^*)/R^*$ | **primary metric**: relative excess risk over the best linear predictor; 0 is optimal |
| `risk_over_var_y` | $R/\operatorname{Var}(y)$ | comparable to standardised `outcome_mse` without test noise |
| `risk_excess`, `risk_intercept` | the quadratic term and $c^2$ | where the excess comes from |
| `R_star`, `var_y`, `g_y` | $R^*$, $\operatorname{Var}(y)$, $\beta^\top G\beta/\beta^\top\beta$ | per dataset |
| `kappa2_y` | $\lVert P_{\mathrm{span}}u\rVert^2/\lVert u\rVert^2$, $u = \Psi_0^{-1/2}\Lambda_0\beta$, span $=\operatorname{col}(\Psi_0^{-1/2}\hat L_{raw})$ | how much of the whitened outcome direction the fitted loadings span (rotation invariant); 1 is perfect |
| `kappa2_min_rel` | the same with $u_k = \Psi_0^{-1/2}\lambda_k$, minimum over factors with $\beta_k\ne0$ | worst-recovered relevant factor |
| `span_rank` | rank of the fitted span | caveat: switched-off columns still add span, so a K = 15 fit gets about $(15-K)/P$ of $\kappa^2$ by chance |
| `g_hat` | $\big((\mu-1-\gamma)+\sqrt{(\mu-1-\gamma)^2-4\gamma}\big)/2$ if $\mu > (1+\sqrt\gamma)^2$, else NaN | strength of the outcome-relevant component estimated from the training data ($\gamma = P/n$) |
| `g_hat_mu`, `g_hat_rank`, `g_hat_overlap`, `g_hat_above_threshold` | $\mu$: sample eigenvalue of the training views whitened by the method's fitted noise, for the eigenvector with the largest overlap with the true outcome direction (an oracle choice, diagnostic only) | |

$\hat L_{raw}$: fitted loadings times each feature's training sd (VIPER: posterior
means; PCA: components; MOFA: weights). Fitted noise for `g_hat`: VIPER
$1/E_q[1/\psi_j]$; PCA the PPCA maximum-likelihood isotropic noise (discarded
variance $/(P-K)$); oracle the true $\Psi$. MOFA's noise is not saved, and ridge /
elastic net have no factors, so those columns are empty for them.

**Linear coefficients by method.** VIPER-Shared: closed-form prediction
(`python/viper_predict.py`), $b_{s,l} = \operatorname{diag}(w_l)M_l\bar A^{-1}m_\beta$
(Option 1, `viper_k*`) or its Monte Carlo analogue over $q$ (Option 2,
`viper_k*_mc`); local SVI is not used. PCA + OLS: $b_s = V\,\mathrm{coef}$.
MOFA + OLS: $b_s = W(W^\top W)^{-1}\mathrm{coef}$ (OLS without intercept on the
training factors). Ridge / elastic net: their coefficients and intercept.

**Single-relevant-factor theory** (`risk_metrics.theory_*`; $\beta$ the relevant
coefficient, $g$ its strength): $\kappa^2_{PCA} = (1-\gamma/g^2)/(1+\gamma/g)$ if
$g>\sqrt\gamma$ else 0; $\kappa^2_{OL} = g/(g+\gamma)$;
$R(\kappa^2) = \sigma_y^2 + \beta^2/(1+g\kappa^2)$; headroom
$H = R(\kappa^2_{PCA}) - R(\kappa^2_{OL})$.

## VIPER-Shared test prediction: closed form vs local SVI

`python/viper_predict.py` computes test scores from the trained guide without
local SVI ($M_l$, $S_l$ the loading means / variances, $w_l = a_{\psi,l}/b_{\psi,l} = E_q[1/\psi]$):
$\bar A = I + \sum_l[(M_l w_l)^\top M_l + \operatorname{diag}(\sum_j w_{lj}S_l[j,:])]$,
$m_z = \bar A^{-1}\sum_l M_l^\top\operatorname{diag}(w_l)x_l$, covariance $\bar A^{-1}$
or $\operatorname{diag}(1/\bar A_{kk})$, $\hat y = m_z^\top m_\beta$, predictive variance
$E_q[\sigma_y^2] + \operatorname{tr}(S_\beta\,\mathrm{cov}_z) + m_\beta^\top\mathrm{cov}_z m_\beta + m_z^\top S_\beta m_z$.
This is the exact optimum of the local ELBO that local SVI maximises.
`python/validate_viper_predict.py` (→ `viper_predict_validation.csv`) checks it on
three misaligned fits: the closed form's local ELBO exceeds local SVI's for every
test sample, so the remaining differences are local SVI's optimisation error.

## Known caveats

- Results produced before 9 June 2026 (commit `9df3c33`) inferred test scores
  partly from the test outcome. Their test-outcome metrics (`outcome_mse`,
  coverage, `test_rse`) are invalid; training-set structure and covariance
  metrics are unaffected. This applies to the aligned shared-model study
  (`sim/results/integration_shared/metrics`, May 2026) and the single-view grid
  (`sim/results/matrixdecomp`, April 2026).
- `joint_rse` for the shared model uses the product of posterior means, which can
  differ slightly from the posterior mean of the product used for the other models.
