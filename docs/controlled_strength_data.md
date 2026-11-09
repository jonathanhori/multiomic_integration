# Controlled-strength simulation data

How the weak-factor study's data are generated. Code:
`R/sim_methods.R::generate_factor_data_shared_controlled` (training replicate) and
`generate_factor_data_shared_controlled_testing` (test set); unit checks in
`R/check_controlled_generator.R` and `python/check_controlled_pipeline.py`.

## Why a new generator

In the earlier generators (`generate_factor_data_shared*`) each feature's noise
variance is tied to its own signal, $\psi_j = \lVert\lambda_j\rVert^2/\mathrm{SNR}_x$.
Then every factor's strength $g_k$ grows linearly in $P$, so every factor is
strong ($g \gg \sqrt{\gamma}$, $\gamma = P/n$) and unsupervised methods find the
outcome-relevant direction as well as a supervised one (the misaligned cells have
$g_y/\sqrt\gamma$ between 22 and 346; `exact_risk.csv`). Supervision can only help
when the relevant factor is weak, $g \approx \sqrt\gamma$. This generator fixes the
noise and sets the relevant factor's strength directly.

## Data-generating process (one replicate)

$L$ views with $p_l$ features ($P = \sum_l p_l$), $K = K_{nuis} + 1$ factors, $N$ samples.

$$
z_i \sim N(0, I_K),\qquad
x_{il} = \Lambda_l z_i + \varepsilon_{il},\ \ \varepsilon_{il} \sim N(0, I_{p_l}),\qquad
y_i = \beta_{k^*} z_{ik^*} + \theta^\top\varepsilon_i + e_i,\ \ e_i \sim N(0, \sigma_y^2).
$$

| Component | Rule | Default |
|---|---|---|
| Noise | $\psi_j = 1$ for every feature (independent of the loadings) | |
| Relevant factor $k^*$ | column `k_star` | last column ($K$) |
| Support $S$ | $s = \operatorname{round}(g_{target}/a^2)$ features; view $l$ gets $\lfloor s/L\rfloor$, the first $s \bmod L$ views one more; uniform at random within each view | |
| Relevant loadings | $\Lambda_{jk^*} \sim N(0, a^2)$ on $S$, 0 elsewhere | $a = 0.5$ |
| Strength | $g = \sum_{j\in S}\Lambda_{jk^*}^2/\psi_j$; $E[g] = s a^2$; returned as `g_true` | |
| Nuisance factors | $K_{nuis}$ columns; on features **not** in $S$: $\Lambda_{jk} = N(0, \sigma_{nuis}^2)\times\mathrm{Bernoulli}(d)$; 0 on $S$ | $K_{nuis}=4$, $\sigma_{nuis}=1$, $d=0.75$ |
| Outcome coefficients | $\beta_k = 0$ for nuisance factors, $\beta_{k^*} \sim N(0,1)$ | |
| Outcome noise | $\sigma_y^2 = \beta_{k^*}^2/\mathrm{SNR}_y$ | $\mathrm{SNR}_y = 2$ |
| Outside-factor signal | off unless $r_{out} > 0$: $n_{out}$ features drawn uniformly from those not in $S$; $\theta$ zero elsewhere, $\theta_{out}\sim N(0,I)$ rescaled to $\lVert\theta\rVert^2 = \frac{r_{out}}{1-r_{out}}(\beta_{k^*}^2+\sigma_y^2)$, so $\theta^\top\varepsilon$ explains a share $r_{out}$ of $\operatorname{Var}(y)$ | $r_{out}=0$, $n_{out}=5$ |
| Test set | same $\Lambda$, $\psi$, $\beta$, $\sigma_y^2$, $\theta$, supports; fresh $z$, $\varepsilon$, $e$ | |

Consequences:

- Supports are disjoint, so $G = \sum_l\Lambda_l^\top\Psi_l^{-1}\Lambda_l$ has row
  and column $k^*$ exactly zero off the diagonal and `g_true` $= G_{k^*k^*}$ is an
  eigenvalue of $G$.
- Nuisance factors are strong: $g_k \approx d\,\sigma_{nuis}^2 (P - s)$, e.g. about 450 at $P = 600$.
- $\operatorname{Var}(y) = \beta_{k^*}^2 + \sigma_y^2 + \lVert\theta\rVert^2$.
- Population oracle: $\operatorname{Cov}(x, y) = \Lambda\beta + \Psi\theta$,
  $b^* = \Sigma_x^{-1}\operatorname{Cov}(x,y)$, $R^* = \operatorname{Var}(y) - \operatorname{Cov}(x,y)^\top b^*$
  (`controlled_oracle` in R, `risk_metrics.oracle` in Python); with $r_{out}=0$,
  $R^* = \sigma_y^2 + \beta^\top A^{-1}\beta$, $A = I + G$.

**Realised vs target strength.** $g$ is a sum of $s$ scaled $\chi^2_1$ draws, so its
coefficient of variation is about $\sqrt{2/s}$: 1.4 at $s = 1$, 0.45 at $s = 10$,
0.1 at $s = 200$. At the smallest targets the support has one to three features
(and some views have none), and rounding moves $E[g]$ off $g_{target}$ (0.25 vs
0.33 at $s=1$). Analyses should put the realised `g_true` on the x-axis.
`rescale_to_target = TRUE` rescales the support loadings so that $g = g_{target}$
exactly. **TODO (design choice, user):** whether to use it.

## Returned fields

The fields of `generate_factor_data_shared_asymmetric` (`L, N, p_l, K, sparsity,
beta, snr_x, snr_y, factor_scale, beta_weights, X_l, y, Lambda_l, Z`), so
`run_methods.load_rds` / `process_data` work unchanged, plus:

| Field | Content |
|---|---|
| `psi` | list of $p_l$-vectors of noise variances (all 1) |
| `sigma2_y` | outcome noise variance |
| `g_true`, `g_target`, `a`, `s`, `k_star` | realised and target strength, loading sd, support size, relevant column (1-based) |
| `support` | list per view of the 1-based indices of $S$ |
| `theta_out`, `out_idx`, `r_out` | outside-factor coefficients (list of $p_l$-vectors), their 1-based feature indices per view, target share |
| `rescale_to_target` | whether loadings were rescaled to hit $g_{target}$ |
| `G` | $K\times K$ strength matrix |

Notes: `snr_x` here is each view's realised mean signal-to-noise ratio
$\sum\Lambda_l^2/p_l$ (the noise is not tied to it); `sparsity` is $1-d$;
`beta_weights` is the indicator of $k^*$; `factor_scale` is all 1.

## Checks (`R/check_controlled_generator.R`, `python/check_controlled_pipeline.py`)

- mean realised $g$ equals $s a^2$ (200 replicates at four targets) and `g_true`
  equals the sum of squared support loadings; $G_{k^*,-k^*} = 0$ exactly
- no nuisance loading on $S$, relevant loadings only on $S$, outside features not
  in $S$; the testing companion keeps $\Theta_0$ and draws new scores
- $\operatorname{Var}(y)$ decomposition and outside share on 40 000 samples
- $R^*$ equals the oracle predictor's MSE on 40 000 test samples, with and without
  the outside term; with $r_{out}=0$ also equal to `oracle_from_x` $=\beta^\top E[z\mid x]$
- through Python: `process_data` keeps all features, `risk_metrics` reads `psi`,
  `sigma2_y`, `theta_out`, its $R^*$ matches R's, and a ridge fit's exact risk
  matches its large-sample MSE

## Datasets

Phase 2 data will be written to
`sim/data/multi_view_shared_controlled/k.5/{train,test}/` (grid and stems are
defined in Phase 2).
