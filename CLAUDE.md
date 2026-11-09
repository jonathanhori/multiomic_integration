# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements **Bayesian supervised factor models for multiomic data integration** — fitting shared and view-specific latent factors across multiple omics data views while predicting a continuous outcome. Multiple models and inference backends are compared:

- **Python/Pyro**: Stochastic variational inference (SVI) — three model variants
- **R/jafar**: MCMC-based Gibbs sampling (reference implementation)
- **R/MOFA2**: Unsupervised factor analysis (comparison baseline)

## Environment

```bash
conda activate multiomic   # Python 3.11, PyTorch, Pyro
```

The conda environment spec is in `environment.yml`. Additional pip packages (torch, pyro-ppl, rpy2, etc.) are installed separately.

## Architecture

### Intended Data Flow

The pipeline is still being developed — individual scripts are not yet fully wired end to end. The intended flow is:

```
R (generate_data.Rmd)
  → sim/data/{train,test}/*.rds          (simulated multiview datasets)
    → python/data_utils.py                (rpy2 loads RDS, converts to tensors)
      → python/run_methods.py             (one pipeline for all models; model_type selects)
        → python/handler.py               (ModelHandler: SVI training loop)
          → python/model.py               (SupMultiviewDecomp)
          → python/multiview_shared.py    (SupMultiviewShared)
          → python/matrix_decomp.py       (MatrixDecomp)
            → python/sim_run.py           (orchestrates full study per model)
              → sim/results/*/            (saved models + metrics)
                → sim/evaluation.Rmd / analysis.Rmd  (visualize and summarize)
```

Interactive notebooks in `sim/` (`.qmd` Quarto, `.Rmd` R Markdown) are the primary way to run analyses during development.

---

## Python Modules (`python/`)

### Models

| File | Class | Role |
|------|-------|------|
| `model.py` | `SupMultiviewDecomp` | Full supervised multiview model with shared **and** view-specific factors; dense or MGP (sparse) priors |
| `multiview_shared.py` | `SupMultiviewShared` | Simplified variant — shared factors only (no view-specific factors); otherwise identical structure |
| `matrix_decomp.py` | `MatrixDecomp` | Single-view supervised/unsupervised matrix factorization baseline |

### Model Structures

#### `SupMultiviewDecomp` (model.py)

- **Z** (n × k): shared scores — predictive of outcome y
- **Φ_l** (n × k_l): view-specific scores per view l
- **Λ_l** (p_l × k): shared loadings per view l
- **Γ_l** (p_l × k_l): view-specific loadings per view l
- **Reconstruction**: X_l = Z @ Λ_l^T + Φ_l @ Γ_l^T + E_l
- **Outcome**: y_i ~ N(Z_i @ β, σ_y²)
- **Priors**: dense (Inverse-Gamma) or sparse (Multiplicative Gamma Process / MGP)
- Orthogonality of shared scores Z enforced via QR projection after the final training epoch
- Methods: `forward_mgp()` / `guide_mgp()`, `forward_dense()` / `guide_dense()`, `predict_forward()` / `predict_guide()` (local inference for test observations)

#### `SupMultiviewShared` (multiview_shared.py)

- **Z** (n × k): shared scores — predictive of outcome y
- **Λ_l** (p_l × k): shared loadings per view l
- **Reconstruction**: X_l = Z @ Λ_l^T + E_l (no view-specific component)
- **Outcome**: y_i ~ N(Z_i @ β, σ_y²)
- Same prior options (dense / MGP) and training interface as `SupMultiviewDecomp`
- Used to isolate the effect of view-specific factors from shared structure

#### `MatrixDecomp` (matrix_decomp.py)

- **Z** (n × k): latent scores
- **Λ** (p × k): loadings
- **Reconstruction**: X = Z @ Λ^T + E (single view only)
- Optional supervised outcome regression from Z
- Same prior options (dense / MGP)

### Inference & Training

| File | Class/Function | Role |
|------|----------------|------|
| `handler.py` | `ModelHandler` | Training loop: SVI with minibatching, AdagradRMSProp or StepLR scheduler, convergence detection, orthogonal projection of scores at convergence |
| `run_methods.py` | many | One pipeline for all three models, chosen with `model_type="single_view" \| "shared" \| "viper"`: `load_rds()`, `process_data()`, `train_model()`, `train_model_locally()`, `evaluate_fitted_model()`, `load_model()`. The module docstring documents every `model_config` / `train_config` key. Old `_shared` / `_single_view` names remain as thin wrappers for existing scripts. `run_cv_grid()` / `run_cv_grid_shared()` for cross-validation |
| `data_utils.py` | many | `r_to_torch()`, `floatmatrix_to_torch()`, `load_and_process_rds_data_for_condition()`, `eval_performance()`, `calc_all_structures_with_rescaling()` |
| `constants.py` | `Sites`, `Params` | Canonical string names for all Pyro sample sites and variational parameters |
| `viper_predict.py` | many | Closed-form test-time scores / predictions for `SupMultiviewShared` from the param store (mean-field Option 1, Monte Carlo Option 2), with linear coefficients; `expected_tau()` for MGP pruning |
| `risk_metrics.py` | many | Exact risk R(b,c) of linear predictors from the true parameters, oracle b*/R*, `kappa2_hat`, `g_hat`, single-factor theory curves (docs/evaluation_metrics.md) |

### Entry Points

| File | Purpose |
|------|---------|
| `sim_run.py` | Full simulation study for `SupMultiviewDecomp`; iterates over `(n, p, SNR, sparsity, k_delta)` grid; outputs to `sim/results/integration/` |
| `sim_run_shared.py` | Same study for `SupMultiviewShared`; data from `sim/data/multi_view_shared/k.{K}/`; outputs to `sim/results/integration_shared/` |
| `hyperparam_run.py` | Grid search over training hyperparameters (LR, decay, minibatch) for `SupMultiviewDecomp` |
| `matrixdecomp_hyperparam_run.py` | Same grid search for `MatrixDecomp` baseline |
| `sim_run_shared_parallel.py` | Shared-model study incl. misaligned scenarios (`--task-id` for cluster arrays) |
| `misaligned_exact_risk.py` | Exact risk / kappa2 / g_hat for all linear methods on the misaligned datasets → `exact_risk.csv` |
| `check_controlled_pipeline.py` | Controlled-strength data through load_rds → process_data → risk_metrics (R* and exact-risk checks) |
| `validate_viper_predict.py` | Closed-form vs local-SVI VIPER-Shared prediction on 3 misaligned fits |
| `outcome_weight_sweep.py` | `outcome_weight` × minibatch sweep across misaligned scenarios; resumable CSV, plotted in `sim/exploration/shared_outcome_weight.qmd` |
| `validate_settings.py` | Candidate optimiser settings (current / default / long) + all comparison methods on the controlled grid's test sets, with factor-recovery measures → `integration_shared_controlled/validation/`; read in `sim/exploration/settings_validation.qmd` |
| `alpha_zero_controlled.py` | Supervision vs sparsity: outcome_weight 0 vs 1 on the Optuna c = 1 folds, scored every 100 epochs |

---

## R Modules (`R/`)

| File | Role |
|------|------|
| `sim_methods.R` | `generate_factor_loadings()`, `generate_factor_scores()`, `generate_factor_data()` — synthetic data generation with controllable SNR, sparsity, view-specific structure |
| `sim_methods.R` (controlled) | `generate_factor_data_shared_controlled()` / `_testing()` / `controlled_oracle()` — unit noise, relevant factor with set strength g on its own support, outside-factor term; documented in `docs/controlled_strength_data.md`, checked by `check_controlled_generator.R` |
| `data_methods.R` | `load_data()`, `standardize_views()`, `summarise_post_samples()`, `eval_performance()` — preprocessing and MCMC output handling |
| `model_run.R` | `run_and_eval_model()` — fits `jafar` MCMC models; manages train/test splits; saves metrics to `sim/results/jafar/` |
| `mofa_run.R` | `fit_mofa()`, `project_mofa_test()`, `fit_outcome_mofa()`, `predict_outcome_mofa()`, `eval_mofa_shared()` — MOFA2 factor analysis wrapper for comparison |

---

## Evaluation Pipeline

Each model is evaluated on the same simulation conditions using these steps:

1. **Load data** via `load_and_process_rds_data_for_condition()` (rpy2 → tensors)
2. **Normalize** features to zero mean / unit variance per view
3. **Train global model** on training split via `ModelHandler.do_inference()`
4. **Local inference** on held-out test observations — infer Z and Φ_l (loadings frozen)
5. **Compute metrics** via `evaluate_fitted_model(..., model_type=...)`, which also returns posterior samples:
   - **RSE** (squared relative error): `||X_est - X_true||_F^2 / ||X_true||_F^2` per view; `test_rse` compares test reconstructions with the observed test views (floor ≈ 1/(1+snr_x))
   - **Outcome MSE**: mean squared error on test y predictions
   - **95% Credible Interval Coverage**: proportion of true values inside posterior intervals
   - **Covariance Error**: Frobenius norm between estimated and true covariance structures
6. **Save** metrics as RDS / CSV to `sim/results/{model_name}/metrics/`, models to `sim/results/{model_name}/models/`

File naming convention for saved artefacts:
```
run_autonormalguide_n{n}p{p}_snr{snr_x}.{snr_y}_sparse{sparsity}_deltak{k_delta}_rep{rep}.pth
```

### Key Training Hyperparameters

**`sim_run.py` defaults** (SupMultiviewDecomp):
```python
MIN_EPOCHS = 100
NUM_EPOCHS = 1000
MINIBATCH_SIZE = 32
initial_lr = 0.005
gamma = 0.1          # final_lr = gamma * initial_lr
N_POSTERIOR_SAMPLES = 500
VARIATIONAL_TOL = 0.3
```

**`sim_run_shared.py` defaults** (SupMultiviewShared):
```python
MIN_EPOCHS = 200
NUM_EPOCHS = 1000
ETA = 0.5            # AdagradRMSProp
DELTA = 1e-16
T = 1.0
```

---

## Simulation Notebooks (`sim/`)

| File | Purpose |
|------|---------|
| `generate_data.Rmd` | Generates synthetic multiview datasets; controls n, p, K, K_l, SNR, sparsity |
| `python_integration.qmd` | Documented single-condition pipeline for `SupMultiviewShared` (load → evaluate → interpret) |
| `python_factorization.qmd` | Documented single-condition pipeline for `MatrixDecomp` |
| `python_viper.qmd` | Pipeline skeleton for `SupMultiviewDecomp` (VIPER) |
| `exploration/` | Focused experiments: outcome weight, stopping rule / SVD init, CV + Optuna, variance diagnostics |
| `archive/` | Notebooks with past (non-current) results and earlier pipeline notebooks, kept unchanged; `archive/README.md` lists the past experiments |
| `simulation_summary.qmd` | Current simulation results only (misaligned study, all comparison methods) |
| `integration_tuning.qmd` | Older hyperparameter tuning exploration (SupMultiviewDecomp) |
| `tcga_real_data.qmd` | Real-data application on TCGA cancer dataset |

---

## Data Paths

| Path | Contents |
|------|----------|
| `sim/data_ortho/{train,test}/` | VIPER (view-specific factor) data used by `hyperparam_run.py` and `python_viper.qmd`. (`sim_run.py` still points at `sim/data/train/`, which does not exist.) |
| `sim/data/multi_view_shared_misaligned/k.{K}/` | Misaligned-scenario data for the shared model |
| `sim/data/multi_view_shared/k.{K}/` | Data for shared-structure simulations |
| `sim/data/multi_view_shared_controlled/k.5/` | (Phase 2, not yet generated) controlled-strength data from `generate_factor_data_shared_controlled`; see `docs/controlled_strength_data.md` |
| `sim/data/single_view/` | Single-view data for `MatrixDecomp` |
| `sim/data/single_view_withoutcome/` | Single-view data with outcome |
| `sim/results/integration/` | `SupMultiviewDecomp` models and metrics |
| `sim/results/integration_shared/` | `SupMultiviewShared` models and metrics |
| `sim/results/matrixdecomp/` | `MatrixDecomp` models and metrics |
| `sim/results/jafar/` | jafar MCMC models and metrics |
| `data/tcga/` | Real TCGA omics data |
| `analysis/` | Real-data analyses (one folder each, with its own `results/` and `figures/`); never write real-data outputs to `sim/`. `analysis/tcga_coadread/`: TMB from expression + TSS200 methylation (`tcga_coadread.qmd`, `compare_methods.py`, `fit_mofa.R`) |

---

## R–Python Bridge

`rpy2` is used to load `.rds` files from R into Python. Key conversion utilities in `data_utils.py`:
- `r_to_torch()`: recursively converts R lists/matrices/dataframes to Python dicts/tensors
- `floatmatrix_to_torch()`: handles R column-major matrix layout → PyTorch tensor
- Data is loaded per condition via `load_and_process_rds_data_for_condition()`
