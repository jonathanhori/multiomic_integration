# Shared-model results with the default guide start (October–November 2026)

Fits of SupMultiviewShared on TCGA COAD/READ with the default initial guide standard
deviations (init_scale_Lambda = init_scale_Z = 1). Superseded by the low-variance
guide start (init_scale_Lambda = 0.01, init_scale_Z = 0.1), with the same optimiser
(eta 0.25, t 0.1, minibatch 0.15). MOFA, PCA, ridge and elastic-net results do not
depend on this setting and were kept in ../.

- convergence_check.csv / convergence_loss_*.csv: K = 20, fixed 100–1600 epochs (split 42)
- method_comparison_splits.csv, method_predictions.csv: all methods, five splits, K = 20
  (the shared rows were removed from the current copies)
- splits/seed*/shared_factors.npz: K = 20 factors per split
- k_sweep/: K = 20, 50, 100 per split (seed*/k*/shared_factors.npz, k_sweep_splits.csv),
  K = 100 convergence check (convergence_k100*), OLS-beta refit, and selection/
  (K_max = 20–150, 800 epochs, splits 42 and 2, with parameter stores)
- model_expr_meth_tss200_k20*.pth: the notebook's saved fit
    stale_outputs/   figures/tables from the default-guide notebook no longer produced by it
