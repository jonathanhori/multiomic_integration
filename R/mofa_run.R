library(MOFA2)
library(tidyverse)

# ---------------------------------------------------------------------------
# Core MOFA2 helpers
# ---------------------------------------------------------------------------

# Fit MOFA2 on training data.
#
# X_l_train : list of (n x p_l) matrices — samples as rows (will be transposed
#             internally for MOFA2's features-x-samples convention)
# K         : number of latent factors
# Returns   : trained MOFA2 object
fit_mofa <- function(X_l_train, K, seed = 123, verbose = FALSE) {
  mofa_data <- lapply(seq_along(X_l_train), function(l) t(X_l_train[[l]]))
  names(mofa_data) <- paste0("view", seq_along(X_l_train))

  mofa_obj <- create_mofa(mofa_data)

  model_opts <- get_default_model_options(mofa_obj)
  model_opts$num_factors <- K

  train_opts <- get_default_training_options(mofa_obj)
  train_opts$seed    <- seed
  train_opts$verbose <- verbose

  mofa_obj <- prepare_mofa(
    mofa_obj,
    model_options   = model_opts,
    training_options = train_opts
  )
  run_mofa(mofa_obj, use_basilisk = TRUE)
}


# Project new observations onto MOFA factor space via OLS.
#
# Stacks all views horizontally and solves Z = X_concat W_concat (W^T W)^{-1}.
# This is equivalent to the GLS estimator when noise variances are equal, and
# is fast + closed-form (no second MOFA run required).
#
# mofa_obj  : trained MOFA2 object
# X_l_test  : list of (n_test x p_l) matrices — same feature order as training
# Returns   : (n_test x K) factor matrix
project_mofa_test <- function(mofa_obj, X_l_test) {
  W_list   <- get_weights(mofa_obj)          # list: p_l x K per view
  X_concat <- do.call(cbind, X_l_test)       # n_test x sum(p_l)
  W_concat <- do.call(rbind, W_list)         # sum(p_l) x K

  X_concat %*% W_concat %*% solve(t(W_concat) %*% W_concat)
}


# Fit OLS regression y ~ Z on training factors (no intercept; y is centered).
fit_outcome_mofa <- function(mofa_obj, y_train) {
  Z_train <- get_factors(mofa_obj)[[1]]      # (n_train x K)
  stats::lm(y_train ~ Z_train - 1)
}


# Predict outcome for test observations given their projected factors.
#
# Returns a matrix with columns fit / lwr / upr when interval != "none",
# or a plain numeric vector when interval = "none".
predict_outcome_mofa <- function(lm_fit, Z_test, interval = "none", level = 0.95) {
  stats::predict(lm_fit,
                 newdata  = list(Z_train = Z_test),
                 interval = interval,
                 level    = level)
}


# Evaluate structure recovery, test reconstruction, and outcome prediction.
#
# Returns a data.frame with columns matching evaluate_fitted_model_shared():
#   view, joint_rse, test_rse, joint_95p_coverage (NA),
#   outcome_mse, outcome_95p_coverage (NA)
#
# joint_rse  : RSE of training structure reconstruction vs rescaled sim structure
# test_rse   : RSE of test data reconstruction (Z_test @ W_l^T vs X_l_test_std)
eval_mofa_shared <- function(mofa_obj,
                              Z_test,
                              y_test_std,
                              lm_fit,
                              sim_data,
                              X_l_means,
                              X_l_sds,
                              col_filters,
                              train_idx,
                              X_l_test_std) {
  L      <- length(sim_data$Lambda_l)
  W_list <- get_weights(mofa_obj)            # list: p_l x K per view
  Z_hat  <- get_factors(mofa_obj)[[1]]       # n_train x K

  # Simulated structure in standardised space: (Z @ Lambda^T - mean) / sd
  # This matches scale_sim_struct() in data_utils.py.
  sim_joint_train <- lapply(seq_len(L), function(l) {
    Lambda_filt  <- sim_data$Lambda_l[[l]][col_filters[[l]], , drop = FALSE]
    struct_raw   <- sim_data$Z[train_idx, ] %*% t(Lambda_filt)
    sweep(sweep(struct_raw, 2, X_l_means[[l]], `-`), 2, X_l_sds[[l]], `/`)
  })

  est_joint_train <- lapply(seq_len(L), function(l) {
    Z_hat %*% t(W_list[[l]])
  })

  # joint_rse: ||sim_struct_std - est_struct||_F^2 / ||sim_struct_std||_F^2
  joint_rse <- sapply(seq_len(L), function(l) {
    diff  <- sim_joint_train[[l]] - est_joint_train[[l]]
    norm(diff, "F")^2 / norm(sim_joint_train[[l]], "F")^2
  })

  # test_rse: ||X_l_test_std - Z_test @ W_l^T||_F^2 / ||X_l_test_std||_F^2
  test_rse <- sapply(seq_len(L), function(l) {
    recon <- Z_test %*% t(W_list[[l]])
    diff  <- X_l_test_std[[l]] - recon
    norm(diff, "F")^2 / norm(X_l_test_std[[l]], "F")^2
  })

  # Outcome — prediction intervals from OLS (conditional on Z_test as fixed)
  pred_int    <- predict_outcome_mofa(lm_fit, Z_test,
                                      interval = "prediction", level = 0.95)
  y_pred      <- pred_int[, "fit"]
  outcome_mse <- mean((y_pred - y_test_std)^2)
  outcome_95p_coverage <- mean(y_test_std >= pred_int[, "lwr"] &
                                y_test_std <= pred_int[, "upr"])

  data.frame(
    view                 = seq_len(L) - 1L,  # 0-indexed to match Python
    joint_rse            = joint_rse,
    test_rse             = test_rse,
    joint_95p_coverage   = NA_real_,
    outcome_mse          = outcome_mse,
    outcome_95p_coverage = outcome_95p_coverage
  )
}


# Evaluate within-view and cross-view covariance reconstruction.
#
# Matches eval_cov_shared() in data_utils.py.
# cov_frob = ||est_cov - sim_cov||_F^2 / (p_l * p_m)
#
# W_list              : MOFA loadings list, p_l x K per view
# sim_data            : original RDS data list (contains Lambda_l)
# X_l_sds             : list of per-feature sds from training data
# col_filters         : list of logical vectors (TRUE = kept feature)
#
# Returns a data.frame: view_l, view_m, cov_type ("within"/"cross"), cov_frob
eval_cov_mofa <- function(W_list, sim_data, X_l_sds, col_filters) {
  L <- length(W_list)

  # Simulated loadings in standardised space: Lambda_std = Lambda_raw / sd
  # Matches scale_sim_cov(raw_cov, sd_l, sd_m) = diag(1/sd_l) @ raw_cov @ diag(1/sd_m)
  sim_Lambda_std <- lapply(seq_len(L), function(l) {
    Lambda_filt <- sim_data$Lambda_l[[l]][col_filters[[l]], , drop = FALSE]
    sweep(Lambda_filt, 1, X_l_sds[[l]], `/`)
  })

  rows <- list()
  for (l in seq_len(L)) {
    for (m in seq(l, L)) {
      est_cov <- W_list[[l]] %*% t(W_list[[m]])
      sim_cov <- sim_Lambda_std[[l]] %*% t(sim_Lambda_std[[m]])
      p_l     <- nrow(W_list[[l]])
      p_m     <- nrow(W_list[[m]])
      rows[[length(rows) + 1L]] <- data.frame(
        view_l   = l - 1L,
        view_m   = m - 1L,
        cov_type = if (l == m) "within" else "cross",
        cov_frob = norm(est_cov - sim_cov, "F")^2 / (p_l * p_m)
      )
    }
  }
  do.call(rbind, rows)
}


# ---------------------------------------------------------------------------
# Simulation sweep (mirrors sim_run_shared.py)
# ---------------------------------------------------------------------------

# High-level wrapper: load one sim rep, fit MOFA2, evaluate, return metric row.
#
# Parameters mirror sim_run_shared.py grid variables.
run_and_eval_mofa_shared <- function(n, p, snr_x, snr_y, sparsity, k_delta,
                                      rep,
                                      train_data_root,
                                      test_data_root,
                                      metric_out_path,
                                      K_true      = 5,
                                      seed        = 123,
                                      align_label = NULL) {
  # cond_dir and out_name differ between standard and asymmetric modes.
  if (is.null(align_label)) {
    cond_dir <- sprintf("n%sp%s_snr%s.%s_sparse%s", n, p, snr_x, snr_y, sparsity)
    out_name <- sprintf("mofa_n%sp%s_snr%s.%s_sparse%s_deltak%s_rep%s",
                        n, p, snr_x, snr_y, sparsity, k_delta, rep)
  } else {
    cond_dir <- sprintf("n%sp%s_snr%s.%s_sparse%s_align_%s",
                        n, p, snr_x, snr_y, sparsity, align_label)
    out_name <- sprintf("mofa_asym_n%sp%s_align%s_deltak%s_rep%s",
                        n, p, align_label, k_delta, rep)
  }
  file_name   <- sprintf("sim_data_shared_rep%s.rds", rep)
  metric_file <- file.path(metric_out_path, paste0(out_name, ".rds"))

  message("  rep ", rep, ": metric_file=", metric_file)
  if (file.exists(metric_file)) {
    message("  rep ", rep, ": already done, skipping.")
    return(invisible(NULL))
  }

  train_path <- file.path(train_data_root, cond_dir, file_name)
  test_path  <- file.path(test_data_root,  cond_dir, file_name)
  message("  rep ", rep, ": train_path=", train_path)
  message("  rep ", rep, ": test_path=", test_path)

  if (!file.exists(train_path) || !file.exists(test_path)) {
    message("  rep ", rep, ": data not found, skipping.")
    return(invisible(NULL))
  }

  train_data <- readRDS(train_path)
  test_data  <- readRDS(test_path)
  message("  rep ", rep, ": data loaded OK")

  X_l_train <- train_data$X_l
  X_l_test  <- test_data$X_l
  y_train   <- as.numeric(train_data$y)
  y_test    <- as.numeric(test_data$y)
  n_train   <- nrow(X_l_train[[1]])

  # Remove zero-variance columns (using training data)
  col_filters <- lapply(X_l_train, function(X) apply(X, 2, stats::sd) > 0)
  X_l_train   <- lapply(seq_along(X_l_train),
                        function(l) X_l_train[[l]][, col_filters[[l]], drop = FALSE])
  X_l_test    <- lapply(seq_along(X_l_test),
                        function(l) X_l_test[[l]][,  col_filters[[l]], drop = FALSE])

  # Standardise with training statistics
  X_l_means <- lapply(X_l_train, colMeans)
  X_l_sds   <- lapply(X_l_train, function(X) apply(X, 2, stats::sd))

  standardise <- function(X, means, sds) {
    sweep(sweep(X, 2, means, `-`), 2, sds, `/`)
  }
  X_l_train_std <- mapply(standardise, X_l_train, X_l_means, X_l_sds, SIMPLIFY = FALSE)
  X_l_test_std  <- mapply(standardise, X_l_test,  X_l_means, X_l_sds, SIMPLIFY = FALSE)

  y_mean <- mean(y_train)
  y_sd   <- stats::sd(y_train)
  y_train_std <- (y_train - y_mean) / y_sd
  y_test_std  <- (y_test  - y_mean) / y_sd

  K_fit     <- K_true + k_delta
  train_idx <- seq_len(n_train)

  tik <- Sys.time()
  tryCatch({
    message("  rep ", rep, ": fitting MOFA (K=", K_fit, ")")
    mofa_obj <- fit_mofa(X_l_train_std, K = K_fit, seed = seed)
    tok <- Sys.time()
    message("  rep ", rep, ": MOFA fit done")

    message("  rep ", rep, ": fitting outcome model")
    lm_fit <- fit_outcome_mofa(mofa_obj, y_train_std)
    message("  rep ", rep, ": outcome model done")

    message("  rep ", rep, ": projecting test data")
    Z_test <- project_mofa_test(mofa_obj, X_l_test_std)
    message("  rep ", rep, ": Z_test dim=", paste(dim(Z_test), collapse="x"))

    message("  rep ", rep, ": evaluating shared structure")
    eval_tbl <- eval_mofa_shared(
      mofa_obj, Z_test, y_test_std, lm_fit,
      train_data, X_l_means, X_l_sds, col_filters, train_idx,
      X_l_test_std = X_l_test_std
    ) |>
      mutate(
        n           = n,
        p           = p,
        snr_x       = snr_x,
        snr_y       = snr_y,
        rep         = rep,
        sparsity    = sparsity,
        k_delta     = k_delta,
        runtime     = as.numeric(difftime(tok, tik, units = "secs")),
        align_label = if (is.null(align_label)) NA_character_ else align_label
      )
    message("  rep ", rep, ": eval_tbl done, nrow=", nrow(eval_tbl))

    message("  rep ", rep, ": evaluating covariance")
    W_list  <- get_weights(mofa_obj)
    cov_tbl <- eval_cov_mofa(W_list, train_data, X_l_sds, col_filters) |>
      mutate(
        n           = n,
        p           = p,
        snr_x       = snr_x,
        snr_y       = snr_y,
        rep         = rep,
        sparsity    = sparsity,
        k_delta     = k_delta,
        align_label = if (is.null(align_label)) NA_character_ else align_label
      )
    message("  rep ", rep, ": cov_tbl done, nrow=", nrow(cov_tbl))

    message("  rep ", rep, ": saving to ", metric_out_path)
    dir.create(metric_out_path, recursive = TRUE, showWarnings = FALSE)
    saveRDS(eval_tbl, metric_file)
    saveRDS(cov_tbl, file.path(metric_out_path, paste0(out_name, "_cov.rds")))
    message("  rep ", rep, ": files saved OK")

    message("  rep ", rep, ": done.  outcome_mse=",
            round(eval_tbl$outcome_mse[1], 4),
            "  mean_test_rse=", round(mean(eval_tbl$test_rse), 4))
    return(eval_tbl)

  }, error = function(e) {
    message("  rep ", rep, ": FAILED — ", conditionMessage(e))
    return(invisible(NULL))
  })
}


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

if (!interactive()) {
  K_TRUE <- 5
  REPS   <- 10

  # ---------------------------------------------------------------------------
  # Rscript mofa_run.R [mode] [task_id]
  #   mode    : "standard" (default) or "asymmetric"
  #   task_id : optional integer row index; omit to run all rows
  # ---------------------------------------------------------------------------
  cli_args <- commandArgs(trailingOnly = TRUE)
  mode     <- if (length(cli_args) >= 1) cli_args[1] else "standard"
  task_id  <- if (length(cli_args) >= 2) as.integer(cli_args[2]) else NULL

  if (mode == "standard") {
    TRAIN_ROOT <- "~/multiomic_integration/sim/data/multi_view_shared/k.5/train"
    TEST_ROOT  <- "~/multiomic_integration/sim/data/multi_view_shared/k.5/test"
    METRIC_OUT <- "~/multiomic_integration/sim/results/integration_shared/metrics_mofa"

    sim_grid <- expand.grid(
      n           = c(100, 500, 1000),
      p           = c(100, 500, 1000),
      snr_x       = c(2),
      snr_y       = c(2),
      sparsity    = c(0.25),
      k_delta     = c(0, 10),
      align_label = NA_character_,
      stringsAsFactors = FALSE
    )

  } else if (mode == "asymmetric") {
    TRAIN_ROOT <- "~/multiomic_integration/sim/data/multi_view_shared_misaligned/k.5/train"
    TEST_ROOT  <- "~/multiomic_integration/sim/data/multi_view_shared_misaligned/k.5/test"
    METRIC_OUT <- "~/multiomic_integration/sim/results/integration_shared_misaligned/metrics_mofa"

    sim_grid <- expand.grid(
      n           = c(100, 500, 1000),
      p           = c(50, 100, 500, 1000),
      snr_x       = 2,
      snr_y       = 2,
      sparsity    = 0.25,
      k_delta     = c(0, 10),
      align_label = c("symmetric", "mild_misalignment",
                      "strong_misalignment", "anti_aligned"),
      stringsAsFactors = FALSE
    )

  } else {
    stop("Unknown mode '", mode, "'. Use 'standard' or 'asymmetric'.")
  }

  dir.create(path.expand(METRIC_OUT), recursive = TRUE, showWarnings = FALSE)

  rows_to_run <- if (!is.null(task_id)) task_id else seq_len(nrow(sim_grid))

  for (i in rows_to_run) {
    row   <- sim_grid[i, ]
    al    <- if (is.na(row$align_label)) NULL else row$align_label
    label <- if (is.null(al)) "" else paste0(" align=", al)
    message("\n=== n=", row$n, " p=", row$p,
            " snr=", row$snr_x, ".", row$snr_y,
            " sparse=", row$sparsity,
            " k_delta=", row$k_delta, label, " ===")

    for (rep in seq_len(REPS)) {
      run_and_eval_mofa_shared(
        n               = row$n,
        p               = row$p,
        snr_x           = row$snr_x,
        snr_y           = row$snr_y,
        sparsity        = row$sparsity,
        k_delta         = row$k_delta,
        rep             = rep,
        train_data_root = TRAIN_ROOT,
        test_data_root  = TEST_ROOT,
        metric_out_path = METRIC_OUT,
        K_true          = K_TRUE,
        align_label     = al
      )
    }
  }
}
