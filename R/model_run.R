

run_and_eval_model <- function(n, p, snr_x, snr_y, rep, sparsity,
                               in_path_base,
                               in_name_base,
                               out_path_base,
                               out_name_base) {
  model_out_data_path <- file.path(out_path_base,
                                   "models",
                                   sprintf(out_name_base,  n, p, snr_x, snr_y, sparsity, rep))
  # Skip dataset if model already run
  if (file.exists(model_out_data_path)) {NA}
  else {
    
    # Load + process data
    run_data <- load_data(in_path_base, in_name_base,
                          n, p, snr_x, snr_y, sparsity, rep)
    X_l <- run_data$X_l
    y <- run_data$y
    
    # Perform train/test split
    if (TRAINING_SPLIT) {
      set.seed(RANDOM_SEED)
      training_idx <- sample.int(n = n, size = floor(TRAINING_SIZE * n))
      
      X_l_train <- lapply(X_l, function(X) X[training_idx, ])
      X_l_test <- lapply(X_l, function(X) X[-training_idx, ])
      
      y_train <- y[training_idx]
      y_test <- y[-training_idx]
    } else {
      X_l_train <- X_l
      X_l_test <- X_l
      
      y_train <- y
      y_test <- y
    }
    
    X_l_list_column_filters <- lapply(X_l_train, function(X) zero_variance_col_filter(X))
    
    X_l_train <- lapply(1:length(X_l_train), 
                        function(l) X_l_train[[l]][, X_l_list_column_filters[[l]]])
    X_l_test <- lapply(1:length(X_l_test), 
                       function(l) X_l_test[[l]][, X_l_list_column_filters[[l]]])
    
    # Calculate mean and sd of each feature - for rescaling simulated structures
    X_l_means <- lapply(X_l_train, function(X) {
      means <- X |> colMeans()
      means
      # t(replicate(nrow(X), means))
    })
    X_l_sds <- lapply(X_l_train, function(X) {
      sds <- X |> colVars() |> sqrt()
      sds
      # diag(1 / sds)
    })
    
    y_mean <- mean(y_train)
    y_sd <- sd(y_train)
    
    log_message("Standardizing")
    # Standardize using training set means/sds
    X_l_train.clean <- standardize_views(X_l_train,
                                         list(means = X_l_means,
                                              sds = X_l_sds))
    X_l_test.clean <- standardize_views(X_l_test,
                                        list(means = X_l_means,
                                             sds = X_l_sds))
    
    y_train.clean <- (y_train - y_mean) / y_sd
    y_test.clean <- (y_test - y_mean) / y_sd
    
    
    # use default K0. 
    # FIT MODEL
    log_message("Fitting:")
    tik <- Sys.time()
    tryCatch({
      mcmc_supervised <- gibbs_jafar(X_m = X_l_train.clean, 
                                     y = y_train.clean, 
                                     # K0=K0, 
                                     # K0_m=c(10, 10, 10),
                                     tMCMC=tMCMC, 
                                     tBurnIn=tBurnIn, 
                                     tThin=tThin,
                                     hyperparams = list(seed = mcmc_seed))
    }, error = function(e) {
      message(e)
      log_message(e)
    })
    tok <- Sys.time()
    runtime <- tok - tik
    
    # Export model
    # model_out_data_path <- file.path(out_path_base,
    #                            "models",
    #                            sprintf(out_name_base,  n, p, snr_x, snr_y, rep))
    log_message("Writing model:", as.character(model_out_data_path))
    dir.create(dirname(model_out_data_path), recursive = TRUE, showWarnings = TRUE)
    saveRDS(mcmc_supervised,
            model_out_data_path)
    
    tryCatch({
      post_samples_y <- obtain_outcome_predictions(mcmc_supervised,
                                                   X_l_test.clean)
      
      # simulated structures need to be filtered for training data and zero 
      #     variance column filters
      training_sim_decomp <- extract_sim_decomp(run_data)
      training_sim_decomp$Z <- training_sim_decomp$Z[training_idx, ]
      training_sim_decomp$Phi <- lapply(training_sim_decomp$Phi, 
                                        function(Phi) Phi[training_idx, ])
      training_sim_decomp$Lambda_l <- lapply(1:length(training_sim_decomp$Lambda_l), 
                                             function(l) {
                                               training_sim_decomp$Lambda_l[[l]][X_l_list_column_filters[[l]], ]
                                             })
      training_sim_decomp$Gamma_l <- lapply(1:length(training_sim_decomp$Gamma_l), 
                                            function(l) {
                                              training_sim_decomp$Gamma_l[[l]][X_l_list_column_filters[[l]], ]
                                            })
      
      eval_structures <- calc_structures_with_rescaling(mcmc_supervised,
                                                        X_l_means,
                                                        X_l_sds,
                                                        post_samples_y,
                                                        training_sim_decomp,
                                                        y_test.clean)
      
      eval_table <- eval_performance(eval_structures$sim_structures,
                                     eval_structures$est_structures,
                                     eval_structures$sim_outcome,
                                     eval_structures$est_outcome,
                                     eval_structures$outcome_intervals) |> 
        mutate(n = n,
               p = p,
               snr_x = snr_x,
               snr_y = snr_y,
               rep = rep,
               sparsity = sparsity,
               time = as.numeric(difftime(tok, tik, units = "secs"))
        )
      
      # covariance_eval <- eval_model(est_obj,
      #                               sim_obj,
      #                               "covariance",
      #                               "difference_norm") |> 
      #   mutate(n = n,
      #          p = p,
      #          snr_x = snr_x,
      #          snr_y = snr_y,
      #          sparsity = sparsity,
      #          rep = rep
      #   )
      
      
      ### Export results
      # eval_table.collected <- bind_rows(eval_table.collected,
      #                                   eval_table)
      metric_out_data_path <- file.path(out_path_base,
                                        "metrics",
                                        sprintf(out_name_base, n, p, snr_x, snr_y, sparsity, rep))
      # cov_out_data_path <- file.path(out_path_base,
      #                                "covariances",
      #                                sprintf(out_name_base, n, p, snr_x, snr_y, sparsity, rep))
      log_message("Writing metrics:", as.character(metric_out_data_path))
      dir.create(dirname(metric_out_data_path), recursive = TRUE, showWarnings = TRUE)
      # dir.create(dirname(cov_out_data_path), recursive = TRUE, showWarnings = TRUE)
      
      stopifnot(exists("eval_table"))
      
      saveRDS(eval_table,
              metric_out_data_path)
      
      # saveRDS(covariance_eval,
      #         cov_out_data_path)
    }, error = function(e) {
      message(e)
      log_message(e)
    })
  }
  
}
