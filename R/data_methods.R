
# colMeans(X_l[[1]])
# colVars(X_l[[1]], std = TRUE)

library(Rfast)


load_data <- function(path_base, name_base, 
                      n, p, snr_x, snr_y, sparsity, rep,
                      log = TRUE) {
  out_path <- sprintf(path_base, n, p, snr_x, snr_y, sparsity)
  data_path <- file.path(file.path(out_path,
                                   sprintf(name_base, rep)))
  if (log) {
    log_message("Loading:", as.character(data_path))
  }
  
  readRDS(as.character(data_path))
}


standardize_X <- function(data, training_data = NA) {
  if (is.na(training_data)) {
    col_means <- colMeans(data)
    col_sds <- colVars(data, std = TRUE)
  }
  
  scale(data, center = col_means, scale = col_sds)
}

standardize_views <- function(view_list, 
                              training_list = NA) {
  if (is.na(training_list)){
    lapply(1:length(view_list),
           function(l) standardize_X(view_list[[l]]))
  } else {
    print(training_list)
    print("training list passed, function not written")
    return(NA)
  }
}

standardize_outcome <- function(data, training_data = NA) {
  if (is.na(training_data)) {
    scale(data)
    # mean <- mean(y)
    # col_means <- colMeans(data)
    # col_sds <- colVars(data, std = TRUE)
  }
  
  # scale(data, center = col_means, scale = col_sds)
}

# mean(y)
# 
# scale(y)
# calc_jafar_posterior_means
summarise_post_samples <- function(mcmc_supervised) {
  
  ######
  # Posterior means
  jafar_loadings.shared_mean <- lapply(mcmc_supervised$Lambda_m,
                                  function(mat) {
                                    apply(mat, c(2, 3), mean)
                                  })
  
  jafar_loadings.view_specific_mean <- lapply(mcmc_supervised$Gamma_m,
                                         function(mat) {
                                           apply(mat, c(2, 3), mean)
                                         })
  
  
  jafar_scores.shared_mean <- apply(mcmc_supervised$eta, c(2, 3), mean)
  
  jafar_scores.view_specific_mean <- lapply(mcmc_supervised$phi_m,
                                       function(mat) {
                                         apply(mat, c(2, 3), mean)
                                       })
  
  
  ######
  # Percentile: 2.5%
  jafar_loadings.shared_lower <- lapply(mcmc_supervised$Lambda_m,
                                       function(mat) {
                                         apply(mat, c(2, 3), quantile, probs = 0.025)
                                       })
  
  jafar_loadings.view_specific_lower <- lapply(mcmc_supervised$Gamma_m,
                                              function(mat) {
                                                apply(mat, c(2, 3), quantile, probs = 0.025)
                                              })
  
  
  jafar_scores.shared_lower <- apply(mcmc_supervised$eta, c(2, 3), quantile, probs = 0.025)
  
  jafar_scores.view_specific_lower <- lapply(mcmc_supervised$phi_m,
                                            function(mat) {
                                              apply(mat, c(2, 3), quantile, probs = 0.025)
                                            })
  
  ######
  # Percentile: 97.5%
  jafar_loadings.shared_upper <- lapply(mcmc_supervised$Lambda_m,
                                       function(mat) {
                                         apply(mat, c(2, 3), quantile, probs = 0.975)
                                       })
  
  jafar_loadings.view_specific_upper <- lapply(mcmc_supervised$Gamma_m,
                                              function(mat) {
                                                apply(mat, c(2, 3), quantile, probs = 0.975)
                                              })
  
  
  jafar_scores.shared_upper <- apply(mcmc_supervised$eta, c(2, 3), quantile, probs = 0.975)
  
  jafar_scores.view_specific_upper <- lapply(mcmc_supervised$phi_m,
                                            function(mat) {
                                              apply(mat, c(2, 3), quantile, probs = 0.975)
                                            })
  
  return(list(
    mean = list(
      shared_scores = jafar_scores.shared_mean,
      view_scores = jafar_scores.view_specific_mean,
      shared_loadings = jafar_loadings.shared_mean,
      view_loadings = jafar_loadings.view_specific_mean
    ),
    q2.5 = list(
      shared_scores = jafar_scores.shared_lower,
      view_scores = jafar_scores.view_specific_lower,
      shared_loadings = jafar_loadings.shared_lower,
      view_loadings = jafar_loadings.view_specific_lower
    ),
    q97.5 = list(
      shared_scores = jafar_scores.shared_upper,
      view_scores = jafar_scores.view_specific_upper,
      shared_loadings = jafar_loadings.shared_upper,
      view_loadings = jafar_loadings.view_specific_upper
    )
  )
  )
}


compute_structure <- function(scores,
                              loadings) {
  scores %*% t(loadings)
}

calc_all_structures <- function(shared_scores,
                                view_scores,
                                shared_loadings,
                                view_loadings,
                                data_mean = NA,
                                data_sd = NA) {
  struct <- list(
    # Z %* % Lambda'
    joint_structure = lapply(1:length(shared_loadings), 
                             function(l) {
                               compute_structure(shared_scores,
                                                 shared_loadings[[l]])
                             }
    ),
    # Phi %*% Gamma'
    view_structure = lapply(1:length(view_scores), 
                            function(l) {
                              compute_structure(view_scores[[l]],
                                                view_loadings[[l]])
                            }
    ),
    # Z %* % Lambda' + Phi %*% Gamma'
    data_reconstruction = lapply(1:length(view_scores), 
                                 function(l) {
                                   compute_structure(shared_scores,
                                                     shared_loadings[[l]]) + 
                                     compute_structure(view_scores[[l]],
                                                       view_loadings[[l]])
                                 }
    ),
    # Lambda_l %*% Lambda_m'
    # all pairwise products
    covariances = lapply(seq_len(length(shared_loadings)), function(i) {
      lapply(seq_len(length(shared_loadings)), function(j) {
        if (i <= j) { # products are symmetric, only consider lower triangle
          A <- shared_loadings[[i]]
          B <- shared_loadings[[j]]
          prod <- A %*% t(B)
          return(prod)  # divide by the dimension of the resulting product
        }
      })
    })
  )
  
  # Rescale using mean and covariance matrices. For comparability of estimated quantities (of rescaled data)
  #   with simulated data structure (which is not scaled beforehand)
  # Note data_sd is diagonal
  if (all(is.na(data_mean)) & all(is.na(data_sd))) {
    NA
  } else {
    struct[["joint_structure"]] <- lapply(1:length(struct[["joint_structure"]]),
                                        function(l) {
                                          # print(struct$joint_structure[[l]])
                                          # print(length(struct["joint_structure"][[l]]))
                                          struct[["joint_structure"]][[l]] %*% data_sd[[l]]
                                        })
    struct[["view_structure"]] <- lapply(1:length(struct[["view_structure"]]),
                                        function(l) {
                                          struct[["view_structure"]][[l]] %*% data_sd[[l]]
                                        })
    struct[["data_reconstruction"]] <- lapply(1:length(struct[["data_reconstruction"]]),
                                       function(l) {
                                         (struct[["data_reconstruction"]][[l]] - data_mean[[l]]) %*% data_sd[[l]]
                                       })
    for (l in 1:length(struct[["covariances"]])) {
      for (m in 1:length(struct[["covariances"]][[l]])) {
        if (is.matrix(struct[["covariances"]][[l]][[m]])) {
          struct[["covariances"]][[l]][[m]] <- data_sd[[l]] %*% struct[["covariances"]][[l]][[m]] %*% data_sd[[m]]
        }
      }
    }
    # struct["covariances"] <- lapply(1:length(struct["covariances"]),
    #                                         function(l) {
    #                                           data_sd[[l]] %*% struct["covariances"][[l]] %*% data_sd[[l]]
    #                                         })
  }
  return(struct)
}


eval_model <- function(est_obj,
                       sim_obj,
                       quantity,
                       metric) {
  # For a given quantity (e.g. joint_structure, view_structure, data_reconstruction),
  #   compute the desired metric (e.g. relative squared error, difference norm)
  
  # Extract quantity to evaluate from model and simulation settings
  est_quantity = switch(quantity,
                        joint_structure = {est_obj$joint_structure},
                        view_structure = {est_obj$view_structure},
                        data_reconstruction = {est_obj$data_reconstruction},
                        covariance = {est_obj$covariances})
  sim_quantity = switch(quantity,
                        joint_structure = {sim_obj$joint_structure},
                        view_structure = {sim_obj$view_structure},
                        data_reconstruction = {sim_obj$data_reconstruction},
                        covariance = {sim_obj$covariances})
  
  if (quantity == "covariance") {
    if (metric != "difference_norm") {
      message("Can only evalaute difference norm for cov")
      return(NA)
    }
    results = lapply(1:length(est_quantity),
                     function(i) {
                       lapply(1:length(sim_quantity),
                              function(j) {
                                if (i <= j) {
                                
                                diff <- sim_quantity[[i]][[j]] - est_quantity[[i]][[j]]
                                difference_norm = norm(diff, type = "F") / (nrow(diff) * ncol(diff))
                                
                                return(list(
                                  l_1 = i,
                                  l_2 = j,
                                  quantity = quantity,
                                  metric = metric,
                                  result = difference_norm
                                ))
                                }
                              }
                       )
                     }
    ) |> bind_rows()
  } else {
    results = lapply(1:length(sim_obj$view_structure),
                     function(l) {
                       return(list(
                         l = l,
                         quantity = quantity,
                         metric = metric,
                         result = switch(
                           metric,
                           rse = {norm(sim_quantity[[l]] - 
                                         est_quantity[[l]], type = "F")^2 / norm(sim_quantity[[l]], type = "F")^2},
                           difference_norm = {norm(sim_quantity[[l]] - 
                                                     est_quantity[[l]], type = "F")^2}
                         )
                       )
                       )
                     }
    ) |> bind_rows()
  }
  
  return(results)
}


eval_post_coverage <- function(data, 
                               lower, 
                               upper) {
  mean(data > lower & data < upper)
}


eval_performance <- function(mod, run_data) {
  # n, p, snr_x, snr_y, rep, sparsity) {
  # X_l <- run_data$X_l
  # y <- run_data$y
  # log_message("Standardizing")
  X_l.clean <- standardize_views(run_data$X_l)
  
  # mod <- readRDS(model_out_data_path)
  
  # Calculate mean and sd of each feature - for rescaling simulated structures
  X_l_means <- lapply(run_data$X_l, function(X) {
    means <- X |> colMeans()
    t(replicate(nrow(X), means))
  }
  )
  
  X_l_sds_inv <- lapply(run_data$X_l, function(X) {
    sds <- X |> colVars() |> sqrt()
    diag(1 / sds)
  }
  )
  
  # Calculate posterior summaries for fitted model
  jafar_posterior_summaries <- summarise_post_samples(mod)
  
  # Calculate structures from fitted model and simulated data
  jafar_posterior_means <- jafar_posterior_summaries$mean
  est_obj <- calc_all_structures(jafar_posterior_means$shared_scores,
                                 jafar_posterior_means$view_scores,
                                 jafar_posterior_means$shared_loadings,
                                 jafar_posterior_means$view_loadings)
  sim_obj <- calc_all_structures(run_data$Z,
                                 run_data$Phi,
                                 run_data$Lambda_l,
                                 run_data$Gamma_l,
                                 data_mean = X_l_means,
                                 data_sd = X_l_sds_inv)
  
  # eval_table <- eval_model(est_obj,
  #                          sim_obj,
  #                          "covariance",
  #                          "difference_norm") |> 
  #   mutate(n = n,
  #          p = p,
  #          snr_x = snr_x,
  #          snr_y = snr_y,
  #          rep = rep
  #   )
  
  # Eval outcome metrics
  y.clean <- standardize_outcome(run_data$y)
  y_pred <- predict_y(X_l.clean, mod)
  y_lower = apply(y_pred$mean, 2, quantile, probs = c(0.025))
  y_upper = apply(y_pred$mean, 2, quantile, probs = c(0.975))
  
  outcome_coverage <- eval_post_coverage(y.clean, y_lower, y_upper)
  
  y_pred.means <- colMeans(y_pred$mean)
  y_mse <- mean((y_pred.means - y.clean)^2)
  
  # Aggregate metrics
  eval_table <- bind_rows(
    eval_model(est_obj,
               sim_obj,
               "joint_structure",
               "rse"),
    eval_model(est_obj,
               sim_obj,
               "view_structure",
               "rse"),
    eval_model(est_obj,
               sim_obj,
               "data_reconstruction",
               "rse"),
    tibble_row(
      l = 0,
      quantity = "outcome",
      metric = "95p_coverage",
      result = outcome_coverage
    ),
    tibble_row(
      l = 0,
      quantity = "outcome",
      metric = "mse",
      result = y_mse
    )
  ) 
  # |> 
  #   mutate(n = n,
  #          p = p,
  #          snr_x = snr_x,
  #          snr_y = snr_y,
  #          rep = rep,
  #          sparsity = loading_sparsity
  #          # time = as.numeric(difftime(tok, tik, units = "secs"))
  #          )
  
  return(eval_table)
}
