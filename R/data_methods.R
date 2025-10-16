
# colMeans(X_l[[1]])
# colVars(X_l[[1]], std = TRUE)

library(Rfast)


load_data <- function(path_base, name_base, 
                      n, p, snr_x, snr_y, rep,
                      log = TRUE) {
  out_path <- sprintf(path_base, n, p, snr_x, snr_y)
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

calc_jafar_posterior_means <- function(mcmc_supervised) {
  jafar_loadings.shared <- lapply(mcmc_supervised$Lambda_m,
                                  function(mat) {
                                    apply(mat, c(2, 3), mean)
                                  })
  
  jafar_loadings.view_specific <- lapply(mcmc_supervised$Gamma_m,
                                         function(mat) {
                                           apply(mat, c(2, 3), mean)
                                         })
  
  
  jafar_scores.shared <- apply(mcmc_supervised$eta, c(2, 3), mean)
  
  jafar_scores.view_specific <- lapply(mcmc_supervised$phi_m,
                                       function(mat) {
                                         apply(mat, c(2, 3), mean)
                                       })
  
  return(list(
    shared_scores = jafar_scores.shared,
    view_scores = jafar_scores.view_specific,
    shared_loadings = jafar_loadings.shared,
    view_loadings = jafar_loadings.view_specific
  ))
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
