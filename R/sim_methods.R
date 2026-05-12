projection_by_qr <- function(X) {
  q = qr.Q(qr(X))
  P = q %*% t(q)
}

projection_by_inverse <- function(X) {
  X %*% solve(t(X) %*% X) %*% t(X)
}

generate_factor_loadings <- function(data_spec, sparsity_prob = 1, K = NA) {
  if (is.na(K)) { # view-specific
    rnorm(n = data_spec$K_l * data_spec$p_l) * 
      rbinom(n = data_spec$K_l * data_spec$p_l,
             size = 1,
             prob = sparsity_prob) |>
      matrix(nrow = data_spec$p_l,
             ncol = data_spec$K_l)
  } else { # shared
    print(data_spec$K)
    rnorm(n = data_spec$K * data_spec$p_l) * 
      rbinom(n = data_spec$K * data_spec$p_l,
             size = 1,
             prob = sparsity_prob) |>
      matrix(nrow = data_spec$p_l,
             ncol = data_spec$K)
  }
}

generate_factor_scores <- function(data_spec, K = NA) {
  
  if (is.na(K)) { # view-specific
    mvtnorm::rmvnorm(n = data_spec$N, mean = rep(0, data_spec$K_l)) 
  } else { # shared
    mvtnorm::rmvnorm(n = data_spec$N, mean = rep(0, K))
  }
  
}

generate_loading_sparsity_mask <- function(data_spec, sparsity_prob = 1, K = NA) {
  if (is.na(K)) { # view-specific
    # rnorm(n = data_spec$K_l * data_spec$p_l) |> #* 
    rbinom(n = data_spec$K_l * data_spec$p_l,
           size = 1,
           prob = sparsity_prob) |>
      matrix(nrow = data_spec$p_l,
             ncol = data_spec$K_l)
  } else { # shared
    # rnorm(n = data_spec$K * data_spec$p_l) |>  #* 
    rbinom(n = data_spec$K * data_spec$p_l,
           size = 1,
           prob = sparsity_prob) |>
      matrix(nrow = data_spec$p_l,
             ncol = data_spec$K)
  }
}


generate_factor_data <- function(L,
                                 N,
                                 p_l_vec,
                                 K,
                                 K_l_vec,
                                 snr_x_vec,
                                 sparsity_vec,
                                 snr_y,
                                 outcome = "all") {
  
  # Set spec for each data view based on input parameters
  sim_data_spec <- lapply(1:L,
                          function(l) list(N = N,
                                           test_N = test_n,
                                           p_l = p_l_vec[[l]],
                                           K = K,
                                           K_l = K_l_vec[[l]],
                                           snr_x = snr_x_vec[[l]],
                                           sparsity = sparsity_vec[[l]])
  )
  
  # Generate shared structure
  sim_loadings.shared <- lapply(sim_data_spec,
                                function(spec) {
                                  generate_factor_loadings(spec, 
                                                           sparsity_prob = 1 - spec$sparsity, 
                                                           K)
                                }
  )
  sim_factors.shared <- generate_factor_scores(sim_data_spec[[1]], K)
  
  # Generate view-specific structure orthogonal to the joint structure
  sim_loadings.view_specific <- lapply(sim_data_spec,
                                       function(spec) {
                                         generate_factor_loadings(spec, 
                                                                  sparsity_prob = 1 - spec$sparsity)
                                       }
  )
  sim_factors.view_specific <- lapply(sim_data_spec,
                                      function(spec) {
                                        generate_factor_scores(spec)
                                      }
  )

  P <- projection_by_qr(sim_factors.shared)
  P_perp <- diag(N) - P
  
  # the below ortho_factors should be used instead of sim_factors above
  ortho_factors.view_specific <- lapply(sim_factors.view_specific,
                                        function(scores) {
                                          ortho_scores <- P_perp %*% scores
                                          ortho_scores_sd <- sd(ortho_scores)
                                          ortho_scores / ortho_scores_sd
                                        }
  )
  
  # Compute the noise term for each data view to obtain desired SNR
  view_vars <- lapply(1:L,
                      function(l) {
                        diag(sim_loadings.shared[[l]] %*% t(sim_loadings.shared[[l]]) + 
                               sim_loadings.view_specific[[l]] %*% t(sim_loadings.view_specific[[l]]))# / N
                      })
  noise_x_views <- lapply(1:length(sim_data_spec),
                          function(l) {
                            
                            noise_x <- matrix(nrow = N, ncol = sim_data_spec[[l]]$p_l)
                            for (i in 1:N) {
                              for (j in 1:sim_data_spec[[l]]$p_l) {
                                noise_x[i, j] <- rnorm(1, sd = sqrt(view_vars[[l]][[j]] / sim_data_spec[[l]]$snr_x))
                              }
                            }
                            noise_x
                          }
  )
  print(dim(sim_factors.shared))
  print(dim(sim_loadings.shared[[1]]))
  
  # Calculate data views based on factor model
  sim_data.x <- lapply(1:length(sim_data_spec), 
                       function(l) {
                         sim_factors.shared %*% t(sim_loadings.shared[[l]]) + 
                           ortho_factors.view_specific[[l]] %*% t(sim_loadings.view_specific[[l]]) +
                           noise_x_views[[l]]
                       }
  )
  
  # Generate outcome with regression coefficients
  #   outcome = "all" ==> response depends on all factors
  #   outcome = "shared" ==> response depends only on shared factors
  if (outcome == "all") {
    # 
    beta_sim <- rnorm(n = K)
    beta_l <- lapply(1:L, 
                     function(l) {
                       rnorm(n = sim_data_spec[[l]]$K_l)
                     }
    )
    
    # lp_view_specific <- lapply(1:L, 
    #                            function(l) {
    #                              sim_factors.view_specific[[l]] %*% beta_l[[l]]
    #                            }
    # )
    cat.sim_factors.view_specific <- do.call("cbind", ortho_factors.view_specific)
    cat.sim_factors <- cbind(sim_factors.shared,
                             cat.sim_factors.view_specific)
    
    sim_beta_cat <- c(beta_sim, beta_l |> unlist())
    ncoef = K + lapply(1:L, function(l) sim_data_spec[[l]]$K_l) |> unlist() |> sum()
    
    outcome_var <- t(sim_beta_cat) %*% sim_beta_cat# / ncoef
    noise_y <- rnorm(n = N, mean = 0, sd = sqrt(outcome_var / snr_y))
    
    sim_data.y <- cat.sim_factors %*% sim_beta_cat + 
      # Reduce("+", lp_view_specific) +
      noise_y
    beta_sim <- c(beta_sim, as.vector(beta_l))
  } else if (outcome == "shared") {
    beta_sim <- rnorm(n = K)
    
    outcome_var <- t(beta_sim) %*% beta_sim# / K
    noise_y <- rnorm(n = N, mean = 0, sd = sqrt(outcome_var / snr_y))
    
    sim_data.y <- sim_factors.shared %*% beta_sim + noise_y
  }
  
  
  return(list(
    L = L,
    N = N,
    p_l = p_l_vec,
    K = K,
    K_l = K_l_vec,
    sparsity = sparsity_vec,
    beta = beta_sim,
    snr_x = snr_x_vec,
    snr_y = snr_y,
    X_l = sim_data.x,
    y = sim_data.y,
    Lambda_l = sim_loadings.shared,
    Gamma_l = sim_loadings.view_specific,
    Z = sim_factors.shared,
    Phi = sim_factors.view_specific,
    Phi_perp = ortho_factors.view_specific
  )
  )
}


generate_factor_data_shared <- function(L,
                                         N,
                                         p_l_vec,
                                         K,
                                         snr_x_vec,
                                         sparsity_vec,
                                         snr_y) {
  # X_l = Z @ Lambda_l^T + E_l,  l = 1..L
  # y   = Z @ beta + e
  # No view-specific factors.

  sim_data_spec <- lapply(1:L,
                          function(l) list(N = N,
                                           p_l = p_l_vec[[l]],
                                           K = K,
                                           snr_x = snr_x_vec[[l]],
                                           sparsity = sparsity_vec[[l]]))

  # View-specific loading matrices, shared scores
  sim_loadings.shared <- lapply(sim_data_spec,
                                function(spec) {
                                  generate_factor_loadings(spec,
                                                           sparsity_prob = 1 - spec$sparsity,
                                                           K)
                                })

  sim_factors.shared <- generate_factor_scores(sim_data_spec[[1]], K)

  # Per-feature noise scaled to desired SNR
  view_vars <- lapply(1:L, function(l) {
    diag(sim_loadings.shared[[l]] %*% t(sim_loadings.shared[[l]]))
  })
  noise_x_views <- lapply(1:L, function(l) {
    matrix(
      rnorm(N * sim_data_spec[[l]]$p_l) *
        rep(sqrt(view_vars[[l]] / sim_data_spec[[l]]$snr_x), each = N),
      nrow = N, ncol = sim_data_spec[[l]]$p_l
    )
  })

  sim_data.x <- lapply(1:L, function(l) {
    sim_factors.shared %*% t(sim_loadings.shared[[l]]) + noise_x_views[[l]]
  })

  beta_sim    <- rnorm(n = K)
  outcome_var <- as.numeric(t(beta_sim) %*% beta_sim)
  noise_y     <- rnorm(n = N, mean = 0, sd = sqrt(outcome_var / snr_y))
  sim_data.y  <- sim_factors.shared %*% beta_sim + noise_y

  return(list(
    L        = L,
    N        = N,
    p_l      = p_l_vec,
    K        = K,
    sparsity = sparsity_vec,
    beta     = beta_sim,
    snr_x    = snr_x_vec,
    snr_y    = snr_y,
    X_l      = sim_data.x,
    y        = sim_data.y,
    Lambda_l = sim_loadings.shared,
    Z        = sim_factors.shared
  ))
}


generate_factor_data_shared_testing <- function(L,
                                                  N,
                                                  p_l_vec,
                                                  K,
                                                  snr_x_vec,
                                                  sparsity_vec,
                                                  snr_y,
                                                  training_loadings,
                                                  training_beta) {
  # Loadings and beta fixed from training; new Z scores drawn for test observations.

  sim_data_spec <- lapply(1:L,
                          function(l) list(N = N,
                                           p_l = p_l_vec[[l]],
                                           K = K,
                                           snr_x = snr_x_vec[[l]],
                                           sparsity = sparsity_vec[[l]]))

  sim_loadings.shared <- training_loadings
  sim_factors.shared  <- generate_factor_scores(sim_data_spec[[1]], K)

  view_vars <- lapply(1:L, function(l) {
    diag(sim_loadings.shared[[l]] %*% t(sim_loadings.shared[[l]]))
  })
  noise_x_views <- lapply(1:L, function(l) {
    matrix(
      rnorm(N * sim_data_spec[[l]]$p_l) *
        rep(sqrt(view_vars[[l]] / sim_data_spec[[l]]$snr_x), each = N),
      nrow = N, ncol = sim_data_spec[[l]]$p_l
    )
  })

  sim_data.x <- lapply(1:L, function(l) {
    sim_factors.shared %*% t(sim_loadings.shared[[l]]) + noise_x_views[[l]]
  })

  beta_sim    <- training_beta
  outcome_var <- as.numeric(t(beta_sim) %*% beta_sim)
  noise_y     <- rnorm(n = N, mean = 0, sd = sqrt(outcome_var / snr_y))
  sim_data.y  <- sim_factors.shared %*% beta_sim + noise_y

  return(list(
    L        = L,
    N        = N,
    p_l      = p_l_vec,
    K        = K,
    sparsity = sparsity_vec,
    beta     = beta_sim,
    snr_x    = snr_x_vec,
    snr_y    = snr_y,
    X_l      = sim_data.x,
    y        = sim_data.y,
    Lambda_l = sim_loadings.shared,
    Z        = sim_factors.shared
  ))
}


generate_single_view_factor_data <- function(N, p_l, K, snr_x, snr_y, sparsity) {
  data_spec <- list(N = N, p_l = p_l, K = K, K_l = K, snr_x = snr_x, sparsity = sparsity)

  sim_loadings <- generate_factor_loadings(data_spec, sparsity_prob = 1 - sparsity, K)
  sim_factors  <- generate_factor_scores(data_spec, K)

  # Per-feature noise variance to achieve desired SNR
  view_vars <- diag(sim_loadings %*% t(sim_loadings))
  noise_x <- matrix(
    rnorm(N * p_l) * rep(sqrt(view_vars / snr_x), each = N),
    nrow = N, ncol = p_l
  )

  X <- sim_factors %*% t(sim_loadings) + noise_x

  beta_sim    <- rnorm(n = K)
  outcome_var <- as.numeric(t(beta_sim) %*% beta_sim)
  noise_y     <- rnorm(n = N, mean = 0, sd = sqrt(outcome_var / snr_y))
  y           <- sim_factors %*% beta_sim + noise_y

  list(N = N, p_l = p_l, K = K, sparsity = sparsity,
       snr_x = snr_x, snr_y = snr_y,
       X_l = list(X), y = y, Z = sim_factors, Lambda_l = list(sim_loadings), beta = beta_sim)
}


generate_single_view_factor_data_testing <- function(N, p_l, K, snr_x, snr_y, sparsity,
                                                     training_loadings, training_beta) {
  data_spec <- list(N = N, p_l = p_l, K = K, K_l = K, snr_x = snr_x, sparsity = sparsity)

  sim_factors   <- generate_factor_scores(data_spec, K)
  sim_loadings  <- training_loadings[[1]]

  view_vars <- diag(sim_loadings %*% t(sim_loadings))
  noise_x <- matrix(
    rnorm(N * p_l) * rep(sqrt(view_vars / snr_x), each = N),
    nrow = N, ncol = p_l
  )

  X <- sim_factors %*% t(sim_loadings) + noise_x

  beta_sim    <- training_beta
  outcome_var <- as.numeric(t(beta_sim) %*% beta_sim)
  noise_y     <- rnorm(n = N, mean = 0, sd = sqrt(outcome_var / snr_y))
  y           <- sim_factors %*% beta_sim + noise_y

  list(N = N, p_l = p_l, K = K, sparsity = sparsity,
       snr_x = snr_x, snr_y = snr_y,
       X_l = list(X), y = y, Z = sim_factors, Lambda_l = list(sim_loadings), beta = beta_sim)
}


generate_factor_data_testing <- function(L,
                                 N,
                                 p_l_vec,
                                 K,
                                 K_l_vec,
                                 snr_x_vec,
                                 sparsity_vec,
                                 snr_y,
                                 training_loadings.shared,
                                 training_loadings.view_specific,
                                 training_beta,
                                 outcome = "all") {
  # Loadings and outcome model coefficients are generated from the training dataset.
  
  # Set spec for each data view based on input parameters
  sim_data_spec <- lapply(1:L,
                          function(l) list(N = N,
                                           test_N = test_n,
                                           p_l = p_l_vec[[l]],
                                           K_l = K_l_vec[[l]],
                                           snr_x = snr_x_vec[[l]],
                                           sparsity = sparsity_vec[[l]])
  )
  
  # # Generate loadings
  # sim_loadings.shared <- lapply(sim_data_spec,
  #                               function(spec) {
  #                                 generate_factor_loadings(spec, 
  #                                                          sparsity_prob = 1 - spec$sparsity, 
  #                                                          K)
  #                               }
  # )
  # sim_loadings.view_specific <- lapply(sim_data_spec,
  #                                      function(spec) {
  #                                        generate_factor_loadings(spec, 
  #                                                                 sparsity_prob = 1 - spec$sparsity)
  #                                      }
  # )
  sim_loadings.shared <- training_loadings.shared
  sim_loadings.view_specific <- training_loadings.view_specific
  
  # Generate factor scores
  # Train
  # sim_factors.shared <- generate_factor_scores(sim_data_spec[[1]], K, TRUE)
  # sim_factors.view_specific <- lapply(sim_data_spec,
  #                                     function(spec) {
  #                                       generate_factor_scores(spec, train = TRUE)
  #                                     }
  # )
  # Test
  sim_factors.shared <- generate_factor_scores(sim_data_spec[[1]], K)
  sim_factors.view_specific <- lapply(sim_data_spec,
                                           function(spec) {
                                             generate_factor_scores(spec)
                                           }
  )
  
  # Compute the noise term for each data view to obtain desired SNR
  view_vars <- lapply(1:L,
                      function(l) {
                        diag(sim_loadings.shared[[l]] %*% t(sim_loadings.shared[[l]]) + 
                               sim_loadings.view_specific[[l]] %*% t(sim_loadings.view_specific[[l]]))# / N
                      })
  noise_x_views <- lapply(1:length(sim_data_spec),
                          function(l) {
                            
                            noise_x <- matrix(nrow = N, ncol = sim_data_spec[[l]]$p_l)
                            for (i in 1:N) {
                              for (j in 1:sim_data_spec[[l]]$p_l) {
                                noise_x[i, j] <- rnorm(1, sd = sqrt(view_vars[[l]][[j]] / sim_data_spec[[l]]$snr_x))
                              }
                            }
                            noise_x
                          }
  )
  
  # Calculate data views based on factor model
  sim_data.x <- lapply(1:length(sim_data_spec), 
                       function(l) {
                         sim_factors.shared %*% t(sim_loadings.shared[[l]]) + 
                           sim_factors.view_specific[[l]] %*% t(sim_loadings.view_specific[[l]]) +
                           noise_x_views[[l]]
                       }
  )
  
  # Generate outcome with regression coefficients
  #   outcome = "all" ==> response depends on all factors
  #   outcome = "shared" ==> response depends only on shared factors
  beta_sim <- training_beta
  if (outcome == "all") {
    # 
    # beta_sim <- rnorm(n = K)
    # beta_l <- lapply(1:L, 
    #                  function(l) {
    #                    rnorm(n = sim_data_spec[[l]]$K_l)
    #                  }
    # )
    
    cat.sim_factors.view_specific <- do.call("cbind", sim_factors.view_specific)
    cat.sim_factors <- cbind(sim_factors.shared,
                             cat.sim_factors.view_specific)
    
    # lp_view_specific <- lapply(1:L, 
    #                            function(l) {
    #                              sim_factors.view_specific[[l]] %*% beta_l[[l]]
    #                            }
    # )
    
    sim_beta_cat <- unlist(beta_sim) #c(beta_sim, beta_l |> unlist())
    # print(unlist(sim_beta_cat))
    
    ncoef = K + lapply(1:L, function(l) sim_data_spec[[l]]$K_l) |> unlist() |> sum()
    
    outcome_var <- t(sim_beta_cat) %*% sim_beta_cat# / ncoef
    noise_y <- rnorm(n = N, mean = 0, sd = sqrt(outcome_var / snr_y))
    
    sim_data.y <- cat.sim_factors %*% sim_beta_cat + 
      # Reduce("+", lp_view_specific) +
      noise_y
    # beta_sim <- c(beta_sim, as.vector(beta_l))
  } else if (outcome == "shared") {
    # beta_sim <- rnorm(n = K)
    
    outcome_var <- t(beta_sim) %*% beta_sim# / K
    noise_y <- rnorm(n = N, mean = 0, sd = sqrt(outcome_var / snr_y))
    
    sim_data.y <- sim_factors.shared %*% beta_sim + noise_y
  }
  
  
  return(list(
    L = L,
    N = N,
    p_l = p_l_vec,
    K = K,
    K_l = K_l_vec,
    sparsity = sparsity_vec,
    beta = beta_sim,
    snr_x = snr_x_vec,
    snr_y = snr_y,
    X_l = sim_data.x,
    y = sim_data.y,
    Lambda_l = sim_loadings.shared,
    Gamma_l = sim_loadings.view_specific,
    Z = sim_factors.shared,
    Phi = sim_factors.view_specific
  )
  )
}
