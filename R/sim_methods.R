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


generate_factor_data_shared_asymmetric <- function(L, N, p_l_vec, K,
                                                    snr_x_vec, sparsity_vec, snr_y,
                                                    factor_scale = rep(1, K),
                                                    beta_weights = rep(1, K)) {
  # X_l = Z @ Lambda_l^T + E_l,  l = 1..L
  # y   = Z @ beta + e
  #
  # factor_scale[k]: multiplies loading column k across all views, so factor k's
  #   contribution to Var(X_l) scales as factor_scale[k]^2 * ||lambda_{l,k}||^2.
  # beta_weights[k]: multiplies the k-th regression coefficient drawn from N(0,1),
  #   so setting beta_weights[k] = 0 removes factor k from the outcome entirely.
  #
  # This allows the outcome-relevant subspace to be misaligned with the high-variance
  # directions of X — the regime where unsupervised methods (MOFA+) fail.

  stopifnot(length(factor_scale) == K, length(beta_weights) == K)

  sim_data_spec <- lapply(1:L,
                          function(l) list(N = N,
                                           p_l = p_l_vec[[l]],
                                           K = K,
                                           snr_x = snr_x_vec[[l]],
                                           sparsity = sparsity_vec[[l]]))

  # Base loadings scaled column-wise: Var(X_l factor k) proportional to factor_scale[k]^2
  sim_loadings.shared <- lapply(sim_data_spec, function(spec) {
    L_base <- generate_factor_loadings(spec, sparsity_prob = 1 - spec$sparsity, K)
    sweep(L_base, 2, factor_scale, `*`)
  })

  sim_factors.shared <- generate_factor_scores(sim_data_spec[[1]], K)

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

  beta_sim    <- rnorm(n = K) * beta_weights
  outcome_var <- as.numeric(t(beta_sim) %*% beta_sim)
  # If all beta_weights are zero, outcome is pure noise (degenerate but safe)
  noise_sd   <- if (outcome_var > 0) sqrt(outcome_var / snr_y) else 1.0
  noise_y    <- rnorm(n = N, mean = 0, sd = noise_sd)
  sim_data.y <- if (outcome_var > 0) sim_factors.shared %*% beta_sim + noise_y else noise_y

  return(list(
    L            = L,
    N            = N,
    p_l          = p_l_vec,
    K            = K,
    sparsity     = sparsity_vec,
    beta         = beta_sim,
    snr_x        = snr_x_vec,
    snr_y        = snr_y,
    factor_scale = factor_scale,
    beta_weights = beta_weights,
    X_l          = sim_data.x,
    y            = sim_data.y,
    Lambda_l     = sim_loadings.shared,
    Z            = sim_factors.shared
  ))
}


generate_factor_data_shared_asymmetric_testing <- function(L, N, p_l_vec, K,
                                                            snr_x_vec, sparsity_vec, snr_y,
                                                            training_loadings,
                                                            training_beta) {
  # Loadings and beta fixed from training; new Z drawn for test observations.

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
  noise_sd   <- if (outcome_var > 0) sqrt(outcome_var / snr_y) else 1.0
  noise_y    <- rnorm(n = N, mean = 0, sd = noise_sd)
  sim_data.y <- if (outcome_var > 0) sim_factors.shared %*% beta_sim + noise_y else noise_y

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


# Estimate SNR = Var(beta^T z) / Var(T) by Monte Carlo.
# z ~ N_K(0, I) and T simulated via inversion: T = exp(beta^T z) * (-log(U))^(1/alpha).
outcome_snr <- function(beta, alpha, n_mc = 10000) {
  K    <- length(beta)
  Z_mc <- matrix(rnorm(n_mc * K), nrow = n_mc, ncol = K)
  lp   <- as.vector(Z_mc %*% beta)         # beta^T z
  U    <- runif(n_mc)
  T_mc <- exp(lp) * (-log(U))^(1 / alpha)  # inversion formula
  var(lp) / var(T_mc)
}


# Simulate right-censored Weibull survival outcomes from pre-simulated factor scores.
#
# Model:  T_i = exp(beta^T z_i) * (-log(U_i))^(1/alpha),  U_i ~ Uniform(0,1)
#         C_i ~ Exponential(gamma_c)  (independent of T_i | z_i)
#         y_i = min(T_i, C_i),  delta_i = 1(T_i <= C_i)
#
# gamma_c is calibrated by bisection on the analytical censoring fraction
#   P(C < T) = mean_i[ 1 - exp(-gamma_c * T_i) ]
# which is monotone increasing in gamma_c and deterministic given T, so
# bisection in log(gamma_c) converges to machine precision in ~50 iterations.
#
# Returns a data.frame with columns y_obs and delta (1 = event, 0 = censored).
simulate_weibull_outcome <- function(Z, beta, alpha, censoring_rate) {
  stopifnot(censoring_rate > 0, censoring_rate < 1, alpha > 0)

  n         <- nrow(Z)
  log_scale <- as.vector(Z %*% beta)
  scale     <- exp(log_scale)

  # Simulate event times via inversion (exact: -log(U) ~ Exp(1))
  U       <- runif(n)
  T_event <- scale * (-log(U))^(1 / alpha)

  # Analytical censoring fraction given gamma_c and fixed T_event
  cens_fraction <- function(gamma_c) mean(1 - exp(-gamma_c * T_event))

  # Bisect in log(gamma_c) space; bounds chosen to bracket [~0, ~1] censoring
  lo <- log(1e-10)
  hi <- log(1e10)
  for (iter in seq_len(200)) {
    mid      <- (lo + hi) / 2
    gamma_c  <- exp(mid)
    emp_rate <- cens_fraction(gamma_c)
    if (emp_rate < censoring_rate) lo <- mid else hi <- mid
    if ((hi - lo) < 1e-12) break
  }
  gamma_c <- exp((lo + hi) / 2)

  # Draw final censoring times with calibrated rate
  C     <- rexp(n, rate = gamma_c)
  y_obs <- pmin(T_event, C)
  delta <- as.integer(T_event <= C)

  data.frame(y_obs = y_obs, delta = delta)
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


#############################################################################
# Controlled-strength shared-factor data (weak-factor study, phases 1-4)
#
# Data-generating process for one replicate (L views, p_l features each,
# P = sum(p_l), K = K_nuis + 1 factors; see docs/controlled_strength_data.md):
#
#   z_i     ~ N(0, I_K)                                  i = 1..N
#   x_il    = Lambda_l z_i + eps_il,  eps_il ~ N(0, I)   (psi_j = 1 for every feature)
#   y_i     = beta_{k*} z_{i,k*} + theta^T eps_i + e_i,  e_i ~ N(0, sigma_y^2)
#
# Relevant factor k* (column k_star):
#   support S of s = round(g_target / a^2) features, split as evenly as possible
#   across views (view l gets floor(s / L) features, the first s mod L views one
#   more), chosen uniformly at random within each view;
#   Lambda_{j,k*} ~ N(0, a^2) for j in S, 0 elsewhere.
#   Its strength is g = sum_{j in S} Lambda_{j,k*}^2 / psi_j (E[g] = s a^2 ~ g_target);
#   the realised g is returned as g_true.
# Nuisance factors (the other K_nuis columns): on features NOT in S only,
#   Lambda_jk = N(0, nuis_sd^2) * Bernoulli(nuis_density).
#   Disjoint supports make row/column k* of G = sum_l Lambda_l^T Psi_l^-1 Lambda_l
#   exactly zero off the diagonal, so g_true is an eigenvalue of G.
# Outcome: beta_k = 0 for nuisance factors, beta_{k*} ~ N(0, 1),
#   sigma_y^2 = beta_{k*}^2 / snr_y.
# Outside-factor signal (r_out > 0, Phase 4): n_out features drawn uniformly from
#   the features NOT in S; theta is zero elsewhere, theta_out ~ N(0, I) rescaled so
#   that ||theta||^2 = r_out / (1 - r_out) * (beta_{k*}^2 + sigma_y^2), i.e. the term
#   theta^T eps (idiosyncratic noise of those features, psi = 1) explains a fraction
#   r_out of Var(y) = beta_{k*}^2 + sigma_y^2 + ||theta||^2.
#
# Returns the fields of generate_factor_data_shared_asymmetric (so
# run_methods.load_rds / process_data work unchanged) plus
#   psi (list of p_l vectors, all 1), sigma2_y, g_true, g_target, a, s, k_star,
#   rescale_to_target,
#   support (list of 1-based feature indices of S per view), theta_out (list of
#   p_l vectors), out_idx (list of 1-based indices of the outside features per
#   view), r_out, G (K x K). snr_x holds each view's realised mean signal-to-noise
#   ratio, sum(Lambda_l^2) / p_l (the noise is not tied to it).
#############################################################################
generate_factor_data_shared_controlled <- function(N, p_l_vec, g_target,
                                                   a = 0.5,
                                                   K_nuis = 4,
                                                   nuis_sd = 1,
                                                   nuis_density = 0.75,
                                                   snr_y = 2,
                                                   k_star = K_nuis + 1,
                                                   r_out = 0,
                                                   n_out = 5,
                                                   rescale_to_target = FALSE) {
  # TODO(user): design choice. With small s (e.g. s = 1 at c = 0.3, gamma = 1.2)
  # the realised g = sum of s squared N(0, a^2) draws has CV ~ sqrt(2/s), so
  # replicates scatter widely around g_target. rescale_to_target = TRUE rescales
  # the support loadings so that g_true == g_target exactly; the default (FALSE)
  # follows the Phase 1 spec and analyses should use the realised g_true.
  L <- length(p_l_vec)
  K <- K_nuis + 1
  P <- sum(p_l_vec)
  s <- round(g_target / a^2)
  stopifnot(s >= 1, s <= P - n_out, k_star >= 1, k_star <= K, r_out >= 0, r_out < 1)

  # Support of the relevant factor, split across views
  s_l <- rep(s %/% L, L) + (seq_len(L) <= s %% L)
  stopifnot(all(s_l <= p_l_vec))
  support <- lapply(1:L, function(l) sort(sample.int(p_l_vec[l], s_l[l])))

  nuis_cols <- setdiff(1:K, k_star)
  Lambda_l <- lapply(1:L, function(l) {
    p <- p_l_vec[l]
    Lam <- matrix(0, p, K)
    S <- support[[l]]
    Lam[S, k_star] <- rnorm(length(S), 0, a)
    off <- setdiff(seq_len(p), S)
    Lam[off, nuis_cols] <- matrix(rnorm(length(off) * K_nuis, 0, nuis_sd) *
                                    rbinom(length(off) * K_nuis, 1, nuis_density),
                                  length(off), K_nuis)
    Lam
  })
  psi <- lapply(p_l_vec, function(p) rep(1, p))
  if (rescale_to_target) {
    g_raw <- sum(sapply(1:L, function(l) sum(Lambda_l[[l]][support[[l]], k_star]^2)))
    Lambda_l <- lapply(Lambda_l, function(Lam) { Lam[, k_star] <- Lam[, k_star] * sqrt(g_target / g_raw); Lam })
  }

  beta <- rep(0, K)
  beta[k_star] <- rnorm(1)
  sigma2_y <- beta[k_star]^2 / snr_y

  # Outside-factor features and coefficients (zero when r_out = 0)
  theta_out <- lapply(p_l_vec, function(p) rep(0, p))
  out_idx <- lapply(1:L, function(l) integer(0))
  if (r_out > 0) {
    off_global <- unlist(lapply(1:L, function(l)
      setdiff(seq_len(p_l_vec[l]), support[[l]]) + c(0, cumsum(p_l_vec))[l]))
    pick <- sort(sample(off_global, n_out))
    view_of <- findInterval(pick - 1, c(0, cumsum(p_l_vec))[-(L + 1)])
    th <- rnorm(n_out)
    th <- th * sqrt(r_out / (1 - r_out) * (beta[k_star]^2 + sigma2_y) / sum(th^2))
    for (l in 1:L) {
      idx <- pick[view_of == l] - c(0, cumsum(p_l_vec))[l]
      out_idx[[l]] <- idx
      theta_out[[l]][idx] <- th[view_of == l]
    }
  }

  G <- Reduce(`+`, lapply(1:L, function(l) t(Lambda_l[[l]]) %*% (Lambda_l[[l]] / psi[[l]])))
  g_true <- G[k_star, k_star]

  draw <- controlled_draw(N, Lambda_l, psi, beta, sigma2_y, theta_out)

  list(
    L            = L,
    N            = N,
    p_l          = p_l_vec,
    K            = K,
    sparsity     = rep(1 - nuis_density, L),
    beta         = beta,
    snr_x        = sapply(Lambda_l, function(Lam) sum(Lam^2) / nrow(Lam)),
    snr_y        = snr_y,
    factor_scale = rep(1, K),
    beta_weights = as.numeric(seq_len(K) == k_star),
    X_l          = draw$X_l,
    y            = draw$y,
    Lambda_l     = Lambda_l,
    Z            = draw$Z,
    psi          = psi,
    sigma2_y     = sigma2_y,
    g_true       = g_true,
    g_target     = g_target,
    a            = a,
    s            = s,
    k_star       = k_star,
    support      = support,
    theta_out    = theta_out,
    out_idx      = out_idx,
    r_out        = r_out,
    rescale_to_target = rescale_to_target,
    G            = G
  )
}


# Test set for generate_factor_data_shared_controlled: the true parameters
# (Lambda_l, psi, beta, sigma2_y, theta_out, supports) are copied from the
# training replicate; scores z, noise eps and outcome noise e are drawn afresh.
generate_factor_data_shared_controlled_testing <- function(N, train_data) {
  draw <- controlled_draw(N, train_data$Lambda_l, train_data$psi, train_data$beta,
                          train_data$sigma2_y, train_data$theta_out)
  out <- train_data
  out$N   <- N
  out$X_l <- draw$X_l
  out$y   <- draw$y
  out$Z   <- draw$Z
  out
}


# One sample of size N from the controlled-strength model with fixed parameters.
controlled_draw <- function(N, Lambda_l, psi, beta, sigma2_y, theta_out) {
  K <- length(beta)
  Z <- matrix(rnorm(N * K), N, K)
  eps <- lapply(seq_along(Lambda_l), function(l)
    matrix(rnorm(N * nrow(Lambda_l[[l]])), N) %*% diag(sqrt(psi[[l]]), nrow(Lambda_l[[l]])))
  X_l <- lapply(seq_along(Lambda_l), function(l) Z %*% t(Lambda_l[[l]]) + eps[[l]])
  y <- Z %*% beta +
    Reduce(`+`, lapply(seq_along(eps), function(l) eps[[l]] %*% theta_out[[l]])) +
    rnorm(N, 0, sqrt(sigma2_y))
  list(X_l = X_l, y = y, Z = Z)
}


# Population oracle for the controlled-strength model (raw scale, true mean 0):
# the best linear predictor b* = Sigma_x^-1 Cov(x, y) and its risk R*, with
# Cov(x, y) = Lambda beta + Psi theta. With theta = 0 these reduce to
# b*_l = Psi_l^-1 Lambda_l A^-1 beta and R* = sigma_y^2 + beta^T A^-1 beta, A = I + G.
controlled_oracle <- function(d) {
  Lam <- do.call(rbind, d$Lambda_l)
  psi <- unlist(d$psi)
  theta <- unlist(d$theta_out)
  beta <- d$beta
  K <- length(beta)
  G <- t(Lam) %*% (Lam / psi)
  A <- diag(K) + G
  PinvLam <- Lam / psi
  b_star <- PinvLam %*% beta + theta - PinvLam %*% solve(A, G %*% beta + t(Lam) %*% theta)
  cxy <- Lam %*% beta + psi * theta
  var_y <- d$sigma2_y + sum(beta^2) + sum(psi * theta^2)
  list(b_star = as.vector(b_star), R_star = as.numeric(var_y - t(cxy) %*% b_star),
       R_star_no_outside = d$sigma2_y + as.numeric(t(beta) %*% solve(A, beta)),
       var_y = var_y, G = G)
}
