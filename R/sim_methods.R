generate_factor_loadings <- function(data_spec, sparsity_prob = 1, K = NA) {
  if (is.na(K)) { # view-specific
    rnorm(n = data_spec$K_l * data_spec$p_l) * 
      rbinom(n = data_spec$K_l * data_spec$p_l, 
             size = 1, 
             prob = sparsity_prob) |> 
      matrix(nrow = data_spec$p_l,
             ncol = data_spec$K_l)
  } else { # shared
    rnorm(n = data_spec$K * data_spec$p_l) * 
      rbinom(n = data_spec$K_l * data_spec$p_l, 
             size = 1, 
             prob = sparsity_prob) |> 
      matrix(nrow = data_spec$p_l,
             ncol = data_spec$K)
  }
}

generate_factor_scores <- function(data_spec, K = NA) {
  if (is.na(K)) { # view-specific
    rmvnorm(n = data_spec$N, mean = rep(0, data_spec$K_l)) 
  } else { # shared
    rmvnorm(n = data_spec$N, mean = rep(0, K))
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
                                           p_l = p_l_vec[[l]],
                                           K_l = K_l_vec[[l]],
                                           snr_x = snr_x_vec[[l]],
                                           sparsity = sparsity_vec[[l]])
  )
  
  # Generate loadings
  sim_loadings.shared <- lapply(sim_data_spec,
                                function(spec) {
                                  generate_factor_loadings(spec, 
                                                           sparsity_prob = 1 - spec$sparsity, 
                                                           K)
                                }
  )
  sim_loadings.view_specific <- lapply(sim_data_spec,
                                       function(spec) {
                                         generate_factor_loadings(spec, 
                                                                  sparsity_prob = 1 - spec$sparsity)
                                       }
  )
  
  # Generate factor scores
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
                               sim_loadings.shared[[l]] %*% t(sim_loadings.shared[[l]])) / N
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
  if (outcome == "all") {
    # 
    beta_sim <- rnorm(n = K)
    beta_l <- lapply(1:L, 
                     function(l) {
                       rnorm(n = sim_data_spec[[l]]$K_l)
                     }
    )
    
    lp_view_specific <- lapply(1:L, 
                               function(l) {
                                 sim_factors.view_specific[[l]] %*% beta_l[[l]]
                               }
    )
    
    sim_beta_cat <- c(beta_sim, beta_l |> unlist())
    ncoef = K + lapply(1:L, function(l) sim_data_spec[[l]]$K_l) |> unlist() |> sum()
    
    outcome_var <- t(sim_beta_cat) %*% sim_beta_cat / ncoef
    noise_y <- rnorm(n = N, mean = 0, sd = sqrt(outcome_var / snr_y))
    
    sim_data.y <- sim_factors.shared %*% beta_sim + 
      Reduce("+", lp_view_specific) +
      noise_y
  } else if (outcome == "shared") {
    beta_sim <- rnorm(n = K)
    
    outcome_var <- t(beta_sim) %*% beta_sim / K
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