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