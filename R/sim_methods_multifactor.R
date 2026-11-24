#############################################################################
# Controlled-strength shared-factor data with several outcome-relevant factors
# (docs/viper_shared_extensions_design.md §1; the two-factor strong + weak study is the first).
#
# Extends generate_factor_data_shared_controlled (R/sim_methods.R, single relevant
# factor) to m relevant factors. Requires R/sim_methods.R to be sourced first
# (uses controlled_draw). For one replicate (L views, p_l features, P = sum p_l,
# K = K_nuis + m factors):
#
#   z_i  ~ N(0, I_K)
#   x_il = Lambda_l z_i + eps_il,   eps_il ~ N(0, I)          (psi_j = 1)
#   y_i  = sum_r beta_r z_{i, k*_r} + e_i,   e_i ~ N(0, sigma_y^2)
#
# Relevant factor r (column k*_r = K_nuis + r):
#   own support S_r of s_r = round(g_target[r] / a^2) features, disjoint from the
#   other relevant supports and from the nuisance loadings, split as evenly as
#   possible across the views in support_views[[r]] (default: all views);
#   Lambda_{j, k*_r} ~ N(0, a^2) for j in S_r, 0 elsewhere.
#   Strength g_r = sum_{j in S_r} Lambda_{j, k*_r}^2 / psi_j (returned as g_true).
#   rescale_to_target = TRUE multiplies factor r's support loadings by
#   sqrt(g_target[r] / g_r), so that g_true == g_target exactly. TRUE is the default
#   (decided 2026-11-18 for every extension study): with small supports the unscaled
#   realised strength is typically 40-70 % off target.
# Nuisance factors (columns 1..K_nuis): features outside every S_r only,
#   Lambda_jk = N(0, nuis_sd^2) * Bernoulli(nuis_density).
#   Disjoint supports make G = sum_l Lambda_l^T Psi_l^-1 Lambda_l block diagonal
#   on the relevant columns, so each g_r is an eigenvalue of G.
# Outcome: signal variance V_s = 1 is split by the fixed shares omega (sum 1):
#   beta_{k*_r} = sign_r * sqrt(omega_r), sign_r = +-1 at random; nuisance beta = 0;
#   sigma_y^2 = 1 / snr_y.  Var(y) = 1 + sigma_y^2.
#   (The single-factor generator draws beta_{k*} ~ N(0, 1) instead; here the
#   shares are part of the design, so beta is fixed up to sign.)
#
# Returns the fields of generate_factor_data_shared_controlled (so
# run_methods.load_rds / process_data and risk_metrics.true_params work
# unchanged), with these differences: g_true, g_target, s, k_star are length-m
# vectors; support is a list over factors of lists over views (1-based indices);
# omega, m, support_views are added; theta_out is all zero (no outside-factor term).
#
# High-PVE predictive factors (omega_nuis; November 2026): omega_nuis is a K_nuis-vector
# of signal shares carried by the nuisance columns, so a nuisance factor (dense loadings
# on ~75 % of the features, PVE ~15 %) can also predict y:
#   beta_k = sign_k * sqrt(omega_nuis[k]) for the nuisance columns with omega_nuis > 0,
#   and sum(omega) + sum(omega_nuis) = 1 (signal variance still 1).
# The default (all zero) draws no extra random numbers, so every earlier design is
# reproduced exactly. Returned as omega_nuis; beta_weights marks every column with
# beta != 0.
#############################################################################
generate_factor_data_shared_multifactor <- function(N, p_l_vec, g_target, omega,
                                                    a = 0.5,
                                                    K_nuis = 4,
                                                    nuis_sd = 1,
                                                    nuis_density = 0.75,
                                                    snr_y = 2,
                                                    support_views = NULL,
                                                    rescale_to_target = TRUE,
                                                    omega_nuis = rep(0, K_nuis)) {
  L <- length(p_l_vec)
  m <- length(g_target)
  stopifnot(length(omega) == m, all(omega > 0), length(omega_nuis) == K_nuis, all(omega_nuis >= 0),
            abs(sum(omega) + sum(omega_nuis) - 1) < 1e-12)
  K <- K_nuis + m
  k_star <- K_nuis + seq_len(m)
  s <- round(g_target / a^2)
  stopifnot(all(s >= 1), sum(s) < sum(p_l_vec))
  if (is.null(support_views)) support_views <- rep(list(seq_len(L)), m)
  stopifnot(length(support_views) == m)

  # Disjoint supports: factor r takes s_r features from the views in
  # support_views[[r]], split as evenly as possible, among features not yet used.
  free <- lapply(p_l_vec, seq_len)
  support <- vector("list", m)
  for (r in seq_len(m)) {
    v <- support_views[[r]]
    s_rv <- rep(s[r] %/% length(v), length(v)) + (seq_along(v) <= s[r] %% length(v))
    support[[r]] <- lapply(seq_len(L), function(l) integer(0))
    for (i in seq_along(v)) {
      l <- v[i]
      stopifnot(s_rv[i] <= length(free[[l]]))
      pick <- if (s_rv[i] > 0) sort(free[[l]][sample.int(length(free[[l]]), s_rv[i])]) else integer(0)
      support[[r]][[l]] <- pick
      free[[l]] <- setdiff(free[[l]], pick)
    }
  }

  Lambda_l <- lapply(seq_len(L), function(l) {
    Lam <- matrix(0, p_l_vec[l], K)
    for (r in seq_len(m)) {
      S <- support[[r]][[l]]
      Lam[S, k_star[r]] <- rnorm(length(S), 0, a)
    }
    off <- free[[l]]
    Lam[off, seq_len(K_nuis)] <- matrix(rnorm(length(off) * K_nuis, 0, nuis_sd) *
                                          rbinom(length(off) * K_nuis, 1, nuis_density),
                                        length(off), K_nuis)
    Lam
  })
  psi <- lapply(p_l_vec, function(p) rep(1, p))
  if (rescale_to_target) {
    for (r in seq_len(m)) {
      g_raw <- sum(sapply(seq_len(L), function(l) sum(Lambda_l[[l]][support[[r]][[l]], k_star[r]]^2)))
      Lambda_l <- lapply(Lambda_l, function(Lam) { Lam[, k_star[r]] <- Lam[, k_star[r]] * sqrt(g_target[r] / g_raw); Lam })
    }
  }

  beta <- rep(0, K)
  beta[k_star] <- sample(c(-1, 1), m, replace = TRUE) * sqrt(omega)
  if (any(omega_nuis > 0)) {
    kn <- which(omega_nuis > 0)
    beta[kn] <- sample(c(-1, 1), length(kn), replace = TRUE) * sqrt(omega_nuis[kn])
  }
  sigma2_y <- 1 / snr_y
  theta_out <- lapply(p_l_vec, function(p) rep(0, p))

  G <- Reduce(`+`, lapply(seq_len(L), function(l) t(Lambda_l[[l]]) %*% (Lambda_l[[l]] / psi[[l]])))
  g_true <- diag(G)[k_star]

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
    beta_weights = as.numeric(beta != 0),
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
    m            = m,
    omega        = omega,
    omega_nuis   = omega_nuis,
    support      = support,
    support_views = support_views,
    theta_out    = theta_out,
    out_idx      = lapply(seq_len(L), function(l) integer(0)),
    r_out        = 0,
    rescale_to_target = rescale_to_target,
    G            = G
  )
}


# Test set: true parameters copied from the training replicate; z, eps, e drawn afresh.
generate_factor_data_shared_multifactor_testing <- function(N, train_data) {
  draw <- controlled_draw(N, train_data$Lambda_l, train_data$psi, train_data$beta,
                          train_data$sigma2_y, train_data$theta_out)
  out <- train_data
  out$N   <- N
  out$X_l <- draw$X_l
  out$y   <- draw$y
  out$Z   <- draw$Z
  out
}
