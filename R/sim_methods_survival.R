#############################################################################
# Right-censored time-to-event outcome for the controlled-strength data
# (docs/viper_shared_extensions_design.md §4; log-normal AFT study).
#
# Requires R/sim_methods.R and R/sim_methods_multifactor.R to be sourced first.
#
# The views, scores and loadings come from generate_factor_data_shared_multifactor
# (rescale_to_target = TRUE; m = 1 for a single relevant factor). The outcome is then
# replaced by a log event time on the same signal:
#
#   log T_i = beta_0 + sum_r beta_r z_{i k*_r} + sigma_y e_i,
#   signal variance sum_r beta_r^2 = 1, sigma_y^2 = 1 / snr_y, beta_0 = 3,
#   e_i with mean 0, variance 1:
#     "normal"   e ~ N(0, 1)                                  (log-normal AFT: the fitted model)
#     "gumbel"   e = (log E + gamma_Euler) / (pi / sqrt(6)),  E ~ Exp(1)   (Weibull truth)
#     "logistic" e = Logistic(0, 1) / (pi / sqrt(3))                       (log-logistic truth)
#   all three by the quantile transform of one Uniform draw U_i, so the error
#   distributions are compared on common random numbers.
#
# Censoring (independent of T given x), calibrated to a target censored fraction
# `cens_rate` of the population (Monte Carlo over z, bisection):
#   "random"     C_i ~ Exp(gamma)
#   "admin"      C_i = min(Exp(gamma), tau), tau = 90th percentile of T (end of follow-up)
#   "covariate"  C_i ~ Exp(gamma_i), log gamma_i = a + b * xbar_i, xbar_i the standardised
#                mean of view 1 (b = 1). Non-informative given x, but Harrell's C is biased.
#   cens_rate = 0: no censoring.
# Observed: y_i = min(T_i, C_i), event_i = 1(T_i <= C_i).
#
# Returned (train): the multi-factor fields, with y = observed time and
#   event, T (true event time), logT, beta0, sigma2_y (log-time scale), error, cens,
#   cens_rate (target), cens_par (calibrated gamma or a, tau, b, xbar_sd),
#   cens_realised (fraction censored in this replicate).
#############################################################################

survival_error <- function(U, error) {
  switch(error,
         normal   = qnorm(U),
         gumbel   = (log(-log(1 - U)) + 0.5772156649015329) / (pi / sqrt(6)),
         logistic = log(U / (1 - U)) / (pi / sqrt(3)),
         stop("error must be normal, gumbel or logistic"))
}

# Population mean of view 1 for given z rows: xbar = z' lambda_bar + eps_bar,
# eps_bar ~ N(0, mean(psi_1) / p_1). Returns list(lambda_bar, sd) for standardising.
view1_mean_par <- function(d) {
  lam_bar <- colMeans(d$Lambda_l[[1]])
  noise_var <- mean(d$psi[[1]]) / nrow(d$Lambda_l[[1]])
  list(lambda_bar = lam_bar, sd = sqrt(sum(lam_bar^2) + noise_var), noise_var = noise_var)
}

# Calibrate the censoring parameters on a Monte Carlo sample of the population.
calibrate_censoring <- function(d, beta0, sigma_y, error, cens, cens_rate, n_mc = 50000, b_cov = 1) {
  if (cens_rate == 0) return(list(type = "none"))
  Zmc <- matrix(rnorm(n_mc * d$K), n_mc, d$K)
  Tmc <- exp(beta0 + drop(Zmc %*% d$beta) + sqrt(sigma_y^2) * survival_error(runif(n_mc), error))
  tau <- if (cens == "admin") unname(quantile(Tmc, 0.9)) else Inf
  lg <- if (cens == "covariate") {
    v <- view1_mean_par(d)
    xbar <- (drop(Zmc %*% v$lambda_bar) + rnorm(n_mc, 0, sqrt(v$noise_var))) / v$sd
    function(a) a + b_cov * xbar
  } else function(a) rep(a, n_mc)
  rate <- function(a) mean(ifelse(Tmc > tau, 1, 1 - exp(-exp(lg(a)) * Tmc)))
  if (rate(-50) > cens_rate) stop(sprintf("administrative censoring alone exceeds the target rate %.2f", cens_rate))
  lo <- -50; hi <- 20
  for (it in 1:100) { mid <- (lo + hi) / 2; if (rate(mid) < cens_rate) lo <- mid else hi <- mid }
  list(type = cens, a = (lo + hi) / 2, tau = tau, b = if (cens == "covariate") b_cov else 0,
       xbar_sd = if (cens == "covariate") view1_mean_par(d)$sd else NA_real_)
}

apply_censoring <- function(d, logT, par) {
  n <- length(logT); Tt <- exp(logT)
  if (par$type == "none") return(list(y = Tt, event = rep(1, n)))
  lg <- if (par$type == "covariate") {
    xbar <- rowMeans(d$X_l[[1]]) / par$xbar_sd
    par$a + par$b * xbar
  } else rep(par$a, n)
  C <- pmin(rexp(n, rate = exp(lg)), par$tau)
  list(y = pmin(Tt, C), event = as.numeric(Tt <= C))
}

# Add the survival outcome to a training set (and its test companion) drawn by
# generate_factor_data_shared_multifactor / _testing. U_train / U_test: Uniform(0, 1)
# draws for the errors (shared across error distributions). The censoring draws use the
# current RNG state.
add_survival_outcome <- function(tr, te, U_train, U_test, error = "normal", cens = "random",
                                 cens_rate = 0.3, beta0 = 3) {
  sigma_y <- sqrt(tr$sigma2_y)
  par <- calibrate_censoring(tr, beta0, sigma_y, error, cens, cens_rate)
  out <- list()
  for (nm in c("train", "test")) {
    d <- if (nm == "train") tr else te
    U <- if (nm == "train") U_train else U_test
    logT <- beta0 + drop(d$Z %*% d$beta) + sigma_y * survival_error(U, error)
    obs <- apply_censoring(d, logT, par)
    d$y <- matrix(obs$y, ncol = 1)
    d$event <- obs$event
    d$T <- exp(logT); d$logT <- logT
    d$beta0 <- beta0; d$error <- error; d$cens <- cens; d$cens_rate <- cens_rate
    d$cens_par <- par; d$cens_realised <- 1 - mean(obs$event)
    out[[nm]] <- d
  }
  out
}
