# Unit checks for generate_factor_data_shared_controlled (R/sim_methods.R).
#
#   1. realised strength g_true vs g_target (and g_true = sum of squared support loadings)
#   2. G[k*, -k*] == 0 exactly (disjoint supports)
#   3. structure: supports disjoint from nuisance loadings and outside features;
#      the testing companion keeps the true parameters
#   4. Var(y) decomposition on a large sample (beta^2 + sigma_y^2 + ||theta||^2,
#      outside share = r_out)
#   5. theory R* (controlled_oracle) vs the empirical MSE of the oracle predictor on a
#      large test set; for r_out = 0 also vs oracle_from_x = beta^T E[z | x]
#
# usage: Rscript R/check_controlled_generator.R      (stops on the first failure)

source(file.path(Sys.getenv("HOME"), "multiomic_integration", "R", "sim_methods.R"))
set.seed(20261021)
ok <- function(cond, msg) { if (!isTRUE(cond)) stop("FAILED: ", msg); cat("  ok  ", msg, "\n") }

# 1-2. Realised strength across replicates, a few (g_target, P) settings
cat("1-2. strength and G\n")
for (cfg in list(c(g = 0.33, P = 600), c(g = 2.45, P = 600), c(g = 10, P = 3000), c(g = 54.8, P = 3000))) {
  gs <- replicate(200, {
    d <- generate_factor_data_shared_controlled(50, rep(cfg["P"] / 3, 3), cfg["g"])
    k <- d$k_star
    stopifnot(abs(d$g_true - sum(sapply(1:3, function(l) sum(d$Lambda_l[[l]][d$support[[l]], k]^2)))) < 1e-10,
              max(abs(d$G[k, -k])) == 0)
    c(d$g_true, d$s)
  })
  cat(sprintf("  g_target %5.2f  s %3d  E[g] = s a^2 = %6.2f  mean(g_true) %6.2f  sd %5.2f  (CV %.2f; theory sqrt(2/s) = %.2f)\n",
              cfg["g"], gs[2, 1], gs[2, 1] * 0.25, mean(gs[1, ]), sd(gs[1, ]),
              sd(gs[1, ]) / mean(gs[1, ]), sqrt(2 / gs[2, 1])))
  ok(abs(mean(gs[1, ]) - gs[2, 1] * 0.25) < 4 * sd(gs[1, ]) / sqrt(200), "mean realised g = s a^2")
}
ok(TRUE, "g_true equals the sum of squared support loadings; G[k*, -k*] == 0 (200 reps each)")

# 3. Structure
cat("3. structure\n")
d <- generate_factor_data_shared_controlled(200, c(200, 200, 200), 5, r_out = 0.25)
k <- d$k_star
for (l in 1:3) {
  S <- d$support[[l]]
  ok(all(d$Lambda_l[[l]][S, -k] == 0), sprintf("view %d: no nuisance loading on S", l))
  ok(all(d$Lambda_l[[l]][-S, k] == 0), sprintf("view %d: relevant loading only on S", l))
  ok(length(intersect(S, d$out_idx[[l]])) == 0, sprintf("view %d: outside features not in S", l))
}
ok(sum(lengths(d$out_idx)) == 5 && sum(unlist(d$theta_out) != 0) == 5, "n_out = 5 outside features")
ok(all(unlist(d$psi) == 1), "psi = 1")
dt <- generate_factor_data_shared_controlled_testing(100, d)
ok(identical(dt$Lambda_l, d$Lambda_l) && identical(dt$beta, d$beta) && identical(dt$theta_out, d$theta_out) &&
     dt$sigma2_y == d$sigma2_y && !identical(dt$Z[1:5, ], d$Z[1:5, ]), "testing companion: same Theta0, fresh Z")

# 4-5. Variance decomposition and oracle risk on a large sample
cat("4-5. Var(y) and R* on large test sets\n")
N_big <- 40000
for (r_out in c(0, 0.25)) {
  for (g in c(1, 10)) {
    d <- generate_factor_data_shared_controlled(200, c(200, 200, 200), g, r_out = r_out)
    big <- generate_factor_data_shared_controlled_testing(N_big, d)
    orc <- controlled_oracle(d)
    X <- do.call(cbind, big$X_l)
    y <- as.vector(big$y)
    th <- unlist(d$theta_out)
    eps <- X - big$Z %*% t(do.call(rbind, d$Lambda_l))
    out_term <- as.vector(eps %*% th)
    var_terms <- c(beta = d$beta[d$k_star]^2, sigma2_y = d$sigma2_y, outside = sum(th^2))
    se <- function(v) sd(v) / sqrt(length(v))
    cat(sprintf("  r_out %.2f g %4.1f: Var(y) emp %.3f theory %.3f | outside share emp %.3f theory %.3f\n",
                r_out, g, var(y), sum(var_terms), var(out_term) / var(y), var_terms[["outside"]] / sum(var_terms)))
    ok(abs(var(y) - sum(var_terms)) < 4 * sum(var_terms) * sqrt(2 / N_big), "Var(y) decomposition")
    ok(abs(var_terms[["outside"]] / sum(var_terms) - r_out) < 1e-12, "outside share = r_out (by construction)")
    sq <- (y - X %*% orc$b_star)^2
    cat(sprintf("        R* theory %.4f  empirical MSE of b*^T x %.4f (se %.4f)\n", orc$R_star, mean(sq), se(sq)))
    ok(abs(mean(sq) - orc$R_star) < 4 * se(sq), "R* matches the oracle's empirical MSE")
    if (r_out == 0) {
      Lam <- do.call(rbind, d$Lambda_l)
      Ez <- X %*% Lam %*% solve(diag(length(d$beta)) + t(Lam) %*% Lam)   # psi = 1
      sq2 <- (y - Ez %*% d$beta)^2
      cat(sprintf("        oracle_from_x MSE (beta^T E[z|x]) %.4f; R*(no outside) %.4f\n",
                  mean(sq2), orc$R_star_no_outside))
      ok(abs(mean(sq2) - orc$R_star) < 4 * se(sq2) && abs(orc$R_star - orc$R_star_no_outside) < 1e-10,
         "oracle_from_x agrees with R* when r_out = 0")
    }
  }
}
cat("\nall checks passed\n")
