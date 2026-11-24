# Generate the right-censored survival datasets (log-normal AFT study,
# docs/viper_shared_extensions_design.md §4).
#
# Generators: generate_factor_data_shared_multifactor (R/sim_methods_multifactor.R,
# rescale_to_target = TRUE) + add_survival_outcome (R/sim_methods_survival.R).
#
# Base cells: n in {100, 200, 500}, P in {600, 3000} (L = 3 views), c in {0.6, 1, 3, 10},
# relevant-factor strength g = c sqrt(gamma), gamma = P / n; K_nuis = 4, a = 0.5,
# nuisance sd 1, density 0.75, snr_y = 2 (signal variance 1, sigma_y^2 = 0.5 on log T),
# beta_0 = 3. Scenarios:
#   S1  log-normal, random censoring, rate in {0, 0.3, 0.6, 0.8}       (main)
#   S2  log-normal, random + administrative censoring, rate in {0.3, 0.6}
#   S3  Weibull (gumbel) and log-logistic errors, random censoring, rate in {0.3, 0.6}
#   S4  log-normal, censoring depending on view 1, rate in {0.3, 0.6}
#   S5  two relevant factors (strong g_1 = 10 sqrt(gamma), weak g_2 = c sqrt(gamma),
#       omega = (0.5, 0.5)), log-normal, random censoring, rate in {0.3, 0.6}
# 10 replicates; training n samples, test TEST_N = 200.
#
# Common random numbers: within a base cell and replicate, the views, scores and error
# uniforms are the same for every censoring level and error distribution of S1-S4
# (seed depends only on n, P, c, rep), so scenarios differ only in what they change.
# S5 has its own base data (another generator design).
#
# Layout: sim/data/multi_view_shared_survival/{train,test}/<cond>/sim_data_surv_<cond>_rep<r>.rds
#   <cond> = <scen>_n{n}_P{P}_c{c}_<error>_<cens>{rate}   e.g. S1_n100_P600_c1_normal_random0.3
#
# usage: Rscript R/generate_survival_data.R [--only <cond>] [--reps r1,r2,...] [--check]

root <- file.path(Sys.getenv("HOME"), "multiomic_integration")
source(file.path(root, "R", "sim_methods.R"))
source(file.path(root, "R", "sim_methods_multifactor.R"))
source(file.path(root, "R", "sim_methods_survival.R"))

N_ARRAY <- c(100, 200, 500)
P_ARRAY <- c(600, 3000)
C_ARRAY <- c(0.6, 1, 3, 10)
REPS    <- 1:10
TEST_N  <- 200
L <- 3
OUT_ROOT <- file.path(root, "sim", "data", "multi_view_shared_survival")

# scenario list: scen, error, cens, rate, two_factor
SCEN <- rbind(
  data.frame(scen = "S1", error = "normal", cens = "random", rate = c(0, 0.3, 0.6, 0.8), two = FALSE),
  data.frame(scen = "S2", error = "normal", cens = "admin", rate = c(0.3, 0.6), two = FALSE),
  data.frame(scen = "S3", error = rep(c("gumbel", "logistic"), each = 2), cens = "random",
             rate = rep(c(0.3, 0.6), 2), two = FALSE),
  data.frame(scen = "S4", error = "normal", cens = "covariate", rate = c(0.3, 0.6), two = FALSE),
  data.frame(scen = "S5", error = "normal", cens = "random", rate = c(0.3, 0.6), two = TRUE))

cond_name <- function(s, n, P, c) sprintf("%s_n%d_P%d_c%s_%s_%s%s", s$scen, n, P, format(c), s$error, s$cens, format(s$rate))

args <- commandArgs(trailingOnly = TRUE)
only <- NULL
if (length(i <- which(args == "--only"))) only <- args[i + 1]
if (length(i <- which(args == "--reps"))) REPS <- as.integer(strsplit(args[i + 1], ",")[[1]])
check <- "--check" %in% args

for (i_n in seq_along(N_ARRAY)) for (i_P in seq_along(P_ARRAY)) for (i_c in seq_along(C_ARRAY)) {
  n <- N_ARRAY[i_n]; P <- P_ARRAY[i_P]; c <- C_ARRAY[i_c]; sg <- sqrt(P / n)
  conds <- sapply(seq_len(nrow(SCEN)), function(j) cond_name(SCEN[j, ], n, P, c))
  todo <- if (is.null(only)) seq_len(nrow(SCEN)) else which(conds == only)
  if (!length(todo)) next
  for (rep in REPS) {
    base <- list()
    for (j in todo) {
      s <- SCEN[j, ]; cond <- conds[j]
      fname <- sprintf("sim_data_surv_%s_rep%d.rds", cond, rep)
      dirs <- file.path(OUT_ROOT, c("train", "test"), cond)
      if (file.exists(file.path(dirs[2], fname))) next
      key <- if (s$two) "two" else "one"
      if (is.null(base[[key]])) {
        # base data: depends only on (n, P, c, rep) and the generator design
        set.seed(20261119 + 100000 * i_n + 10000 * i_P + 100 * i_c + rep + if (s$two) 5000 else 0)
        g_t <- if (s$two) c(10 * sg, c * sg) else c * sg
        om  <- if (s$two) c(0.5, 0.5) else 1
        tr <- generate_factor_data_shared_multifactor(N = n, p_l_vec = diff(round(seq(0, P, length.out = L + 1))),
                                                      g_target = g_t, omega = om, a = 0.5, K_nuis = 4, nuis_sd = 1,
                                                      nuis_density = 0.75, snr_y = 2, rescale_to_target = TRUE)
        te <- generate_factor_data_shared_multifactor_testing(TEST_N, tr)
        base[[key]] <- list(tr = tr, te = te, U_tr = runif(n), U_te = runif(TEST_N), seed = .Random.seed)
      }
      b <- base[[key]]
      # censoring draws: own stream per scenario row
      set.seed(20261119 + 100000 * i_n + 10000 * i_P + 100 * i_c + rep + 1000 * j + 7)
      out <- add_survival_outcome(b$tr, b$te, b$U_tr, b$U_te, error = s$error, cens = s$cens, cens_rate = s$rate)
      stopifnot(isTRUE(all.equal(out$train$g_true, out$train$g_target, tolerance = 1e-10)),
                all(out$train$y > 0), all(out$test$y > 0),
                all(out$train$y <= out$train$T + 1e-12), all((out$train$event == 1) == (abs(out$train$y - out$train$T) < 1e-12 * out$train$T)))
      for (d in dirs) dir.create(d, recursive = TRUE, showWarnings = FALSE)
      saveRDS(out$train, file.path(dirs[1], fname))
      saveRDS(out$test, file.path(dirs[2], fname))
      if (check) cat(sprintf("%-45s rep %d  censored train %.2f test %.2f (target %.2f)\n", cond, rep,
                             out$train$cens_realised, out$test$cens_realised, s$rate))
    }
  }
  if (!check) cat(sprintf("n%d_P%d_c%s done\n", n, P, format(c)))
}
