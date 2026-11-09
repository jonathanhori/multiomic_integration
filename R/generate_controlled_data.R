# Generate the controlled-strength datasets (weak-factor study, Phase 2).
#
# Generator: generate_factor_data_shared_controlled / _testing (R/sim_methods.R);
# the data-generating process is documented in docs/controlled_strength_data.md.
#
# Grid (Phase 2):
#   n in {100, 200, 500}, P in {600, 3000} with L = 3 views of p_l = P / 3,
#   c in {0.3, 0.6, 1, 1.5, 2, 3, 10},  g_target = c * sqrt(gamma), gamma = P / n,
#   K_nuis = 4 (K = 5), a = 0.5, nuisance density 0.75, snr_y = 2, r_out = 0,
#   rescale_to_target = FALSE (realised g_true varies around g_target).
#   Each training replicate has N = n samples; its test set has TEST_N fresh samples
#   with the same true parameters.
#
# Layout:
#   sim/data/multi_view_shared_controlled/k.5/{train,test}/n{n}_P{P}_c{c}/
#       sim_data_controlled_n{n}_P{P}_c{c}_rep{rep}.rds
# Seeds: each replicate uses set.seed(20261021 + 100000 i_n + 10000 i_P + 100 i_c + rep)
# (i_* = position in the grid vectors), so any dataset can be regenerated alone and
# the script resumes (existing test files are skipped).
#
# usage: Rscript R/generate_controlled_data.R [--only n,P,c] [--reps r1,r2,...]

source(file.path(Sys.getenv("HOME"), "multiomic_integration", "R", "sim_methods.R"))

N_ARRAY <- c(100, 200, 500)
P_ARRAY <- c(600, 3000)
C_ARRAY <- c(0.3, 0.6, 1, 1.5, 2, 3, 10)
REPS    <- 1:10          # TODO(user): number of replicates (spec: start with 10)
TEST_N  <- 100           # TODO(user): test-set size (100 as in the misaligned study; the
                         # primary metric, exact risk, does not use the test set)
L <- 3
OUT_ROOT <- file.path(Sys.getenv("HOME"), "multiomic_integration", "sim", "data",
                      "multi_view_shared_controlled", "k.5")

args <- commandArgs(trailingOnly = TRUE)
only <- NULL
if (length(i <- which(args == "--only"))) only <- as.numeric(strsplit(args[i + 1], ",")[[1]])
if (length(i <- which(args == "--reps"))) REPS <- as.integer(strsplit(args[i + 1], ",")[[1]])

cond_name <- function(n, P, c) sprintf("n%d_P%d_c%s", n, P, format(c))

for (i_n in seq_along(N_ARRAY)) for (i_P in seq_along(P_ARRAY)) for (i_c in seq_along(C_ARRAY)) {
  n <- N_ARRAY[i_n]; P <- P_ARRAY[i_P]; c <- C_ARRAY[i_c]
  if (!is.null(only) && !all(c(n, P, c) == only)) next
  cond <- cond_name(n, P, c)
  dirs <- file.path(OUT_ROOT, c("train", "test"), cond)
  for (d in dirs) dir.create(d, recursive = TRUE, showWarnings = FALSE)
  g_target <- c * sqrt(P / n)
  for (rep in REPS) {
    fname <- sprintf("sim_data_controlled_%s_rep%d.rds", cond, rep)
    if (file.exists(file.path(dirs[2], fname))) next
    set.seed(20261021 + 100000 * i_n + 10000 * i_P + 100 * i_c + rep)
    tr <- generate_factor_data_shared_controlled(N = n, p_l_vec = rep(P / L, L), g_target = g_target,
                                                 a = 0.5, K_nuis = 4, nuis_sd = 1, nuis_density = 0.75,
                                                 snr_y = 2, r_out = 0, rescale_to_target = FALSE)
    te <- generate_factor_data_shared_controlled_testing(TEST_N, tr)
    saveRDS(tr, file.path(dirs[1], fname))
    saveRDS(te, file.path(dirs[2], fname))
  }
  cat(sprintf("%s: g_target %.2f, s = %d\n", cond, g_target, round(g_target / 0.25)))
}
