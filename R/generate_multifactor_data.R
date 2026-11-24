# Generate the multi-factor controlled-strength datasets
# (docs/viper_shared_extensions_design.md §1).
#
# Generator: generate_factor_data_shared_multifactor / _testing (R/sim_methods_multifactor.R).
#
# Two-factor study (one strong and one weak outcome-relevant factor), the first
# multi-factor study:
#   m = 2 relevant factors, g_1 = 10 sqrt(gamma) (strong), g_2 = c sqrt(gamma) (weak),
#   gamma = P / n; weak factor's share of the outcome signal omega_2 in {0.25, 0.5, 0.75}
#   (omega = (1 - omega_2, omega_2)); signal variance 1, snr_y = 2.
#   The same (n, P, c) grid as the single-factor study (generate_controlled_data.R):
#   n in {100, 200, 500}, P in {600, 3000} with L = 3 views (p_l = P / 3),
#   c in {0.3, 0.6, 1, 1.5, 2, 3, 10}; plus P = 10000 (views 3334 / 3333 / 3333) with
#   c in {0.3, 1, 3, 10} (C_ARRAY_HIGHDIM). c = 10: the weak factor as strong as the
#   strong one (tie check).
#   K_nuis = 4 (K = 6), a = 0.5, nuisance sd 1, density 0.75, as the single-factor grid;
#   rescale_to_target = TRUE (g_true == g_target for both relevant factors).
#   Each training replicate has N = n samples; its test set TEST_N fresh samples with
#   the same true parameters.
#
# Layout:
#   sim/data/multi_view_shared_multifactor/two_factor/{train,test}/n{n}_P{P}_c{c}_w{omega_2}/
#       sim_data_twofactor_n{n}_P{P}_c{c}_w{omega_2}_rep{rep}.rds
# Seeds: set.seed(20261118 + 100000 i_n + 10000 i_P + 1000 i_w + 100 i_c + rep)
# (i_* = position in the grid vectors), so any dataset can be regenerated alone and
# the script resumes (existing test files are skipped).
#
# Single-factor study (--single; November 2026 rerun of the controlled grid): m = 1
#   relevant factor with g = c sqrt(gamma), omega = 1 (beta = +-1, signal variance 1,
#   sigma_y^2 = 0.5), rescale_to_target = TRUE, the same (n, P, c) cells, K_nuis = 4
#   (K = 5). This is the m = 1 design of the missingness and survival studies, and it
#   replaces the unscaled single-factor data of generate_controlled_data.R, whose
#   realised strength is 0-3.9 x target at c = 0.3 and 0.2-2 x at c = 0.6 (and whose
#   beta ~ N(0, 1) can be near 0).
#   Layout: sim/data/multi_view_shared_multifactor/single_factor/{train,test}/n{n}_P{P}_c{c}/
#       sim_data_singlefactor_n{n}_P{P}_c{c}_rep{rep}.rds
#   Seeds: set.seed(20261122 + 100000 i_n + 10000 i_P + 100 i_c + rep).
#
# High-PVE + weak study (--highpve; November 2026): one predictive factor that is as
#   large as the nuisance factors (nuisance column 1: dense loadings on ~75 % of the
#   features, sd 1, PVE ~15 %) and one weak predictive factor (g = c sqrt(gamma), own
#   support, rescaled to target), plus 3 nuisance factors with beta = 0 (K = 5). The
#   weak factor carries a share omega_2 in {0.25, 0.5, 0.75} of the signal and the large
#   one 1 - omega_2 (omega = omega_2, omega_nuis = (1 - omega_2, 0, 0, 0)); signal variance
#   1, snr_y = 2. Same (n, P, c) cells as the other studies.
#   Layout: sim/data/multi_view_shared_multifactor/high_pve/{train,test}/n{n}_P{P}_c{c}_w{omega_2}/
#       sim_data_highpve_n{n}_P{P}_c{c}_w{omega_2}_rep{rep}.rds
#   Seeds: set.seed(20261124 + 100000 i_n + 10000 i_P + 1000 i_w + 100 i_c + rep).
#
# usage: Rscript R/generate_multifactor_data.R [--single | --highpve] [--only n,P,c(,w)] [--reps r1,r2,...]

root <- file.path(Sys.getenv("HOME"), "multiomic_integration")
source(file.path(root, "R", "sim_methods.R"))
source(file.path(root, "R", "sim_methods_multifactor.R"))

N_ARRAY <- c(100, 200, 500)
P_ARRAY <- c(600, 3000, 10000)
C_ARRAY <- c(0.3, 0.6, 1, 1.5, 2, 3, 10)
C_ARRAY_HIGHDIM <- c(0.3, 1, 3, 10)   # c values generated for P > 3000
W_ARRAY <- c(0.25, 0.5, 0.75)   # omega_2, the weak factor's share of the signal
G_STRONG <- 10                  # strong factor: g_1 = G_STRONG * sqrt(gamma)
REPS    <- 1:10
TEST_N  <- 100
# rescale_to_target = TRUE (decided 2026-11-18, the default for every extension study):
# each relevant factor's support loadings are rescaled so that g_true == g_target
# exactly. Without it the realised strength is a sum of s squared N(0, a^2) draws, and for
# supports of a few features a typical replicate is 40-70 % off target, so results by c
# would be unreliable exactly where the weak factor matters. Checked for every replicate
# below (the script stops on a mismatch).
RESCALE <- TRUE
L <- 3
OUT_ROOT <- file.path(root, "sim", "data", "multi_view_shared_multifactor", "two_factor")
SINGLE <- "--single" %in% commandArgs(trailingOnly = TRUE)
if (SINGLE) {
  OUT_ROOT <- file.path(root, "sim", "data", "multi_view_shared_multifactor", "single_factor")
  W_ARRAY <- 1                  # omega = 1: one relevant factor carries all the signal
}
HIGHPVE <- "--highpve" %in% commandArgs(trailingOnly = TRUE)
stopifnot(!(SINGLE && HIGHPVE))
if (HIGHPVE) OUT_ROOT <- file.path(root, "sim", "data", "multi_view_shared_multifactor", "high_pve")

args <- commandArgs(trailingOnly = TRUE)
only <- NULL
if (length(i <- which(args == "--only"))) only <- as.numeric(strsplit(args[i + 1], ",")[[1]])
if (length(i <- which(args == "--reps"))) REPS <- as.integer(strsplit(args[i + 1], ",")[[1]])

cond_name <- function(n, P, c, w) if (SINGLE) sprintf("n%d_P%d_c%s", n, P, format(c)) else
  sprintf("n%d_P%d_c%s_w%s", n, P, format(c), format(w))

for (i_n in seq_along(N_ARRAY)) for (i_P in seq_along(P_ARRAY))
for (i_w in seq_along(W_ARRAY)) for (i_c in seq_along(C_ARRAY)) {
  n <- N_ARRAY[i_n]; P <- P_ARRAY[i_P]; c <- C_ARRAY[i_c]; w <- W_ARRAY[i_w]
  if (!is.null(only) && !all(c(n, P, c, w)[seq_along(only)] == only)) next
  if (P > 3000 && !(c %in% C_ARRAY_HIGHDIM)) next
  cond <- cond_name(n, P, c, w)
  dirs <- file.path(OUT_ROOT, c("train", "test"), cond)
  for (d in dirs) dir.create(d, recursive = TRUE, showWarnings = FALSE)
  sg <- sqrt(P / n)
  for (rep in REPS) {
    prefix <- if (SINGLE) "singlefactor" else if (HIGHPVE) "highpve" else "twofactor"
    fname <- sprintf("sim_data_%s_%s_rep%d.rds", prefix, cond, rep)
    if (file.exists(file.path(dirs[2], fname))) next
    if (SINGLE) set.seed(20261122 + 100000 * i_n + 10000 * i_P + 100 * i_c + rep) else
      if (HIGHPVE) set.seed(20261124 + 100000 * i_n + 10000 * i_P + 1000 * i_w + 100 * i_c + rep) else
      set.seed(20261118 + 100000 * i_n + 10000 * i_P + 1000 * i_w + 100 * i_c + rep)
    tr <- generate_factor_data_shared_multifactor(
      N = n, p_l_vec = diff(round(seq(0, P, length.out = L + 1))),
      g_target = if (SINGLE || HIGHPVE) c * sg else c(G_STRONG * sg, c * sg),
      omega = if (SINGLE) 1 else if (HIGHPVE) w else c(1 - w, w),
      omega_nuis = if (HIGHPVE) c(1 - w, 0, 0, 0) else rep(0, 4),
      a = 0.5, K_nuis = 4, nuis_sd = 1, nuis_density = 0.75, snr_y = 2,
      rescale_to_target = RESCALE)
    te <- generate_factor_data_shared_multifactor_testing(TEST_N, tr)
    if (RESCALE) {
      stopifnot("g_true != g_target" = isTRUE(all.equal(tr$g_true, tr$g_target, tolerance = 1e-10)),
                "test set strength differs" = identical(te$g_true, tr$g_true),
                "G diagonal != g_target" = isTRUE(all.equal(unname(diag(tr$G)[tr$k_star]), tr$g_target, tolerance = 1e-10)))
    }
    saveRDS(tr, file.path(dirs[1], fname))
    saveRDS(te, file.path(dirs[2], fname))
  }
  cat(cond, "done\n")
}
