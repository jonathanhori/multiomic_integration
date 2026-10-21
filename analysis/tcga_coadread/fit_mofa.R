# Fit MOFA2 + OLS to one train/test split exported by compare_methods.py.
#
# The Python side writes the standardised views and outcome it gives every
# other method, so MOFA sees exactly the same samples, genes and scaling:
#   <split_dir>/x_train_view<l>.csv, x_test_view<l>.csv   (samples x features, no header)
#   <split_dir>/y_train.csv                                (one column, no header)
#
# Writes to <split_dir>:
#   mofa_predictions.csv   fit, lwr, upr for each test sample (95% OLS prediction interval)
#   mofa_z_test.csv        test factors (projection of the test views onto MOFA's weights)
#   mofa_z_train.csv       training factors
#   mofa_weights_view<l>.csv  MOFA weights per view (features x factors)
#   mofa_runtime.txt       seconds to fit MOFA
#
# usage: Rscript fit_mofa.R <split_dir> <n_views> <K> [seed]

suppressPackageStartupMessages({
  library(MOFA2)
})
args <- commandArgs(trailingOnly = TRUE)
split_dir <- args[1]
n_views   <- as.integer(args[2])
K         <- as.integer(args[3])
seed      <- if (length(args) >= 4) as.integer(args[4]) else 123

# Load the MOFA helpers from R/mofa_run.R without running its command-line
# block (a top-level `if (!interactive())`), which would read this script's arguments.
for (expr in parse(file.path(Sys.getenv("HOME"), "multiomic_integration", "R", "mofa_run.R"))) {
  if (!(is.call(expr) && identical(expr[[1]], as.name("if")))) eval(expr, globalenv())
}

read_mat <- function(name) as.matrix(read.csv(file.path(split_dir, name), header = FALSE))
X_train <- lapply(seq_len(n_views), function(l) read_mat(sprintf("x_train_view%d.csv", l)))
X_test  <- lapply(seq_len(n_views), function(l) read_mat(sprintf("x_test_view%d.csv", l)))
y_train <- as.numeric(read_mat("y_train.csv"))

t0 <- Sys.time()
mofa_obj <- fit_mofa(X_train, K, seed = seed)
runtime <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

lm_fit <- fit_outcome_mofa(mofa_obj, y_train)
Z_test <- project_mofa_test(mofa_obj, X_test)
pred   <- predict_outcome_mofa(lm_fit, Z_test, interval = "prediction", level = 0.95)

write.csv(pred, file.path(split_dir, "mofa_predictions.csv"), row.names = FALSE)
write.csv(Z_test, file.path(split_dir, "mofa_z_test.csv"), row.names = FALSE)
write.csv(get_factors(mofa_obj)[[1]], file.path(split_dir, "mofa_z_train.csv"), row.names = FALSE)
W <- get_weights(mofa_obj)
for (l in seq_len(n_views)) {
  write.csv(W[[l]], file.path(split_dir, sprintf("mofa_weights_view%d.csv", l)), row.names = FALSE)
}
writeLines(as.character(runtime), file.path(split_dir, "mofa_runtime.txt"))
cat("MOFA done:", ncol(Z_test), "factors,", round(runtime, 1), "s\n")
