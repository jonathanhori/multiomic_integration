#!/bin/bash
# One-time setup of this repository on the Hoffman2 Slurm cluster.
#
# Run from the repository root as a batch job (not on the login node). The first run
# compiles MOFA2's R dependencies from source, which takes hours on 2 CPUs:
#   cd $SCRATCH/multiomic_integration
#   sbatch --job-name=setup_smoke --account=pi_huazhou --time=08:00:00 --cpus-per-task=8 \
#     --mem=16G --output=cluster/logs/setup_smoke.%j.out --wrap "bash cluster/setup_hoffman2.sh --smoke"
# Each step keeps what it finished, so after a time-out simply submit it again.
#
# What it does (each step is skipped if already done):
#   1. ~/multiomic_integration -> this directory (symlink). The code and the R scripts
#      locate the repository at ~/multiomic_integration; keeping the repository itself in
#      $SCRATCH (or your group's project directory) keeps the data and results out of the
#      small $HOME quota.
#   2. cluster/logs/ for the job logs (Slurm does not create directories).
#   3. the conda environment from cluster/environment_hoffman2.yml, in $SCRATCH/software
#      (cluster/hoffman2_env.sh), then rpy2 linked to the R module. An environment left
#      half-built (time-out, partial purge) is detected by an import check and repaired.
#   4. MOFA2 (Bioconductor) in the R library $SCRATCH/software/R, and its Python backend
#      (basilisk builds it on first use: done here once, so the array tasks do not all
#      build it at once).
# Rerun it after a pause of two weeks or more: $SCRATCH deletes files older than 14 days.
#   5. import checks; with --smoke, one full replicate of each study (~10-20 min).
# Edit cluster/hoffman2_env.sh first if the module versions differ (module spider
# miniforge; h2 modules R).
set -eo pipefail     # no -u: the module and conda shell functions use unset variables
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
echo "== repository: $REPO"

# 1. symlink
LINK="$HOME/multiomic_integration"
if [ -e "$LINK" ] || [ -L "$LINK" ]; then
  if [ "$(cd "$LINK" && pwd -P)" != "$(pwd -P)" ]; then
    echo "ERROR: $LINK exists and is not this repository; move it or link it here by hand." >&2
    exit 1
  fi
  echo "== $LINK already points here"
elif [ "$(pwd -P)" != "$(cd "$HOME" && pwd -P)/multiomic_integration" ]; then
  ln -s "$REPO" "$LINK"; echo "== linked $LINK -> $REPO"
fi

# 2. log directory
mkdir -p cluster/logs

# 3. conda environment, in $SCRATCH (cluster/hoffman2_env.sh says why)
eval "$(grep -E '^(MINIFORGE_MODULE|R_MODULE|ENV_ROOT|CONDA_ENV)=|^export (CONDA_PKGS_DIRS|PIP_CACHE_DIR|BASILISK_EXTERNAL_DIR)=' cluster/hoffman2_env.sh)"
mkdir -p "$ENV_ROOT" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$BASILISK_EXTERNAL_DIR"
module purge
module load "$MINIFORGE_MODULE"
ENV_CHECK="import torch, pyro, zuko, numpy, pandas, scipy, sklearn, sksurv, joblib, optuna, tqdm, pyreadr"
if [ ! -d "$CONDA_ENV/conda-meta" ]; then
  # not there, or a creation that stopped before conda finished (e.g. after a purge or a time-out)
  echo "== creating conda environment $CONDA_ENV"
  conda env create --yes -p "$CONDA_ENV" -f cluster/environment_hoffman2.yml
elif ! "$CONDA_ENV/bin/python" -c "$ENV_CHECK" 2>/dev/null; then
  # conda finished but the pip packages are missing or incomplete (time-out in pip, partial purge)
  echo "== repairing conda environment $CONDA_ENV (some packages missing)"
  conda env update -p "$CONDA_ENV" -f cluster/environment_hoffman2.yml
fi
"$CONDA_ENV/bin/python" -c "$ENV_CHECK" || { echo "ERROR: environment $CONDA_ENV is incomplete" >&2; exit 1; }
source cluster/hoffman2_env.sh
python -c "import rpy2" 2>/dev/null || { echo "== installing rpy2 against $(R --version | head -1)"; pip install rpy2==3.6.4; }

# 4. MOFA2 and its Python backend. R builds every package from source (no Hoffman2 R
# module has the Bioconductor stack), so compile in parallel on the job's CPUs; an
# interrupted install leaves 00LOCK-<pkg> folders that block reinstalling, so clear them.
NCPU="${SLURM_CPUS_PER_TASK:-2}"
export MAKEFLAGS="-j$NCPU"
mkdir -p "$R_LIBS_USER"
for lock in "$R_LIBS_USER"/00LOCK-*; do [ -d "$lock" ] && { echo "== removing stale R lock $lock"; rm -rf "$lock"; }; done
NCPU="$NCPU" Rscript -e '
lib <- Sys.getenv("R_LIBS_USER"); .libPaths(c(lib, .libPaths())); options(Ncpus = as.integer(Sys.getenv("NCPU")))
if (!requireNamespace("MOFA2", quietly = TRUE)) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager", repos = "https://cloud.r-project.org", lib = lib)
  BiocManager::install("MOFA2", ask = FALSE, update = FALSE, lib = lib)
}
# R/mofa_run.R attaches tidyverse; the R module has only part of it (ggplot2, dplyr, ...)
if (!requireNamespace("tidyverse", quietly = TRUE))
  install.packages("tidyverse", repos = "https://cloud.r-project.org", lib = lib)
for (p in c("MOFA2", "tidyverse")) if (!requireNamespace(p, quietly = TRUE)) stop("R package ", p, " failed to install")
cat("MOFA2", as.character(packageVersion("MOFA2")), "\n")'
echo "== building MOFA's Python backend with a tiny fit (first run only takes a few minutes)"
python - <<'EOF'
import os, sys, tempfile, numpy as np, torch
sys.path.insert(1, "python")
from controlled_scoring import fit_mofa
g = torch.Generator().manual_seed(0)
dp = {"L": 2, "train_X_l_list_clean": [torch.randn(60, 20, generator=g) for _ in range(2)],
      "test_X_l_list_clean": [torch.randn(10, 20, generator=g) for _ in range(2)],
      "train_y_clean": torch.randn(60, generator=g)}
out = os.path.join(tempfile.mkdtemp(), "mofa_test.npz")
fit_mofa(dp, out, k_fit=3)
print("MOFA ok:", np.load(out)["Z_train"].shape)
EOF

# 5. checks
python - <<'EOF'
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(1, "python")
import torch, pyro, sklearn, sksurv, rpy2, multiview_shared, run_methods, survival_metrics
print("imports ok: torch", torch.__version__, "pyro", pyro.__version__, "sksurv", sksurv.__version__)
EOF
if [ "${1:-}" = "--smoke" ]; then
  echo "== smoke run: two-factor study task 55 rep 1, survival study task 16 rep 1"
  python python/sim_run_controlled_study.py --task-id 55 --reps 1
  python python/sim_run_survival.py --task-id 16 --reps 1
fi
echo "== setup done. Next: set --account in cluster/submit_*.slurm, then h2 validate / h2 submit them."
