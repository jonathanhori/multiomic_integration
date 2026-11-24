#!/bin/bash
# One-time setup of this repository on the Hoffman2 Slurm cluster.
#
# Run from the repository root, inside an interactive session (not on the login node):
#   srun --time=02:00:00 --cpus-per-task=2 --mem=8G --account=pi_huazhou --pty bash -l
#   cd $SCRATCH/multiomic_integration && bash cluster/setup_hoffman2.sh [--smoke]
#
# What it does (each step is skipped if already done):
#   1. ~/multiomic_integration -> this directory (symlink). The code and the R scripts
#      locate the repository at ~/multiomic_integration; keeping the repository itself in
#      $SCRATCH (or your group's project directory) keeps the data and results out of the
#      small $HOME quota.
#   2. cluster/logs/ for the job logs (Slurm does not create directories).
#   3. the conda environment from cluster/environment_hoffman2.yml, then rpy2 linked to
#      the R module.
#   4. MOFA2 (Bioconductor) in your R library, and its Python backend (basilisk builds it
#      on first use: done here once, so the array tasks do not all build it at once).
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

# 3. conda environment
eval "$(grep -E '^(MINIFORGE_MODULE|R_MODULE|CONDA_ENV)=' cluster/hoffman2_env.sh)"   # the three settings
module purge
module load "$MINIFORGE_MODULE"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "== creating conda environment $CONDA_ENV"
  conda env create -f cluster/environment_hoffman2.yml
fi
source cluster/hoffman2_env.sh
python -c "import rpy2" 2>/dev/null || { echo "== installing rpy2 against $(R --version | head -1)"; pip install rpy2==3.6.4; }

# 4. MOFA2 and its Python backend
Rscript -e '
lib <- Sys.getenv("R_LIBS_USER"); dir.create(lib, recursive = TRUE, showWarnings = FALSE); .libPaths(c(lib, .libPaths()))
if (!requireNamespace("MOFA2", quietly = TRUE)) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager", repos = "https://cloud.r-project.org", lib = lib)
  BiocManager::install("MOFA2", ask = FALSE, update = FALSE, lib = lib)
}
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
