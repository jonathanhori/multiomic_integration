# Job environment for the Hoffman2 Slurm cluster, sourced by the job scripts
# (cluster/submit_*.slurm) and by cluster/setup_hoffman2.sh. Edit the values once.
#
#   MINIFORGE_MODULE  the miniforge version the conda environment was created with
#                     (module spider miniforge; e.g. miniforge/26.5.3-0)
#   R_MODULE          an R module with MOFA2 installed in R_LIBS_USER
#                     (h2 modules R; e.g. R/4.4.0; "R" loads the default version)
#   ENV_ROOT          where the software lives: the conda environment, the R library,
#                     and the download / build caches
#   CONDA_ENV         the conda environment from cluster/environment_hoffman2.yml (a path)
#
# Everything is in $SCRATCH, not $HOME: $HOME is 60 GB and was too slow on 2026-11-24 to
# install the environment (a 3 h setup job timed out in pip). $SCRATCH removes files older
# than 14 days, so after a long pause rerun cluster/setup_hoffman2.sh: it rebuilds whatever
# is missing or broken.

MINIFORGE_MODULE=miniforge/26.5.3-0
R_MODULE=R
ENV_ROOT=$SCRATCH/software
CONDA_ENV=$ENV_ROOT/conda_envs/multiomic

export CONDA_PKGS_DIRS="$ENV_ROOT/conda_pkgs"        # conda's package cache (default ~/.conda/pkgs)
export PIP_CACHE_DIR="$ENV_ROOT/cache/pip"           # default ~/.cache/pip
export BASILISK_EXTERNAL_DIR="$ENV_ROOT/basilisk"    # MOFA2's Python backend (default ~/.cache/R/basilisk)

module purge                       # every Slurm job starts with intel/2026.2.0 loaded; start clean
module load "$MINIFORGE_MODULE"
module load "$R_MODULE"
export R_LIBS_USER="$ENV_ROOT/R/library/$(R --version | head -1 | awk '{print $3}')"   # after module load R
conda activate "$CONDA_ENV"        # activated last so the environment's python comes first on PATH
export PYTHONNOUSERSITE=1          # ignore packages from pip install --user
export R_HOME="$(R RHOME)"         # for rpy2 (run_methods.load_rds)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
