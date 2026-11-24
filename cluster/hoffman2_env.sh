# Job environment for the Hoffman2 Slurm cluster, sourced by the job scripts
# (cluster/submit_*.slurm) and by cluster/setup_hoffman2.sh. Edit the three values once.
#
#   MINIFORGE_MODULE  the miniforge version the conda environment was created with
#                     (module spider miniforge; e.g. miniforge/26.5.3-0)
#   R_MODULE          an R module with MOFA2 installed in your R library
#                     (h2 modules R; e.g. R/4.4.0; "R" loads the default version)
#   CONDA_ENV         the conda environment from cluster/environment_hoffman2.yml

MINIFORGE_MODULE=miniforge/26.5.3-0
R_MODULE=R
CONDA_ENV=multiomic

module purge                       # every Slurm job starts with intel/2026.2.0 loaded; start clean
module load "$MINIFORGE_MODULE"
module load "$R_MODULE"
conda activate "$CONDA_ENV"        # activated last so the environment's python comes first on PATH
export PYTHONNOUSERSITE=1          # ignore packages from pip install --user
export R_HOME="$(R RHOME)"         # for rpy2 (run_methods.load_rds)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
