#!/bin/bash
#$ -cwd
#$ -o cluster/logs/multifactor_B.$JOB_ID.$TASK_ID.log
#$ -j y
#$ -l h_rt=12:00:00
#$ -l h_data=4G
#$ -pe shared 2
#$ -t 1-90:1

# ---------------------------------------------------------------------------
# Hoffman2 SGE array job: multi-factor controlled study, scenario B
# (docs/viper_shared_extensions_design.md §1; python/sim_run_shared_multifactor.py).
#
# One task per cell: n {100, 200, 500} x P {600, 3000} x omega_2 {0.25, 0.5, 0.75}
# x c {0.6, 1, 2, 3, 10} = 90 cells, 10 replicates each. Each task first generates its
# cell's data (R/generate_multifactor_data.R, skips existing files), then fits VIPER
# (package defaults via run_methods.with_defaults), MOFA2 and the baselines, and writes
# sim/results/integration_shared_multifactor/B/scores/<cell>.csv (resumable by rep:
# resubmitting a task skips finished replicates).
#
# Resources: 2 threads per fit as in the local grids. Locally a replicate took
# 1-2 min at P = 600 and up to ~5 min at n = 500, P = 3000, so a task needs
# ~15 min to ~1 h; 12 h leaves room for slow nodes.
#
# Needs on the cluster: the repo (python/, R/, analysis/tcga_coadread/fit_mofa.R),
# conda env "multiomic", and R with MOFA2 (as cluster/submit_mofa_shared.sh).
# Submit from the repo root:   qsub cluster/submit_multifactor_B.sh
# Task -> cell map:            python python/sim_run_shared_multifactor.py --list
# ---------------------------------------------------------------------------

echo "=== Task $SGE_TASK_ID starting on $(hostname) at $(date) ==="
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
source /u/local/apps/miniconda3/latest/etc/profile.d/conda.sh
conda activate multiomic
module load R/4.4.0

python python/sim_run_shared_multifactor.py --task-id "$SGE_TASK_ID" --generate

echo "=== Task $SGE_TASK_ID finished at $(date) ==="
