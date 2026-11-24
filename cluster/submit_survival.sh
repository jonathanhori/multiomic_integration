#!/bin/bash
# SUPERSEDED (2026-11-24): Grid Engine version for the legacy cluster. Hoffman2's
# production cluster runs Slurm: use cluster/submit_survival.slurm.
#$ -cwd
#$ -o cluster/logs/survival.$JOB_ID.$TASK_ID.log
#$ -j y
#$ -l h_rt=12:00:00
#$ -l h_data=4G
#$ -pe shared 2
#$ -t 1-336:1

# ---------------------------------------------------------------------------
# Hoffman2 SGE array job: right-censored survival study, log-normal AFT
# (python/sim_run_survival.py; docs/viper_shared_extensions_design.md §4).
#
# One task per condition (python python/sim_run_survival.py --list): 24 base cells
# (n {100,200,500} x P {600,3000} x c {0.6,1,3,10}) x 14 scenario levels
# (S1 censoring 0/0.3/0.6/0.8; S2 administrative 0.3/0.6; S3 Weibull and log-logistic
# truths 0.3/0.6; S4 censoring depending on view 1 0.3/0.6; S5 two relevant factors
# 0.3/0.6) = 336 tasks, 10 replicates each. Each task generates its data
# (R/generate_survival_data.R; skips existing files) and fits, per replicate: VIPER
# log-normal AFT (closed-form and local prediction), two Gaussian VIPER fits on log y
# (naive, events only), PCA / MOFA + AFT or Cox, elastic-net Cox. Output:
# sim/results/survival_study/scores/<cond>.csv (resumable by replicate).
#
# Time: ~4-5 min per replicate at P = 600 and ~10-15 min at P = 3000 locally (three
# VIPER fits), so a task takes ~1-2.5 h; 12 h is headroom.
#
# Submit from the repo root:   qsub cluster/submit_survival.sh
# ---------------------------------------------------------------------------

echo "=== Task $SGE_TASK_ID starting on $(hostname) at $(date) ==="
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
source /u/local/apps/miniconda3/latest/etc/profile.d/conda.sh
conda activate multiomic
module load R/4.4.0

python python/sim_run_survival.py --task-id "$SGE_TASK_ID"

echo "=== Task $SGE_TASK_ID finished at $(date) ==="
