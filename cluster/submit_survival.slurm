#!/bin/bash
#SBATCH --job-name=survival_study
#SBATCH --account=pi_huazhou
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --array=1-336
#SBATCH --output=cluster/logs/survival.%A_%a.out
##SBATCH --mail-user=CHANGE_ME@mail
##SBATCH --mail-type=END,FAIL

# ---------------------------------------------------------------------------
# Hoffman2 Slurm array job: right-censored survival study, log-normal AFT
# (python/sim_run_survival.py; docs/viper_shared_extensions_design.md §4).
#
# Before the first submission: cluster/setup_hoffman2.sh (environment, MOFA2), and set
# --account above to your group's Slurm account (h2 whoami; pi_<PI's user name>).
# Submit from the repository root (logs go to cluster/logs/, relative to it):
#   h2 validate cluster/submit_survival.slurm
#   h2 submit   cluster/submit_survival.slurm
#
# One task per condition (python python/sim_run_survival.py --list): 24 base cells
# (n {100,200,500} x P {600,3000} x c {0.6,1,3,10}) x 14 scenario levels (S1 censoring
# 0/0.3/0.6/0.8; S2 administrative 0.3/0.6; S3 Weibull and log-logistic truths 0.3/0.6;
# S4 censoring depending on view 1 0.3/0.6; S5 two relevant factors 0.3/0.6) = 336
# tasks, 10 replicates each. Each task generates its data (R/generate_survival_data.R)
# and fits, per replicate: VIPER log-normal AFT (closed-form and local prediction), two
# Gaussian VIPER fits on log y (naive, events only), PCA / MOFA + AFT or Cox,
# elastic-net Cox. Output: sim/results/survival_study/scores/<cond>.csv (resumable by
# replicate).
#
# Time: ~4-5 min per replicate at P = 600 and ~10-15 min at P = 3000 (three VIPER fits),
# so a task takes ~1-2.5 h; 12 h is headroom.
# ---------------------------------------------------------------------------

echo "Task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID started on $(hostname -s) at $(date)"
source cluster/hoffman2_env.sh

python python/sim_run_survival.py --task-id "$SLURM_ARRAY_TASK_ID"

echo "Task $SLURM_ARRAY_TASK_ID ended at $(date)"
