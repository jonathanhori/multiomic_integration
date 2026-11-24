#!/bin/bash
#$ -cwd
#$ -o cluster/logs/mofa_shared.$JOB_ID.$TASK_ID.log
#$ -j y
#$ -l h_rt=24:00:00
#$ -l h_data=16G
#$ -t 1-32:1

# ---------------------------------------------------------------------------
# Hoffman2 SGE array job — MOFA2 simulation sweep (R)
#
# Grid: 4n × 4p × 1snr_x × 1snr_y × 1sparsity × 2k_delta = 32 conditions
# Each task runs all 10 reps for one grid element sequentially.
#
# Resource rationale:
#   Memory: R uses float64 so data footprint is ~2× Python's. Peak ~3GB for
#           n=5000, p=5000, L=3 views; 16G gives ample headroom.
#   Cores:  MOFA2 runs full-batch EM via a Python backend (basilisk). R itself
#           is single-threaded, so 1 slot is appropriate.
#   Time:   MOFA full-batch EM over n=5000, p=5000 is faster per iteration than
#           SVI but still potentially 1–3h per rep; 10 reps = 10–30h worst case.
#           24h covers the typical range with a reasonable safety margin.
#
# Submit from the repo root:
#   qsub cluster/submit_mofa_shared.sh
#
# Monitor:
#   qstat -u $USER
# ---------------------------------------------------------------------------

echo "=== Task $SGE_TASK_ID starting on $(hostname) at $(date) ==="

# --- Load R module -----------------------------------------------------------
. /u/local/Modules/default/init/modules.sh
module load R/4.4.0

# --- Run simulation ----------------------------------------------------------
Rscript R/mofa_run.R "$SGE_TASK_ID"

echo "=== Task $SGE_TASK_ID finished at $(date) ==="
