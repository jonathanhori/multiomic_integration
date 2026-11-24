#!/bin/bash
#$ -cwd
#$ -o cluster/logs/python_shared.$JOB_ID.$TASK_ID.log
#$ -j y
#$ -l h_rt=48:00:00
#$ -l h_data=4G
#$ -pe shared 4
#$ -t 1-32:1

# ---------------------------------------------------------------------------
# Hoffman2 SGE array job — SupMultiviewShared simulation sweep
#
# Grid: 4n × 4p × 1snr_x × 1snr_y × 1sparsity × 2k_delta = 32 conditions
# Each task runs all 10 reps for one grid element sequentially.
#
# Resource rationale:
#   Memory: peak ~1.5GB even for n=5000, p=5000, L=3 views. 4G/slot × 4 slots
#           = 16G total gives ample headroom.
#   Cores:  4 slots let PyTorch parallelize matrix ops (Z_batch @ Lambda_l^T
#           across 3 views). Set OMP/MKL threads to match.
#   Time:   worst case ~11h on 1 core (n=5000, p=5000, 10 reps × 1000 epochs).
#           4 cores cut this to ~4–5h; 48h gives a comfortable safety margin.
#
# Submit from the repo root:
#   qsub cluster/submit_python_shared.sh
#
# Monitor:
#   qstat -u $USER
# ---------------------------------------------------------------------------

echo "=== Task $SGE_TASK_ID starting on $(hostname) at $(date) ==="

# --- Thread control (match requested slots) ----------------------------------
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# --- Conda environment -------------------------------------------------------
# Adjust the path below if your miniconda/conda is installed elsewhere.
source /u/local/apps/miniconda3/latest/etc/profile.d/conda.sh
conda activate multiomic

# --- Run simulation ----------------------------------------------------------
python python/sim_run_shared.py --task-id "$SGE_TASK_ID"

echo "=== Task $SGE_TASK_ID finished at $(date) ==="
