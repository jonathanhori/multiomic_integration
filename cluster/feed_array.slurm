#!/bin/bash
#SBATCH --job-name=feed_array
#SBATCH --account=pi_huazhou
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=cluster/logs/feed_array.%j.out

# ---------------------------------------------------------------------------
# Submits a large array in chunks that fit under the per-user limit on submitted jobs
# (campus24 QOS: 500 submitted, each array task counts; 100 running). Each run submits
# as many of the remaining task ids as fit, then resubmits itself to run again in
# FEED_INTERVAL minutes; it stops when every id has been submitted. Runs on the cluster,
# so nothing depends on a laptop staying awake.
#
#   sbatch cluster/feed_array.slurm <job script> <first id> <last id>
#   e.g. sbatch cluster/feed_array.slurm cluster/submit_survival.slurm 1 336
#
# Progress: cluster/logs/feed_array.<script name>.next (next id to submit) and the
# feed_array.*.out logs. Stop it: scancel --name=feed_array.
# Submit from the repository root.
# ---------------------------------------------------------------------------
set -eo pipefail
SCRIPT="$1"; FIRST="$2"; LAST="$3"
LIMIT="${FEED_LIMIT:-500}"        # per-user submitted-job limit (sacctmgr show qos campus24)
MARGIN="${FEED_MARGIN:-10}"       # slots left free (this job, its successor, other work)
INTERVAL="${FEED_INTERVAL:-30}"   # minutes between runs
STATE="cluster/logs/feed_array.$(basename "$SCRIPT" .slurm).next"
[ -f "$STATE" ] || echo "$FIRST" > "$STATE"
NEXT=$(cat "$STATE")

if [ "$NEXT" -le "$LAST" ]; then
  QUEUED=$(squeue -u "$USER" -h -r | wc -l)               # -r: one line per array task
  ROOM=$(( LIMIT - QUEUED - MARGIN ))
  if [ "$ROOM" -gt 0 ]; then
    END=$(( NEXT + ROOM - 1 )); [ "$END" -gt "$LAST" ] && END="$LAST"
    sbatch --array="${NEXT}-${END}" "$SCRIPT"
    echo "$(date): submitted ${SCRIPT} tasks ${NEXT}-${END} (${QUEUED} jobs were queued)"
    echo $(( END + 1 )) > "$STATE"; NEXT=$(( END + 1 ))
  else
    echo "$(date): no room (${QUEUED} jobs queued); next id ${NEXT}"
  fi
fi

if [ "$NEXT" -le "$LAST" ]; then
  # by path, not $0 (inside a job $0 is Slurm's spooled copy)
  sbatch --begin="now+${INTERVAL}minutes" cluster/feed_array.slurm "$SCRIPT" "$FIRST" "$LAST" >/dev/null
  echo "next run in ${INTERVAL} min"
else
  echo "$(date): all of ${SCRIPT} ${FIRST}-${LAST} submitted; feeder done"
fi
