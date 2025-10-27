#!/usr/bin/env bash

logfile="missing_check.log"
echo "=== Missing files/directories check started at $(date) ===" > "$logfile"

missing=0
for N in 50 100 500 1000 5000; do
  for P in 50 100 1000 10000; do
    for X in 2 1 0.5; do
      for Y in 2 1 0.5; do
        dir="n${N}p${P}_snr${X}.${Y}"
        if [[ ! -d $dir ]]; then
          echo "❌ Missing directory: $dir" | tee -a "$logfile"
          missing=1
          continue
        fi
        for rep in {1..10}; do
          file="${dir}/sim_data_ywithview_rep${rep}.rds"
          if [[ ! -f $file ]]; then
            echo "❌ Missing file: $file" | tee -a "$logfile"
            missing=1
          fi
        done
      done
    done
  done
done

if [[ $missing -eq 0 ]]; then
  echo "✅ All expected directories and files are present!" | tee -a "$logfile"
else
  echo "⚠️  Check complete — missing items logged in $logfile" | tee -a "$logfile"
fi

echo "=== Check finished at $(date) ===" >> "$logfile"
