#!/usr/bin/env bash
# Runs the training jobs listed in a queue file sequentially: one "<run-name> <train.py args...>" per line.
# Idempotent: finished runs (summary.json exists) are skipped, interrupted ones continue from last.pt.
# Meant to be started inside screen/tmux so that an SSH disconnect does not kill training:
#   screen -dmS ft_a bash models/yolo_finetune/queue.sh models/yolo_finetune/experiments/phase1_a.txt
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}
while read -r name args; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    dir=runs/yolo_finetune/$name
    [[ -f $dir/summary.json ]] && { echo "skip $name (done)"; continue; }
    resume=""
    [[ -f $dir/last.pt ]] && resume="--resume"
    mkdir -p "$dir"
    echo "=== $(date '+%F %T') start $name $args $resume" | tee -a "$dir/train.log"
    $PY -m models.yolo_finetune.train --run-name "$name" $args $resume 2>&1 | tee -a "$dir/train.log"
    echo "=== $(date '+%F %T') end $name (exit ${PIPESTATUS[0]})" | tee -a "$dir/train.log"
done < "$1"
