#!/bin/bash
# Extend the winning progress-only run past 24M, in place so the self-play
# league keeps its accumulated snapshot opponents.
cd "$(dirname "$0")"
RUN=runs/exp3_progress_only
LOG=$RUN/train.log
echo "=== extension: goal_radius 0.075, 24M -> 48M ($(date -u))" | tee -a $LOG
uv run klask-train --num-envs 12 --n-steps 1024 --batch-size 1024 --snapshot-freq 100000 \
  --max-steps 450 --reward-profile simple_v5_noaim --vec-env subproc --device cpu \
  --policy-net-arch 256x256x256 --ent-coef 0.01 --output-dir $RUN \
  --goal-radius 0.075 --total-steps 48000000 \
  --resume-from $RUN/latest/final_model.zip 2>&1 | tee -a $LOG
echo "=== extension done at $(date -u)" | tee -a $LOG
