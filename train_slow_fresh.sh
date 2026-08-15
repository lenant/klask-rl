#!/bin/bash
# Slowed board (0.4x), uniform ball and handle spawns. Winning reward from the
# earlier sweep: progress, no distance, no aim. Episodes at 750 steps (25s).
cd "$(dirname "$0")"
RUN=runs/slow_fresh
LOG=$RUN/train.log
mkdir -p $RUN
COMMON="--num-envs 12 --n-steps 1024 --batch-size 1024 --snapshot-freq 100000 \
  --max-steps 750 --reward-profile slow_v1 --vec-env subproc --device cpu \
  --policy-net-arch 256x256x256 --ent-coef 0.01 --output-dir $RUN"

echo "=== stage 1: goal_radius 0.15, 0 -> 6M ($(date -u))" | tee -a $LOG
uv run klask-train $COMMON --goal-radius 0.15 --total-steps 6000000 \
  --bc-samples 12000 --bc-epochs 8 --bc-batch-size 512 2>&1 | tee -a $LOG
echo "=== stage 2: goal_radius 0.11, 6M -> 12M ($(date -u))" | tee -a $LOG
uv run klask-train $COMMON --goal-radius 0.11 --total-steps 12000000 \
  --resume-from $RUN/latest/final_model.zip 2>&1 | tee -a $LOG
echo "=== stage 3: goal_radius 0.075 (real), 12M -> 24M ($(date -u))" | tee -a $LOG
uv run klask-train $COMMON --goal-radius 0.075 --total-steps 24000000 \
  --resume-from $RUN/latest/final_model.zip 2>&1 | tee -a $LOG
echo "=== done at $(date -u)" | tee -a $LOG
