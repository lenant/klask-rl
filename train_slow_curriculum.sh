#!/bin/bash
# Distance curriculum: the ball starts within reach and the cap widens, so the
# policy learns to strike before it has to learn to find. Goal radius shrinks
# on the same schedule.
cd "$(dirname "$0")"
RUN=runs/slow_curriculum
LOG=$RUN/train.log
mkdir -p $RUN
COMMON="--num-envs 12 --n-steps 1024 --batch-size 1024 --snapshot-freq 100000 \
  --max-steps 750 --reward-profile slow_v1 --vec-env subproc --device cpu \
  --policy-net-arch 256x256x256 --ent-coef 0.002 --output-dir $RUN"

echo "=== stage 1: reach 0.25, goal 0.15, -> 5M ($(date -u))" | tee -a $LOG
uv run klask-train $COMMON --spawn-distance 0.25 --goal-radius 0.15 --total-steps 5000000 \
  --bc-samples 12000 --bc-epochs 8 --bc-batch-size 512 2>&1 | tee -a $LOG
echo "=== stage 2: reach 0.5, goal 0.15, -> 10M ($(date -u))" | tee -a $LOG
uv run klask-train $COMMON --spawn-distance 0.5 --goal-radius 0.15 --total-steps 10000000 \
  --resume-from $RUN/latest/final_model.zip 2>&1 | tee -a $LOG
echo "=== stage 3: reach 1.0, goal 0.11, -> 16M ($(date -u))" | tee -a $LOG
uv run klask-train $COMMON --spawn-distance 1.0 --goal-radius 0.11 --total-steps 16000000 \
  --resume-from $RUN/latest/final_model.zip 2>&1 | tee -a $LOG
echo "=== stage 4: no cap, goal 0.075, -> 26M ($(date -u))" | tee -a $LOG
uv run klask-train $COMMON --goal-radius 0.075 --total-steps 26000000 \
  --resume-from $RUN/latest/final_model.zip 2>&1 | tee -a $LOG
echo "=== done at $(date -u)" | tee -a $LOG
