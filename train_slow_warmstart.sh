#!/bin/bash
# Same as train_slow_fresh, but seeded from the 24M model trained on the fast
# board. It already knows to chase and shoot; the open question is whether that
# transfers or fights the new timing and spawn distribution.
cd "$(dirname "$0")"
RUN=runs/slow_warmstart
LOG=$RUN/train.log
mkdir -p $RUN
cp runs/klask/latest/exp3_24M_baseline.zip $RUN/seed.zip
COMMON="--num-envs 12 --n-steps 1024 --batch-size 1024 --snapshot-freq 100000 \
  --max-steps 750 --reward-profile slow_v1 --vec-env subproc --device cpu \
  --policy-net-arch 256x256x256 --ent-coef 0.01 --output-dir $RUN"

echo "=== stage 1: goal_radius 0.15, seeded, -> 6M ($(date -u))" | tee -a $LOG
uv run klask-train $COMMON --goal-radius 0.15 --total-steps 6000000 \
  --resume-from $RUN/seed.zip 2>&1 | tee -a $LOG
echo "=== stage 2: goal_radius 0.11, 6M -> 12M ($(date -u))" | tee -a $LOG
uv run klask-train $COMMON --goal-radius 0.11 --total-steps 12000000 \
  --resume-from $RUN/latest/final_model.zip 2>&1 | tee -a $LOG
echo "=== stage 3: goal_radius 0.075 (real), 12M -> 24M ($(date -u))" | tee -a $LOG
uv run klask-train $COMMON --goal-radius 0.075 --total-steps 24000000 \
  --resume-from $RUN/latest/final_model.zip 2>&1 | tee -a $LOG
echo "=== done at $(date -u)" | tee -a $LOG
