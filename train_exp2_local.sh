#!/bin/bash
# Experiment 2: simple_v5 = simple_v4 plus explicit shot shaping.
#   progress -- pay for driving the puck downfield (telescopes, cannot be farmed)
#   aim      -- pay for a shot whose path reaches the hole, bank shots included
# Both speed-scaled, so a decisive strike beats a dribble.
#
# Runs on the local 18-core machine, so it uses fewer envs than the remote
# experiment and leaves headroom for interactive work.
cd "$(dirname "$0")"
RUN=runs/exp2_v5_aim
LOG=$RUN/train.log
mkdir -p $RUN
COMMON="--num-envs 12 --n-steps 1024 --batch-size 1024 --snapshot-freq 100000 \
  --max-steps 450 --reward-profile simple_v5 --vec-env subproc --device cpu \
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
