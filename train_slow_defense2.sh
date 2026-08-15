#!/bin/bash
# Episode cap 1500 rather than 750: at 750 only 44% of episodes resolve before
# truncation, so most never deliver the terminal reward at all.
# What actually works: seed from the model that already knows the game, and use
# the spawn-distance curriculum to re-teach contact under the new timing before
# widening to the full board. From-scratch stalls here; this does not.
cd "$(dirname "$0")"
RUN=runs/slow_defense2
LOG=$RUN/train.log
mkdir -p $RUN
cp runs/klask/latest/exp3_24M_baseline.zip $RUN/seed.zip
# Carry the seed's self-play league across. Resuming into a fresh output dir
# restores zero checkpoint opponents, so the agent practises only against the
# scripted baselines -- a model forged against 480 opponents then trains
# against passive and random, and gets worse within 100k steps.
# Resuming into a fresh output dir restores zero checkpoint opponents, so a
# model forged against hundreds of snapshots would practise only against the
# scripted baselines and degrade within 100k steps.
mkdir -p $RUN/latest/snapshots
PYTHONPATH=src uv run python scripts/seed_league.py \
  runs/exp3_progress_only/latest/snapshots $RUN/latest/snapshots 24010752 --keep 64 | tee -a $LOG
echo "seeded league: $(ls $RUN/latest/snapshots/*.zip 2>/dev/null | wc -l) opponents" | tee -a $LOG
COMMON="--num-envs 12 --n-steps 1024 --batch-size 1024 --snapshot-freq 100000 \
  --max-steps 1500 --reward-profile slow_v3 --vec-env subproc --device cpu \
  --policy-net-arch 256x256x256 --ent-coef 0.002 --output-dir $RUN"
# The seed carries 24M timesteps, and --resume-from treats --total-steps as an
# absolute target, so every stage budget is offset past it.
echo "=== stage 1: reach 0.3, goal 0.15, +5M ($(date -u))" | tee -a $LOG
uv run klask-train $COMMON --spawn-distance 0.3 --goal-radius 0.15 --total-steps 29000000 \
  --resume-from $RUN/seed.zip 2>&1 | tee -a $LOG
echo "=== stage 2: reach 0.6, goal 0.11, +5M ($(date -u))" | tee -a $LOG
uv run klask-train $COMMON --spawn-distance 0.6 --goal-radius 0.11 --total-steps 34000000 \
  --resume-from $RUN/latest/final_model.zip 2>&1 | tee -a $LOG
echo "=== stage 3: no cap, goal 0.075, +8M ($(date -u))" | tee -a $LOG
uv run klask-train $COMMON --goal-radius 0.075 --total-steps 42000000 \
  --resume-from $RUN/latest/final_model.zip 2>&1 | tee -a $LOG
echo "=== done at $(date -u)" | tee -a $LOG
