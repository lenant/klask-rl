# klask-rl

Magnet-free Klask-like aero hockey simulator for reinforcement learning.

The project exposes a Pymunk physics model, a PettingZoo parallel two-agent
environment, and a Gymnasium self-play wrapper that trains one shared
Stable-Baselines3 PPO policy from mirrored observations.

## Quick start

```bash
uv sync
uv run pytest
uv run klask-train --total-steps 1000 --num-envs 2 --n-steps 64 --batch-size 64
uv run klask-eval --model runs/klask/latest/final_model.zip --opponent random
uv run klask-watch --model runs/klask/latest/final_model.zip --self-play
uv run klask-benchmark --model runs/klask/latest/final_model.zip
```

The environment is intentionally closer to air hockey than full Klask v1:
there are two handles, one puck, walls, goals, and no magnets or biscuits.

## Current trained checkpoint

A local PPO checkpoint is written by training to `runs/klask/latest/final_model.zip`.
The `runs/` directory is ignored because checkpoints and TensorBoard logs are
generated artifacts.

The latest local run used:

```bash
uv run klask-train --total-steps 180000 --num-envs 8 --n-steps 256 \
  --batch-size 512 --snapshot-freq 30000 --max-steps 450 \
  --reward-profile possession --bc-samples 12000 --bc-epochs 8
```

Evaluation commands:

```bash
uv run klask-eval --model runs/klask/latest/final_model.zip --episodes 30 --opponent passive
uv run klask-eval --model runs/klask/latest/final_model.zip --episodes 30 --opponent random
uv run klask-eval --model runs/klask/latest/final_model.zip --episodes 30 --opponent heuristic
uv run klask-eval --model runs/klask/latest/final_model.zip --episodes 30 --self-play
```

`klask-watch` defaults to `--self-play`, so both handles are controlled by the
same trained mirrored policy. `klask-benchmark` runs passive, random, heuristic,
and self-play matchups and reports quality metrics: goal margin, goals per game,
draw rate, defense rate, contact activity, territory, and aggregate quality.

Training supports reward profiles with `--reward-profile balanced|aggressive|defensive|possession`.
The default profile is `possession` and uses behavior-cloning warm start samples
from an active striker expert before PPO self-play.

The best current checkpoint was selected from the possession reward sweep and is
documented in `reports/reward_experiments.md`.
