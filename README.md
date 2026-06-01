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
uv run klask-watch --model runs/klask/latest/final_model.zip --opponent heuristic
```

The environment is intentionally closer to air hockey than full Klask v1:
there are two handles, one puck, walls, goals, and no magnets or biscuits.

## Current trained checkpoint

A local PPO checkpoint is written by training to `runs/klask/latest/final_model.zip`.
The `runs/` directory is ignored because checkpoints and TensorBoard logs are
generated artifacts.

The latest local run used:

```bash
uv run klask-train --total-steps 80000 --num-envs 8 --n-steps 256 \
  --batch-size 512 --snapshot-freq 20000 --max-steps 450
```

Evaluation commands:

```bash
uv run klask-eval --model runs/klask/latest/final_model.zip --episodes 30 --opponent passive
uv run klask-eval --model runs/klask/latest/final_model.zip --episodes 30 --opponent random
uv run klask-eval --model runs/klask/latest/final_model.zip --episodes 30 --opponent heuristic
```
