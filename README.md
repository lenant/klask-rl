# Klask RL

Klask RL is a small reinforcement-learning playground for a Klask-inspired table game.
It is currently closer to fast air hockey than full physical Klask: two handles move on
their own halves of the board, push a puck toward the opposite goal, and must avoid three
magnetic biscuits that can attach to a handle.

The repository contains:

- a Pymunk physics simulator with pygame rendering;
- PettingZoo and Gymnasium environments;
- Stable-Baselines3 PPO self-play training;
- benchmark scripts for comparing checkpoints;
- a simple model-vs-model and human-vs-model viewer.

## Demo

Five self-play episodes recorded from the current PPO leader:

![Klask RL self-play demo](docs/assets/klask-self-play-5-episodes.gif)

[Open the MP4 version](docs/assets/klask-self-play-5-episodes.mp4)

The red and blue circles are the handles, the white circle is the puck, and the small gray
circles are the magnets. The side panels show the current step reward, episode total, and
every reward component for each side.

## Setup

Use Python 3.11 or newer. The project is managed with `uv`.

```bash
uv sync
uv run pytest
```

`ffmpeg` is only required if you want to record new videos.

## Watch A Model

Trained checkpoints and logs are generated artifacts and are ignored by git. The current
local leader is expected at `runs/klask/latest/simple_leader.zip` after copying or training
a model.

Model vs model:

```bash
uv run klask-watch --model runs/klask/latest/simple_leader.zip --self-play --reward-profile simple
```

Human vs model:

```bash
uv run klask-play --model runs/klask/latest/simple_leader.zip --human-side left --reward-profile simple
```

Controls in human mode:

- `W`, `A`, `S`, `D` move the human handle.
- `Space` pauses and resumes the episode.
- `Escape` or closing the window exits.

## Train

For a quick smoke run:

```bash
uv run klask-train --total-steps 10000 --num-envs 2 --n-steps 128 \
  --batch-size 128 --snapshot-freq 5000 --reward-profile simple
```

The current stronger setup uses a larger PPO network, behavior-cloning warm start, parallel
self-play environments, simple reward, and automatic snapshots:

```bash
uv run klask-train --total-steps 5000000 --num-envs 16 --n-steps 1024 \
  --batch-size 1024 --snapshot-freq 100000 --max-steps 450 \
  --reward-profile simple --bc-samples 12000 --bc-epochs 8 \
  --bc-batch-size 512 --vec-env subproc --device cuda \
  --policy-net-arch 256x256x256 \
  --output-dir runs/remote_simple_ppo_256x3_5m
```

Use `--device cpu` or `--device auto` on machines without CUDA.

Training writes checkpoints under:

- `runs/.../latest/final_model.zip`
- `runs/.../latest/snapshots/policy_<step>.zip`
- `runs/.../tensorboard/`

## Evaluate

Evaluate against one opponent:

```bash
uv run klask-eval --model runs/klask/latest/simple_leader.zip \
  --opponent heuristic --episodes 30 --reward-profile simple
```

Run the built-in benchmark suite:

```bash
uv run klask-benchmark --model runs/klask/latest/simple_leader.zip \
  --episodes 20 --reward-profile simple
```

`klask-benchmark` tests passive, random, heuristic, and self-play matchups. The aggregate
quality score combines attack, defense, contact activity, territory, and decisiveness.

During long training runs, benchmark snapshots automatically and keep only new leaders:

```bash
PYTHONPATH=src uv run python scripts/benchmark_snapshots.py \
  --snapshot-dir runs/remote_simple_ppo_256x3_5m/latest/snapshots \
  --csv runs/remote_simple_ppo_256x3_5m/benchmark_results.csv \
  --leader-dir runs/remote_simple_ppo_256x3_5m/leaders \
  --leader-copy runs/klask/latest/simple_leader.zip \
  --episodes 20 --reward-profile simple
```

The best 5M-step run so far used `--policy-net-arch 256x256x256` with the simple reward
profile. Its final leader reached aggregate quality `5.393`, with `33` goals for and `1`
goal against across the four 20-episode benchmark matchups.

## Reward Profile

The current recommended training profile is `simple`. Pass it explicitly to training,
because the training CLI still defaults to the older `possession` profile.

`simple` gives:

- `+12 / -12` terminal reward for scoring or conceding;
- `-2.0` once when a new magnet attaches to the handle;
- up to `-0.05` per step while an unattached magnet is pulling toward the handle, scaled by
  closeness;
- `-0.01` per step while the puck is on the player's own side;
- `0` reward when the puck is on the other side, unless one of the rules above applies.

Older reward profiles (`balanced`, `aggressive`, `defensive`, and `possession`) remain in
the code for experiments.

## Model Inputs

Both sides use the same policy. Observations are mirrored so the learning side always sees
itself as playing left to right.

The observation vector has 33 floats:

- own handle position and velocity;
- opponent handle position and velocity;
- puck position, velocity, and relative vector from the own handle;
- score difference and time remaining;
- for each of the three magnets: position, velocity, and attachment state;
- own and opponent magnet attachment counts.

The action is a 2D continuous vector in `[-1, 1]` that controls handle movement.

The current leader uses Stable-Baselines3 PPO with an `MlpPolicy` and hidden layers
`256x256x256` for the policy and value networks.

## Record A Demo

Record five model-vs-model episodes to an MP4:

```bash
PYTHONPATH=src uv run python scripts/record_demo.py \
  --model runs/klask/latest/simple_leader.zip \
  --output docs/assets/klask-self-play-5-episodes.mp4 \
  --episodes 5 --reward-profile simple
```

## Project Layout

- `src/klask_rl/physics.py` contains the board physics and renderer.
- `src/klask_rl/envs.py` contains the PettingZoo and Gymnasium environments.
- `src/klask_rl/cli.py` contains training, evaluation, watch, benchmark, and play commands.
- `src/klask_rl/opponents.py` contains passive, random, heuristic, striker, and checkpoint opponents.
- `src/klask_rl/training.py` contains the behavior-cloning warm start.
- `scripts/benchmark_snapshots.py` evaluates snapshots and copies new leaders.
- `scripts/record_demo.py` records self-play videos.
