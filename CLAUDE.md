# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Klask RL: a reinforcement-learning playground for the table game Klask (40 × 30 cm board at 1 unit = 20 cm, circular goal holes sunk into each half, magnetic "biscuits" that attach to handles). Pymunk physics, PettingZoo/Gymnasium envs, Stable-Baselines3 PPO self-play training. Python ≥3.11, managed with `uv`.

## Commands

```bash
uv sync                                  # install deps
uv run pytest                            # full test suite (~80s; includes PPO/BC smoke tests)
uv run pytest tests/test_envs.py         # one file
uv run pytest tests/test_envs.py::test_parallel_env_api   # one test
uv run ruff check                        # lint (line-length 100, target py311)
```

CLI entry points (all defined in `src/klask_rl/cli.py`): `klask-train`, `klask-eval`, `klask-watch`, `klask-benchmark`, `klask-play`, run via `uv run klask-train ...`. `klask-watch`/`klask-play` open a pygame window. Standalone scripts need the src path: `PYTHONPATH=src uv run python scripts/benchmark_snapshots.py ...`.

Quick training smoke run:

```bash
uv run klask-train --total-steps 10000 --num-envs 2 --n-steps 128 \
  --batch-size 128 --snapshot-freq 5000 --reward-profile simple
```

Trained models/logs live under `runs/` (gitignored). Commands in the README expect a model at `runs/klask/latest/simple_leader.zip` (downloadable from the GitHub release `v0.1-demo`).

## Architecture

Dependency order: `config.py` → `physics.py` → `opponents.py` → `envs.py` → `training.py` → `cli.py`.

- `config.py` — frozen dataclasses `ArenaConfig` (physics/board constants) and `RewardConfig`, plus `REWARD_PROFILES` (`balanced`, `aggressive`, `defensive`, `possession`, `simple`). `OBSERVATION_SIZE` (33) is computed here from the magnet count; changing observation layout means updating this constant and `_make_observation` in `envs.py` together.
- `physics.py` — `KlaskPhysics`: Pymunk simulation (solid walls on all four sides, handles clamped to their own halves, puck, 3 magnets with attraction/attachment logic, a circular goal hole in each half) plus the pygame renderer with a reward-component overlay. Energy loss is explicit, not left to `space.damping` (which is per *second* in pymunk, so the old 0.995 was ~0.04% per substep — effectively frictionless): `_decelerate` applies viscous drag plus a constant deceleration per body each substep, and the constant term is what brings a slow puck to a full stop so the policy sees resting balls. Shape elasticities multiply in pymunk, so the puck is kept perfectly elastic and each partner carries the whole pair restitution (`wall_elasticity`, `handle_elasticity`); manual containment reflections use the `puck_wall_elasticity`/`magnet_wall_elasticity` properties to stay consistent with what pymunk resolves. Actions are handle *target* velocities, slewed toward by `_steer_handles` under `max_handle_acceleration` each substep rather than applied instantly — handles are kinematic, so an unbounded action would let them reverse direction within one 33 ms control step and make every contact a full-power shot. Because the handle now carries momentum, its velocity in the observation is real state rather than a restatement of the last action, and `_clamp_handles` cancels only the axis that hit its bound — zeroing the whole vector pins an accelerating handle against a wall it is held into. Handles cannot cross the halfway line, so a puck that comes to rest in the other half is unreachable by everyone and the rest of the episode is dead time the learner cannot affect — measured at 41-88% of steps before this was handled. `_maybe_serve_stuck_puck` re-serves the puck after `dead_ball_steps` motionless control steps. Against a real opponent it fires ~0.1x per episode, so it is a backstop rather than a game mechanic, and the threshold stays generous on purpose so agents still have to learn to fetch a resting puck. An episode ends by `score_reason` of `"goal"` (ball captured by either hole — the hole's owner concedes), `"klask"` (handle falls into its own hole), or `"magnets"` (a handle collecting `magnet_score_threshold` = 2 magnets scores for the opponent). A free biscuit that slides into a hole is removed from the space and sits inert at the hole center until reset (`magnet_in_hole`).
- `envs.py` — two envs:
  - `KlaskParallelEnv` (PettingZoo `ParallelEnv`, agents `"left"`/`"right"`): steps both agents, computes rewards.
  - `SelfPlayKlaskEnv` (Gymnasium, single-agent): wraps the parallel env, drives the other side with an `OpponentPolicy`, and optionally randomizes which side the learner plays each episode.
- `opponents.py` — `OpponentPolicy` protocol and implementations: `Passive`, `Random`, `Heuristic`, `Striker` (the BC expert), `SB3CheckpointOpponent` (lazy-loads a PPO zip), and `OpponentPool` (random sampling over baselines + frozen checkpoints for league-style self-play). The scripted experts share `_attack_target`: they wind up at a standoff behind the puck, then must *commit* through it — a standoff alone is only a shot if something else closes the gap, which stopped being true once the puck could come to rest. A puck resting on the agent's own back wall has no room behind it, so there they hug the wall and sweep across its line instead of pushing it into the boards. They also skirt their own hole when chasing deep, since the chase now reaches far enough to klask.
- `training.py` — behavior-cloning warm start: collects `StrikerOpponent` expert transitions and does MSE regression on the tanh of the policy mean before PPO starts.
- `cli.py` — Typer commands. Training builds a vec env of `SelfPlayKlaskEnv`s each holding an `OpponentPool`; `SelfPlaySnapshotCallback` periodically saves snapshots and injects them into every env's pool (via `env_method("add_opponent_checkpoint", ...)`), so the agent trains against progressively stronger frozen copies of itself. `--resume-from` restores a checkpoint and re-registers existing snapshots as opponents.

### Key conventions

- **Canonical (mirrored) coordinates**: both sides share one policy. Observations and actions are expressed as if the agent always attacks left-to-right; for the right agent, x components are negated on the way in and out (`_make_observation`, `_canonical_action_to_world` in `envs.py`, `_human_action_to_canonical` in `cli.py`). Any new observation/action feature must respect this mirroring — see `test_magnet_observation_is_side_canonical`.
- **Zero-sum reward**: per-step shaping components are computed per agent and then differenced against the opponent's (`own - opponent's`), except `own_side` which stays one-sided; the terminal goal reward is added on top. Reward components are itemized in `REWARD_COMPONENTS` and surfaced through `info["reward_components"]` and the render overlay.
- **Reward profile mismatch**: the recommended profile is `simple`, but the training CLI still defaults to `possession` — pass `--reward-profile simple` explicitly. Eval/watch/benchmark/play already default to `simple`.
- **CLI duplication**: every command exists twice in `cli.py` — as a standalone Typer app callback (`train_entry` etc., backing the `klask-train`-style entry points) and as a subcommand of `app` (`klask-rl train`). Both delegate to shared `run_*` functions, but option signatures/defaults are duplicated: change both when adding or modifying options (tests check `--help` output of the standalone apps).
