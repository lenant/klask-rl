from __future__ import annotations

import time
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from klask_rl.config import AGENTS, OPPONENT, REWARD_PROFILES, ArenaConfig
from klask_rl.envs import KlaskParallelEnv, SelfPlayKlaskEnv
from klask_rl.opponents import (
    HeuristicOpponent,
    OpponentPolicy,
    OpponentPool,
    PassiveOpponent,
    RandomOpponent,
    SB3CheckpointOpponent,
    StrikerOpponent,
)
from klask_rl.training import pretrain_policy_with_behavior_cloning

console = Console()
app = typer.Typer(add_completion=False)
train_app = typer.Typer(add_completion=False)
eval_app = typer.Typer(add_completion=False)
watch_app = typer.Typer(add_completion=False)
benchmark_app = typer.Typer(add_completion=False)
play_app = typer.Typer(add_completion=False)
SNAPSHOT_RE = re.compile(r"policy_(\d+)\.zip$")


@dataclass(frozen=True)
class EvaluationSummary:
    episodes: int
    opponent: str
    deterministic: bool
    wins: int
    losses: int
    draws: int
    mean_reward: float
    mean_length: float
    mean_final_puck_x: float
    mean_contact_steps: float

    @property
    def goals_for(self) -> int:
        return self.wins

    @property
    def goals_against(self) -> int:
        return self.losses

    @property
    def draw_rate(self) -> float:
        return self.draws / max(1, self.episodes)

    @property
    def defense_rate(self) -> float:
        return 1.0 - (self.losses / max(1, self.episodes))

    @property
    def goals_per_game(self) -> float:
        return (self.wins + self.losses) / max(1, self.episodes)

    @property
    def quality_score(self) -> float:
        attack = 3.0 * self.wins / max(1, self.episodes)
        defense = 2.0 * self.defense_rate
        activity = min(2.0, self.mean_contact_steps / 3.0)
        territory = max(-1.0, min(1.0, self.mean_final_puck_x))
        decisiveness = min(1.0, self.goals_per_game)
        return attack + defense + activity + territory + decisiveness

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "episodes": self.episodes,
            "opponent": self.opponent,
            "deterministic": self.deterministic,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "goals_for": self.goals_for,
            "goals_against": self.goals_against,
            "draw_rate": self.draw_rate,
            "defense_rate": self.defense_rate,
            "goals_per_game": self.goals_per_game,
            "quality_score": self.quality_score,
            "mean_reward": self.mean_reward,
            "mean_length": self.mean_length,
            "mean_final_puck_x": self.mean_final_puck_x,
            "mean_contact_steps": self.mean_contact_steps,
        }


class SelfPlaySnapshotCallback(BaseCallback):
    def __init__(
        self,
        pool: OpponentPool,
        snapshot_dir: Path,
        save_freq: int,
        initial_last_save: int = 0,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.pool = pool
        self.snapshot_dir = snapshot_dir
        self.save_freq = max(1, save_freq)
        self._last_save = initial_last_save

    def _on_training_start(self) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save < self.save_freq:
            return True
        self._last_save = self.num_timesteps
        path = self.snapshot_dir / f"policy_{self.num_timesteps}"
        self.model.save(path)
        checkpoint = path.with_suffix(".zip")
        self.pool.add_checkpoint(checkpoint)
        if self.training_env is not None:
            self.training_env.env_method("add_opponent_checkpoint", str(checkpoint))
        if self.verbose:
            console.print(f"saved self-play snapshot: {checkpoint}")
        return True


def make_named_opponent(name: str, seed: int = 0) -> OpponentPolicy:
    if name == "heuristic":
        return HeuristicOpponent()
    if name == "random":
        return RandomOpponent(seed=seed)
    if name == "passive":
        return PassiveOpponent()
    if name == "striker":
        return StrikerOpponent()
    raise typer.BadParameter("opponent must be one of: heuristic, random, passive, striker")


def make_training_pool(seed: int) -> OpponentPool:
    return OpponentPool(
        [
            PassiveOpponent(),
            RandomOpponent(seed=seed + 1),
            StrikerOpponent(),
            HeuristicOpponent(),
            HeuristicOpponent(aggression=4.0),
        ],
        seed=seed,
    )


def snapshot_step(path: Path) -> int | None:
    match = SNAPSHOT_RE.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group(1))


def existing_snapshot_paths(snapshot_dir: Path, max_step: int | None = None) -> list[Path]:
    snapshots: list[tuple[int, Path]] = []
    for path in snapshot_dir.glob("policy_*.zip"):
        step = snapshot_step(path)
        if step is None:
            continue
        if max_step is not None and step > max_step:
            continue
        snapshots.append((step, path))
    return [path for _, path in sorted(snapshots)]


def add_opponent_checkpoints(pool: OpponentPool, env: VecEnv, checkpoint_paths: list[Path]) -> None:
    for checkpoint_path in checkpoint_paths:
        pool.add_checkpoint(checkpoint_path)
        env.env_method("add_opponent_checkpoint", str(checkpoint_path))


def build_vec_env(
    num_envs: int,
    seed: int,
    max_steps: int | None,
    reward_profile: str,
    vec_env: str,
) -> VecEnv:
    def make_env(rank: int):
        def _factory():
            pool = make_training_pool(seed + rank)
            env = SelfPlayKlaskEnv(
                opponent=pool,
                max_steps=max_steps,
                reward_profile=reward_profile,
            )
            return Monitor(env)

        return _factory

    factories = [make_env(rank) for rank in range(num_envs)]
    if vec_env == "dummy":
        env = DummyVecEnv(factories)
    elif vec_env == "subproc":
        env = SubprocVecEnv(factories, start_method="fork")
    else:
        raise typer.BadParameter("vec-env must be one of: dummy, subproc")
    env.seed(seed)
    return env


def parse_policy_net_arch(value: str) -> tuple[int, ...]:
    separators_normalized = value.replace("x", ",").replace("X", ",")
    try:
        sizes = tuple(int(part.strip()) for part in separators_normalized.split(",") if part.strip())
    except ValueError as exc:
        raise typer.BadParameter("policy-net-arch must be like 64,64 or 256x256x256") from exc
    if not sizes:
        raise typer.BadParameter("policy-net-arch must include at least one hidden layer size")
    if any(size <= 0 for size in sizes):
        raise typer.BadParameter("policy-net-arch hidden sizes must be positive")
    return sizes


def run_train(
    total_steps: int,
    num_envs: int,
    output_dir: Path,
    seed: int,
    n_steps: int,
    batch_size: int,
    snapshot_freq: int,
    max_steps: int | None,
    reward_profile: str,
    bc_samples: int,
    bc_epochs: int,
    bc_batch_size: int,
    vec_env: str,
    device: str,
    policy_net_arch: str,
    resume_from: Path | None,
    resume_opponent_checkpoints: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_dir = output_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = latest_dir / "snapshots"
    net_arch = parse_policy_net_arch(policy_net_arch)
    pool = make_training_pool(seed)
    env = build_vec_env(
        num_envs=num_envs,
        seed=seed,
        max_steps=max_steps,
        reward_profile=reward_profile,
        vec_env=vec_env,
    )
    if resume_from is not None:
        if not resume_from.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_from}")
        model = PPO.load(resume_from, env=env, device=device)
        model.verbose = 1
        checkpoint_paths = existing_snapshot_paths(snapshot_dir, max_step=model.num_timesteps)
        if resume_opponent_checkpoints >= 0:
            checkpoint_paths = checkpoint_paths[-resume_opponent_checkpoints:]
        add_opponent_checkpoints(pool, env, checkpoint_paths)
        initial_last_save = (model.num_timesteps // snapshot_freq) * snapshot_freq
        remaining_steps = max(0, total_steps - model.num_timesteps)
        console.print(
            {
                "resume_from": str(resume_from),
                "starting_num_timesteps": model.num_timesteps,
                "target_total_steps": total_steps,
                "remaining_steps": remaining_steps,
                "restored_opponent_checkpoints": len(checkpoint_paths),
                "resume_opponent_checkpoints": resume_opponent_checkpoints,
                "next_snapshot_at": initial_last_save + snapshot_freq,
            }
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=5,
            learning_rate=3e-4,
            gamma=0.985,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=str(output_dir / "tensorboard"),
            seed=seed,
            device=device,
            policy_kwargs={"net_arch": list(net_arch)},
            verbose=1,
        )
        initial_last_save = 0
        remaining_steps = total_steps
        console.print({"policy_net_arch": net_arch})
        bc_stats = pretrain_policy_with_behavior_cloning(
            model,
            samples=bc_samples,
            epochs=bc_epochs,
            batch_size=bc_batch_size,
            seed=seed,
            max_steps=max_steps,
            reward_profile=reward_profile,
        )
        if bc_stats.samples:
            console.print(
                {
                    "behavior_cloning_samples": bc_stats.samples,
                    "behavior_cloning_epochs": bc_stats.epochs,
                    "behavior_cloning_final_loss": bc_stats.final_loss,
                }
            )
    callback = SelfPlaySnapshotCallback(
        pool=pool,
        snapshot_dir=snapshot_dir,
        save_freq=snapshot_freq,
        initial_last_save=initial_last_save,
        verbose=1,
    )
    if remaining_steps > 0:
        model.learn(
            total_timesteps=remaining_steps,
            callback=callback,
            progress_bar=False,
            reset_num_timesteps=resume_from is None,
        )
    final_path = latest_dir / "final_model"
    model.save(final_path)
    env.close()
    console.print(f"saved final model: {final_path.with_suffix('.zip')}")
    console.print(f"self-play snapshots available: {len(pool.checkpoint_paths)}")


def run_eval(
    model_path: Path,
    episodes: int,
    opponent_model: Path | None,
    opponent: str,
    self_play: bool,
    seed: int,
    deterministic: bool,
    render: bool,
    max_steps: int | None,
    reward_profile: str,
) -> EvaluationSummary:
    model = PPO.load(model_path, device="cpu")
    if self_play and opponent_model is None:
        opponent_model = model_path
    opponent_policy = (
        SB3CheckpointOpponent(opponent_model)
        if opponent_model
        else make_named_opponent(opponent, seed=seed + 1000)
    )
    wins = 0
    losses = 0
    draws = 0
    rewards: list[float] = []
    lengths: list[int] = []
    final_puck_x: list[float] = []
    contact_steps: list[int] = []

    for episode in range(episodes):
        env = SelfPlayKlaskEnv(
            opponent=opponent_policy,
            render_mode="human" if render else None,
            max_steps=max_steps,
            reward_profile=reward_profile,
        )
        obs, info = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        length = 0
        contacts = 0
        final_info = info
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, final_info = env.step(action)
            episode_reward += reward
            length += 1
            contacts += int(final_info.get("contacts", {}).get(final_info.get("learning_side"), False))
            done = terminated or truncated
            if render:
                env.render()
                while env.base_env.physics.paused:
                    time.sleep(env.base_env.arena_config.control_dt)
                    env.render()
                time.sleep(env.base_env.arena_config.control_dt)
        scored_by = final_info.get("scored_by")
        learning_side = final_info.get("learning_side")
        if scored_by == learning_side:
            wins += 1
        elif scored_by is None:
            draws += 1
        else:
            losses += 1
        rewards.append(episode_reward)
        lengths.append(length)
        final_puck_x.append(float(obs[8]))
        contact_steps.append(contacts)
        env.close()

    summary = EvaluationSummary(
        episodes=episodes,
        opponent="self-play" if self_play else ("checkpoint" if opponent_model else opponent),
        deterministic=deterministic,
        wins=wins,
        losses=losses,
        draws=draws,
        mean_reward=float(np.mean(rewards)) if rewards else 0.0,
        mean_length=float(np.mean(lengths)) if lengths else 0.0,
        mean_final_puck_x=float(np.mean(final_puck_x)) if final_puck_x else 0.0,
        mean_contact_steps=float(np.mean(contact_steps)) if contact_steps else 0.0,
    )
    console.print(summary.as_dict())
    return summary


def run_benchmark(
    model_path: Path,
    episodes: int,
    seed: int,
    deterministic: bool,
    max_steps: int | None,
    reward_profile: str,
) -> dict[str, float | int | str]:
    summaries = [
        run_eval(
            model_path,
            episodes,
            None,
            "passive",
            False,
            seed,
            deterministic,
            False,
            max_steps,
            reward_profile,
        ),
        run_eval(
            model_path,
            episodes,
            None,
            "random",
            False,
            seed + 100,
            deterministic,
            False,
            max_steps,
            reward_profile,
        ),
        run_eval(
            model_path,
            episodes,
            None,
            "heuristic",
            False,
            seed + 200,
            deterministic,
            False,
            max_steps,
            reward_profile,
        ),
        run_eval(
            model_path,
            episodes,
            None,
            "heuristic",
            True,
            seed + 300,
            deterministic,
            False,
            max_steps,
            reward_profile,
        ),
    ]
    aggregate_quality = float(np.mean([summary.quality_score for summary in summaries]))
    aggregate_goals_for = int(sum(summary.goals_for for summary in summaries))
    aggregate_goals_against = int(sum(summary.goals_against for summary in summaries))
    result = {
        "model": str(model_path),
        "episodes_per_matchup": episodes,
        "aggregate_quality": aggregate_quality,
        "aggregate_goals_for": aggregate_goals_for,
        "aggregate_goals_against": aggregate_goals_against,
        "aggregate_goal_margin": aggregate_goals_for - aggregate_goals_against,
    }
    console.print(result)
    return result


def _scaled_world_action(x: float, y: float, action_scale: float) -> np.ndarray:
    action = np.array([x, y], dtype=np.float32)
    norm = float(np.linalg.norm(action))
    if norm > 1.0:
        action /= norm
    return np.clip(action_scale, 0.0, 1.0) * action


def _world_action_from_key_state(
    is_pressed: Callable[[object], bool],
    left_key: object,
    right_key: object,
    up_key: object,
    down_key: object,
    action_scale: float,
) -> np.ndarray:
    x = float(is_pressed(right_key)) - float(is_pressed(left_key))
    y = float(is_pressed(up_key)) - float(is_pressed(down_key))
    return _scaled_world_action(x, y, action_scale)


def _poll_pygame_keys(pygame_module: object) -> object:
    pygame_module.event.pump()
    return pygame_module.key.get_pressed()


def _human_world_action_from_keys(pygame_module: object, keys: object, action_scale: float) -> np.ndarray:
    return _world_action_from_key_state(
        keys.__getitem__,
        pygame_module.K_a,
        pygame_module.K_d,
        pygame_module.K_w,
        pygame_module.K_s,
        action_scale,
    )


def _human_world_action(pygame_module: object, action_scale: float) -> np.ndarray:
    keys = _poll_pygame_keys(pygame_module)
    return _human_world_action_from_keys(pygame_module, keys, action_scale)


def _key_is_pressed(keys: object, key: int) -> bool:
    return bool(keys[key])


def _human_action_to_canonical(human_side: str, world_action: np.ndarray) -> np.ndarray:
    if human_side == "left":
        return np.asarray(world_action, dtype=np.float32)
    if human_side == "right":
        return np.array([-world_action[0], world_action[1]], dtype=np.float32)
    raise typer.BadParameter("human-side must be one of: left, right")


def run_play(
    model_path: Path,
    human_side: str,
    episodes: int,
    seed: int,
    max_steps: int | None,
    deterministic: bool,
    action_scale: float,
    reward_profile: str,
) -> None:
    if human_side not in AGENTS:
        raise typer.BadParameter("human-side must be one of: left, right")

    import pygame

    model = PPO.load(model_path, device="cpu")
    arena_config = ArenaConfig()
    if max_steps is not None:
        arena_config = replace(arena_config, max_steps=max_steps)
    env = KlaskParallelEnv(arena_config=arena_config, reward_profile=reward_profile)
    model_side = OPPONENT[human_side]
    console.print(
        {
            "human_side": human_side,
            "model_side": model_side,
            "reward_profile": reward_profile,
            "controls": "WASD",
            "pause": "space",
            "quit": "close window or Escape",
        }
    )

    try:
        for episode in range(episodes):
            observations, infos = env.reset(seed=seed + episode)
            done = False
            final_info = infos[human_side]
            env.render()
            while not done and not env.physics.quit_requested:
                keys = _poll_pygame_keys(pygame)
                if _key_is_pressed(keys, pygame.K_ESCAPE):
                    env.physics.quit_requested = True
                    break

                while env.physics.paused and not env.physics.quit_requested:
                    time.sleep(env.arena_config.control_dt)
                    env.render()
                    keys = _poll_pygame_keys(pygame)
                    if _key_is_pressed(keys, pygame.K_ESCAPE):
                        env.physics.quit_requested = True
                        break
                if env.physics.quit_requested:
                    break

                human_action = _human_action_to_canonical(
                    human_side,
                    _human_world_action_from_keys(pygame, keys, action_scale),
                )
                model_action, _ = model.predict(observations[model_side], deterministic=deterministic)
                actions = {
                    human_side: human_action,
                    model_side: np.asarray(model_action, dtype=np.float32),
                }
                observations, _, terminations, truncations, infos = env.step(actions)
                final_info = infos.get(human_side, final_info)
                env.render()
                done = bool(terminations.get(human_side, False) or truncations.get(human_side, False))
                time.sleep(env.arena_config.control_dt)

            if env.physics.quit_requested:
                break
            console.print(
                {
                    "episode": episode + 1,
                    "score": final_info.get("score", {}),
                    "scored_by": final_info.get("scored_by"),
                    "score_reason": final_info.get("score_reason"),
                    "steps": final_info.get("steps"),
                }
            )
            time.sleep(0.4)
    finally:
        env.close()


@train_app.callback(invoke_without_command=True)
def train_entry(
    total_steps: Annotated[int, typer.Option(help="Total PPO environment steps.")] = 100_000,
    num_envs: Annotated[int, typer.Option(help="Number of sequential vector envs.")] = 4,
    output_dir: Annotated[Path, typer.Option(help="Directory for models and logs.")] = Path(
        "runs/klask"
    ),
    seed: Annotated[int, typer.Option(help="Random seed.")] = 7,
    n_steps: Annotated[int, typer.Option(help="PPO rollout steps per env.")] = 1024,
    batch_size: Annotated[int, typer.Option(help="PPO minibatch size.")] = 256,
    snapshot_freq: Annotated[int, typer.Option(help="Steps between self-play snapshots.")] = 10_000,
    max_steps: Annotated[
        int | None, typer.Option(help="Optional max steps per episode.")
    ] = 450,
    reward_profile: Annotated[
        str, typer.Option(help=f"Reward profile: {', '.join(sorted(REWARD_PROFILES))}.")
    ] = "possession",
    bc_samples: Annotated[int, typer.Option(help="Expert samples for behavior-cloning warm start.")] = 4096,
    bc_epochs: Annotated[int, typer.Option(help="Behavior-cloning epochs before PPO.")] = 4,
    bc_batch_size: Annotated[int, typer.Option(help="Behavior-cloning batch size.")] = 256,
    vec_env: Annotated[str, typer.Option(help="Vector env backend: dummy or subproc.")] = "dummy",
    device: Annotated[str, typer.Option(help="SB3 device: auto, cpu, cuda, cuda:0, etc.")] = "auto",
    policy_net_arch: Annotated[
        str, typer.Option(help="PPO MLP hidden sizes, e.g. 64,64 or 256x256x256.")
    ] = "64,64",
    resume_from: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Optional PPO checkpoint to continue from. "
                "When set, total-steps is treated as the final target timestep."
            )
        ),
    ] = None,
    resume_opponent_checkpoints: Annotated[
        int,
        typer.Option(
            help=(
                "How many recent checkpoint opponents to restore on resume. "
                "Use -1 to restore every historical snapshot."
            )
        ),
    ] = 64,
) -> None:
    run_train(
        total_steps,
        num_envs,
        output_dir,
        seed,
        n_steps,
        batch_size,
        snapshot_freq,
        max_steps,
        reward_profile,
        bc_samples,
        bc_epochs,
        bc_batch_size,
        vec_env,
        device,
        policy_net_arch,
        resume_from,
        resume_opponent_checkpoints,
    )


@eval_app.callback(invoke_without_command=True)
def eval_entry(
    model: Annotated[Path, typer.Option(help="Path to a PPO .zip model.")],
    episodes: Annotated[int, typer.Option(help="Evaluation episodes.")] = 20,
    opponent_model: Annotated[
        Path | None, typer.Option(help="Optional opponent checkpoint.")
    ] = None,
    opponent: Annotated[
        str,
        typer.Option(help="Opponent if no checkpoint is supplied: heuristic, random, passive, striker."),
    ] = "heuristic",
    self_play: Annotated[bool, typer.Option(help="Use the same model to control both sides.")] = False,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 11,
    deterministic: Annotated[bool, typer.Option(help="Use deterministic policy actions.")] = True,
    render: Annotated[bool, typer.Option(help="Render with pygame.")] = False,
    max_steps: Annotated[
        int | None, typer.Option(help="Optional max steps per evaluation episode.")
    ] = 450,
    reward_profile: Annotated[
        str, typer.Option(help=f"Reward profile: {', '.join(sorted(REWARD_PROFILES))}.")
    ] = "simple",
) -> None:
    run_eval(
        model,
        episodes,
        opponent_model,
        opponent,
        self_play,
        seed,
        deterministic,
        render,
        max_steps,
        reward_profile,
    )


@watch_app.callback(invoke_without_command=True)
def watch_entry(
    model: Annotated[Path, typer.Option(help="Path to a PPO .zip model.")],
    opponent_model: Annotated[
        Path | None, typer.Option(help="Optional opponent checkpoint.")
    ] = None,
    opponent: Annotated[
        str,
        typer.Option(help="Opponent if no checkpoint is supplied: heuristic, random, passive, striker."),
    ] = "heuristic",
    self_play: Annotated[bool, typer.Option(help="Use the same model to control both sides.")] = True,
    episodes: Annotated[int, typer.Option(help="Episodes to render.")] = 5,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 17,
    max_steps: Annotated[
        int | None, typer.Option(help="Optional max steps per watch episode.")
    ] = 450,
    reward_profile: Annotated[
        str, typer.Option(help=f"Reward profile: {', '.join(sorted(REWARD_PROFILES))}.")
    ] = "simple",
) -> None:
    run_eval(
        model_path=model,
        episodes=episodes,
        opponent_model=opponent_model,
        opponent=opponent,
        self_play=self_play,
        seed=seed,
        deterministic=True,
        render=True,
        max_steps=max_steps,
        reward_profile=reward_profile,
    )


@benchmark_app.callback(invoke_without_command=True)
def benchmark_entry(
    model: Annotated[Path, typer.Option(help="Path to a PPO .zip model.")],
    episodes: Annotated[int, typer.Option(help="Episodes per matchup.")] = 20,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 31,
    deterministic: Annotated[bool, typer.Option(help="Use deterministic policy actions.")] = True,
    max_steps: Annotated[
        int | None, typer.Option(help="Optional max steps per evaluation episode.")
    ] = 450,
    reward_profile: Annotated[
        str, typer.Option(help=f"Reward profile: {', '.join(sorted(REWARD_PROFILES))}.")
    ] = "simple",
) -> None:
    run_benchmark(model, episodes, seed, deterministic, max_steps, reward_profile)


@play_app.callback(invoke_without_command=True)
def play_entry(
    model: Annotated[Path, typer.Option(help="Path to a PPO .zip model.")],
    human_side: Annotated[
        str, typer.Option(help="Side controlled by WASD: left or right.")
    ] = "left",
    episodes: Annotated[int, typer.Option(help="Episodes to play.")] = 5,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 17,
    max_steps: Annotated[
        int | None, typer.Option(help="Optional max steps per played episode.")
    ] = 450,
    deterministic: Annotated[bool, typer.Option(help="Use deterministic model actions.")] = True,
    action_scale: Annotated[float, typer.Option(help="Human action strength, 0.0 to 1.0.")] = 1.0,
    reward_profile: Annotated[
        str, typer.Option(help=f"Reward profile: {', '.join(sorted(REWARD_PROFILES))}.")
    ] = "simple",
) -> None:
    run_play(model, human_side, episodes, seed, max_steps, deterministic, action_scale, reward_profile)


@app.command()
def train(
    total_steps: int = 100_000,
    num_envs: int = 4,
    output_dir: Path = Path("runs/klask"),
    seed: int = 7,
    n_steps: int = 1024,
    batch_size: int = 256,
    snapshot_freq: int = 10_000,
    max_steps: int | None = 450,
    reward_profile: str = "possession",
    bc_samples: int = 4096,
    bc_epochs: int = 4,
    bc_batch_size: int = 256,
    vec_env: str = "dummy",
    device: str = "auto",
    policy_net_arch: str = "64,64",
    resume_from: Path | None = None,
    resume_opponent_checkpoints: int = 64,
) -> None:
    run_train(
        total_steps,
        num_envs,
        output_dir,
        seed,
        n_steps,
        batch_size,
        snapshot_freq,
        max_steps,
        reward_profile,
        bc_samples,
        bc_epochs,
        bc_batch_size,
        vec_env,
        device,
        policy_net_arch,
        resume_from,
        resume_opponent_checkpoints,
    )


@app.command(name="eval")
def eval_command(
    model: Path,
    episodes: int = 20,
    opponent_model: Path | None = None,
    opponent: str = "heuristic",
    self_play: bool = False,
    seed: int = 11,
    deterministic: bool = True,
    render: bool = False,
    max_steps: int | None = 450,
    reward_profile: str = "simple",
) -> None:
    run_eval(
        model,
        episodes,
        opponent_model,
        opponent,
        self_play,
        seed,
        deterministic,
        render,
        max_steps,
        reward_profile,
    )


@app.command()
def watch(
    model: Path,
    opponent_model: Path | None = None,
    opponent: str = "heuristic",
    self_play: bool = True,
    episodes: int = 5,
    seed: int = 17,
    max_steps: int | None = 450,
    reward_profile: str = "simple",
) -> None:
    run_eval(
        model,
        episodes,
        opponent_model,
        opponent,
        self_play,
        seed,
        deterministic=True,
        render=True,
        max_steps=max_steps,
        reward_profile=reward_profile,
    )


@app.command()
def benchmark(
    model: Path,
    episodes: int = 20,
    seed: int = 31,
    deterministic: bool = True,
    max_steps: int | None = 450,
    reward_profile: str = "simple",
) -> None:
    run_benchmark(model, episodes, seed, deterministic, max_steps, reward_profile)


@app.command()
def play(
    model: Path,
    human_side: str = "left",
    episodes: int = 5,
    seed: int = 17,
    max_steps: int | None = 450,
    deterministic: bool = True,
    action_scale: float = 1.0,
    reward_profile: str = "simple",
) -> None:
    run_play(model, human_side, episodes, seed, max_steps, deterministic, action_scale, reward_profile)


def main() -> None:
    app()


def train_main() -> None:
    train_app()


def eval_main() -> None:
    eval_app()


def watch_main() -> None:
    watch_app()


def benchmark_main() -> None:
    benchmark_app()


def play_main() -> None:
    play_app()
