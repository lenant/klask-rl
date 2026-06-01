from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from klask_rl.envs import SelfPlayKlaskEnv
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
    def __init__(self, pool: OpponentPool, snapshot_dir: Path, save_freq: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.pool = pool
        self.snapshot_dir = snapshot_dir
        self.save_freq = max(1, save_freq)
        self._last_save = 0

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


def build_vec_env(
    pool: OpponentPool,
    num_envs: int,
    seed: int,
    max_steps: int | None,
    reward_profile: str,
) -> DummyVecEnv:
    def make_env(rank: int):
        def _factory():
            env = SelfPlayKlaskEnv(
                opponent=pool,
                max_steps=max_steps,
                reward_profile=reward_profile,
            )
            return Monitor(env)

        return _factory

    env = DummyVecEnv([make_env(rank) for rank in range(num_envs)])
    env.seed(seed)
    return env


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
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_dir = output_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    pool = OpponentPool(
        [
            PassiveOpponent(),
            RandomOpponent(seed=seed + 1),
            StrikerOpponent(),
            HeuristicOpponent(),
            HeuristicOpponent(aggression=4.0),
        ],
        seed=seed,
    )
    env = build_vec_env(
        pool=pool,
        num_envs=num_envs,
        seed=seed,
        max_steps=max_steps,
        reward_profile=reward_profile,
    )
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
        verbose=1,
    )
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
        snapshot_dir=latest_dir / "snapshots",
        save_freq=snapshot_freq,
        verbose=1,
    )
    model.learn(total_timesteps=total_steps, callback=callback, progress_bar=False)
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
) -> EvaluationSummary:
    model = PPO.load(model_path)
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
) -> dict[str, float | int | str]:
    summaries = [
        run_eval(model_path, episodes, None, "passive", False, seed, deterministic, False, max_steps),
        run_eval(model_path, episodes, None, "random", False, seed + 100, deterministic, False, max_steps),
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
        ),
        run_eval(model_path, episodes, None, "heuristic", True, seed + 300, deterministic, False, max_steps),
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
        str, typer.Option(help="Reward profile: balanced, aggressive, defensive, possession.")
    ] = "possession",
    bc_samples: Annotated[int, typer.Option(help="Expert samples for behavior-cloning warm start.")] = 4096,
    bc_epochs: Annotated[int, typer.Option(help="Behavior-cloning epochs before PPO.")] = 4,
    bc_batch_size: Annotated[int, typer.Option(help="Behavior-cloning batch size.")] = 256,
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
) -> None:
    run_eval(model, episodes, opponent_model, opponent, self_play, seed, deterministic, render, max_steps)


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
) -> None:
    run_benchmark(model, episodes, seed, deterministic, max_steps)


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
) -> None:
    run_eval(model, episodes, opponent_model, opponent, self_play, seed, deterministic, render, max_steps)


@app.command()
def watch(
    model: Path,
    opponent_model: Path | None = None,
    opponent: str = "heuristic",
    self_play: bool = True,
    episodes: int = 5,
    seed: int = 17,
    max_steps: int | None = 450,
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
    )


@app.command()
def benchmark(
    model: Path,
    episodes: int = 20,
    seed: int = 31,
    deterministic: bool = True,
    max_steps: int | None = 450,
) -> None:
    run_benchmark(model, episodes, seed, deterministic, max_steps)


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
