from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from stable_baselines3 import PPO

from klask_rl.envs import SelfPlayKlaskEnv
from klask_rl.opponents import OpponentPolicy, PassiveOpponent, RandomOpponent, StrikerOpponent


@dataclass(frozen=True)
class BehaviorCloningStats:
    samples: int
    epochs: int
    final_loss: float


def collect_expert_transitions(
    *,
    samples: int,
    seed: int,
    max_steps: int | None,
    reward_profile: str,
    expert: OpponentPolicy | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect canonical observations and expert actions for warm-start training."""

    rng = np.random.default_rng(seed)
    expert = expert or StrikerOpponent()
    opponents: list[OpponentPolicy] = [
        PassiveOpponent(),
        RandomOpponent(seed=seed + 1),
        StrikerOpponent(aggression=3.2),
    ]
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    env = SelfPlayKlaskEnv(
        opponent=opponents[0],
        randomize_side=True,
        max_steps=max_steps,
        reward_profile=reward_profile,
    )
    obs, _ = env.reset(seed=seed)
    while len(observations) < samples:
        env.opponent = opponents[int(rng.integers(0, len(opponents)))]
        action = expert.act(obs)
        observations.append(obs.copy())
        actions.append(action.copy())
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
    env.close()
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def pretrain_policy_with_behavior_cloning(
    model: PPO,
    *,
    samples: int,
    epochs: int,
    batch_size: int,
    seed: int,
    max_steps: int | None,
    reward_profile: str,
) -> BehaviorCloningStats:
    if samples <= 0 or epochs <= 0:
        return BehaviorCloningStats(samples=0, epochs=0, final_loss=0.0)

    observations, expert_actions = collect_expert_transitions(
        samples=samples,
        seed=seed,
        max_steps=max_steps,
        reward_profile=reward_profile,
    )
    device = model.policy.device
    optimizer = model.policy.optimizer
    rng = np.random.default_rng(seed + 17)
    final_loss = 0.0

    model.policy.train()
    for _ in range(epochs):
        indices = rng.permutation(samples)
        for start in range(0, samples, batch_size):
            batch_indices = indices[start : start + batch_size]
            obs_tensor = torch.as_tensor(observations[batch_indices], device=device)
            action_tensor = torch.as_tensor(expert_actions[batch_indices], device=device)
            distribution = model.policy.get_distribution(obs_tensor)
            mean_actions = distribution.distribution.mean
            loss = torch.nn.functional.mse_loss(torch.tanh(mean_actions), action_tensor)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()
            final_loss = float(loss.detach().cpu())

    return BehaviorCloningStats(samples=samples, epochs=epochs, final_loss=final_loss)
