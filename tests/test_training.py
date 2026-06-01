from __future__ import annotations

from stable_baselines3 import PPO

from klask_rl.envs import SelfPlayKlaskEnv
from klask_rl.training import pretrain_policy_with_behavior_cloning


def test_behavior_cloning_warm_start_smoke() -> None:
    env = SelfPlayKlaskEnv(max_steps=10, reward_profile="aggressive")
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        seed=9,
        verbose=0,
    )
    stats = pretrain_policy_with_behavior_cloning(
        model,
        samples=16,
        epochs=1,
        batch_size=8,
        seed=9,
        max_steps=10,
        reward_profile="aggressive",
    )
    assert stats.samples == 16
    assert stats.epochs == 1
    assert stats.final_loss >= 0.0
