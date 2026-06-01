from __future__ import annotations

from stable_baselines3 import PPO

from klask_rl.envs import SelfPlayKlaskEnv
from klask_rl.opponents import HeuristicOpponent


def test_short_ppo_training_smoke(tmp_path) -> None:
    env = SelfPlayKlaskEnv(opponent=HeuristicOpponent(), max_steps=15)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        gamma=0.95,
        seed=5,
        verbose=0,
    )
    model.learn(total_timesteps=16, progress_bar=False)
    path = tmp_path / "model"
    model.save(path)
    assert path.with_suffix(".zip").exists()
