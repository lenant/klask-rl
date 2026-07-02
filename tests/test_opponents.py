from __future__ import annotations

import numpy as np

from klask_rl.envs import SelfPlayKlaskEnv
from klask_rl.opponents import OpponentPool, PassiveOpponent


class _ConstantOpponent:
    def __init__(self, value: float) -> None:
        self.value = value

    def act(self, observation: np.ndarray) -> np.ndarray:
        del observation
        return np.full(2, self.value, dtype=np.float32)


def test_pool_acts_coherently_between_resamples() -> None:
    pool = OpponentPool([_ConstantOpponent(-1.0), _ConstantOpponent(1.0)], seed=3)
    obs = np.zeros(33, dtype=np.float32)

    seen: set[float] = set()
    for _ in range(20):
        pool.resample()
        episode_actions = {float(pool.act(obs)[0]) for _ in range(5)}
        assert len(episode_actions) == 1
        seen.update(episode_actions)
    assert seen == {-1.0, 1.0}


def test_selfplay_reset_resamples_pool_opponent() -> None:
    class _CountingPool(OpponentPool):
        def __init__(self) -> None:
            super().__init__([PassiveOpponent()], seed=0)
            self.resamples = 0

        def resample(self) -> None:
            self.resamples += 1
            super().resample()

    pool = _CountingPool()
    env = SelfPlayKlaskEnv(opponent=pool, max_steps=10)
    env.reset(seed=1)
    env.reset(seed=2)
    assert pool.resamples == 2
    env.close()
