from __future__ import annotations

import numpy as np
from pettingzoo.test import parallel_api_test
from stable_baselines3.common.env_checker import check_env

from klask_rl.envs import KlaskParallelEnv, SelfPlayKlaskEnv
from klask_rl.opponents import HeuristicOpponent


def test_parallel_env_api() -> None:
    env = KlaskParallelEnv()
    parallel_api_test(env, num_cycles=5)


def test_selfplay_env_passes_sb3_checker() -> None:
    env = SelfPlayKlaskEnv(opponent=HeuristicOpponent(), max_steps=20)
    check_env(env, warn=True)


def test_right_agent_action_is_mirrored_to_world() -> None:
    env = KlaskParallelEnv()
    np.testing.assert_allclose(env._canonical_action_to_world("left", np.array([1.0, 0.5])), [1.0, 0.5])
    np.testing.assert_allclose(env._canonical_action_to_world("right", np.array([1.0, 0.5])), [-1.0, 0.5])


def test_reset_is_deterministic_for_same_seed() -> None:
    env_a = KlaskParallelEnv()
    env_b = KlaskParallelEnv()
    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)
    np.testing.assert_allclose(obs_a["left"], obs_b["left"])
    np.testing.assert_allclose(obs_a["right"], obs_b["right"])
