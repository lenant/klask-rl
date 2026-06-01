from __future__ import annotations

import numpy as np
from pettingzoo.test import parallel_api_test
from stable_baselines3.common.env_checker import check_env

from klask_rl.config import AGENTS, OBSERVATION_SIZE
from klask_rl.envs import REWARD_COMPONENTS, KlaskParallelEnv, SelfPlayKlaskEnv
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
    assert obs_a["left"].shape == (OBSERVATION_SIZE,)


def test_magnet_observation_is_side_canonical() -> None:
    env = KlaskParallelEnv()
    env.reset(seed=124)
    env.physics.magnet_bodies[0].position = (0.2, -0.1)
    env.physics.magnet_bodies[0].velocity = (0.4, -0.2)
    env.physics.magnet_attached_to[0] = "left"

    left_obs = env._make_observation("left")
    right_obs = env._make_observation("right")

    assert left_obs[16] == -right_obs[16]
    assert left_obs[17] == right_obs[17]
    assert left_obs[18] == -right_obs[18]
    assert left_obs[19] == right_obs[19]
    assert left_obs[20] == 1.0
    assert right_obs[20] == -1.0


def test_reward_profiles_change_reward_weights() -> None:
    balanced = KlaskParallelEnv(reward_profile="balanced")
    aggressive = KlaskParallelEnv(reward_profile="aggressive")
    assert aggressive.reward_config.progress > balanced.reward_config.progress
    assert aggressive.reward_config.terminal_goal > balanced.reward_config.terminal_goal


def test_reward_overlay_tracks_step_and_episode_rewards() -> None:
    env = KlaskParallelEnv(render_mode="rgb_array")
    env.reset(seed=42)
    actions = {agent: np.zeros(2, dtype=np.float32) for agent in AGENTS}

    _, rewards_1, _, _, _ = env.step(actions)
    _, rewards_2, _, _, infos = env.step(actions)

    for agent in AGENTS:
        assert infos[agent]["rewards"][agent] == rewards_2[agent]
        assert infos[agent]["episode_rewards"][agent] == rewards_1[agent] + rewards_2[agent]
        components = infos[agent]["reward_components"][agent]
        assert tuple(components) == REWARD_COMPONENTS
        assert sum(components.values()) == rewards_2[agent]

    frame = env.render()
    assert frame is not None
    assert frame.shape == (env.physics._surface_size[1], env.physics._surface_size[0], 3)


def test_magnet_risk_is_zero_sum_reward_component() -> None:
    env = KlaskParallelEnv()
    env.reset(seed=43)
    handle_position = env.physics.handle_bodies["left"].position
    env.physics.magnet_bodies[0].position = (handle_position.x + 0.18, handle_position.y)
    env.physics.magnet_bodies[0].velocity = (0.0, 0.0)

    actions = {agent: np.zeros(2, dtype=np.float32) for agent in AGENTS}
    _, _, _, _, infos = env.step(actions)

    left_magnet = infos["left"]["reward_components"]["left"]["magnet"]
    right_magnet = infos["right"]["reward_components"]["right"]["magnet"]
    assert left_magnet < 0.0
    assert right_magnet > 0.0
    assert left_magnet == -right_magnet


def test_magnet_scoring_terminates_episode_with_reason() -> None:
    env = KlaskParallelEnv()
    env.reset(seed=44)
    cfg = env.physics.config
    handle_position = env.physics.handle_bodies["left"].position
    env.physics.magnet_bodies[0].position = (
        handle_position.x + cfg.handle_radius + cfg.magnet_radius,
        handle_position.y,
    )
    env.physics.magnet_bodies[1].position = (
        handle_position.x,
        handle_position.y + cfg.handle_radius + cfg.magnet_radius,
    )
    env.physics.magnet_bodies[0].velocity = (0.0, 0.0)
    env.physics.magnet_bodies[1].velocity = (0.0, 0.0)

    actions = {agent: np.zeros(2, dtype=np.float32) for agent in AGENTS}
    _, _, terminations, _, infos = env.step(actions)

    assert terminations == {"left": True, "right": True}
    assert infos["left"]["score"] == {"left": 0, "right": 1}
    assert infos["left"]["scored_by"] == "right"
    assert infos["left"]["score_reason"] == "magnets"
    assert infos["left"]["magnet_counts"]["left"] == 2
