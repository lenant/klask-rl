from __future__ import annotations

import numpy as np

from klask_rl.config import AGENTS
from klask_rl.envs import KlaskParallelEnv, SelfPlayKlaskEnv
from klask_rl.opponents import HeuristicOpponent, OpponentPool, PassiveOpponent, StrikerOpponent


class _ConstantOpponent:
    def __init__(self, value: float) -> None:
        self.value = value

    def act(self, observation: np.ndarray) -> np.ndarray:
        del observation
        return np.full(2, self.value, dtype=np.float32)


def test_scripted_experts_attack_a_resting_puck() -> None:
    """A puck that has come to rest must still get hit, not stared at.

    The experts wind up at a standoff behind the puck; without a commit step
    they park there forever once the puck stops moving on its own.
    """
    for expert in (StrikerOpponent(), HeuristicOpponent()):
        env = KlaskParallelEnv(reward_profile="simple")
        observations, _ = env.reset(seed=4)
        env.physics.puck_body.position = (-0.2, 0.1)
        env.physics.puck_body.velocity = (0.0, 0.0)
        observations = {agent: env._make_observation(agent) for agent in AGENTS}

        struck = False
        for _ in range(120):
            actions = {agent: expert.act(observations[agent]) for agent in env.agents}
            observations, _, terminations, truncations, _ = env.step(actions)
            if env.physics.puck_body.velocity.length > env.arena_config.max_puck_speed * 0.15:
                struck = True
                break
            if any(terminations.values()) or any(truncations.values()):
                break
        assert struck, f"{type(expert).__name__} never struck the resting puck"
        env.close()


def test_scripted_experts_keep_working_a_puck_on_their_back_wall() -> None:
    """A puck pinned on the back wall leaves no room to get behind it.

    Driving straight at it only presses it into the boards, so the expert works
    it from the side. It will not always free it from a corner, but it has to
    keep the ball alive rather than settle into a standoff and let the rest of
    the episode run out with a dead puck.
    """
    for expert in (StrikerOpponent(), HeuristicOpponent()):
        env = KlaskParallelEnv(reward_profile="simple")
        env.reset(seed=5)
        env.physics.puck_body.position = (-0.94, -0.60)
        env.physics.puck_body.velocity = (0.0, 0.0)
        env.physics.handle_bodies["left"].position = (-0.6, -0.3)
        observations = {agent: env._make_observation(agent) for agent in AGENTS}

        cfg = env.arena_config
        strikes = 0
        motionless = 0
        longest_motionless = 0
        # The same rally takes more control steps on a slower board.
        window = int(300 * 1.8 / cfg.max_handle_speed)
        for _ in range(window):
            actions = {agent: expert.act(observations[agent]) for agent in env.agents}
            observations, _, terminations, truncations, _ = env.step(actions)
            speed = env.physics.puck_body.velocity.length
            strikes += speed > cfg.max_puck_speed * 0.15
            motionless = motionless + 1 if speed == 0.0 else 0
            longest_motionless = max(longest_motionless, motionless)
            if any(terminations.values()) or any(truncations.values()):
                break
        name = type(expert).__name__
        # Repeated engagement, not a specific count -- a slower board makes each
        # hit gentler relative to the cap. The dead-time check below is the one
        # that actually pins the behaviour.
        assert strikes > 4, f"{name} barely touched the puck on the back wall"
        assert longest_motionless < cfg.dead_ball_steps + 30, (
            f"{name} let the puck sit dead for {longest_motionless} steps"
        )
        env.close()


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
