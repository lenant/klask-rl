from __future__ import annotations

import numpy as np

from klask_rl.config import AGENTS, OBSERVATION_SIZE, ArenaConfig
from klask_rl.envs import KlaskParallelEnv, SelfPlayKlaskEnv
from klask_rl.opponents import (
    HeuristicOpponent,
    OpponentPool,
    PassiveOpponent,
    PlannerOpponent,
    StrikerOpponent,
)


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


def _rest_the_puck(env: KlaskParallelEnv, x: float, y: float) -> dict[str, np.ndarray]:
    """Put a motionless puck at (x, y) with the magnets parked out of the way."""
    env.physics.puck_body.position = (x, y)
    env.physics.puck_body.velocity = (0.0, 0.0)
    env.physics.handle_bodies["left"].position = (-0.5, 0.0)
    env.physics.handle_bodies["left"].velocity = (0.0, 0.0)
    env.physics.handle_bodies["right"].position = (0.95, -0.70)
    for body, spot in zip(
        env.physics.magnet_bodies, [(0.94, 0.68), (0.94, 0.40), (0.94, -0.68)], strict=False
    ):
        body.position = spot
        body.velocity = (0.0, 0.0)
    return {agent: env._make_observation(agent) for agent in AGENTS}


def test_planner_frees_a_resting_puck_from_the_edges_and_corners() -> None:
    """The cases the trained policies handle worst: a ball at rest against the boards.

    There is no room to get behind a ball touching a wall, so the strike point
    has to be clamped into the half and the shot re-derived from where the
    handle can actually stand. Without that the planner circles a reachable ball
    until the dead-ball timer re-serves it.
    """
    cfg = ArenaConfig()
    edge_x = cfg.half_width - cfg.wall_radius - cfg.puck_radius
    edge_y = cfg.half_height - cfg.wall_radius - cfg.puck_radius
    spots = {
        "back corner": (-edge_x, edge_y),
        "back wall": (-edge_x, 0.30),
        "top wall": (-0.35, edge_y),
        "front corner": (-0.10, edge_y),
        "beside own hole": (-0.62, 0.14),
    }
    for name, (x, y) in spots.items():
        env = KlaskParallelEnv(reward_profile="slow_v2")
        env.reset(seed=17)
        observations = _rest_the_puck(env, x, y)
        planner, passive = PlannerOpponent(), PassiveOpponent()

        struck = None
        for step in range(env.arena_config.dead_ball_steps - 10):
            actions = {
                "left": planner.act(observations["left"]),
                "right": passive.act(observations["right"]),
            }
            observations, _, terminations, truncations, _ = env.step(actions)
            if env.physics.puck_body.velocity.length > env.arena_config.max_puck_speed * 0.1:
                struck = step
                break
            if any(terminations.values()) or any(truncations.values()):
                break
        env.close()
        assert struck is not None, f"planner never struck the puck resting at the {name}"


def test_planner_aims_better_than_the_scripted_experts() -> None:
    """The point of searching over shots: the ball should end up going at the hole.

    Measured on the strongest shot of each rally, because a brush while orbiting
    is not the shot the planner intended to take.
    """
    spots = [(-0.45, 0.0), (-0.25, -0.30), (-0.80, -0.42), (-0.35, 0.55), (-0.60, 0.35)]

    def best_aim(policy) -> list[float]:
        results = []
        for x, y in spots:
            env = KlaskParallelEnv(reward_profile="slow_v2")
            env.reset(seed=17)
            observations = _rest_the_puck(env, x, y)
            passive = PassiveOpponent()
            aim = 0.0
            for _ in range(120):
                actions = {
                    "left": policy.act(observations["left"]),
                    "right": passive.act(observations["right"]),
                }
                observations, _, terminations, truncations, _ = env.step(actions)
                if env.physics.puck_body.velocity.length > env.arena_config.max_puck_speed * 0.35:
                    aim = max(aim, env.physics.shot_on_target("right", max_reflections=3))
                if any(terminations.values()) or any(truncations.values()):
                    break
            results.append(aim)
            env.close()
        return results

    planner = float(np.mean(best_aim(PlannerOpponent())))
    striker = float(np.mean(best_aim(StrikerOpponent())))
    assert planner > striker, f"planner aim {planner:.2f} did not beat striker {striker:.2f}"
    assert planner > 0.5, f"planner shots are not on target ({planner:.2f})"


def test_planner_never_plans_a_shot_into_its_own_hole() -> None:
    """A plan that concedes must never outrank one that does not.

    The search does allow an own-goal path as a last resort -- when the ball is
    already rolling into the hole, every option is one -- so this checks the
    positions where a safe shot exists, which is all of them here.
    """
    planner = PlannerOpponent()
    trajectory = planner.trajectory
    observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    own_hole = np.asarray(planner.own_hole)
    for x in np.linspace(-0.90, -0.10, 9):
        for y in np.linspace(-0.65, 0.65, 9):
            contact = np.array([x, y])
            if float(np.hypot(*(contact - own_hole))) <= planner.config.puck_capture_radius:
                continue  # already in the hole: the point is lost either way
            plan = planner._plan(
                np.array([-0.5, 0.0]), contact, np.array([0.6, 0.0]), observation
            )
            assert plan is not None, f"no plan at ({x:.2f}, {y:.2f})"
            path = trajectory.march(
                (contact[0], contact[1]),
                tuple(np.asarray(plan.launch) * planner.launch_speed),
                max_reflections=planner.max_reflections,
            )
            assert path.fell_into != "left", f"planned an own goal from ({x:.2f}, {y:.2f})"


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
