from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ArenaConfig:
    # Real Klask playing field is 40 x 30 cm; 1 unit = 20 cm.
    width: float = 2.0
    height: float = 1.5
    goal_radius: float = 0.075
    goal_center_offset: float = 0.225
    wall_radius: float = 0.018
    puck_radius: float = 0.035
    handle_radius: float = 0.04
    magnet_count: int = 3
    magnet_radius: float = 0.026
    magnet_mass: float = 0.03
    magnet_attraction_range: float = 0.22
    magnet_attraction_strength: float = 3.0
    magnet_max_force: float = 0.8
    max_magnet_speed: float = 1.5
    magnet_friction: float = 0.75
    magnet_linear_friction: float = 3.0
    magnet_slide_friction: float = 1.0
    magnet_stop_speed: float = 0.02
    magnet_elasticity: float = 0.35
    magnet_attach_frames: int = 2
    magnet_release_distance: float = 0.16
    magnet_score_threshold: int = 2
    puck_start_min_x_fraction: float = 0.125
    puck_start_max_x_fraction: float = 0.25
    puck_mass: float = 0.045
    max_handle_speed: float = 1.8
    # Actions are target velocities, but a hand (or a gantry) cannot change
    # velocity instantly. Bounding the change keeps the handle moving
    # continuously and makes shot power depend on the run-up, instead of every
    # contact landing at full speed.
    max_handle_acceleration: float = 18.0
    max_puck_speed: float = 3.2
    physics_dt: float = 1.0 / 120.0
    frame_skip: int = 4
    # Pair restitutions. Shape elasticities multiply in pymunk, so the puck is
    # kept perfectly elastic and each partner carries the whole coefficient.
    puck_elasticity: float = 1.0
    wall_elasticity: float = 0.55
    handle_elasticity: float = 0.70
    wall_friction: float = 0.6
    puck_friction: float = 0.2
    # Rolling resistance: a linear (viscous) term plus a constant deceleration.
    # The constant term is what actually brings a slow ball to rest, so the
    # policy sees stationary pucks the way it will on the real board.
    puck_linear_drag: float = 0.15
    puck_rolling_friction: float = 0.22
    puck_stop_speed: float = 0.02
    # Handles are confined to their own half, so a puck that stops in the other
    # half is unreachable by everyone and the rest of the episode is dead. Put
    # it back in play after this many motionless control steps, the way a player
    # would re-serve a stuck ball. 0 disables. Against a real opponent this
    # fires ~0.1x per episode -- it is a backstop, not a game mechanic. Kept
    # generous so a clumsy early policy still has to learn to go fetch a
    # resting puck rather than wait the timer out.
    dead_ball_steps: int = 60
    # Velocity decay is modelled explicitly per body, so the space adds none.
    damping: float = 1.0
    max_steps: int = 900

    @property
    def half_width(self) -> float:
        return self.width / 2.0

    @property
    def half_height(self) -> float:
        return self.height / 2.0

    @property
    def goal_center_x(self) -> float:
        return self.half_width - self.goal_center_offset

    @property
    def puck_wall_elasticity(self) -> float:
        """Restitution pymunk resolves a puck/wall contact with."""
        return self.puck_elasticity * self.wall_elasticity

    @property
    def magnet_wall_elasticity(self) -> float:
        """Restitution pymunk resolves a magnet/wall contact with."""
        return self.magnet_elasticity * self.wall_elasticity

    @property
    def puck_capture_radius(self) -> float:
        return math.sqrt(self.goal_radius**2 - self.puck_radius**2)

    @property
    def handle_klask_radius(self) -> float:
        return self.goal_radius - self.handle_radius / 2.0

    @property
    def magnet_capture_radius(self) -> float:
        return math.sqrt(self.goal_radius**2 - self.magnet_radius**2)

    def goal_center(self, side: str) -> tuple[float, float]:
        if side == "left":
            return (-self.goal_center_x, 0.0)
        if side == "right":
            return (self.goal_center_x, 0.0)
        raise ValueError(f"unknown side {side!r}; expected 'left' or 'right'")

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.frame_skip


@dataclass(frozen=True)
class RewardConfig:
    terminal_goal: float = 12.0
    progress: float = 0.6
    # Reward for a shot whose path leads into the opponent's hole, counting
    # bank shots off the boards -- scoring off a wall is normal in Klask, and
    # rewarding only straight-on shots would teach otherwise.
    aim: float = 0.0
    aim_reflections: int = 2
    puck_position: float = 0.02
    puck_speed: float = 0.08
    contact: float = 0.08
    puck_distance: float = 0.025
    defense: float = 0.04
    own_goal_danger: float = 0.05
    magnet_attached_penalty: float = 0.05
    magnet_proximity_penalty: float = 0.01
    magnet_attach_penalty: float = 0.0
    magnet_pull_penalty: float = 0.0
    own_side_penalty: float = 0.0
    time_penalty: float = 0.0005
    action_penalty: float = 0.0003


REWARD_PROFILES: dict[str, RewardConfig] = {
    "balanced": RewardConfig(),
    "aggressive": RewardConfig(
        terminal_goal=14.0,
        progress=0.9,
        puck_position=0.04,
        puck_speed=0.14,
        contact=0.12,
        puck_distance=0.035,
        defense=0.025,
        own_goal_danger=0.035,
        time_penalty=0.0002,
        action_penalty=0.00015,
    ),
    "defensive": RewardConfig(
        terminal_goal=14.0,
        progress=0.45,
        puck_position=0.01,
        puck_speed=0.05,
        contact=0.07,
        puck_distance=0.03,
        defense=0.09,
        own_goal_danger=0.12,
        time_penalty=0.0003,
        action_penalty=0.0002,
    ),
    "possession": RewardConfig(
        terminal_goal=12.0,
        progress=0.55,
        puck_position=0.02,
        puck_speed=0.08,
        contact=0.18,
        puck_distance=0.07,
        defense=0.045,
        own_goal_danger=0.06,
        time_penalty=0.0002,
        action_penalty=0.0001,
    ),
    "simple": RewardConfig(
        terminal_goal=12.0,
        progress=0.0,
        puck_position=0.0,
        puck_speed=0.0,
        contact=0.0,
        puck_distance=0.0,
        defense=0.0,
        own_goal_danger=0.0,
        magnet_attached_penalty=0.0,
        magnet_proximity_penalty=0.0,
        magnet_attach_penalty=2.0,
        magnet_pull_penalty=0.05,
        own_side_penalty=0.01,
        time_penalty=0.0,
        action_penalty=0.0,
    ),
    # simple, rebalanced against the edge-hugging equilibrium: the pull penalty
    # is weak enough not to dominate, staying near the puck earns more than
    # hiding, and stalling with the puck on the own side is costly.
    "simple_v2": RewardConfig(
        terminal_goal=12.0,
        progress=0.0,
        puck_position=0.0,
        puck_speed=0.0,
        contact=0.0,
        puck_distance=0.02,
        defense=0.0,
        own_goal_danger=0.0,
        magnet_attached_penalty=0.0,
        magnet_proximity_penalty=0.0,
        magnet_attach_penalty=2.0,
        magnet_pull_penalty=0.01,
        own_side_penalty=0.04,
        time_penalty=0.0,
        action_penalty=0.0,
    ),
    # simple_v2 rebalanced for a puck that can come to rest. own_side is a
    # one-sided tax, so with a ball that stays where it stops the best play
    # became "knock it into their half and abandon it": both agents park, the
    # ball is never played, episodes run to the cap, and own_side (-18.5/ep
    # measured) swamps terminal_goal (+12). Cut it to a nudge. puck_distance
    # carries the load instead -- it is differenced, so being nearer the puck
    # than the opponent pays, which is what breaks a mutual-ignore standoff.
    "simple_v3": RewardConfig(
        terminal_goal=12.0,
        progress=0.0,
        puck_position=0.0,
        puck_speed=0.0,
        contact=0.0,
        puck_distance=0.05,
        defense=0.0,
        own_goal_danger=0.0,
        magnet_attached_penalty=0.0,
        magnet_proximity_penalty=0.0,
        magnet_attach_penalty=2.0,
        magnet_pull_penalty=0.01,
        own_side_penalty=0.005,
        # NOTE: time_penalty is inert. Every component except own_side is
        # differenced against the opponent's, and both agents get the same
        # time term, so it cancels to exactly zero. Making it one-sided would
        # give a real anti-stall lever, but it silently changes the four
        # legacy profiles, so it is left alone deliberately.
        time_penalty=0.0,
        action_penalty=0.0,
    ),
    # Experiment 1. simple_v2 with puck_distance removed. distance rewards
    # standing next to the puck, which directly opposes own_side's "clear it
    # off your half" -- and clearing requires striking. With a puck that can
    # rest, loitering beside it was the better deal (measured 2% strike rate on
    # resting balls). Dropping distance leaves own_side as a clean strike
    # incentive.
    "simple_v4": RewardConfig(
        terminal_goal=12.0,
        progress=0.0,
        puck_position=0.0,
        puck_speed=0.0,
        contact=0.0,
        puck_distance=0.0,
        defense=0.0,
        own_goal_danger=0.0,
        magnet_attached_penalty=0.0,
        magnet_proximity_penalty=0.0,
        magnet_attach_penalty=2.0,
        magnet_pull_penalty=0.01,
        own_side_penalty=0.04,
        time_penalty=0.0,
        action_penalty=0.0,
    ),
    # Experiment 2. simple_v4 plus explicit shot shaping: progress for driving
    # the puck downfield, aim for pointing it at the hole including bank shots.
    # progress is already speed-scaled (it is a per-step displacement) and aim
    # is scaled by puck speed, so both pay more for a decisive strike.
    "simple_v5": RewardConfig(
        terminal_goal=12.0,
        progress=0.3,
        aim=0.15,
        aim_reflections=2,
        puck_position=0.0,
        puck_speed=0.0,
        contact=0.0,
        puck_distance=0.0,
        defense=0.0,
        own_goal_danger=0.0,
        magnet_attached_penalty=0.0,
        magnet_proximity_penalty=0.0,
        magnet_attach_penalty=2.0,
        magnet_pull_penalty=0.01,
        own_side_penalty=0.04,
        time_penalty=0.0,
        action_penalty=0.0,
    ),
}


def reward_profile(name: str) -> RewardConfig:
    try:
        return REWARD_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(REWARD_PROFILES))
        raise ValueError(f"unknown reward profile {name!r}; expected one of: {known}") from exc


AGENTS: tuple[str, str] = ("left", "right")
OPPONENT: dict[str, str] = {"left": "right", "right": "left"}
MAGNET_FEATURES = 5
OBSERVATION_SIZE = 16 + ArenaConfig.magnet_count * MAGNET_FEATURES + 2
