from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArenaConfig:
    width: float = 2.0
    height: float = 1.2
    goal_width: float = 0.34
    wall_radius: float = 0.018
    puck_radius: float = 0.035
    handle_radius: float = 0.07
    magnet_count: int = 3
    magnet_radius: float = 0.026
    magnet_mass: float = 0.03
    magnet_attraction_range: float = 0.22
    magnet_attraction_strength: float = 3.0
    magnet_max_force: float = 0.8
    max_magnet_speed: float = 1.5
    magnet_friction: float = 0.75
    magnet_linear_friction: float = 3.0
    magnet_elasticity: float = 0.35
    magnet_attach_frames: int = 2
    magnet_release_distance: float = 0.16
    magnet_score_threshold: int = 2
    puck_start_min_x_fraction: float = 0.125
    puck_start_max_x_fraction: float = 0.25
    puck_mass: float = 0.045
    max_handle_speed: float = 1.8
    max_puck_speed: float = 3.2
    physics_dt: float = 1.0 / 120.0
    frame_skip: int = 4
    wall_elasticity: float = 0.92
    handle_elasticity: float = 0.78
    puck_friction: float = 0.2
    damping: float = 0.995
    max_steps: int = 900

    @property
    def half_width(self) -> float:
        return self.width / 2.0

    @property
    def half_height(self) -> float:
        return self.height / 2.0

    @property
    def goal_half_width(self) -> float:
        return self.goal_width / 2.0

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.frame_skip


@dataclass(frozen=True)
class RewardConfig:
    terminal_goal: float = 12.0
    progress: float = 0.6
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
