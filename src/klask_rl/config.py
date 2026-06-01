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
    time_penalty: float = 0.0005
    action_penalty: float = 0.0003


AGENTS: tuple[str, str] = ("left", "right")
OPPONENT: dict[str, str] = {"left": "right", "right": "left"}
OBSERVATION_SIZE = 16
