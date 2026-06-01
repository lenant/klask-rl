from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pymunk

from klask_rl.config import AGENTS, ArenaConfig


@dataclass(frozen=True)
class PhysicsStepResult:
    scored_by: str | None
    contacts: dict[str, bool]


class KlaskPhysics:
    """Pymunk-backed puck and handle simulation.

    Coordinates are centered on the arena. The left goal is at negative x and
    the right goal is at positive x.
    """

    def __init__(self, config: ArenaConfig | None = None) -> None:
        self.config = config or ArenaConfig()
        self.space: pymunk.Space
        self.puck_body: pymunk.Body
        self.puck_shape: pymunk.Circle
        self.handle_bodies: dict[str, pymunk.Body] = {}
        self.handle_shapes: dict[str, pymunk.Circle] = {}
        self._screen: Any | None = None
        self._clock: Any | None = None
        self._surface_size = (900, int(900 * self.config.height / self.config.width))
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        cfg = self.config
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = cfg.damping
        self.handle_bodies = {}
        self.handle_shapes = {}

        self._add_walls()

        puck_moment = pymunk.moment_for_circle(cfg.puck_mass, 0.0, cfg.puck_radius)
        self.puck_body = pymunk.Body(cfg.puck_mass, puck_moment)
        self.puck_body.position = (
            rng.uniform(-0.08, 0.08),
            rng.uniform(-0.08, 0.08),
        )
        self.puck_body.velocity = (
            rng.uniform(-0.25, 0.25),
            rng.uniform(-0.25, 0.25),
        )
        self.puck_shape = pymunk.Circle(self.puck_body, cfg.puck_radius)
        self.puck_shape.elasticity = 0.9
        self.puck_shape.friction = cfg.puck_friction
        self.space.add(self.puck_body, self.puck_shape)

        self._add_handle("left", (-0.55, rng.uniform(-0.08, 0.08)))
        self._add_handle("right", (0.55, rng.uniform(-0.08, 0.08)))

    def _add_walls(self) -> None:
        cfg = self.config
        hw = cfg.half_width
        hh = cfg.half_height
        gap = cfg.goal_half_width
        static = self.space.static_body
        wall_segments = [
            ((-hw, hh), (hw, hh)),
            ((-hw, -hh), (hw, -hh)),
            ((-hw, -hh), (-hw, -gap)),
            ((-hw, gap), (-hw, hh)),
            ((hw, -hh), (hw, -gap)),
            ((hw, gap), (hw, hh)),
        ]
        for start, end in wall_segments:
            shape = pymunk.Segment(static, start, end, cfg.wall_radius)
            shape.elasticity = cfg.wall_elasticity
            shape.friction = 0.6
            self.space.add(shape)

    def _add_handle(self, agent: str, position: tuple[float, float]) -> None:
        cfg = self.config
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        shape = pymunk.Circle(body, cfg.handle_radius)
        shape.elasticity = cfg.handle_elasticity
        shape.friction = 0.8
        self.space.add(body, shape)
        self.handle_bodies[agent] = body
        self.handle_shapes[agent] = shape

    def step(self, world_actions: dict[str, np.ndarray]) -> PhysicsStepResult:
        cfg = self.config
        for agent in AGENTS:
            action = np.asarray(world_actions.get(agent, np.zeros(2)), dtype=np.float64)
            action = np.clip(action, -1.0, 1.0)
            self.handle_bodies[agent].velocity = tuple(action * cfg.max_handle_speed)

        scored_by: str | None = None
        contacts = {agent: False for agent in AGENTS}
        for _ in range(cfg.frame_skip):
            self.space.step(cfg.physics_dt)
            self._clamp_handles()
            self._limit_puck_speed()
            for agent in AGENTS:
                contacts[agent] = contacts[agent] or self._is_touching(agent)
            scored_by = self._detect_goal()
            if scored_by is not None:
                break

        return PhysicsStepResult(scored_by=scored_by, contacts=contacts)

    def _clamp_handles(self) -> None:
        cfg = self.config
        y_min = -cfg.half_height + cfg.handle_radius
        y_max = cfg.half_height - cfg.handle_radius
        x_limits = {
            "left": (-cfg.half_width + cfg.handle_radius, -cfg.handle_radius),
            "right": (cfg.handle_radius, cfg.half_width - cfg.handle_radius),
        }
        for agent, body in self.handle_bodies.items():
            x_min, x_max = x_limits[agent]
            x = float(np.clip(body.position.x, x_min, x_max))
            y = float(np.clip(body.position.y, y_min, y_max))
            if x != body.position.x or y != body.position.y:
                body.position = (x, y)
                body.velocity = (0.0, 0.0)

    def _limit_puck_speed(self) -> None:
        velocity = self.puck_body.velocity
        speed = velocity.length
        if speed > self.config.max_puck_speed:
            self.puck_body.velocity = velocity * (self.config.max_puck_speed / speed)

    def _is_touching(self, agent: str) -> bool:
        cfg = self.config
        delta = self.puck_body.position - self.handle_bodies[agent].position
        return delta.length <= cfg.puck_radius + cfg.handle_radius + 0.012

    def _detect_goal(self) -> str | None:
        cfg = self.config
        puck = self.puck_body
        x = puck.position.x
        y = puck.position.y
        if x > cfg.half_width and abs(y) <= cfg.goal_half_width:
            return "left"
        if x < -cfg.half_width and abs(y) <= cfg.goal_half_width:
            return "right"

        if x > cfg.half_width and abs(y) > cfg.goal_half_width:
            puck.position = (cfg.half_width - cfg.puck_radius, y)
            puck.velocity = (-abs(puck.velocity.x) * cfg.wall_elasticity, puck.velocity.y)
        elif x < -cfg.half_width and abs(y) > cfg.goal_half_width:
            puck.position = (-cfg.half_width + cfg.puck_radius, y)
            puck.velocity = (abs(puck.velocity.x) * cfg.wall_elasticity, puck.velocity.y)
        return None

    def snapshot(self) -> dict[str, np.ndarray]:
        return {
            "puck_pos": np.array(self.puck_body.position, dtype=np.float32),
            "puck_vel": np.array(self.puck_body.velocity, dtype=np.float32),
            "left_pos": np.array(self.handle_bodies["left"].position, dtype=np.float32),
            "left_vel": np.array(self.handle_bodies["left"].velocity, dtype=np.float32),
            "right_pos": np.array(self.handle_bodies["right"].position, dtype=np.float32),
            "right_vel": np.array(self.handle_bodies["right"].velocity, dtype=np.float32),
        }

    def render(self, mode: str = "human") -> np.ndarray | None:
        import pygame

        cfg = self.config
        width, height = self._surface_size
        if mode == "human":
            if self._screen is None:
                pygame.init()
                self._screen = pygame.display.set_mode((width, height))
                pygame.display.set_caption("klask-rl")
                self._clock = pygame.time.Clock()
            surface = self._screen
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
        else:
            surface = pygame.Surface((width, height))

        def to_screen(point: tuple[float, float] | pymunk.Vec2d) -> tuple[int, int]:
            x = int((point[0] + cfg.half_width) / cfg.width * width)
            y = int((cfg.half_height - point[1]) / cfg.height * height)
            return x, y

        def to_px(radius: float) -> int:
            return max(2, int(radius / cfg.width * width))

        surface.fill((22, 92, 98))
        pygame.draw.rect(surface, (238, 232, 212), pygame.Rect(0, 0, width, height), 3)
        goal_px = int(cfg.goal_width / cfg.height * height)
        pygame.draw.rect(
            surface,
            (235, 81, 75),
            pygame.Rect(0, height // 2 - goal_px // 2, 8, goal_px),
        )
        pygame.draw.rect(
            surface,
            (74, 126, 234),
            pygame.Rect(width - 8, height // 2 - goal_px // 2, 8, goal_px),
        )
        pygame.draw.line(surface, (200, 222, 220), (width // 2, 0), (width // 2, height), 1)
        pygame.draw.circle(surface, (245, 245, 240), to_screen(self.puck_body.position), to_px(cfg.puck_radius))
        pygame.draw.circle(
            surface,
            (235, 81, 75),
            to_screen(self.handle_bodies["left"].position),
            to_px(cfg.handle_radius),
        )
        pygame.draw.circle(
            surface,
            (74, 126, 234),
            to_screen(self.handle_bodies["right"].position),
            to_px(cfg.handle_radius),
        )

        if mode == "human":
            pygame.display.flip()
            if self._clock is not None:
                self._clock.tick(int(1.0 / cfg.control_dt))
            return None
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))

    def close(self) -> None:
        if self._screen is not None:
            import pygame

            pygame.display.quit()
            self._screen = None
            self._clock = None
