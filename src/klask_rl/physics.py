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
            self._contain_puck()
            scored_by = self._detect_goal()
            if scored_by is not None:
                break
            self._separate_handles_from_puck()
            self._limit_puck_speed()
            for agent in AGENTS:
                contacts[agent] = contacts[agent] or self._is_touching(agent)

        return PhysicsStepResult(scored_by=scored_by, contacts=contacts)

    def _handle_bounds(self, agent: str) -> tuple[float, float, float, float]:
        cfg = self.config
        y_min = -cfg.half_height + cfg.handle_radius
        y_max = cfg.half_height - cfg.handle_radius
        x_limits = {
            "left": (-cfg.half_width + cfg.handle_radius, -cfg.handle_radius),
            "right": (cfg.handle_radius, cfg.half_width - cfg.handle_radius),
        }
        x_min, x_max = x_limits[agent]
        return x_min, x_max, y_min, y_max

    def _clamped_handle_position(self, agent: str, position: pymunk.Vec2d) -> pymunk.Vec2d:
        x_min, x_max, y_min, y_max = self._handle_bounds(agent)
        return pymunk.Vec2d(
            float(np.clip(position.x, x_min, x_max)),
            float(np.clip(position.y, y_min, y_max)),
        )

    def _non_overlapping_handle_position(
        self,
        agent: str,
        current_position: pymunk.Vec2d,
        preferred_position: pymunk.Vec2d,
        puck_position: pymunk.Vec2d,
        min_distance: float,
    ) -> pymunk.Vec2d:
        x_min, x_max, y_min, y_max = self._handle_bounds(agent)
        min_distance_sq = min_distance * min_distance
        candidates = [
            self._clamped_handle_position(agent, preferred_position),
            pymunk.Vec2d(x_min, y_min),
            pymunk.Vec2d(x_min, y_max),
            pymunk.Vec2d(x_max, y_min),
            pymunk.Vec2d(x_max, y_max),
        ]

        for x in (x_min, x_max):
            dx = x - puck_position.x
            remaining = min_distance_sq - dx * dx
            if remaining >= 0.0:
                dy = float(np.sqrt(remaining))
                for y in (puck_position.y - dy, puck_position.y + dy):
                    if y_min <= y <= y_max:
                        candidates.append(pymunk.Vec2d(x, y))

        for y in (y_min, y_max):
            dy = y - puck_position.y
            remaining = min_distance_sq - dy * dy
            if remaining >= 0.0:
                dx = float(np.sqrt(remaining))
                for x in (puck_position.x - dx, puck_position.x + dx):
                    if x_min <= x <= x_max:
                        candidates.append(pymunk.Vec2d(x, y))

        valid_candidates = [
            candidate
            for candidate in candidates
            if (candidate - puck_position).length >= min_distance - 1e-9
        ]
        if valid_candidates:
            return min(valid_candidates, key=lambda candidate: (candidate - current_position).length)

        return max(candidates, key=lambda candidate: (candidate - puck_position).length)

    def _clamp_handles(self) -> None:
        for agent, body in self.handle_bodies.items():
            clamped = self._clamped_handle_position(agent, body.position)
            if clamped.x != body.position.x or clamped.y != body.position.y:
                body.position = clamped
                body.velocity = (0.0, 0.0)

    def _contain_puck(self) -> None:
        cfg = self.config
        puck = self.puck_body
        x = puck.position.x
        y = puck.position.y
        vx = puck.velocity.x
        vy = puck.velocity.y
        y_min = -cfg.half_height + cfg.puck_radius
        y_max = cfg.half_height - cfg.puck_radius
        changed = False

        if y > y_max:
            y = y_max
            vy = -abs(vy) * cfg.wall_elasticity
            changed = True
        elif y < y_min:
            y = y_min
            vy = abs(vy) * cfg.wall_elasticity
            changed = True

        if changed:
            puck.position = (x, y)
            puck.velocity = (vx, vy)

    def _separate_handles_from_puck(self) -> None:
        cfg = self.config
        min_distance = cfg.puck_radius + cfg.handle_radius + 1e-6
        puck_position = self.puck_body.position
        for agent, body in self.handle_bodies.items():
            delta = body.position - puck_position
            distance = delta.length
            if distance >= min_distance:
                continue

            if distance > 1e-9:
                normal = delta / distance
            else:
                fallback_x = -1.0 if agent == "left" else 1.0
                normal = pymunk.Vec2d(fallback_x, 0.0)

            body.position = self._non_overlapping_handle_position(
                agent=agent,
                current_position=body.position,
                preferred_position=puck_position + normal * min_distance,
                puck_position=puck_position,
                min_distance=min_distance,
            )

            corrected_delta = body.position - puck_position
            corrected_distance = corrected_delta.length
            if corrected_distance > 1e-9:
                corrected_normal = corrected_delta / corrected_distance
                inward_speed = body.velocity.dot(corrected_normal)
                if inward_speed < 0.0:
                    body.velocity = body.velocity - corrected_normal * inward_speed

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
