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
    score_reason: str | None
    magnet_counts: dict[str, int]


RewardOverlay = dict[str, dict[str, Any]]

REWARD_COMPONENT_LABELS: tuple[tuple[str, str], ...] = (
    ("progress", "progress"),
    ("position", "position"),
    ("speed", "speed"),
    ("contact", "contact"),
    ("distance", "distance"),
    ("defense", "defense"),
    ("danger", "danger"),
    ("time", "time"),
    ("action", "action"),
    ("magnet_attach", "mag attach"),
    ("magnet_pull", "mag pull"),
    ("own_side", "own side"),
    ("terminal", "terminal"),
)


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
        self.magnet_bodies: list[pymunk.Body] = []
        self.magnet_shapes: list[pymunk.Circle] = []
        self.magnet_attached_to: list[str | None] = []
        self.magnet_attracted_to: list[str | None] = []
        self._magnet_attached_offsets: list[pymunk.Vec2d] = []
        self._magnet_contact_frames: list[dict[str, int]] = []
        self.paused = False
        self.quit_requested = False
        self._screen: Any | None = None
        self._clock: Any | None = None
        arena_width = 900
        self._panel_width = 230
        self._panel_gap = 12
        self._arena_size = (arena_width, int(arena_width * self.config.height / self.config.width))
        self._surface_size = (
            self._arena_size[0] + 2 * (self._panel_width + self._panel_gap),
            self._arena_size[1],
        )
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        cfg = self.config
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = cfg.damping
        self.handle_bodies = {}
        self.handle_shapes = {}
        self.magnet_bodies = []
        self.magnet_shapes = []
        self.magnet_attached_to = []
        self.magnet_attracted_to = []
        self._magnet_attached_offsets = []
        self._magnet_contact_frames = []
        self.paused = False
        self.quit_requested = False

        self._add_walls()

        puck_moment = pymunk.moment_for_circle(cfg.puck_mass, 0.0, cfg.puck_radius)
        self.puck_body = pymunk.Body(cfg.puck_mass, puck_moment)
        self.puck_body.position = self._sample_puck_start_position(rng)
        self.puck_body.velocity = (
            rng.uniform(-0.25, 0.25),
            rng.uniform(-0.25, 0.25),
        )
        self.puck_shape = pymunk.Circle(self.puck_body, cfg.puck_radius)
        self.puck_shape.elasticity = 0.9
        self.puck_shape.friction = cfg.puck_friction
        self.space.add(self.puck_body, self.puck_shape)

        self._add_handle("left", self._sample_handle_start_position("left", rng))
        self._add_handle("right", self._sample_handle_start_position("right", rng))
        for position in self._magnet_start_positions():
            self._add_magnet(position)

    def _sample_puck_start_position(self, rng: np.random.Generator) -> tuple[float, float]:
        cfg = self.config
        min_x = cfg.width * cfg.puck_start_min_x_fraction
        max_x = cfg.width * cfg.puck_start_max_x_fraction
        side = -1.0 if rng.random() < 0.5 else 1.0
        return (
            side * rng.uniform(min_x, max_x),
            rng.uniform(-0.08, 0.08),
        )

    def _sample_handle_start_position(self, agent: str, rng: np.random.Generator) -> tuple[float, float]:
        cfg = self.config
        preferred_x = -0.55 if agent == "left" else 0.55
        preferred = pymunk.Vec2d(preferred_x, rng.uniform(-0.08, 0.08))
        safe_position = self._non_overlapping_handle_position(
            agent=agent,
            current_position=preferred,
            preferred_position=preferred,
            puck_position=self.puck_body.position,
            min_distance=cfg.puck_radius + cfg.handle_radius + 0.02,
        )
        return (safe_position.x, safe_position.y)

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

    def _magnet_start_positions(self) -> list[tuple[float, float]]:
        positions = [(0.0, -0.24), (0.0, 0.0), (0.0, 0.24)]
        return positions[: self.config.magnet_count]

    def _add_magnet(self, position: tuple[float, float]) -> None:
        cfg = self.config
        moment = pymunk.moment_for_circle(cfg.magnet_mass, 0.0, cfg.magnet_radius)
        body = pymunk.Body(cfg.magnet_mass, moment)
        body.position = position
        shape = pymunk.Circle(body, cfg.magnet_radius)
        shape.elasticity = cfg.magnet_elasticity
        shape.friction = cfg.magnet_friction
        self.space.add(body, shape)
        self.magnet_bodies.append(body)
        self.magnet_shapes.append(shape)
        self.magnet_attached_to.append(None)
        self.magnet_attracted_to.append(None)
        self._magnet_attached_offsets.append(pymunk.Vec2d.zero())
        self._magnet_contact_frames.append({agent: 0 for agent in AGENTS})

    def step(self, world_actions: dict[str, np.ndarray]) -> PhysicsStepResult:
        cfg = self.config
        for agent in AGENTS:
            action = np.asarray(world_actions.get(agent, np.zeros(2)), dtype=np.float64)
            action = np.clip(action, -1.0, 1.0)
            self.handle_bodies[agent].velocity = tuple(action * cfg.max_handle_speed)

        scored_by: str | None = None
        score_reason: str | None = None
        contacts = {agent: False for agent in AGENTS}
        for _ in range(cfg.frame_skip):
            self._apply_magnet_forces()
            self._pin_attached_magnets()
            self.space.step(cfg.physics_dt)
            self._clamp_handles()
            self._pin_attached_magnets()
            self._contain_puck()
            self._contain_magnets()
            scored_by = self._detect_goal()
            if scored_by is not None:
                score_reason = "goal"
                break
            self._separate_handles_from_puck()
            self._separate_handles_from_magnets()
            self._contain_magnets()
            self._update_magnet_attachment_state()
            self._pin_attached_magnets()
            scored_by = self._detect_magnet_score()
            if scored_by is not None:
                score_reason = "magnets"
                break
            self._apply_magnet_friction()
            self._limit_puck_speed()
            self._limit_magnet_speeds()
            for agent in AGENTS:
                contacts[agent] = contacts[agent] or self._is_touching(agent)

        return PhysicsStepResult(
            scored_by=scored_by,
            contacts=contacts,
            score_reason=score_reason,
            magnet_counts=self.magnet_attachment_counts(),
        )

    def _magnet_attraction_force(self, distance: float) -> float:
        cfg = self.config
        if distance <= 1e-9 or distance >= cfg.magnet_attraction_range:
            return 0.0
        closeness = 1.0 - distance / cfg.magnet_attraction_range
        return min(cfg.magnet_max_force, cfg.magnet_attraction_strength * closeness * closeness)

    def _apply_magnet_forces(self) -> None:
        self.magnet_attracted_to = [None for _ in self.magnet_bodies]
        for index, magnet_body in enumerate(self.magnet_bodies):
            if self.magnet_attached_to[index] is not None:
                self.magnet_attracted_to[index] = self.magnet_attached_to[index]
                continue
            strongest_agent: str | None = None
            strongest_force = 0.0
            for agent, handle_body in self.handle_bodies.items():
                delta = handle_body.position - magnet_body.position
                distance = delta.length
                magnitude = self._magnet_attraction_force(distance)
                if magnitude <= 0.0:
                    continue
                normal = delta / distance
                force = normal * magnitude
                magnet_body.apply_force_at_world_point(force, magnet_body.position)
                if magnitude > strongest_force:
                    strongest_force = magnitude
                    strongest_agent = agent
            self.magnet_attracted_to[index] = strongest_agent

    def _magnet_bounds(self) -> tuple[float, float, float, float]:
        cfg = self.config
        return (
            -cfg.half_width + cfg.magnet_radius,
            cfg.half_width - cfg.magnet_radius,
            -cfg.half_height + cfg.magnet_radius,
            cfg.half_height - cfg.magnet_radius,
        )

    def _clamped_magnet_position(self, position: pymunk.Vec2d) -> pymunk.Vec2d:
        x_min, x_max, y_min, y_max = self._magnet_bounds()
        return pymunk.Vec2d(
            float(np.clip(position.x, x_min, x_max)),
            float(np.clip(position.y, y_min, y_max)),
        )

    def _pin_attached_magnets(self) -> None:
        for index, attached_agent in enumerate(self.magnet_attached_to):
            if attached_agent is None:
                continue
            handle_body = self.handle_bodies[attached_agent]
            magnet_body = self.magnet_bodies[index]
            target = self._clamped_magnet_position(handle_body.position + self._magnet_attached_offsets[index])
            magnet_body.position = target
            magnet_body.velocity = handle_body.velocity
            magnet_body.angular_velocity = 0.0

    def _contain_magnets(self) -> None:
        cfg = self.config
        x_min, x_max, y_min, y_max = self._magnet_bounds()
        for index, body in enumerate(self.magnet_bodies):
            if self.magnet_attached_to[index] is not None:
                continue
            x = body.position.x
            y = body.position.y
            vx = body.velocity.x
            vy = body.velocity.y
            changed = False

            if x > x_max:
                x = x_max
                vx = -abs(vx) * cfg.magnet_elasticity
                changed = True
            elif x < x_min:
                x = x_min
                vx = abs(vx) * cfg.magnet_elasticity
                changed = True

            if y > y_max:
                y = y_max
                vy = -abs(vy) * cfg.magnet_elasticity
                changed = True
            elif y < y_min:
                y = y_min
                vy = abs(vy) * cfg.magnet_elasticity
                changed = True

            if changed:
                body.position = (x, y)
                body.velocity = (vx, vy)

    def _separate_handles_from_magnets(self) -> None:
        cfg = self.config
        min_distance = cfg.magnet_radius + cfg.handle_radius + 1e-6
        for index, magnet_body in enumerate(self.magnet_bodies):
            if self.magnet_attached_to[index] is not None:
                continue
            for agent, handle_body in self.handle_bodies.items():
                delta = magnet_body.position - handle_body.position
                distance = delta.length
                if distance >= min_distance:
                    continue

                if distance > 1e-9:
                    normal = delta / distance
                else:
                    fallback_x = -1.0 if agent == "left" else 1.0
                    normal = pymunk.Vec2d(fallback_x, 0.0)

                magnet_body.position = handle_body.position + normal * min_distance
                inward_speed = magnet_body.velocity.dot(normal)
                if inward_speed < 0.0:
                    magnet_body.velocity = magnet_body.velocity - normal * inward_speed

    def _update_magnet_attachment_state(self) -> None:
        cfg = self.config
        attach_distance = cfg.handle_radius + cfg.magnet_radius + 0.008
        for index, magnet_body in enumerate(self.magnet_bodies):
            attached_agent = self.magnet_attached_to[index]
            if attached_agent is not None:
                self._magnet_contact_frames[index][attached_agent] = cfg.magnet_attach_frames
                continue

            candidate: str | None = None
            candidate_distance = float("inf")
            for agent, handle_body in self.handle_bodies.items():
                distance = (magnet_body.position - handle_body.position).length
                if distance <= attach_distance:
                    self._magnet_contact_frames[index][agent] += 1
                    if (
                        self._magnet_contact_frames[index][agent] >= cfg.magnet_attach_frames
                        and distance < candidate_distance
                    ):
                        candidate = agent
                        candidate_distance = distance
                else:
                    self._magnet_contact_frames[index][agent] = 0
            if candidate is not None:
                self.magnet_attached_to[index] = candidate
                self._magnet_attached_offsets[index] = self._sticky_offset(
                    self.magnet_bodies[index].position - self.handle_bodies[candidate].position,
                    candidate,
                )

    def _sticky_offset(self, offset: pymunk.Vec2d, agent: str) -> pymunk.Vec2d:
        distance = offset.length
        if distance > 1e-9:
            normal = offset / distance
        else:
            normal = pymunk.Vec2d(-1.0 if agent == "left" else 1.0, 0.0)
        return normal * (self.config.handle_radius + self.config.magnet_radius)

    def magnet_attachment_counts(self) -> dict[str, int]:
        return {
            agent: sum(1 for attached_agent in self.magnet_attached_to if attached_agent == agent)
            for agent in AGENTS
        }

    def magnet_risk(self, agent: str) -> dict[str, float]:
        cfg = self.config
        attached = float(sum(1 for attached_agent in self.magnet_attached_to if attached_agent == agent))
        proximity = 0.0
        handle_position = self.handle_bodies[agent].position
        for index, magnet_body in enumerate(self.magnet_bodies):
            if self.magnet_attached_to[index] is not None:
                continue
            distance = (magnet_body.position - handle_position).length
            if distance < cfg.magnet_attraction_range:
                proximity += 1.0 - distance / cfg.magnet_attraction_range
        return {"attached": attached, "proximity": proximity}

    def _detect_magnet_score(self) -> str | None:
        counts = self.magnet_attachment_counts()
        threshold = self.config.magnet_score_threshold
        if counts["left"] >= threshold and counts["left"] >= counts["right"]:
            return "right"
        if counts["right"] >= threshold:
            return "left"
        return None

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

    def _limit_magnet_speeds(self) -> None:
        max_speed = self.config.max_magnet_speed
        for index, body in enumerate(self.magnet_bodies):
            if self.magnet_attached_to[index] is not None:
                continue
            speed = body.velocity.length
            if speed > max_speed:
                body.velocity = body.velocity * (max_speed / speed)

    def _apply_magnet_friction(self) -> None:
        multiplier = max(0.0, 1.0 - self.config.magnet_linear_friction * self.config.physics_dt)
        for index, body in enumerate(self.magnet_bodies):
            if self.magnet_attached_to[index] is not None:
                continue
            body.velocity = body.velocity * multiplier
            body.angular_velocity *= multiplier

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
            "magnet_pos": np.array([body.position for body in self.magnet_bodies], dtype=np.float32),
            "magnet_vel": np.array([body.velocity for body in self.magnet_bodies], dtype=np.float32),
        }

    def render(self, mode: str = "human", reward_overlay: RewardOverlay | None = None) -> np.ndarray | None:
        import pygame

        cfg = self.config
        width, height = self._surface_size
        arena_width, arena_height = self._arena_size
        arena_left = self._panel_width + self._panel_gap
        if mode == "human":
            if self._screen is None:
                pygame.init()
                self._screen = pygame.display.set_mode((width, height))
                pygame.display.set_caption("klask-rl")
                self._clock = pygame.time.Clock()
            surface = self._screen
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit_requested = True
                    self.close()
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.quit_requested = True
                    self.close()
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.paused = not self.paused
        else:
            surface = pygame.Surface((width, height))

        def to_screen(point: tuple[float, float] | pymunk.Vec2d) -> tuple[int, int]:
            x = arena_left + int((point[0] + cfg.half_width) / cfg.width * arena_width)
            y = int((cfg.half_height - point[1]) / cfg.height * arena_height)
            return x, y

        def to_px(radius: float) -> int:
            return max(2, int(radius / cfg.width * arena_width))

        surface.fill((14, 18, 24))
        arena_rect = pygame.Rect(arena_left, 0, arena_width, arena_height)
        pygame.draw.rect(surface, (22, 92, 98), arena_rect)
        pygame.draw.rect(surface, (238, 232, 212), arena_rect, 3)
        goal_px = int(cfg.goal_width / cfg.height * arena_height)
        pygame.draw.rect(
            surface,
            (235, 81, 75),
            pygame.Rect(arena_left, arena_height // 2 - goal_px // 2, 8, goal_px),
        )
        pygame.draw.rect(
            surface,
            (74, 126, 234),
            pygame.Rect(arena_left + arena_width - 8, arena_height // 2 - goal_px // 2, 8, goal_px),
        )
        pygame.draw.line(
            surface,
            (200, 222, 220),
            (arena_left + arena_width // 2, 0),
            (arena_left + arena_width // 2, arena_height),
            1,
        )
        for index, magnet_body in enumerate(self.magnet_bodies):
            owner = self.magnet_attached_to[index] or self.magnet_attracted_to[index]
            outline = (218, 224, 222)
            if owner == "left":
                outline = (255, 128, 122)
            elif owner == "right":
                outline = (120, 162, 255)
            center = to_screen(magnet_body.position)
            radius = to_px(cfg.magnet_radius)
            pygame.draw.circle(surface, (226, 230, 224), center, radius)
            pygame.draw.circle(surface, outline, center, radius + 3, 2)
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
        if reward_overlay is not None:
            self._draw_reward_overlay(pygame, surface, reward_overlay)
        if self.paused:
            self._draw_pause_indicator(pygame, surface)

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
            self.paused = False

    def _draw_reward_overlay(self, pygame: Any, surface: Any, reward_overlay: RewardOverlay) -> None:
        pygame.font.init()
        height = self._surface_size[1]
        panel_height = height - 32
        panel_y = (height - panel_height) // 2
        left_rect = pygame.Rect(8, panel_y, self._panel_width - 16, panel_height)
        right_rect = pygame.Rect(
            self._surface_size[0] - self._panel_width + 8,
            panel_y,
            self._panel_width - 16,
            panel_height,
        )
        title_font = pygame.font.SysFont("Arial", 18, bold=True)
        label_font = pygame.font.SysFont("Arial", 13)
        value_font = pygame.font.SysFont("Arial", 13, bold=True)
        total_font = pygame.font.SysFont("Arial", 18, bold=True)

        def draw_panel(agent: str, rect: Any, color: tuple[int, int, int]) -> None:
            values = reward_overlay.get(agent, {})
            step_reward = float(values.get("step", 0.0))
            episode_reward = float(values.get("episode", 0.0))
            components = values.get("components", {})
            pygame.draw.rect(surface, (30, 36, 42), rect, border_radius=6)
            pygame.draw.rect(surface, color, rect, width=2, border_radius=6)

            title = title_font.render(agent.upper(), True, color)
            step_label = label_font.render("step reward", True, (190, 200, 204))
            episode_label = label_font.render("episode total", True, (190, 200, 204))
            step_value = total_font.render(f"{step_reward:+.3f}", True, (242, 245, 242))
            episode_value = total_font.render(f"{episode_reward:+.2f}", True, (242, 245, 242))

            surface.blit(title, (rect.x + 14, rect.y + 12))
            surface.blit(step_label, (rect.x + 14, rect.y + 42))
            surface.blit(step_value, (rect.x + 14, rect.y + 58))
            surface.blit(episode_label, (rect.x + 14, rect.y + 84))
            surface.blit(episode_value, (rect.x + 14, rect.y + 100))

            divider_y = rect.y + 132
            pygame.draw.line(surface, (76, 86, 94), (rect.x + 12, divider_y), (rect.right - 12, divider_y), 1)
            row_y = divider_y + 14
            row_height = 27
            for component, label in REWARD_COMPONENT_LABELS:
                value = float(components.get(component, 0.0)) if isinstance(components, dict) else 0.0
                label_surface = label_font.render(label, True, (184, 194, 200))
                value_surface = value_font.render(f"{value:+.3f}", True, (238, 241, 238))
                surface.blit(label_surface, (rect.x + 14, row_y))
                surface.blit(value_surface, (rect.right - value_surface.get_width() - 14, row_y))
                row_y += row_height

        draw_panel("left", left_rect, (235, 81, 75))
        draw_panel("right", right_rect, (74, 126, 234))

    def _draw_pause_indicator(self, pygame: Any, surface: Any) -> None:
        font = pygame.font.SysFont("Arial", 28, bold=True)
        label = font.render("PAUSED", True, (242, 245, 242))
        padding_x = 20
        padding_y = 10
        rect = label.get_rect()
        rect.center = (self._surface_size[0] // 2, 34)
        background = rect.inflate(padding_x * 2, padding_y * 2)
        pygame.draw.rect(surface, (18, 22, 28), background, border_radius=6)
        pygame.draw.rect(surface, (238, 232, 212), background, width=1, border_radius=6)
        surface.blit(label, rect)
