from __future__ import annotations

from dataclasses import replace

import numpy as np
import pymunk
import pytest

from klask_rl.config import ArenaConfig
from klask_rl.physics import KlaskPhysics


def _steps_for(physics: KlaskPhysics, substeps: int) -> int:
    """Control steps needed to accumulate `substeps` physics substeps."""
    return -(-substeps // physics.config.frame_skip) + 1


def _steps_to_full_speed(cfg: ArenaConfig) -> int:
    """Control steps for a handle to ramp from rest to its speed cap."""
    return -(-int(cfg.max_handle_speed / (cfg.max_handle_acceleration * cfg.control_dt)) // 1) + 1


def test_puck_in_right_hole_scores_left() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=1)
    hole = physics.config.goal_center("right")
    physics.puck_body.position = hole
    physics.puck_body.velocity = (0.0, 0.0)
    result = physics.step({"left": np.zeros(2), "right": np.zeros(2)})
    assert result.scored_by == "left"
    assert result.score_reason == "goal"
    assert np.allclose(np.array(physics.puck_body.position), np.array(hole))


def test_puck_rolling_into_own_hole_scores_opponent() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=1)
    cfg = physics.config
    hole = cfg.goal_center("left")
    physics.puck_body.position = (hole[0] + cfg.puck_capture_radius + 0.02, 0.0)
    physics.puck_body.velocity = (-1.0, 0.0)
    physics.handle_bodies["left"].position = (-0.4, 0.4)
    result = physics.step({"left": np.zeros(2), "right": np.zeros(2)})
    assert result.scored_by == "right"
    assert result.score_reason == "goal"


def test_goal_radius_override_widens_capture_zone() -> None:
    offset = 0.12

    physics = KlaskPhysics()
    physics.reset(seed=1)
    hole = physics.config.goal_center("right")
    physics.puck_body.position = (hole[0] - offset, hole[1])
    physics.puck_body.velocity = (0.0, 0.0)
    result = physics.step({"left": np.zeros(2), "right": np.zeros(2)})
    assert result.scored_by is None

    physics = KlaskPhysics(replace(ArenaConfig(), goal_radius=0.15))
    physics.reset(seed=1)
    hole = physics.config.goal_center("right")
    physics.puck_body.position = (hole[0] - offset, hole[1])
    physics.puck_body.velocity = (0.0, 0.0)
    result = physics.step({"left": np.zeros(2), "right": np.zeros(2)})
    assert result.scored_by == "left"
    assert result.score_reason == "goal"


def test_handle_in_own_hole_scores_opponent() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=1)
    physics.handle_bodies["left"].position = physics.config.goal_center("left")
    result = physics.step({"left": np.zeros(2), "right": np.zeros(2)})
    assert result.scored_by == "right"
    assert result.score_reason == "klask"


def test_handle_can_hover_at_hole_edge_without_klask() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=1)
    cfg = physics.config
    hole = cfg.goal_center("left")
    physics.handle_bodies["left"].position = (hole[0] + cfg.handle_klask_radius + 0.01, 0.0)
    for _ in range(10):
        result = physics.step({"left": np.zeros(2), "right": np.zeros(2)})
        assert result.scored_by is None


def test_puck_bounces_off_solid_end_wall_at_former_gate_center() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=1)
    cfg = physics.config
    physics.puck_body.position = (cfg.half_width - cfg.puck_radius - 0.005, 0.0)
    physics.puck_body.velocity = (2.0, 0.0)
    result = physics.step({"left": np.zeros(2), "right": np.zeros(2)})
    assert result.scored_by is None
    assert physics.puck_body.position.x <= cfg.half_width - cfg.puck_radius + 1e-6
    assert physics.puck_body.velocity.x <= 0.0


def test_stuck_puck_is_served_back_into_play() -> None:
    """A puck at rest in the far half is unreachable: handles cannot cross."""
    physics = KlaskPhysics()
    physics.reset(seed=1)
    cfg = physics.config
    stuck = (0.85, 0.6)
    physics.puck_body.position = stuck
    physics.puck_body.velocity = (0.0, 0.0)
    physics.handle_bodies["left"].position = (-0.5, 0.0)
    physics.handle_bodies["right"].position = (0.5, -0.6)

    for _ in range(cfg.dead_ball_steps - 1):
        result = physics.step({"left": np.zeros(2), "right": np.zeros(2)})
        assert not result.served
    assert np.allclose(np.array(physics.puck_body.position), np.array(stuck))

    result = physics.step({"left": np.zeros(2), "right": np.zeros(2)})
    assert result.served
    assert physics.serves == 1
    assert (physics.puck_body.position - pymunk.Vec2d(*stuck)).length > 0.1
    assert physics.puck_body.velocity.length > 0.0
    for body in physics.handle_bodies.values():
        separation = (body.position - physics.puck_body.position).length
        assert separation > cfg.puck_radius + cfg.handle_radius


def test_moving_puck_is_never_served() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=2)
    physics.puck_body.position = (0.0, 0.55)
    physics.puck_body.velocity = (0.9, 0.0)
    physics.handle_bodies["left"].position = (-0.9, -0.6)
    physics.handle_bodies["right"].position = (0.9, -0.6)

    for _ in range(physics.config.dead_ball_steps + 20):
        result = physics.step({"left": np.zeros(2), "right": np.zeros(2)})
        if result.scored_by is not None:
            break
        if physics.puck_body.velocity.length == 0.0:
            break
        assert not result.served


def test_handle_accelerates_instead_of_jumping_to_speed() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=1)
    cfg = physics.config
    # Drive along y, which has no half-board clamp to zero the velocity.
    physics.handle_bodies["left"].position = (-0.5, -0.3)
    physics.puck_body.position = (0.5, 0.55)
    physics.puck_body.velocity = (0.0, 0.0)

    forward = np.array([0.0, 1.0])
    physics.step({"left": forward, "right": np.zeros(2)})
    first = physics.handle_bodies["left"].velocity.y
    assert 0.0 < first < cfg.max_handle_speed

    for _ in range(_steps_to_full_speed(cfg)):
        physics.step({"left": forward, "right": np.zeros(2)})
    assert physics.handle_bodies["left"].velocity.y == pytest.approx(cfg.max_handle_speed, abs=1e-6)

    # Reversing has to bleed through zero rather than flipping sign in one step.
    physics.step({"left": -forward, "right": np.zeros(2)})
    assert physics.handle_bodies["left"].velocity.y > 0.0


def test_handle_slides_along_a_boundary_it_is_pushed_into() -> None:
    """Only the blocked axis may be cancelled at a bound.

    Zeroing the whole velocity pins the handle: under the acceleration limit it
    rebuilds a fraction of its speed per substep and loses it again on contact,
    so holding a diagonal into a wall stops it dead instead of sliding.
    """
    cfg = ArenaConfig()
    diagonal = float(np.sqrt(0.5))
    for position, action in (
        ((-cfg.half_width + cfg.handle_radius, 0.0), np.array([-diagonal, diagonal])),
        ((-cfg.handle_radius, 0.0), np.array([diagonal, diagonal])),
        ((-0.5, cfg.half_height - cfg.handle_radius), np.array([-diagonal, diagonal])),
    ):
        physics = KlaskPhysics(cfg)
        physics.reset(seed=1)
        physics.puck_body.position = (0.5, -0.5)
        physics.puck_body.velocity = (0.0, 0.0)
        for index, body in enumerate(physics.magnet_bodies):
            body.position = (0.3 + 0.25 * index, 0.6)
            body.velocity = (0.0, 0.0)
        physics.handle_bodies["left"].position = position
        physics.handle_bodies["left"].velocity = (0.0, 0.0)

        start = physics.handle_bodies["left"].position
        steps = 3 * _steps_to_full_speed(cfg)
        for _ in range(steps):
            physics.step({"left": action, "right": np.zeros(2)})
        travelled = (physics.handle_bodies["left"].position - start).length
        floor = 0.25 * cfg.max_handle_speed * steps * cfg.control_dt
        assert travelled > floor, f"handle stuck at boundary {position}: moved {travelled:.3f}"


def test_shot_power_scales_with_run_up() -> None:
    """Every contact used to be a full-power shot; a short run-up must be softer."""

    def strike(gap: float) -> float:
        physics = KlaskPhysics()
        physics.reset(seed=1)
        cfg = physics.config
        physics.handle_bodies["right"].position = (0.9, -0.6)
        for index, body in enumerate(physics.magnet_bodies):
            body.position = (-0.3 + 0.3 * index, -0.65)
            body.velocity = (0.0, 0.0)
        puck_x = -0.3
        physics.puck_body.position = (puck_x, 0.55)
        physics.puck_body.velocity = (0.0, 0.0)
        contact = cfg.puck_radius + cfg.handle_radius
        physics.handle_bodies["left"].position = (puck_x - contact - gap, 0.55)
        physics.handle_bodies["left"].velocity = (0.0, 0.0)
        for _ in range(40):
            physics.step({"left": np.array([1.0, 0.0]), "right": np.zeros(2)})
            if physics.puck_body.velocity.length > 1e-6:
                return physics.puck_body.velocity.length
        raise AssertionError("handle never reached the puck")

    nudge = strike(0.01)
    full = strike(0.4)
    assert nudge < full * 0.6
    assert full > ArenaConfig().max_puck_speed * 0.7


def test_puck_loses_speed_bouncing_off_a_wall() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=1)
    physics.handle_bodies["left"].position = (-0.9, -0.6)
    physics.handle_bodies["right"].position = (0.9, -0.6)
    # A lane clear of the goal holes, which sit on y = 0.
    physics.puck_body.position = (0.0, 0.55)
    physics.puck_body.velocity = (2.0, 0.0)

    incoming = 2.0
    for _ in range(200):
        before = physics.puck_body.velocity.x
        physics.step({"left": np.zeros(2), "right": np.zeros(2)})
        after = physics.puck_body.velocity.x
        if after < 0.0 <= before:
            assert abs(after) < incoming * 0.7
            return
        incoming = abs(after)
    raise AssertionError("puck never bounced off the end wall")


def test_rolling_puck_comes_to_rest() -> None:
    # No re-serve: this is about friction, and a puck left at rest long enough
    # would otherwise be put back into play and be moving again at the end.
    physics = KlaskPhysics(replace(ArenaConfig(), dead_ball_steps=0))
    physics.reset(seed=1)
    cfg = physics.config
    physics.handle_bodies["left"].position = (-0.9, -0.6)
    physics.handle_bodies["right"].position = (0.9, -0.6)
    physics.puck_body.position = (0.0, 0.55)
    physics.puck_body.velocity = (cfg.max_puck_speed * 0.3, 0.0)

    speeds = []
    roll_out = physics._stopping_distance(cfg.max_puck_speed * 0.3) / (cfg.max_puck_speed * 0.05)
    for _ in range(int(roll_out / cfg.control_dt) + 200):
        physics.step({"left": np.zeros(2), "right": np.zeros(2)})
        speeds.append(physics.puck_body.velocity.length)

    assert speeds[0] < 0.4
    assert speeds[-1] == 0.0


def test_handle_is_clamped_to_own_half() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=2)
    physics.handle_bodies["left"].position = (0.5, 0.0)
    physics.step({"left": np.array([1.0, 0.0]), "right": np.zeros(2)})
    assert physics.handle_bodies["left"].position.x <= -physics.config.handle_radius


def test_puck_speed_is_capped() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=3)
    physics.puck_body.velocity = (100.0, 0.0)
    physics.step({"left": np.zeros(2), "right": np.zeros(2)})
    assert physics.puck_body.velocity.length <= physics.config.max_puck_speed + 1e-6


def test_puck_is_contained_by_top_and_bottom_walls() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=4)

    physics.puck_body.position = (0.0, physics.config.half_height + 0.2)
    physics.puck_body.velocity = (0.0, 1.0)
    physics.step({"left": np.zeros(2), "right": np.zeros(2)})
    assert physics.puck_body.position.y <= physics.config.half_height - physics.config.puck_radius
    assert physics.puck_body.velocity.y <= 0.0

    physics.puck_body.position = (0.0, -physics.config.half_height - 0.2)
    physics.puck_body.velocity = (0.0, -1.0)
    physics.step({"left": np.zeros(2), "right": np.zeros(2)})
    assert physics.puck_body.position.y >= -physics.config.half_height + physics.config.puck_radius
    assert physics.puck_body.velocity.y >= 0.0


def test_handle_cannot_overlap_puck_when_pinning_corner() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=5)
    cfg = physics.config
    physics.puck_body.position = (
        -cfg.half_width + cfg.puck_radius + 0.02,
        cfg.half_height - cfg.puck_radius - 0.02,
    )
    physics.puck_body.velocity = (0.0, 0.0)
    physics.handle_bodies["left"].position = (-0.75, 0.35)

    min_distance = cfg.puck_radius + cfg.handle_radius
    for _ in range(80):
        physics.step({"left": np.array([-1.0, 1.0]), "right": np.zeros(2)})
        separation = (physics.handle_bodies["left"].position - physics.puck_body.position).length
        assert separation >= min_distance - 1e-6
        assert physics.puck_body.position.y <= cfg.half_height - cfg.puck_radius


def test_handle_overlap_is_resolved_when_puck_blocks_corner_escape_direction() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=6)
    cfg = physics.config
    physics.puck_body.position = (-0.9, 0.5)
    physics.puck_body.velocity = (0.0, 0.0)
    physics.handle_bodies["left"].position = (
        -cfg.half_width + cfg.handle_radius,
        cfg.half_height - cfg.handle_radius,
    )

    physics.step({"left": np.zeros(2), "right": np.zeros(2)})

    separation = (physics.handle_bodies["left"].position - physics.puck_body.position).length
    assert separation >= cfg.puck_radius + cfg.handle_radius - 1e-6


def test_space_toggles_pause_in_human_render(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame

    physics = KlaskPhysics()
    physics.render("human")

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
    physics.render("human")
    assert physics.paused

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
    physics.render("human")
    assert not physics.paused

    physics.close()


def test_magnets_are_created_inside_arena() -> None:
    physics = KlaskPhysics()
    cfg = physics.config

    assert len(physics.magnet_bodies) == cfg.magnet_count
    for body in physics.magnet_bodies:
        assert -cfg.half_width + cfg.magnet_radius <= body.position.x <= cfg.half_width - cfg.magnet_radius
        assert -cfg.half_height + cfg.magnet_radius <= body.position.y <= cfg.half_height - cfg.magnet_radius


def test_magnets_start_on_center_line() -> None:
    physics = KlaskPhysics()
    spacing = physics.config.half_height / 2.0

    assert [body.position.x for body in physics.magnet_bodies] == [0.0, 0.0, 0.0]
    assert [body.position.y for body in physics.magnet_bodies] == [-spacing, 0.0, spacing]
    assert spacing <= physics.config.half_height - physics.config.magnet_radius


def test_puck_starts_on_left_or_right_quarter_without_handle_overlap() -> None:
    physics = KlaskPhysics()
    cfg = physics.config
    min_x = cfg.width * cfg.puck_start_min_x_fraction
    max_x = cfg.width * cfg.puck_start_max_x_fraction
    min_handle_distance = cfg.puck_radius + cfg.handle_radius

    for seed in range(30):
        physics.reset(seed=seed)
        puck_position = physics.puck_body.position
        assert min_x <= abs(puck_position.x) <= max_x
        for body in physics.handle_bodies.values():
            assert (body.position - puck_position).length >= min_handle_distance


def test_magnet_attraction_fades_with_distance() -> None:
    physics = KlaskPhysics()
    cfg = physics.config

    near_force = physics._magnet_attraction_force(cfg.magnet_attraction_range * 0.25)
    far_force = physics._magnet_attraction_force(cfg.magnet_attraction_range * 0.75)

    assert near_force > far_force > 0.0
    assert physics._magnet_attraction_force(cfg.magnet_attraction_range + 0.01) == 0.0


def test_far_magnet_does_not_chase_handler() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=6)
    cfg = physics.config
    physics.handle_bodies["left"].position = (-0.4, 0.0)
    physics.handle_bodies["left"].velocity = (0.0, 0.0)
    physics.handle_bodies["right"].position = (0.8, 0.3)
    physics.magnet_bodies[0].position = (-0.4 + cfg.magnet_attraction_range + 0.04, 0.0)
    physics.magnet_bodies[0].velocity = (0.0, 0.0)

    initial_position = np.array(physics.magnet_bodies[0].position)
    physics.step({"left": np.zeros(2), "right": np.zeros(2)})

    final_position = np.array(physics.magnet_bodies[0].position)
    assert np.linalg.norm(final_position - initial_position) < 1e-6


def test_free_magnet_slows_down_without_attraction() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=6)
    physics.handle_bodies["left"].position = (-0.85, 0.45)
    physics.handle_bodies["right"].position = (0.85, -0.45)
    physics.magnet_bodies[0].position = (0.0, 0.0)
    physics.magnet_bodies[0].velocity = (1.0, 0.0)

    initial_speed = physics.magnet_bodies[0].velocity.length
    for _ in range(20):
        physics.step({"left": np.zeros(2), "right": np.zeros(2)})
    final_speed = physics.magnet_bodies[0].velocity.length

    assert final_speed < initial_speed * 0.35


def test_handler_can_run_away_from_nearby_magnet() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=7)
    physics.handle_bodies["left"].position = (-0.24, 0.0)
    physics.handle_bodies["left"].velocity = (0.0, 0.0)
    physics.magnet_bodies[0].position = (-0.48, 0.0)
    physics.magnet_bodies[0].velocity = (0.0, 0.0)

    initial_distance = (physics.magnet_bodies[0].position - physics.handle_bodies["left"].position).length
    for _ in range(8):
        physics.step({"left": np.array([1.0, 0.0]), "right": np.zeros(2)})
    final_distance = (physics.magnet_bodies[0].position - physics.handle_bodies["left"].position).length

    assert final_distance > initial_distance


def test_nearby_magnet_moves_toward_stationary_handler() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=8)
    physics.handle_bodies["left"].position = (-0.4, 0.0)
    physics.handle_bodies["right"].position = (0.8, 0.3)
    physics.magnet_bodies[0].position = (-0.24, 0.0)
    physics.magnet_bodies[0].velocity = (0.0, 0.0)

    initial_distance = (physics.magnet_bodies[0].position - physics.handle_bodies["left"].position).length
    for _ in range(5):
        physics.step({"left": np.zeros(2), "right": np.zeros(2)})
    final_distance = (physics.magnet_bodies[0].position - physics.handle_bodies["left"].position).length

    assert final_distance < initial_distance - 0.01


def test_sustained_contact_marks_magnet_attached() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=9)
    cfg = physics.config
    handle_position = (-0.4, 0.0)
    physics.puck_body.position = (0.3, 0.45)
    physics.puck_body.velocity = (0.0, 0.0)
    physics.handle_bodies["left"].position = handle_position
    physics.magnet_bodies[0].position = (handle_position[0] + cfg.handle_radius + cfg.magnet_radius, 0.0)
    physics.magnet_bodies[0].velocity = (0.0, 0.0)

    for _ in range(_steps_for(physics, physics.config.magnet_attach_frames)):
        physics.step({"left": np.zeros(2), "right": np.zeros(2)})

    assert physics.magnet_attached_to[0] == "left"
    assert physics.magnet_attachment_counts()["left"] == 1


def test_attached_magnet_sticks_to_moving_handler() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=10)
    cfg = physics.config
    physics.handle_bodies["left"].position = (-0.4, 0.0)
    physics.magnet_bodies[0].position = (-0.4 + cfg.handle_radius + cfg.magnet_radius, 0.0)
    physics.magnet_bodies[0].velocity = (0.0, 0.0)
    for _ in range(_steps_for(physics, physics.config.magnet_attach_frames)):
        physics.step({"left": np.zeros(2), "right": np.zeros(2)})
    assert physics.magnet_attached_to[0] == "left"

    initial_offset = physics.magnet_bodies[0].position - physics.handle_bodies["left"].position
    for _ in range(4):
        physics.step({"left": np.array([1.0, 0.0]), "right": np.zeros(2)})
    final_offset = physics.magnet_bodies[0].position - physics.handle_bodies["left"].position

    assert physics.magnet_attached_to[0] == "left"
    assert (final_offset - initial_offset).length < 1e-6
    assert final_offset.length <= cfg.handle_radius + cfg.magnet_radius + 1e-6


def test_two_attached_magnets_score_for_opponent() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=11)
    cfg = physics.config
    handle_position = (-0.4, 0.0)
    physics.puck_body.position = (0.3, 0.45)
    physics.puck_body.velocity = (0.0, 0.0)
    physics.handle_bodies["left"].position = handle_position
    physics.magnet_bodies[0].position = (handle_position[0] + cfg.handle_radius + cfg.magnet_radius, 0.0)
    physics.magnet_bodies[1].position = (handle_position[0], cfg.handle_radius + cfg.magnet_radius)
    physics.magnet_bodies[0].velocity = (0.0, 0.0)
    physics.magnet_bodies[1].velocity = (0.0, 0.0)

    for _ in range(_steps_for(physics, physics.config.magnet_attach_frames)):
        result = physics.step({"left": np.zeros(2), "right": np.zeros(2)})

    assert result.scored_by == "right"
    assert result.score_reason == "magnets"
    assert result.magnet_counts["left"] == 2


def test_magnet_in_hole_becomes_inert() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=12)
    cfg = physics.config
    hole = cfg.goal_center("left")
    physics.puck_body.position = (0.5, 0.45)
    physics.puck_body.velocity = (0.0, 0.0)
    physics.handle_bodies["left"].position = (-0.4, 0.4)
    physics.magnet_bodies[0].position = (hole[0] + 0.01, 0.0)
    physics.magnet_bodies[0].velocity = (0.0, 0.0)

    result = physics.step({"left": np.zeros(2), "right": np.zeros(2)})
    assert result.scored_by is None
    assert physics.magnet_in_hole[0]
    assert np.allclose(np.array(physics.magnet_bodies[0].position), np.array(hole))
    assert physics.magnet_bodies[0].velocity.length == 0.0

    physics.handle_bodies["left"].position = (hole[0] + cfg.handle_klask_radius + 0.01, 0.0)
    for _ in range(5):
        result = physics.step({"left": np.zeros(2), "right": np.zeros(2)})
    assert result.scored_by is None
    assert physics.magnet_attached_to[0] is None
    assert physics.magnet_risk("left")["proximity"] == 0.0

    physics.handle_bodies["left"].position = (-0.4, 0.4)
    physics.puck_body.position = (hole[0] + cfg.puck_capture_radius + 0.02, 0.0)
    physics.puck_body.velocity = (-1.0, 0.0)
    result = physics.step({"left": np.zeros(2), "right": np.zeros(2)})
    assert result.scored_by == "right"
    assert result.score_reason == "goal"


def test_shot_on_target_scores_a_straight_shot() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=1)
    hole = physics.config.goal_center("right")
    physics.puck_body.position = (-0.4, hole[1])
    physics.puck_body.velocity = (2.0, 0.0)
    assert physics.shot_on_target("right") == 1.0
    # Same puck, aimed away from the hole.
    physics.puck_body.velocity = (-2.0, 0.0)
    assert physics.shot_on_target("right") < 1.0


def _bank_shots(physics: KlaskPhysics, speed: float) -> list[float]:
    """Angles from the centre that only reach the hole after a bounce."""
    found = []
    for degrees in range(0, 360, 2):
        radians = np.radians(degrees)
        physics.puck_body.position = (-0.3, 0.0)
        physics.puck_body.velocity = pymunk.Vec2d(np.cos(radians), np.sin(radians)) * speed
        if physics.shot_on_target("right", max_reflections=0) < 1.0 and (
            physics.shot_on_target("right", max_reflections=2) >= 1.0
        ):
            found.append(radians)
    return found


def test_shot_on_target_finds_shots_that_only_work_off_a_wall() -> None:
    """Banking is a normal way to score, so bounces must widen what counts."""
    physics = KlaskPhysics()
    physics.reset(seed=1)
    assert _bank_shots(physics, physics.config.max_puck_speed), "no bank shots found"


def test_predicted_bank_shots_mostly_are_goals() -> None:
    """The predictor earns reward, so its verdicts must track the simulator.

    Guards against the bank branch decaying into noise: it once reflected
    specularly and ignored friction, which made it barely better than chance.
    """
    physics = KlaskPhysics()
    physics.reset(seed=1)
    speed = physics.config.max_puck_speed
    angles = _bank_shots(physics, speed)

    goals = 0
    for radians in angles:
        physics.reset(seed=1)
        physics.handle_bodies["left"].position = (-0.94, -0.70)
        physics.handle_bodies["right"].position = (0.94, -0.70)
        for index, body in enumerate(physics.magnet_bodies):
            body.position = (-0.5 + 0.5 * index, -0.70)
            body.velocity = (0.0, 0.0)
        physics.puck_body.position = (-0.3, 0.0)
        physics.puck_body.velocity = pymunk.Vec2d(np.cos(radians), np.sin(radians)) * speed
        for _ in range(150):
            result = physics.step({"left": np.zeros(2), "right": np.zeros(2)})
            if result.scored_by is not None:
                goals += result.score_reason == "goal" and result.scored_by == "left"
                break
            if physics.puck_body.velocity.length == 0.0:
                break
    assert goals / len(angles) > 0.5, f"only {goals}/{len(angles)} predicted bank shots scored"


def test_shot_on_target_is_zero_for_a_resting_puck() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=1)
    physics.puck_body.position = physics.config.goal_center("right")
    physics.puck_body.velocity = (0.0, 0.0)
    assert physics.shot_on_target("right") == 0.0


def test_shot_on_target_handles_a_puck_outside_the_bounce_box() -> None:
    """_contain_puck tolerates the puck past where pymunk resolves a bounce.

    Marching from out there finds no wall ahead and reports a miss, even though
    the simulator bounces it straight into the hole.
    """
    physics = KlaskPhysics()
    physics.reset(seed=1)
    cfg = physics.config
    hole = cfg.goal_center("right")
    physics.puck_body.position = (cfg.half_width - cfg.puck_radius, hole[1])
    physics.puck_body.velocity = (2.0, 0.0)
    assert physics.shot_on_target("right") == 1.0


def test_n_key_requests_skipping_the_episode(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame

    physics = KlaskPhysics()
    physics.render("human")
    assert not physics.skip_requested

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n))
    physics.render("human")
    assert physics.skip_requested

    # A skip must also release a pause, or the loop would sit in the pause
    # branch waiting for a keypress that already happened.
    physics.paused = True
    physics.skip_requested = False
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n))
    physics.render("human")
    assert physics.skip_requested
    assert not physics.paused

    physics.close()
    assert not physics.skip_requested
