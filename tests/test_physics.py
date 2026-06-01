from __future__ import annotations

import numpy as np

from klask_rl.physics import KlaskPhysics


def test_goal_detection_right_gate_scores_left() -> None:
    physics = KlaskPhysics()
    physics.reset(seed=1)
    physics.puck_body.position = (physics.config.half_width + 0.01, 0.0)
    physics.puck_body.velocity = (0.5, 0.0)
    result = physics.step({"left": np.zeros(2), "right": np.zeros(2)})
    assert result.scored_by == "left"


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

    assert [body.position.x for body in physics.magnet_bodies] == [0.0, 0.0, 0.0]
    assert [body.position.y for body in physics.magnet_bodies] == [-0.24, 0.0, 0.24]


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
    physics.handle_bodies["right"].position = (0.8, 0.0)
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
    physics.handle_bodies["right"].position = (0.8, 0.0)
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

    result = physics.step({"left": np.zeros(2), "right": np.zeros(2)})

    assert result.scored_by == "right"
    assert result.score_reason == "magnets"
    assert result.magnet_counts["left"] == 2
