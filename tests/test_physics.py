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
