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
