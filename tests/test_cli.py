from __future__ import annotations

import numpy as np
from typer.testing import CliRunner

from klask_rl.cli import (
    _human_action_to_canonical,
    _scaled_world_action,
    _world_action_from_key_state,
    parse_policy_net_arch,
    play_app,
    train_app,
    watch_app,
)


def test_scaled_world_action_normalizes_diagonal_input() -> None:
    action = _scaled_world_action(1.0, 1.0, 1.0)

    assert np.isclose(np.linalg.norm(action), 1.0)
    np.testing.assert_allclose(action, [0.70710677, 0.70710677])


def test_right_human_action_is_mirrored_to_canonical_space() -> None:
    world_action = np.array([-1.0, 0.5], dtype=np.float32)

    np.testing.assert_allclose(_human_action_to_canonical("left", world_action), [-1.0, 0.5])
    np.testing.assert_allclose(_human_action_to_canonical("right", world_action), [1.0, 0.5])


def test_wasd_key_state_maps_to_world_action() -> None:
    pressed = {"w", "a"}

    action = _world_action_from_key_state(
        pressed.__contains__,
        left_key="a",
        right_key="d",
        up_key="w",
        down_key="s",
        action_scale=1.0,
    )

    np.testing.assert_allclose(action, [-0.70710677, 0.70710677])


def test_play_help_exposes_reward_profile() -> None:
    result = CliRunner().invoke(play_app, ["--help"])

    assert result.exit_code == 0
    assert "--reward-profile" in result.output


def test_watch_help_exposes_reward_profile() -> None:
    result = CliRunner().invoke(watch_app, ["--help"])

    assert result.exit_code == 0
    assert "--reward-profile" in result.output


def test_train_help_exposes_policy_net_arch() -> None:
    result = CliRunner().invoke(train_app, ["--help"])

    assert result.exit_code == 0
    assert "--policy-net-arch" in result.output
    assert "--resume-from" in result.output
    assert "--resume-opponent-checkpoints" in result.output


def test_parse_policy_net_arch_accepts_commas_and_x() -> None:
    assert parse_policy_net_arch("256,256,256") == (256, 256, 256)
    assert parse_policy_net_arch("256x256x256") == (256, 256, 256)
