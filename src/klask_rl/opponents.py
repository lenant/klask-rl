from __future__ import annotations

import random
from pathlib import Path
from typing import Protocol

import numpy as np

from klask_rl.config import ArenaConfig


class OpponentPolicy(Protocol):
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return a canonical normalized action for one agent."""


# Observations are normalized per axis (x by half_width, y by half_height), so
# the reachable box is not square. These are all in that canonical frame.
_ARENA = ArenaConfig()
CONTACT_GAP = 0.08
ALIGN_GAP = 0.16
STRIKE_OVERSHOOT = 0.3
REACH_X_MIN = -(_ARENA.half_width - _ARENA.handle_radius) / _ARENA.half_width
REACH_Y = (_ARENA.half_height - _ARENA.handle_radius) / _ARENA.half_height
OWN_HOLE_X = -_ARENA.goal_center_x / _ARENA.half_width
# Roughly twice the klask radius, and deliberately below ALIGN_GAP: a skirt
# wider than the alignment window would hold the handle permanently off the
# puck's line, so it would wind up next to the hole and never commit.
HOLE_CLEARANCE_X = 2.0 * _ARENA.handle_klask_radius / _ARENA.half_width
HOLE_CLEARANCE_Y = 0.75 * ALIGN_GAP
# Sideways offset used to prise a puck off the agent's own back wall; wider than
# the contact gap so the handle arrives beside the puck rather than jammed on it.
WALL_ESCAPE_Y = 0.14


# Radius of the keep-out bubble around the agent's own hole, per axis. Skirting
# only the *target* is not enough: the handle can drive through the hole on its
# way to a perfectly safe target, which is how deep starts turned into klasks.
HOLE_AVOID_X = 0.17
HOLE_AVOID_Y = 0.24


def _avoid_own_hole(action: np.ndarray, own_x: float, own_y: float) -> np.ndarray:
    """Bend an action away from the agent's own hole as it gets close.

    A potential field rather than a target offset, so it applies whatever the
    handle happens to be doing -- including travelling somewhere else entirely.
    """
    offset_x = (own_x - OWN_HOLE_X) / HOLE_AVOID_X
    offset_y = own_y / HOLE_AVOID_Y
    danger = offset_x * offset_x + offset_y * offset_y
    if danger >= 1.0:
        return action

    away = np.array([offset_x, offset_y], dtype=np.float32)
    magnitude = float(np.linalg.norm(away))
    if magnitude < 1e-6:
        away = np.array([1.0, 1.0], dtype=np.float32)
        magnitude = float(np.linalg.norm(away))
    away = away / magnitude

    # Cancel any component heading into the hole before adding the push out.
    # Blending instead leaves a net inward action through the outer half of the
    # bubble, and by the time it reverses the handle can no longer brake inside
    # the klask radius -- which is exactly how it was still falling in.
    inward = -float(np.dot(action, away))
    if inward > 0.0:
        action = action + away * inward
    urgency = 1.0 - float(np.sqrt(danger))
    return np.clip(action + away * (urgency * 2.0), -1.0, 1.0)


# Magnets sit on the centre line, exactly where the experts like to wait, and
# collecting two of them loses the point. Same keep-out treatment as the hole.
BASE_OBSERVATION_FEATURES = 16
MAGNET_AVOID_X = _ARENA.magnet_attraction_range / _ARENA.half_width
MAGNET_AVOID_Y = _ARENA.magnet_attraction_range / _ARENA.half_height


def _push_out_of(action: np.ndarray, offset_x: float, offset_y: float) -> np.ndarray:
    """Cancel motion into a hazard and add a push away, given a normalised offset."""
    danger = offset_x * offset_x + offset_y * offset_y
    if danger >= 1.0:
        return action
    away = np.array([offset_x, offset_y], dtype=np.float32)
    magnitude = float(np.linalg.norm(away))
    if magnitude < 1e-6:
        away = np.array([1.0, 1.0], dtype=np.float32)
        magnitude = float(np.linalg.norm(away))
    away = away / magnitude
    inward = -float(np.dot(action, away))
    if inward > 0.0:
        action = action + away * inward
    urgency = 1.0 - float(np.sqrt(danger))
    return action + away * (urgency * 2.0)


def _avoid_magnets(action: np.ndarray, own_x: float, own_y: float, observation: np.ndarray) -> np.ndarray:
    """Steer clear of loose magnets. Two of them on your handle loses the point."""
    for index in range(_ARENA.magnet_count):
        base = BASE_OBSERVATION_FEATURES + index * 5
        if base + 4 >= observation.shape[0]:
            break
        if abs(float(observation[base + 4])) > 0.5:
            continue  # already attached to someone; avoiding it changes nothing
        action = _push_out_of(
            action,
            (own_x - float(observation[base])) / MAGNET_AVOID_X,
            (own_y - float(observation[base + 1])) / MAGNET_AVOID_Y,
        )
    return np.clip(action, -1.0, 1.0)


def _lined_up_behind_puck(own_x: float, own_y: float, puck_x: float, puck_y: float) -> bool:
    """True once the handle sits behind the puck and roughly on its line."""
    return own_x < puck_x - CONTACT_GAP * 0.5 and abs(own_y - puck_y) < ALIGN_GAP


def _skirt_own_hole(target_x: float, target_y: float) -> tuple[float, float]:
    """Steer a deep target around the agent's own hole so it does not klask."""
    if abs(target_x - OWN_HOLE_X) > HOLE_CLEARANCE_X or abs(target_y) >= HOLE_CLEARANCE_Y:
        return target_x, target_y
    return target_x, HOLE_CLEARANCE_Y if target_y >= 0.0 else -HOLE_CLEARANCE_Y


def _attack_target(
    own_x: float,
    own_y: float,
    puck_x: float,
    puck_y: float,
    standoff: float,
) -> tuple[float, float]:
    """Wind up behind the puck, commit through it, or prise it off the back wall.

    Winding up at a standoff is only a shot if something then closes the gap, so
    once the handle is in position it has to drive through the ball. A puck at
    rest against the agent's own back wall leaves no room to get behind it at
    all, and pushing straight at it only presses it into the wall. There the
    handle hugs the wall and sweeps across the puck's line instead, aiming past
    it to whichever side it is not currently on: each crossing knocks the puck
    along the boards until it is back in open play.
    """
    if _lined_up_behind_puck(own_x, own_y, puck_x, puck_y):
        target_x = puck_x + STRIKE_OVERSHOOT
    elif puck_x - standoff < REACH_X_MIN:
        sweep = -WALL_ESCAPE_Y if own_y > puck_y else WALL_ESCAPE_Y
        return _skirt_own_hole(REACH_X_MIN, float(np.clip(puck_y + sweep, -REACH_Y, REACH_Y)))
    else:
        target_x = puck_x - standoff
    return _skirt_own_hole(target_x, float(np.clip(puck_y, -REACH_Y, REACH_Y)))


class HeuristicOpponent:
    """Small goalie/striker baseline used for bootstrapping and evaluation."""

    def __init__(self, aggression: float = 2.8) -> None:
        self.aggression = aggression

    def act(self, observation: np.ndarray) -> np.ndarray:
        own_x, own_y = observation[0], observation[1]
        puck_x, puck_y = observation[8], observation[9]
        puck_vx = observation[10]

        if puck_x < 0.05 or puck_vx < -0.05:
            target_x, target_y = _attack_target(own_x, own_y, puck_x, puck_y, standoff=0.16)
        else:
            target_x = -0.55
            target_y = np.clip(puck_y * 0.75, -0.65, 0.65)

        action = np.clip(
            np.array([target_x - own_x, target_y - own_y], dtype=np.float32) * self.aggression,
            -1.0,
            1.0,
        )
        # Hole avoidance goes last: a klask loses the point outright, so it
        # must be able to override the magnet push rather than the reverse.
        action = _avoid_magnets(action, own_x, own_y, observation)
        return _avoid_own_hole(action, own_x, own_y)


class StrikerOpponent:
    """Aggressive expert used for behavior-cloning warm starts and baselines."""

    def __init__(self, aggression: float = 5.0) -> None:
        self.aggression = aggression

    def act(self, observation: np.ndarray) -> np.ndarray:
        own_x, own_y = observation[0], observation[1]
        puck_x, puck_y = observation[8], observation[9]
        puck_vx = observation[10]

        danger = puck_x < -0.55 and abs(puck_y) < 0.72
        if danger and puck_vx < 0.0:
            target_x, target_y = _attack_target(own_x, own_y, puck_x, puck_y, standoff=0.10)
        elif puck_x > 0.0:
            target_x = -0.08
            target_y = float(np.clip(puck_y, -REACH_Y, REACH_Y))
        else:
            target_x, target_y = _attack_target(own_x, own_y, puck_x, puck_y, standoff=0.14)

        action = np.clip(
            np.array([target_x - own_x, target_y - own_y], dtype=np.float32) * self.aggression,
            -1.0,
            1.0,
        )
        # Hole avoidance goes last: a klask loses the point outright, so it
        # must be able to override the magnet push rather than the reverse.
        action = _avoid_magnets(action, own_x, own_y, observation)
        return _avoid_own_hole(action, own_x, own_y)


class PassiveOpponent:
    def act(self, observation: np.ndarray) -> np.ndarray:
        del observation
        return np.zeros(2, dtype=np.float32)


class RandomOpponent:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> np.ndarray:
        del observation
        return self._rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


class SB3CheckpointOpponent:
    def __init__(self, model_path: str | Path, deterministic: bool = True) -> None:
        self.model_path = Path(model_path)
        self.deterministic = deterministic
        self._model = None

    def act(self, observation: np.ndarray) -> np.ndarray:
        if self._model is None:
            from stable_baselines3 import PPO

            self._model = PPO.load(self.model_path, device="cpu")
        action, _ = self._model.predict(observation, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32)


class OpponentPool:
    """Sample heuristic and frozen checkpoint opponents for league-style self-play.

    One opponent is sampled per episode (via ``resample``) so the opponent plays a
    coherent policy within an episode instead of a per-step mixture.
    """

    def __init__(self, opponents: list[OpponentPolicy] | None = None, seed: int | None = None) -> None:
        self._opponents: list[OpponentPolicy] = opponents or [HeuristicOpponent()]
        self._checkpoint_paths: list[Path] = []
        self._rng = random.Random(seed)
        self._current: OpponentPolicy = self._opponents[0]

    @property
    def checkpoint_paths(self) -> tuple[Path, ...]:
        return tuple(self._checkpoint_paths)

    def add_checkpoint(self, path: str | Path) -> None:
        checkpoint = Path(path)
        if checkpoint.exists() and checkpoint not in self._checkpoint_paths:
            self._checkpoint_paths.append(checkpoint)
            self._opponents.append(SB3CheckpointOpponent(checkpoint))

    def resample(self) -> None:
        self._current = self._rng.choice(self._opponents)

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self._current.act(observation)
