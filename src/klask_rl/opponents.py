from __future__ import annotations

import random
from pathlib import Path
from typing import Protocol

import numpy as np


class OpponentPolicy(Protocol):
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return a canonical normalized action for one agent."""


class HeuristicOpponent:
    """Small goalie/striker baseline used for bootstrapping and evaluation."""

    def __init__(self, aggression: float = 2.8) -> None:
        self.aggression = aggression

    def act(self, observation: np.ndarray) -> np.ndarray:
        own_x, own_y = observation[0], observation[1]
        puck_x, puck_y = observation[8], observation[9]
        puck_vx = observation[10]

        if puck_x < 0.05 or puck_vx < -0.05:
            target_x = np.clip(puck_x - 0.16, -0.82, -0.08)
            target_y = puck_y
        else:
            target_x = -0.55
            target_y = np.clip(puck_y * 0.75, -0.65, 0.65)

        action = np.array([target_x - own_x, target_y - own_y], dtype=np.float32)
        return np.clip(action * self.aggression, -1.0, 1.0)


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
            target_x = np.clip(puck_x - 0.10, -0.88, -0.18)
            target_y = np.clip(puck_y, -0.78, 0.78)
        elif puck_x > 0.0:
            target_x = -0.08
            target_y = np.clip(puck_y, -0.82, 0.82)
        else:
            target_x = np.clip(puck_x - 0.14, -0.88, -0.08)
            target_y = np.clip(puck_y, -0.82, 0.82)

        action = np.array([target_x - own_x, target_y - own_y], dtype=np.float32)
        return np.clip(action * self.aggression, -1.0, 1.0)


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
    """Sample heuristic and frozen checkpoint opponents for league-style self-play."""

    def __init__(self, opponents: list[OpponentPolicy] | None = None, seed: int | None = None) -> None:
        self._opponents: list[OpponentPolicy] = opponents or [HeuristicOpponent()]
        self._checkpoint_paths: list[Path] = []
        self._rng = random.Random(seed)

    @property
    def checkpoint_paths(self) -> tuple[Path, ...]:
        return tuple(self._checkpoint_paths)

    def add_checkpoint(self, path: str | Path) -> None:
        checkpoint = Path(path)
        if checkpoint.exists() and checkpoint not in self._checkpoint_paths:
            self._checkpoint_paths.append(checkpoint)
            self._opponents.append(SB3CheckpointOpponent(checkpoint))

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self._rng.choice(self._opponents).act(observation)
