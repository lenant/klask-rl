from __future__ import annotations

from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from klask_rl.config import AGENTS, OBSERVATION_SIZE, OPPONENT, ArenaConfig, RewardConfig
from klask_rl.opponents import HeuristicOpponent, OpponentPolicy
from klask_rl.physics import KlaskPhysics


class KlaskParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "klask_parallel_v0"}

    def __init__(
        self,
        arena_config: ArenaConfig | None = None,
        reward_config: RewardConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        self.arena_config = arena_config or ArenaConfig()
        self.reward_config = reward_config or RewardConfig()
        self.render_mode = render_mode
        self.physics = KlaskPhysics(self.arena_config)
        self.possible_agents = list(AGENTS)
        self.agents: list[str] = []
        self.observation_spaces = {
            agent: spaces.Box(-1.0, 1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
            for agent in AGENTS
        }
        self.action_spaces = {
            agent: spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32) for agent in AGENTS
        }
        self.steps = 0
        self.scores = {agent: 0 for agent in AGENTS}

    def observation_space(self, agent: str) -> spaces.Box:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Box:
        return self.action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        del options
        self.physics.reset(seed=seed)
        self.agents = list(AGENTS)
        self.steps = 0
        self.scores = {agent: 0 for agent in AGENTS}
        observations = {agent: self._make_observation(agent) for agent in self.agents}
        infos = {agent: {"score": self.scores.copy()} for agent in self.agents}
        return observations, infos

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if not self.agents:
            return {}, {}, {}, {}, {}

        active_agents = list(self.agents)
        previous_puck_x = {
            agent: float(self._make_observation(agent)[8]) for agent in active_agents
        }
        canonical_actions = {
            agent: np.asarray(actions.get(agent, np.zeros(2)), dtype=np.float32)
            for agent in active_agents
        }
        world_actions = {
            agent: self._canonical_action_to_world(agent, canonical_actions[agent])
            for agent in active_agents
        }

        result = self.physics.step(world_actions)
        self.steps += 1
        if result.scored_by is not None:
            self.scores[result.scored_by] += 1

        own_terms = {
            agent: self._shaping_reward(
                agent=agent,
                previous_puck_x=previous_puck_x[agent],
                action=canonical_actions[agent],
                contact=result.contacts[agent],
            )
            for agent in active_agents
        }
        rewards = {
            agent: own_terms[agent] - own_terms[OPPONENT[agent]] for agent in active_agents
        }
        if result.scored_by is not None:
            rewards[result.scored_by] += self.reward_config.terminal_goal
            rewards[OPPONENT[result.scored_by]] -= self.reward_config.terminal_goal

        terminated = result.scored_by is not None
        truncated = self.steps >= self.arena_config.max_steps
        terminations = {agent: terminated for agent in active_agents}
        truncations = {agent: truncated for agent in active_agents}
        observations = {agent: self._make_observation(agent) for agent in active_agents}
        infos = {
            agent: {
                "score": self.scores.copy(),
                "scored_by": result.scored_by,
                "contacts": result.contacts.copy(),
                "steps": self.steps,
            }
            for agent in active_agents
        }

        if terminated or truncated:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def _make_observation(self, agent: str) -> np.ndarray:
        cfg = self.arena_config
        snap = self.physics.snapshot()
        sign = 1.0 if agent == "left" else -1.0
        opponent = OPPONENT[agent]
        own_pos = snap[f"{agent}_pos"]
        own_vel = snap[f"{agent}_vel"]
        opp_pos = snap[f"{opponent}_pos"]
        opp_vel = snap[f"{opponent}_vel"]
        puck_pos = snap["puck_pos"]
        puck_vel = snap["puck_vel"]

        observation = np.array(
            [
                sign * own_pos[0] / cfg.half_width,
                own_pos[1] / cfg.half_height,
                sign * own_vel[0] / cfg.max_handle_speed,
                own_vel[1] / cfg.max_handle_speed,
                sign * opp_pos[0] / cfg.half_width,
                opp_pos[1] / cfg.half_height,
                sign * opp_vel[0] / cfg.max_handle_speed,
                opp_vel[1] / cfg.max_handle_speed,
                sign * puck_pos[0] / cfg.half_width,
                puck_pos[1] / cfg.half_height,
                sign * puck_vel[0] / cfg.max_puck_speed,
                puck_vel[1] / cfg.max_puck_speed,
                sign * (puck_pos[0] - own_pos[0]) / cfg.width,
                (puck_pos[1] - own_pos[1]) / cfg.height,
                (self.scores[agent] - self.scores[opponent]) / 3.0,
                1.0 - min(1.0, self.steps / cfg.max_steps),
            ],
            dtype=np.float32,
        )
        return np.clip(observation, -1.0, 1.0)

    def _canonical_action_to_world(self, agent: str, action: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if agent == "left":
            return clipped
        return np.array([-clipped[0], clipped[1]], dtype=np.float32)

    def _shaping_reward(
        self,
        agent: str,
        previous_puck_x: float,
        action: np.ndarray,
        contact: bool,
    ) -> float:
        obs = self._make_observation(agent)
        puck_x = float(obs[8])
        puck_y = float(obs[9])
        puck_vx = float(obs[10])
        own_y = float(obs[1])
        puck_distance = float(np.linalg.norm(obs[12:14]))

        progress = self.reward_config.progress * (puck_x - previous_puck_x)
        position = self.reward_config.puck_position * puck_x
        speed = self.reward_config.puck_speed * puck_vx
        contact_bonus = self.reward_config.contact if contact else 0.0
        distance_bonus = self.reward_config.puck_distance * (1.0 - min(1.0, puck_distance * 2.0))
        defensive_need = max(0.0, -puck_x)
        y_alignment = 1.0 - min(1.0, abs(own_y - puck_y))
        defense = self.reward_config.defense * defensive_need * y_alignment
        goal_danger = self.reward_config.own_goal_danger * defensive_need * (1.0 - abs(puck_y))
        action_cost = self.reward_config.action_penalty * float(np.dot(action, action))
        return (
            progress
            + position
            + speed
            + contact_bonus
            + distance_bonus
            + defense
            - goal_danger
            - self.reward_config.time_penalty
            - action_cost
        )

    def render(self) -> np.ndarray | None:
        return self.physics.render(self.render_mode or "human")

    def close(self) -> None:
        self.physics.close()


class SelfPlayKlaskEnv(gym.Env):
    metadata = KlaskParallelEnv.metadata

    def __init__(
        self,
        opponent: OpponentPolicy | None = None,
        arena_config: ArenaConfig | None = None,
        reward_config: RewardConfig | None = None,
        render_mode: str | None = None,
        randomize_side: bool = True,
        max_steps: int | None = None,
    ) -> None:
        arena_config = arena_config or ArenaConfig()
        if max_steps is not None:
            arena_config = replace(arena_config, max_steps=max_steps)
        self.base_env = KlaskParallelEnv(arena_config, reward_config, render_mode)
        self.opponent = opponent or HeuristicOpponent()
        self.randomize_side = randomize_side
        self.learning_side = "left"
        self.observation_space = self.base_env.observation_space("left")
        self.action_space = self.base_env.action_space("left")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        observations, infos = self.base_env.reset(seed=seed, options=options)
        if self.randomize_side:
            self.learning_side = str(self.np_random.choice(AGENTS))
        else:
            self.learning_side = "left"
        info = infos[self.learning_side] | {"learning_side": self.learning_side}
        return observations[self.learning_side], info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        opponent_side = OPPONENT[self.learning_side]
        opponent_obs = self.base_env._make_observation(opponent_side)
        opponent_action = self.opponent.act(opponent_obs)
        actions = {
            self.learning_side: np.asarray(action, dtype=np.float32),
            opponent_side: opponent_action,
        }
        observations, rewards, terminations, truncations, infos = self.base_env.step(actions)
        if observations:
            observation = observations[self.learning_side]
        else:
            observation = self.base_env._make_observation(self.learning_side)
        info = infos.get(self.learning_side, {}) | {"learning_side": self.learning_side}
        return (
            observation,
            float(rewards.get(self.learning_side, 0.0)),
            bool(terminations.get(self.learning_side, False)),
            bool(truncations.get(self.learning_side, False)),
            info,
        )

    def render(self) -> np.ndarray | None:
        return self.base_env.render()

    def close(self) -> None:
        self.base_env.close()
