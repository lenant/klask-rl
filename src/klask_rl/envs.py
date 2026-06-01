from __future__ import annotations

from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from klask_rl.config import AGENTS, OBSERVATION_SIZE, OPPONENT, ArenaConfig, RewardConfig
from klask_rl.config import reward_profile as get_reward_profile
from klask_rl.opponents import HeuristicOpponent, OpponentPolicy
from klask_rl.physics import KlaskPhysics


REWARD_COMPONENTS: tuple[str, ...] = (
    "progress",
    "position",
    "speed",
    "contact",
    "distance",
    "defense",
    "danger",
    "time",
    "action",
    "magnet_attach",
    "magnet_pull",
    "own_side",
    "terminal",
)


class KlaskParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "klask_parallel_v0"}

    def __init__(
        self,
        arena_config: ArenaConfig | None = None,
        reward_config: RewardConfig | None = None,
        render_mode: str | None = None,
        reward_profile: str = "balanced",
    ) -> None:
        self.arena_config = arena_config or ArenaConfig()
        self.reward_config = reward_config or get_reward_profile(reward_profile)
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
        self.last_rewards = {agent: 0.0 for agent in AGENTS}
        self.episode_rewards = {agent: 0.0 for agent in AGENTS}
        self.last_reward_components = {
            agent: {component: 0.0 for component in REWARD_COMPONENTS} for agent in AGENTS
        }

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
        self.last_rewards = {agent: 0.0 for agent in AGENTS}
        self.episode_rewards = {agent: 0.0 for agent in AGENTS}
        self.last_reward_components = {
            agent: {component: 0.0 for component in REWARD_COMPONENTS} for agent in AGENTS
        }
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
        previous_magnet_attached_to = list(self.physics.magnet_attached_to)

        result = self.physics.step(world_actions)
        self.steps += 1
        if result.scored_by is not None:
            self.scores[result.scored_by] += 1
        new_magnet_attachments = {
            agent: sum(
                1
                for before, after in zip(
                    previous_magnet_attached_to, self.physics.magnet_attached_to, strict=True
                )
                if before != agent and after == agent
            )
            for agent in active_agents
        }

        own_components = {
            agent: self._shaping_reward_components(
                agent=agent,
                previous_puck_x=previous_puck_x[agent],
                action=canonical_actions[agent],
                contact=result.contacts[agent],
                new_magnet_attachments=new_magnet_attachments[agent],
            )
            for agent in active_agents
        }
        reward_components = {
            agent: {
                component: own_components[agent][component] - own_components[OPPONENT[agent]][component]
                for component in REWARD_COMPONENTS
            }
            for agent in active_agents
        }
        for agent in active_agents:
            reward_components[agent]["own_side"] = own_components[agent]["own_side"]
        if result.scored_by is not None:
            reward_components[result.scored_by]["terminal"] += self.reward_config.terminal_goal
            reward_components[OPPONENT[result.scored_by]]["terminal"] -= self.reward_config.terminal_goal
        rewards = {
            agent: float(sum(reward_components[agent].values())) for agent in active_agents
        }
        self.last_rewards = {agent: float(rewards.get(agent, 0.0)) for agent in AGENTS}
        self.last_reward_components = {
            agent: {
                component: float(reward_components.get(agent, {}).get(component, 0.0))
                for component in REWARD_COMPONENTS
            }
            for agent in AGENTS
        }
        for agent in active_agents:
            self.episode_rewards[agent] += self.last_rewards[agent]

        terminated = result.scored_by is not None
        truncated = self.steps >= self.arena_config.max_steps
        terminations = {agent: terminated for agent in active_agents}
        truncations = {agent: truncated for agent in active_agents}
        observations = {agent: self._make_observation(agent) for agent in active_agents}
        infos = {
            agent: {
                "score": self.scores.copy(),
                "scored_by": result.scored_by,
                "score_reason": result.score_reason,
                "contacts": result.contacts.copy(),
                "magnet_counts": result.magnet_counts.copy(),
                "steps": self.steps,
                "rewards": self.last_rewards.copy(),
                "episode_rewards": self.episode_rewards.copy(),
                "reward_components": {
                    side: components.copy()
                    for side, components in self.last_reward_components.items()
                },
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
        magnet_pos = snap["magnet_pos"]
        magnet_vel = snap["magnet_vel"]

        base_observation = [
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
        ]
        magnet_features: list[float] = []
        for index in range(cfg.magnet_count):
            pos = magnet_pos[index]
            vel = magnet_vel[index]
            attached_to = self.physics.magnet_attached_to[index]
            if attached_to == agent:
                attached_state = 1.0
            elif attached_to == opponent:
                attached_state = -1.0
            else:
                attached_state = 0.0
            magnet_features.extend(
                [
                    sign * pos[0] / cfg.half_width,
                    pos[1] / cfg.half_height,
                    sign * vel[0] / cfg.max_puck_speed,
                    vel[1] / cfg.max_puck_speed,
                    attached_state,
                ]
            )

        observation = np.array(
            [
                *base_observation,
                *magnet_features,
                self.physics.magnet_attachment_counts()[agent] / cfg.magnet_score_threshold,
                self.physics.magnet_attachment_counts()[opponent] / cfg.magnet_score_threshold,
            ],
            dtype=np.float32,
        )
        return np.clip(observation, -1.0, 1.0)

    def _canonical_action_to_world(self, agent: str, action: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if agent == "left":
            return clipped
        return np.array([-clipped[0], clipped[1]], dtype=np.float32)

    def _magnet_pull_risk(self, agent: str) -> float:
        cfg = self.arena_config
        handle_position = self.physics.handle_bodies[agent].position
        attach_distance = cfg.handle_radius + cfg.magnet_radius + 0.008
        active_distance = max(1e-9, cfg.magnet_attraction_range - attach_distance)
        risk = 0.0
        for index, magnet_body in enumerate(self.physics.magnet_bodies):
            if self.physics.magnet_attached_to[index] is not None:
                continue
            distance = (magnet_body.position - handle_position).length
            if distance >= cfg.magnet_attraction_range:
                continue
            closeness = (cfg.magnet_attraction_range - max(distance, attach_distance)) / active_distance
            risk += float(np.clip(closeness, 0.0, 1.0) ** 2)
        return risk

    def _shaping_reward_components(
        self,
        agent: str,
        previous_puck_x: float,
        action: np.ndarray,
        contact: bool,
        new_magnet_attachments: int = 0,
    ) -> dict[str, float]:
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
        magnet_risk = self.physics.magnet_risk(agent)
        magnet_attach = -(
            self.reward_config.magnet_attached_penalty * magnet_risk["attached"]
            + self.reward_config.magnet_attach_penalty * float(new_magnet_attachments)
        )
        magnet_pull = -(
            self.reward_config.magnet_proximity_penalty * magnet_risk["proximity"]
            + self.reward_config.magnet_pull_penalty * self._magnet_pull_risk(agent)
        )
        own_side = -self.reward_config.own_side_penalty if puck_x < 0.0 else 0.0
        return {
            "progress": progress,
            "position": position,
            "speed": speed,
            "contact": contact_bonus,
            "distance": distance_bonus,
            "defense": defense,
            "danger": -goal_danger,
            "time": -self.reward_config.time_penalty,
            "action": -action_cost,
            "magnet_attach": magnet_attach,
            "magnet_pull": magnet_pull,
            "own_side": own_side,
            "terminal": 0.0,
        }

    def _shaping_reward(
        self,
        agent: str,
        previous_puck_x: float,
        action: np.ndarray,
        contact: bool,
    ) -> float:
        return float(
            sum(
                self._shaping_reward_components(
                    agent=agent,
                    previous_puck_x=previous_puck_x,
                    action=action,
                    contact=contact,
                ).values()
            )
        )

    def render(self) -> np.ndarray | None:
        reward_overlay = {
            agent: {
                "step": self.last_rewards[agent],
                "episode": self.episode_rewards[agent],
                "components": self.last_reward_components[agent].copy(),
            }
            for agent in AGENTS
        }
        return self.physics.render(self.render_mode or "human", reward_overlay=reward_overlay)

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
        reward_profile: str = "balanced",
    ) -> None:
        arena_config = arena_config or ArenaConfig()
        if max_steps is not None:
            arena_config = replace(arena_config, max_steps=max_steps)
        self.base_env = KlaskParallelEnv(arena_config, reward_config, render_mode, reward_profile)
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

    def add_opponent_checkpoint(self, checkpoint_path: str) -> bool:
        add_checkpoint = getattr(self.opponent, "add_checkpoint", None)
        if add_checkpoint is None:
            return False
        add_checkpoint(checkpoint_path)
        return True
