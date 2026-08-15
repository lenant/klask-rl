from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from klask_rl.config import ArenaConfig
from klask_rl.physics import PuckPath, PuckTrajectory


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


class _HoleGeometry:
    """Keep-out sizes around the agent's own hole.

    These follow goal_radius, which the curriculum changes between stages. Held
    per instance rather than as module constants: sized for the default 0.075
    hole, the experts klasked 20% of the time at the 0.15 hole of stage 1.
    """

    def __init__(self, config: ArenaConfig | None = None) -> None:
        arena = config or _ARENA
        klask = arena.handle_klask_radius
        self.avoid_x = 3.0 * klask / arena.half_width
        self.avoid_y = 3.0 * klask / arena.half_height
        self.safe_y = 2.0 * klask / arena.half_height
        self.magnet_avoid_x = arena.magnet_attraction_range / arena.half_width
        self.magnet_avoid_y = arena.magnet_attraction_range / arena.half_height
        self.magnet_capture = arena.magnet_capture_radius
        self.half_width = arena.half_width
        self.half_height = arena.half_height
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


def _avoid_own_hole(action: np.ndarray, own_x: float, own_y: float, geometry: _HoleGeometry) -> np.ndarray:
    """Bend an action away from the agent's own hole as it gets close.

    A potential field rather than a target offset, so it applies whatever the
    handle happens to be doing -- including travelling somewhere else entirely.
    """
    offset_x = (own_x - OWN_HOLE_X) / geometry.avoid_x
    offset_y = own_y / geometry.avoid_y
    danger = offset_x * offset_x + offset_y * offset_y
    if danger >= 1.0:
        return action

    if own_x < OWN_HOLE_X:
        # Behind the hole, "directly away" means deeper into the back wall, and
        # every route back into play then reads as heading inward -- which
        # pinned the handle against the boards for whole episodes. Escape
        # sideways instead, which clears the hole without trapping it.
        sideways = 1.0 if own_y >= 0.0 else -1.0
        urgency = 1.0 - float(np.sqrt(danger))
        forward, lateral = float(action[0]), float(action[1])
        if lateral * sideways < 0.0:
            # Chasing the puck's y pulls straight back onto the hole's line and
            # exactly cancelled the escape, so the handle hovered. Drop that
            # component rather than blending against it.
            lateral = 0.0
        if abs(own_y) < geometry.safe_y and forward > 0.0:
            # Coming back into play means crossing the hole's line, so get
            # clear of it sideways before moving up the board at all.
            forward = 0.0
        return np.array([forward, lateral + sideways * urgency * 2.0], dtype=np.float32)

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


def _is_in_a_hole(x: float, y: float, geometry: _HoleGeometry) -> bool:
    """A magnet sitting in a hole is inert and cannot be collected."""
    captured = geometry.magnet_capture
    for hole_x in (OWN_HOLE_X, -OWN_HOLE_X):
        dx = (x - hole_x) * geometry.half_width
        dy = (y - 0.0) * geometry.half_height
        if dx * dx + dy * dy <= captured * captured:
            return True
    return False


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


def _avoid_magnets(action: np.ndarray, own_x: float, own_y: float, observation: np.ndarray, geometry: _HoleGeometry) -> np.ndarray:
    """Steer clear of loose magnets. Two of them on your handle loses the point."""
    for index in range(_ARENA.magnet_count):
        base = BASE_OBSERVATION_FEATURES + index * 5
        if base + 4 >= observation.shape[0]:
            break
        if abs(float(observation[base + 4])) > 0.5:
            continue  # already attached to someone; avoiding it changes nothing
        magnet_x = float(observation[base])
        magnet_y = float(observation[base + 1])
        if _is_in_a_hole(magnet_x, magnet_y, geometry):
            continue  # captured and inert: keeping out only fences off the goal
        action = _push_out_of(
            action,
            (own_x - float(observation[base])) / geometry.magnet_avoid_x,
            (own_y - float(observation[base + 1])) / geometry.magnet_avoid_y,
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

    def __init__(self, aggression: float = 2.8, arena_config: ArenaConfig | None = None) -> None:
        self.aggression = aggression
        self.geometry = _HoleGeometry(arena_config)

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
        action = _avoid_magnets(action, own_x, own_y, observation, self.geometry)
        return _avoid_own_hole(action, own_x, own_y, self.geometry)


class StrikerOpponent:
    """Aggressive expert used for behavior-cloning warm starts and baselines."""

    def __init__(self, aggression: float = 5.0, arena_config: ArenaConfig | None = None) -> None:
        self.aggression = aggression
        self.geometry = _HoleGeometry(arena_config)

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
        action = _avoid_magnets(action, own_x, own_y, observation, self.geometry)
        return _avoid_own_hole(action, own_x, own_y, self.geometry)


@dataclass(frozen=True)
class ShotPlan:
    """A place to strike from, the heading the ball leaves on, and how good it is."""

    stand: tuple[float, float]
    launch: tuple[float, float]
    depth: float
    score: float
    scores_goal: bool
    reflections: int


class PlannerOpponent:
    """Rule-based striker: search for a shot at the opponent's hole, then take it.

    Two halves, and both are needed. The *aim* half samples handle positions on
    a ring around the ball, walks each resulting shot through the same rolling
    and bouncing model the simulation resolves (``PuckTrajectory``), and keeps
    whichever ends up nearest the opponent's hole. Bank shots therefore fall out
    of the same search as straight ones, and nothing that would drop the ball
    into our own hole survives it.

    The *approach* half is what makes it work on a ball at rest. Only ring
    positions the handle can legally occupy are candidates, so a ball tucked
    against a wall or wedged in a corner is answered with the best shot the
    remaining geometry allows -- a nudge along the boards, or a lift off the
    wall -- instead of a shove straight into the wall it is already touching.
    Getting to the chosen spot orbits the ball at arm's length rather than
    driving through it, which is what stops the handle knocking the ball away
    while it is still lining up.
    """

    def __init__(
        self,
        arena_config: ArenaConfig | None = None,
        *,
        angle_samples: int = 64,
        strike_tolerance: float = 0.012,
        strike_fraction: float = 0.85,
        lateral_gain: float = 10.0,
        lateral_damping: float = 0.9,
        lead_cap: int = 60,
        guard_x: float = -0.45,
        defend_radius: float = 0.34,
        max_reflections: int = 3,
        min_launch_run: float = 0.21,
        switch_margin: float = 15.0,
        keeper_weight: float = 1.0,
        save_first: bool = True,
        hole_margin: float = 0.02,
    ) -> None:
        self.config = arena_config or _ARENA
        self.trajectory = PuckTrajectory(self.config)
        self.geometry = _HoleGeometry(self.config)
        self.angle_samples = angle_samples
        self.strike_tolerance = strike_tolerance
        self.lateral_gain = lateral_gain
        self.lateral_damping = lateral_damping
        self.lead_cap = lead_cap
        self.guard_x = guard_x
        self.defend_radius = defend_radius
        self.max_reflections = max_reflections
        self.min_launch_run = min_launch_run
        self.switch_margin = switch_margin
        self.keeper_weight = keeper_weight
        self.save_first = save_first
        self.hole_margin = hole_margin
        self._committed: int | None = None

        cfg = self.config
        # Canonical coordinates: we always attack +x, which is the left agent's
        # own frame, so its bounds and its hole are the ones that apply.
        self.reach_x = (-cfg.half_width + cfg.handle_radius, -cfg.handle_radius)
        self.reach_y = (-cfg.half_height + cfg.handle_radius, cfg.half_height - cfg.handle_radius)
        self.target_hole = cfg.goal_center("right")
        self.own_hole = cfg.goal_center("left")
        self.contact_distance = cfg.puck_radius + cfg.handle_radius
        self.orbit_radius = self.contact_distance + 0.07
        self.line_clearance = 0.02
        # Enough run-up to reach the speed cap: max_speed**2 / (2 * accel),
        # with margin, so the strike lands at full power rather than mid-ramp.
        self.run_up = 1.5 * cfg.max_handle_speed**2 / (2.0 * cfg.max_handle_acceleration)
        # A resting ball leaves at (1 + e) times the handle speed along the
        # contact normal. Discounted, because the handle rarely arrives at the
        # cap and an overestimate turns unreachable bank shots into plans.
        pair_elasticity = cfg.puck_elasticity * cfg.handle_elasticity
        self.launch_speed = (1.0 + pair_elasticity) * cfg.max_handle_speed * strike_fraction

    # -- geometry helpers ------------------------------------------------

    def _unpack(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        own = np.array(
            [observation[0] * cfg.half_width, observation[1] * cfg.half_height], dtype=np.float64
        )
        own_velocity = np.array(
            [observation[2], observation[3]], dtype=np.float64
        ) * cfg.max_handle_speed
        puck = np.array(
            [observation[8] * cfg.half_width, observation[9] * cfg.half_height], dtype=np.float64
        )
        puck_velocity = np.array(
            [observation[10], observation[11]], dtype=np.float64
        ) * cfg.max_puck_speed
        return own, own_velocity, puck, puck_velocity

    def _within_reach(self, point: np.ndarray) -> bool:
        return (
            self.reach_x[0] <= point[0] <= self.reach_x[1]
            and self.reach_y[0] <= point[1] <= self.reach_y[1]
        )

    def _klasks(self, point: np.ndarray) -> bool:
        offset = point - np.asarray(self.own_hole)
        # A margin on top of the klask radius: the handle carries momentum into
        # the spot it is aiming for, so standing on the rim is standing in it.
        return bool(np.hypot(*offset) < self.config.handle_klask_radius + self.config.handle_radius)

    def _intercept(
        self, own: np.ndarray, puck: np.ndarray, puck_velocity: np.ndarray
    ) -> np.ndarray:
        """Where the ball and the handle can actually meet.

        Solved as a fixed point -- where will the ball be by the time I reach
        where the ball will be -- rather than by a fixed lookahead. Against a
        ball coming at us the difference is the whole game: aim at where it is
        now and the shot search puts the strike point *behind* it, so the handle
        retreats toward its own goal to get there and lets the ball run past.
        """
        cfg = self.config
        if float(np.hypot(*puck_velocity)) <= cfg.puck_stop_speed:
            return puck
        # Spin-up costs roughly the time to reach the cap, on top of the travel.
        spin_up = cfg.max_handle_speed / cfg.max_handle_acceleration / cfg.control_dt
        target = puck
        for _ in range(3):
            gap = max(0.0, float(np.hypot(*(target - own))) - self.contact_distance)
            steps = int(
                np.clip(gap / (cfg.max_handle_speed * cfg.control_dt) + spin_up, 0, self.lead_cap)
            )
            position, _ = self.trajectory.advance(
                (puck[0], puck[1]), (puck_velocity[0], puck_velocity[1]), steps * cfg.frame_skip
            )
            target = np.array([position.x, position.y], dtype=np.float64)
        return target

    # -- the shot search -------------------------------------------------

    def _free_magnets(self, observation: np.ndarray) -> list[np.ndarray]:
        cfg = self.config
        magnets: list[np.ndarray] = []
        for index in range(cfg.magnet_count):
            base = BASE_OBSERVATION_FEATURES + index * 5
            if base + 4 >= observation.shape[0]:
                break
            if abs(float(observation[base + 4])) > 0.5:
                continue  # already attached to someone
            if _is_in_a_hole(float(observation[base]), float(observation[base + 1]), self.geometry):
                continue  # captured and inert
            magnets.append(
                np.array(
                    [
                        float(observation[base]) * cfg.half_width,
                        float(observation[base + 1]) * cfg.half_height,
                    ],
                    dtype=np.float64,
                )
            )
        return magnets

    def _score_shot(
        self,
        contact: np.ndarray,
        stand: np.ndarray,
        launch: np.ndarray,
        own: np.ndarray,
        keeper: np.ndarray,
        magnets: list[np.ndarray],
    ) -> tuple[float, bool, int]:
        cfg = self.config
        path = self.trajectory.march(
            (contact[0], contact[1]),
            (launch[0] * self.launch_speed, launch[1] * self.launch_speed),
            max_reflections=self.max_reflections,
        )
        if path.fell_into == "left":
            # Never something to aim for, but when the ball is already rolling
            # at our own hole every option can be one, and the least bad of them
            # still beats standing off and watching it go in.
            travelled = float(np.hypot(*(np.asarray(path.rest) - contact)))
            return -5000.0 + 20.0 * travelled, False, path.reflections

        # A bank off a wall the ball is already resting against is a shot on
        # paper only: the handle is still right there when the ball comes back
        # off the boards, so what happens is a squirt along the wall in whatever
        # direction the contact happened to favour. Such a shot is still worth
        # playing to free a trapped ball -- it just must not be *aimed* with.
        blind = path.reflections >= 1 and self._first_leg(path) < self.min_launch_run
        scores_goal = path.fell_into == "right" and not blind

        # There is somebody standing in front of that hole. The trajectory model
        # knows about walls and friction but not about the other handle, so a
        # shot straight down the middle reads as perfect and gets saved -- and
        # the rebound comes back at our own goal. Discounting by how squarely
        # the keeper is in the way is what makes the search go round them.
        obstructed = self.keeper_weight * max(
            0.0,
            1.0
            - path.closest_approach(keeper) / (cfg.puck_radius + cfg.handle_radius),
        )

        if scores_goal:
            # Fewer walls in the way means fewer places for the model to be
            # wrong, so a direct shot outranks a bank that scores just as well.
            score = (200.0 - 15.0 * path.reflections) * (1.0 - 0.85 * obstructed)
        else:
            rest = np.asarray(path.rest)
            gained = float(np.hypot(*(contact - self.target_hole))) - float(
                np.hypot(*(rest - self.target_hole))
            )
            score = 25.0 * gained
            # A near miss is a good shot the model got slightly wrong, and it
            # leaves the ball deep in their half either way.
            miss = path.closest_approach(self.target_hole) - cfg.puck_capture_radius
            score += 20.0 * max(0.0, 1.0 - miss / (4.0 * cfg.puck_radius))
            # Never trade a shot for leaving the ball sitting on our own hole.
            danger = float(np.hypot(*(rest - self.own_hole)))
            score -= 80.0 * max(0.0, 1.0 - danger / (3.0 * cfg.puck_capture_radius))
            score -= 40.0 * obstructed
            if blind:
                score = 0.3 * score - 20.0

        # Standing on a loose magnet collects it, and two of them lose the point.
        for magnet in magnets:
            reach = cfg.magnet_attraction_range
            overlap = reach - float(np.hypot(*(stand - magnet)))
            if overlap > 0.0:
                score -= 40.0 * (overlap / reach)

        score -= 14.0 * self._reposition_cost(own, contact, stand)
        return score, scores_goal, path.reflections

    def _reposition_cost(
        self, own: np.ndarray, contact: np.ndarray, stand: np.ndarray
    ) -> float:
        """Path length of the orbit the approach will actually walk."""
        from_ball = own - contact
        radius = float(np.hypot(*from_ball))
        if radius < 1e-6:
            return float(np.hypot(*(stand - contact)))
        to_stand = stand - contact
        cosine = float(
            np.clip(
                np.dot(from_ball, to_stand) / (radius * max(1e-9, float(np.hypot(*to_stand)))),
                -1.0,
                1.0,
            )
        )
        swing = float(np.arccos(cosine))
        return self.orbit_radius * swing + abs(radius - float(np.hypot(*to_stand)))

    @staticmethod
    def _first_leg(path: PuckPath) -> float:
        if len(path.points) < 2:
            return 0.0
        start, end = path.points[0], path.points[1]
        return float(np.hypot(end[0] - start[0], end[1] - start[1]))

    def _plan(
        self,
        own: np.ndarray,
        contact: np.ndarray,
        keeper: np.ndarray,
        observation: np.ndarray,
    ) -> ShotPlan | None:
        magnets = self._free_magnets(observation)
        scored: dict[int, ShotPlan] = {}
        for index in range(self.angle_samples):
            angle = 2.0 * np.pi * index / self.angle_samples
            offset = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            # Clamp the ideal stand-off into the half rather than dropping it.
            # Against a wall the ideal spot is inside the boards for every
            # direction worth playing, and rejecting those leaves a ball in the
            # corner untouched -- measured at zero contacts in 150 steps. The
            # handle can still stand closer in and shove; what the shot then
            # does is decided by where it can actually stand, so re-derive the
            # launch from that rather than from the direction we asked for.
            stand = self._legal(contact + offset * self.contact_distance)
            launch = contact - stand
            reach = float(np.hypot(*launch))
            if reach < 0.3 * self.contact_distance:
                continue  # squeezed onto the ball: the push direction is noise
            launch = launch / reach
            score, scores_goal, reflections = self._score_shot(
                contact, stand, launch, own, keeper, magnets
            )
            if not np.isfinite(score):
                continue
            scored[index] = ShotPlan(
                stand=(float(stand[0]), float(stand[1])),
                launch=(float(launch[0]), float(launch[1])),
                depth=reach,
                score=score,
                scores_goal=scores_goal,
                reflections=reflections,
            )
        if not scored:
            self._committed = None
            return None

        best_index = max(scored, key=lambda index: scored[index].score)
        # Stick with the shot already being lined up unless a new one is clearly
        # better. Re-running the search every step otherwise swaps between
        # near-equal candidates and the handle orbits the ball forever without
        # ever arriving -- measured at 149 steps to a first touch.
        held = scored.get(self._committed) if self._committed is not None else None
        if held is not None and held.score + self.switch_margin >= scored[best_index].score:
            return held
        self._committed = best_index
        return scored[best_index]

    # -- execution -------------------------------------------------------

    def _steer(self, own: np.ndarray, target: np.ndarray, speed: float) -> np.ndarray:
        """Command a velocity toward ``target``, easing off as it is reached.

        The handle brakes within ``max_handle_speed**2 / (2 * accel)``, so the
        gain that just avoids overshoot is the speed cap over that distance.
        """
        cfg = self.config
        offset = target - own
        distance = float(np.hypot(*offset))
        if distance < 1e-9:
            return np.zeros(2, dtype=np.float32)
        brake = cfg.max_handle_speed**2 / (2.0 * cfg.max_handle_acceleration)
        wanted = min(speed, speed * distance / max(brake, 1e-6))
        command = offset / distance * (wanted / cfg.max_handle_speed)
        return np.clip(command, -1.0, 1.0).astype(np.float32)

    def _approach(
        self,
        own: np.ndarray,
        own_velocity: np.ndarray,
        contact: np.ndarray,
        plan: ShotPlan,
    ) -> np.ndarray:
        """Line up on the shot line, then run down it.

        The ball leaves along the contact normal, so the launch direction is
        decided entirely by *where the handle is* at the moment of impact, not
        by where it was heading. Being a centimetre off the line at contact
        turns a shot on goal into a wide one -- asin(0.02 / 0.075) is 15
        degrees. So this servos the lateral error to near zero at a standoff and
        only then drives forward, rather than steering at a point past the ball
        and hoping the geometry works out.
        """
        cfg = self.config
        launch = np.asarray(plan.launch)
        behind = -launch
        offset = own - contact
        along = float(np.dot(offset, behind))
        sideways = offset - behind * along
        error = float(np.hypot(*sideways))

        if along >= plan.depth and error <= self.strike_tolerance:
            # On the line and behind the ball: run it down at full speed, with
            # a damped correction so the last centimetres do not drift off.
            lateral_velocity = own_velocity - behind * float(np.dot(own_velocity, behind))
            command = launch + (
                -sideways * self.lateral_gain - lateral_velocity * self.lateral_damping
            ) / cfg.max_handle_speed
            return np.clip(command, -1.0, 1.0).astype(np.float32)

        if along >= self.contact_distance + self.line_clearance:
            # Behind the ball but off its line. Sliding across at this depth
            # cannot touch the ball, so go straight to the wind-up spot.
            run_up = self._fit_run_up(contact, launch, plan.depth)
            return self._steer(own, self._legal(contact + behind * run_up), cfg.max_handle_speed)

        # Beside or in front of the ball: orbit round to the strike side at
        # arm's length. Cutting straight across would clip the ball and knock
        # it somewhere nobody planned.
        radius = float(np.hypot(*offset))
        if radius < 1e-6:
            return self._steer(own, contact + behind * self.orbit_radius, cfg.max_handle_speed)
        heading = offset / radius
        swing = float(np.arccos(float(np.clip(np.dot(heading, behind), -1.0, 1.0))))
        turn = float(np.sign(heading[0] * behind[1] - heading[1] * behind[0]) or 1.0)
        # Judge the whole arc, not just the next waypoint. Testing one step at a
        # time makes the choice flip as the handle moves -- when the short way
        # round clips the hole keep-out only in its middle, the handle sets off
        # each way in turn and oscillates in place instead of going round.
        short = self._arc_blocked(contact, heading, turn, swing)
        long_way = self._arc_blocked(contact, heading, -turn, 2.0 * np.pi - swing)
        if short == 0 or short <= long_way:
            direction, travel = turn, swing
        else:
            direction, travel = -turn, 2.0 * np.pi - swing
        waypoint = contact + self._rotate(heading, direction * min(travel, 0.7)) * self.orbit_radius
        return self._steer(own, self._legal(waypoint), cfg.max_handle_speed)

    @staticmethod
    def _rotate(vector: np.ndarray, angle: float) -> np.ndarray:
        cosine, sine = float(np.cos(angle)), float(np.sin(angle))
        return np.array(
            [
                vector[0] * cosine - vector[1] * sine,
                vector[0] * sine + vector[1] * cosine,
            ],
            dtype=np.float64,
        )

    def _arc_blocked(
        self, contact: np.ndarray, heading: np.ndarray, direction: float, swing: float
    ) -> int:
        """How many sample points along this way round the ball are unusable."""
        blocked = 0
        samples = 8
        for step in range(1, samples + 1):
            point = contact + self._rotate(
                heading, direction * swing * step / samples
            ) * self.orbit_radius
            if not self._within_reach(point) or self._klasks(point):
                blocked += 1
        return blocked

    def _fit_run_up(self, contact: np.ndarray, launch: np.ndarray, depth: float) -> float:
        """How far behind the ball there is room to wind up.

        Enough to reach the speed cap where the board allows it, and whatever is
        left when it does not -- against a wall the answer is often nothing, and
        a short shove is still better than refusing to take the shot.
        """
        wanted = depth + self.run_up
        for fraction in (1.0, 0.75, 0.5, 0.3):
            distance = max(depth, wanted * fraction)
            if self._within_reach(contact - launch * distance):
                return distance
        return depth

    def _defend(self, own: np.ndarray, puck: np.ndarray, puck_velocity: np.ndarray) -> np.ndarray:
        """Hold the ball's line into our half; fall back to a goalie arc."""
        cfg = self.config
        if float(np.hypot(*puck_velocity)) > cfg.puck_stop_speed:
            path = self.trajectory.march(
                (puck[0], puck[1]),
                (puck_velocity[0], puck_velocity[1]),
                max_reflections=self.max_reflections,
            )
            crossing = path.crosses(self.guard_x)
            if crossing is not None:
                target = np.array([self.guard_x, crossing], dtype=np.float64)
                return self._steer(own, self._legal(target), cfg.max_handle_speed)

        to_puck = puck - np.asarray(self.own_hole)
        distance = float(np.hypot(*to_puck))
        if distance < 1e-6:
            target = np.array([self.guard_x, 0.0], dtype=np.float64)
        else:
            target = np.asarray(self.own_hole) + to_puck / distance * self.defend_radius
        return self._steer(own, self._legal(target), cfg.max_handle_speed)

    def _threatens_own_hole(self, puck: np.ndarray, puck_velocity: np.ndarray) -> bool:
        if float(np.hypot(*puck_velocity)) <= self.config.puck_stop_speed:
            return False
        path = self.trajectory.march(
            (puck[0], puck[1]),
            (puck_velocity[0], puck_velocity[1]),
            max_reflections=self.max_reflections,
        )
        return path.fell_into == "left"

    def _save(self, own: np.ndarray, contact: np.ndarray) -> np.ndarray:
        """Get between the ball and our own hole, and never mind the shot.

        Lining a shot up means standing behind the ball, which when the ball is
        already rolling goalward means standing *between it and our own goal
        line* and then having to be perfectly on the shot line before it
        arrives. Measured, that is how every conceded goal happened: attacking,
        deep, with the ball going past. Blocking needs no alignment at all, and
        the handle sitting goal-side sends the rebound back up the board.
        """
        goalward = contact - np.asarray(self.own_hole)
        distance = float(np.hypot(*goalward))
        if distance < 1e-6:
            return self._steer(own, contact, self.config.max_handle_speed)
        block = contact - goalward / distance * self.contact_distance
        return self._steer(own, self._legal(block), self.config.max_handle_speed)

    def _legal(self, target: np.ndarray) -> np.ndarray:
        """Clamp a target into the half, and off our own hole."""
        clamped = np.array(
            [
                np.clip(target[0], *self.reach_x),
                np.clip(target[1], *self.reach_y),
            ],
            dtype=np.float64,
        )
        offset = clamped - np.asarray(self.own_hole)
        distance = float(np.hypot(*offset))
        keep_out = self.config.handle_klask_radius + self.config.handle_radius
        if distance >= keep_out:
            return clamped
        if distance < 1e-6:
            offset = np.array([-1.0, 1.0], dtype=np.float64)
            distance = float(np.hypot(*offset))
        return np.asarray(self.own_hole) + offset / distance * keep_out

    def _keep_clear(
        self,
        action: np.ndarray,
        own: np.ndarray,
        own_velocity: np.ndarray,
        hazard: np.ndarray,
        rim: float,
    ) -> np.ndarray:
        """Cancel motion into a hazard, but only when it is actually on the way in.

        Two things have to be true before this fires: the handle's own path
        enters the rim, and there is no longer room to brake before it does. A
        fixed keep-out bubble instead has to assume the worst case, which at
        full speed is 0.09 across -- wide enough to fence off the ground behind
        a ball resting near the hole, and it left the planner circling a
        perfectly reachable ball for 150 steps without ever touching it. A plain
        radial-speed test has the same problem for a different reason: it fires
        on anything moving *past* the hazard as hard as on something aimed at it.
        """
        cfg = self.config
        away = own - hazard
        distance = float(np.hypot(*away))
        if distance < 1e-6:
            return np.array([1.0, 0.0], dtype=np.float32)
        direction = away / distance
        speed = float(np.hypot(*own_velocity))
        urgent = distance < rim + self.hole_margin
        if not urgent and speed > 1e-6:
            along = float(np.dot(-away, own_velocity / speed))
            miss_squared = distance * distance - along * along
            if along > 0.0 and miss_squared < rim * rim:
                entry = along - float(np.sqrt(max(0.0, rim * rim - miss_squared)))
                braking = speed * speed / (2.0 * cfg.max_handle_acceleration)
                urgent = entry <= braking + self.hole_margin
        if not urgent:
            return action

        commanded = -float(np.dot(action, direction))
        if commanded > 0.0:
            action = action + direction * commanded
        urgency = float(np.clip(1.0 - (distance - rim) / max(rim, 1e-6), 0.0, 1.0))
        return np.clip(action + direction * 2.0 * urgency, -1.0, 1.0)

    def act(self, observation: np.ndarray) -> np.ndarray:
        cfg = self.config
        own, own_velocity, puck, puck_velocity = self._unpack(observation)

        # The handle stops at the halfway line, so a ball beyond its contact
        # reach cannot be played at all and the only useful thing to do is get
        # in the way of wherever it comes back.
        contact = self._intercept(own, puck, puck_velocity)
        reachable = contact[0] <= self.reach_x[1] + self.contact_distance
        keeper = np.array(
            [observation[4] * cfg.half_width, observation[5] * cfg.half_height], dtype=np.float64
        )
        if reachable and self.save_first and self._threatens_own_hole(puck, puck_velocity):
            action = self._save(own, contact)
        else:
            plan = self._plan(own, contact, keeper, observation) if reachable else None
            if plan is None:
                action = self._defend(own, puck, puck_velocity)
            else:
                action = self._approach(own, own_velocity, contact, plan)

        # Hole avoidance goes last: a klask loses the point outright, so it must
        # be able to override the magnet push rather than the reverse.
        for magnet in self._free_magnets(observation):
            action = self._keep_clear(
                action, own, own_velocity, magnet, cfg.handle_radius + cfg.magnet_radius
            )
        return self._keep_clear(
            action,
            own,
            own_velocity,
            np.asarray(self.own_hole),
            cfg.handle_klask_radius + 0.5 * cfg.handle_radius,
        )


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
