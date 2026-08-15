from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ArenaConfig:
    """Board and physics constants.

    The speed-related values are the original ones put through
    ``scale_arena_speeds`` at 0.4: a slower board, with control_dt left
    alone so the policy still decides 30 times a second and therefore
    steers more finely. Retune with that helper rather than by hand --
    velocities, accelerations, drag coefficients and step budgets each
    take a different power of the scale, and moving one alone silently
    changes how far the puck rolls or how hard a shot lands.
    """

    # Real Klask playing field is 40 x 30 cm; 1 unit = 20 cm.
    width: float = 2.0
    height: float = 1.5
    goal_radius: float = 0.075
    goal_center_offset: float = 0.225
    wall_radius: float = 0.018
    puck_radius: float = 0.035
    handle_radius: float = 0.04
    magnet_count: int = 3
    magnet_radius: float = 0.026
    magnet_mass: float = 0.03
    magnet_attraction_range: float = 0.22
    magnet_attraction_strength: float = 0.48
    magnet_max_force: float = 0.128
    max_magnet_speed: float = 0.6
    magnet_friction: float = 0.75
    magnet_linear_friction: float = 1.2
    magnet_slide_friction: float = 0.16
    magnet_stop_speed: float = 0.008
    magnet_elasticity: float = 0.35
    magnet_attach_frames: int = 5
    magnet_release_distance: float = 0.16
    magnet_score_threshold: int = 2
    # The puck serves anywhere on the board, clear of the holes. It used to
    # land only in a narrow band in front of the near handle, so a ball that
    # had got past the agent was a state it never saw and had no answer to.
    puck_start_margin: float = 0.02
    # Curriculum: cap how far the ball may serve from the handle in whose half
    # it lands, so early training is guaranteed reachable contact. 0 means no
    # cap, i.e. the ball anywhere and the handle anywhere.
    puck_start_max_handle_distance: float = 0.0
    # Extra room beyond the radius at which a body is actually captured. Both
    # clearances are derived from goal_radius rather than fixed, because the
    # curriculum overrides it: at goal_radius 0.15 the puck capture radius is
    # 0.146, so a fixed 0.11 clearance served the ball straight into the goal.
    start_hole_margin: float = 0.04
    puck_mass: float = 0.045
    max_handle_speed: float = 0.72
    # Actions are target velocities, but a hand (or a gantry) cannot change
    # velocity instantly. Bounding the change keeps the handle moving
    # continuously and makes shot power depend on the run-up, instead of every
    # contact landing at full speed.
    max_handle_acceleration: float = 2.88
    max_puck_speed: float = 1.28
    physics_dt: float = 1.0 / 120.0
    frame_skip: int = 4
    # Pair restitutions. Shape elasticities multiply in pymunk, so the puck is
    # kept perfectly elastic and each partner carries the whole coefficient.
    puck_elasticity: float = 1.0
    wall_elasticity: float = 0.55
    handle_elasticity: float = 0.70
    wall_friction: float = 0.6
    puck_friction: float = 0.2
    # Rolling resistance: a linear (viscous) term plus a constant deceleration.
    # The constant term is what actually brings a slow ball to rest, so the
    # policy sees stationary pucks the way it will on the real board.
    puck_linear_drag: float = 0.09
    puck_rolling_friction: float = 0.0528
    puck_stop_speed: float = 0.008
    # Handles are confined to their own half, so a puck that stops in the other
    # half is unreachable by everyone and the rest of the episode is dead. Put
    # it back in play after this many motionless control steps, the way a player
    # would re-serve a stuck ball. 0 disables. Against a real opponent this
    # fires ~0.1x per episode -- it is a backstop, not a game mechanic. Kept
    # generous so a clumsy early policy still has to learn to go fetch a
    # resting puck rather than wait the timer out.
    dead_ball_steps: int = 150
    # Velocity decay is modelled explicitly per body, so the space adds none.
    damping: float = 1.0
    max_steps: int = 2250

    @property
    def half_width(self) -> float:
        return self.width / 2.0

    @property
    def half_height(self) -> float:
        return self.height / 2.0

    @property
    def goal_center_x(self) -> float:
        return self.half_width - self.goal_center_offset

    @property
    def puck_wall_elasticity(self) -> float:
        """Restitution pymunk resolves a puck/wall contact with."""
        return self.puck_elasticity * self.wall_elasticity

    @property
    def magnet_wall_elasticity(self) -> float:
        """Restitution pymunk resolves a magnet/wall contact with."""
        return self.magnet_elasticity * self.wall_elasticity

    @property
    def puck_capture_radius(self) -> float:
        return math.sqrt(self.goal_radius**2 - self.puck_radius**2)

    @property
    def puck_start_hole_clearance(self) -> float:
        """How far a served puck must sit from a hole to not fall straight in."""
        return self.puck_capture_radius + self.start_hole_margin

    @property
    def handle_start_hole_clearance(self) -> float:
        """How far a placed handle must sit from its own hole to not klask."""
        return self.handle_klask_radius + self.start_hole_margin

    @property
    def handle_klask_radius(self) -> float:
        return self.goal_radius - self.handle_radius / 2.0

    @property
    def magnet_capture_radius(self) -> float:
        return math.sqrt(self.goal_radius**2 - self.magnet_radius**2)

    def goal_center(self, side: str) -> tuple[float, float]:
        if side == "left":
            return (-self.goal_center_x, 0.0)
        if side == "right":
            return (self.goal_center_x, 0.0)
        raise ValueError(f"unknown side {side!r}; expected 'left' or 'right'")

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.frame_skip


@dataclass(frozen=True)
class RewardConfig:
    terminal_goal: float = 12.0
    progress: float = 0.6
    # Reward for a shot whose path leads into the opponent's hole, counting
    # bank shots off the boards -- scoring off a wall is normal in Klask, and
    # rewarding only straight-on shots would teach otherwise.
    aim: float = 0.0
    aim_reflections: int = 2
    puck_position: float = 0.02
    puck_speed: float = 0.08
    contact: float = 0.08
    puck_distance: float = 0.025
    defense: float = 0.04
    own_goal_danger: float = 0.05
    magnet_attached_penalty: float = 0.05
    magnet_proximity_penalty: float = 0.01
    magnet_attach_penalty: float = 0.0
    magnet_pull_penalty: float = 0.0
    own_side_penalty: float = 0.0
    time_penalty: float = 0.0005
    action_penalty: float = 0.0003


REWARD_PROFILES: dict[str, RewardConfig] = {
    "balanced": RewardConfig(),
    "aggressive": RewardConfig(
        terminal_goal=14.0,
        progress=0.9,
        puck_position=0.04,
        puck_speed=0.14,
        contact=0.12,
        puck_distance=0.035,
        defense=0.025,
        own_goal_danger=0.035,
        time_penalty=0.0002,
        action_penalty=0.00015,
    ),
    "defensive": RewardConfig(
        terminal_goal=14.0,
        progress=0.45,
        puck_position=0.01,
        puck_speed=0.05,
        contact=0.07,
        puck_distance=0.03,
        defense=0.09,
        own_goal_danger=0.12,
        time_penalty=0.0003,
        action_penalty=0.0002,
    ),
    "possession": RewardConfig(
        terminal_goal=12.0,
        progress=0.55,
        puck_position=0.02,
        puck_speed=0.08,
        contact=0.18,
        puck_distance=0.07,
        defense=0.045,
        own_goal_danger=0.06,
        time_penalty=0.0002,
        action_penalty=0.0001,
    ),
    "simple": RewardConfig(
        terminal_goal=12.0,
        progress=0.0,
        puck_position=0.0,
        puck_speed=0.0,
        contact=0.0,
        puck_distance=0.0,
        defense=0.0,
        own_goal_danger=0.0,
        magnet_attached_penalty=0.0,
        magnet_proximity_penalty=0.0,
        magnet_attach_penalty=2.0,
        magnet_pull_penalty=0.05,
        own_side_penalty=0.01,
        time_penalty=0.0,
        action_penalty=0.0,
    ),
    # simple, rebalanced against the edge-hugging equilibrium: the pull penalty
    # is weak enough not to dominate, staying near the puck earns more than
    # hiding, and stalling with the puck on the own side is costly.
    "simple_v2": RewardConfig(
        terminal_goal=12.0,
        progress=0.0,
        puck_position=0.0,
        puck_speed=0.0,
        contact=0.0,
        puck_distance=0.02,
        defense=0.0,
        own_goal_danger=0.0,
        magnet_attached_penalty=0.0,
        magnet_proximity_penalty=0.0,
        magnet_attach_penalty=2.0,
        magnet_pull_penalty=0.01,
        own_side_penalty=0.04,
        time_penalty=0.0,
        action_penalty=0.0,
    ),
    # simple_v2 rebalanced for a puck that can come to rest. own_side is a
    # one-sided tax, so with a ball that stays where it stops the best play
    # became "knock it into their half and abandon it": both agents park, the
    # ball is never played, episodes run to the cap, and own_side (-18.5/ep
    # measured) swamps terminal_goal (+12). Cut it to a nudge. puck_distance
    # carries the load instead -- it is differenced, so being nearer the puck
    # than the opponent pays, which is what breaks a mutual-ignore standoff.
    "simple_v3": RewardConfig(
        terminal_goal=12.0,
        progress=0.0,
        puck_position=0.0,
        puck_speed=0.0,
        contact=0.0,
        puck_distance=0.05,
        defense=0.0,
        own_goal_danger=0.0,
        magnet_attached_penalty=0.0,
        magnet_proximity_penalty=0.0,
        magnet_attach_penalty=2.0,
        magnet_pull_penalty=0.01,
        own_side_penalty=0.005,
        # NOTE: time_penalty is inert. Every component except own_side is
        # differenced against the opponent's, and both agents get the same
        # time term, so it cancels to exactly zero. Making it one-sided would
        # give a real anti-stall lever, but it silently changes the four
        # legacy profiles, so it is left alone deliberately.
        time_penalty=0.0,
        action_penalty=0.0,
    ),
    # Experiment 1. simple_v2 with puck_distance removed. distance rewards
    # standing next to the puck, which directly opposes own_side's "clear it
    # off your half" -- and clearing requires striking. With a puck that can
    # rest, loitering beside it was the better deal (measured 2% strike rate on
    # resting balls). Dropping distance leaves own_side as a clean strike
    # incentive.
    "simple_v4": RewardConfig(
        terminal_goal=12.0,
        progress=0.0,
        puck_position=0.0,
        puck_speed=0.0,
        contact=0.0,
        puck_distance=0.0,
        defense=0.0,
        own_goal_danger=0.0,
        magnet_attached_penalty=0.0,
        magnet_proximity_penalty=0.0,
        magnet_attach_penalty=2.0,
        magnet_pull_penalty=0.01,
        own_side_penalty=0.04,
        time_penalty=0.0,
        action_penalty=0.0,
    ),
    # simple_v5 with a single reflection. Depth 2 predicts a bank shot with
    # 68% accuracy against depth 1's 79%, the marginal second-bounce shots
    # landing near chance -- and at depth 2 aim did not beat plain progress
    # (89-70 to progress-only over 200 games, p=0.15). This tests whether the
    # sharper signal is what the idea needed.
    "simple_v5_aim1": RewardConfig(
        terminal_goal=12.0,
        progress=0.3,
        aim=0.15,
        aim_reflections=1,
        puck_position=0.0,
        puck_speed=0.0,
        contact=0.0,
        puck_distance=0.0,
        defense=0.0,
        own_goal_danger=0.0,
        magnet_attached_penalty=0.0,
        magnet_proximity_penalty=0.0,
        magnet_attach_penalty=2.0,
        magnet_pull_penalty=0.01,
        own_side_penalty=0.04,
        time_penalty=0.0,
        action_penalty=0.0,
    ),
    # simple_v5_noaim re-balanced for the 0.4x board. Per-step costs have to
    # scale with it: the same rally now takes 2.5x the control steps, so a flat
    # own_side integrated to about -10 an episode against a terminal of 12 --
    # the ratio that produced the mutual-park stalemate before. progress is
    # deliberately NOT scaled: it telescopes to (final_x - initial_x), so its
    # episode total is already dilation-invariant.
    "slow_v1": RewardConfig(
        terminal_goal=12.0,
        progress=0.3,
        aim=0.0,
        puck_position=0.0,
        puck_speed=0.0,
        contact=0.0,
        puck_distance=0.0,
        defense=0.0,
        own_goal_danger=0.0,
        magnet_attached_penalty=0.0,
        magnet_proximity_penalty=0.0,
        magnet_attach_penalty=2.0,
        magnet_pull_penalty=0.004,
        own_side_penalty=0.016,
        time_penalty=0.0,
        action_penalty=0.0,
    ),
    # slow_v1 plus defensive shaping. On the slow board with the ball serving
    # anywhere -- including behind the handle, near its own goal -- slow_v1
    # concedes 30 goals to the fast-board model's 13 while scoring the same,
    # and nothing in it rewards defending except the sparse terminal. Both new
    # terms are per-step, so they carry the 0.4 dilation factor.
    "slow_v2": RewardConfig(
        terminal_goal=12.0,
        progress=0.3,
        aim=0.0,
        puck_position=0.0,
        puck_speed=0.0,
        contact=0.0,
        puck_distance=0.0,
        defense=0.016,
        own_goal_danger=0.02,
        magnet_attached_penalty=0.0,
        magnet_proximity_penalty=0.0,
        magnet_attach_penalty=2.0,
        magnet_pull_penalty=0.004,
        own_side_penalty=0.016,
        time_penalty=0.0,
        action_penalty=0.0,
    ),
    # slow_v2 with the defensive terms roughly doubled. slow_v2 cut conceding
    # from 30 to 18 against the seed's 13 without costing any offence -- it
    # still scores 25 -- so there is room to lean harder on the term that is
    # visibly working. Watch that it does not tip into passivity: if goals
    # scored fall below the seed's 25, this has gone too far.
    "slow_v3": RewardConfig(
        terminal_goal=12.0,
        progress=0.3,
        aim=0.0,
        puck_position=0.0,
        puck_speed=0.0,
        contact=0.0,
        puck_distance=0.0,
        defense=0.032,
        own_goal_danger=0.04,
        magnet_attached_penalty=0.0,
        magnet_proximity_penalty=0.0,
        magnet_attach_penalty=2.0,
        magnet_pull_penalty=0.004,
        own_side_penalty=0.016,
        time_penalty=0.0,
        action_penalty=0.0,
    ),
    # Ablation of simple_v5: progress without aim. simple_v5 beat the previous
    # leader 39-12, but it changed two things at once, so this isolates whether
    # the bank-shot aim term earned its keep or plain "drive it downfield" did
    # all the work.
    "simple_v5_noaim": RewardConfig(
        terminal_goal=12.0,
        progress=0.3,
        aim=0.0,
        puck_position=0.0,
        puck_speed=0.0,
        contact=0.0,
        puck_distance=0.0,
        defense=0.0,
        own_goal_danger=0.0,
        magnet_attached_penalty=0.0,
        magnet_proximity_penalty=0.0,
        magnet_attach_penalty=2.0,
        magnet_pull_penalty=0.01,
        own_side_penalty=0.04,
        time_penalty=0.0,
        action_penalty=0.0,
    ),
    # Experiment 2. simple_v4 plus explicit shot shaping: progress for driving
    # the puck downfield, aim for pointing it at the hole including bank shots.
    # progress is already speed-scaled (it is a per-step displacement) and aim
    # is scaled by puck speed, so both pay more for a decisive strike.
    "simple_v5": RewardConfig(
        terminal_goal=12.0,
        progress=0.3,
        aim=0.15,
        aim_reflections=2,
        puck_position=0.0,
        puck_speed=0.0,
        contact=0.0,
        puck_distance=0.0,
        defense=0.0,
        own_goal_danger=0.0,
        magnet_attached_penalty=0.0,
        magnet_proximity_penalty=0.0,
        magnet_attach_penalty=2.0,
        magnet_pull_penalty=0.01,
        own_side_penalty=0.04,
        time_penalty=0.0,
        action_penalty=0.0,
    ),
}


def scale_arena_speeds(config: ArenaConfig, scale: float) -> ArenaConfig:
    """Slow the whole game down by ``scale`` without changing its geometry.

    This is a time dilation, so the constants do not all move together. Under
    t -> t/scale: velocities take one factor, accelerations and forces take two
    (length/time squared), a viscous coefficient is a reciprocal time so it
    takes one, and anything counted in steps lasts longer so it divides.

    ``control_dt`` deliberately stays put. That is the whole point: the board
    plays out slower while the policy still decides 30 times a second, so it
    gets finer control -- and the same rally then needs more steps, which is
    why the step budgets grow.
    """
    if scale <= 0.0:
        raise ValueError(f"speed scale must be positive, got {scale}")
    squared = scale * scale

    def steps(count: int) -> int:
        return max(1, round(count / scale))

    return replace(
        config,
        # velocities
        max_handle_speed=config.max_handle_speed * scale,
        max_puck_speed=config.max_puck_speed * scale,
        max_magnet_speed=config.max_magnet_speed * scale,
        puck_stop_speed=config.puck_stop_speed * scale,
        magnet_stop_speed=config.magnet_stop_speed * scale,
        # accelerations and forces
        max_handle_acceleration=config.max_handle_acceleration * squared,
        puck_rolling_friction=config.puck_rolling_friction * squared,
        magnet_slide_friction=config.magnet_slide_friction * squared,
        magnet_attraction_strength=config.magnet_attraction_strength * squared,
        magnet_max_force=config.magnet_max_force * squared,
        # reciprocal-time coefficients
        puck_linear_drag=config.puck_linear_drag * scale,
        magnet_linear_friction=config.magnet_linear_friction * scale,
        # durations, counted in steps
        max_steps=steps(config.max_steps),
        dead_ball_steps=steps(config.dead_ball_steps),
        magnet_attach_frames=steps(config.magnet_attach_frames),
    )


def reward_profile(name: str) -> RewardConfig:
    try:
        return REWARD_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(REWARD_PROFILES))
        raise ValueError(f"unknown reward profile {name!r}; expected one of: {known}") from exc


AGENTS: tuple[str, str] = ("left", "right")
OPPONENT: dict[str, str] = {"left": "right", "right": "left"}
MAGNET_FEATURES = 5
OBSERVATION_SIZE = 16 + ArenaConfig.magnet_count * MAGNET_FEATURES + 2
