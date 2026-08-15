# A rule-based planner, and what it took to make it play

`PlannerOpponent` searches for a shot at the opponent's hole and takes it. It
beats `exp3_24M_baseline` -- the best model in the repo -- and it beats the
scripted expert the behaviour cloning uses by a wide margin.

| match (sides swapped, 1500-step episodes) | result | p |
| --- | --- | --- |
| vs `exp3_24M_baseline`, seed A | 42-17 | 0.002 |
| vs `exp3_24M_baseline`, seed B | 46-29 | 0.064 |
| **both seeds pooled** | **88-46** | **0.0003** |
| vs `slow_defense1500` (best slow-board model) | 40-11 | 0.0001 |
| vs `StrikerOpponent` (the BC expert) | 35-11 | 0.0005 |

Everything below was found by measuring. The first working version scored a
mean shot-on-target of **0.04**, which is worse than the scripted striker it was
meant to replace.

## The design, in one paragraph

Sample handle positions on a ring around the ball. For each, the launch
direction is fixed by geometry -- the ball leaves along the contact normal, so
it is decided by *where the handle stands*, not by where it was heading. March
each resulting shot through `PuckTrajectory`, the same rolling-and-bouncing
model the simulation resolves, and keep whichever ends nearest the hole. Bank
shots, corner recoveries and straight shots all fall out of the one search,
because they differ only in which ring positions the geometry still allows.

## Five things it needed, each measured failing without them

**Run down the shot line, not at a point past the ball.** The launch direction
is the contact normal, so being 2 cm off the line at impact is `asin(0.02 /
0.075)` = 15 degrees off target. Steering at a point beyond the ball and
letting the geometry work out gives glancing contact: mean aim **0.04**.
Servoing the lateral error to under 12 mm at a standoff and only then driving
forward gives **0.46** on the same probe.

**Clamp the strike point into the half; do not reject it.** Against a wall the
ideal stand-off is inside the boards for every direction worth playing --
the handle centre can only get 13 mm past a ball resting on the top wall.
Rejecting those candidates left a ball in the corner **untouched for 150
steps**, until the dead-ball timer re-served it. Clamping the point into the
reachable half and re-deriving the launch from where the handle can actually
stand recovers the ball, and the plan then carries its own achievable depth,
because the strike phase can never reach a nominal 75 mm behind a wall ball.

**Hold the chosen shot.** Re-running the search every step swaps between
near-equal candidates, and the handle orbits a reachable ball forever: **149
steps to a first touch**. A `switch_margin` of 15 points fixes it. The orbit
direction needs the same treatment for a different reason -- testing only the
next waypoint makes the choice flip as the handle moves, so the whole arc is
judged at once.

**Solve interception as a fixed point.** Aiming at where an incoming ball *is*
puts the strike point behind it, so the handle retreats toward its own goal to
get there and lets the ball run past. "Where will the ball be when I reach
where the ball will be", iterated three times, is what makes it meet the ball.

**Size the hazard guard by braking distance, not by a bubble.** A fixed
keep-out has to assume worst-case braking, which at full speed is 0.09 across
-- wide enough to fence off the ground behind a ball resting near the hole. A
plain radial-speed test has the same problem for a different reason: it fires
on anything moving *past* the hole as hard as on something aimed at it. The
test that works asks whether the handle's own path enters the rim, and whether
there is still room to stop.

## Saving beats shooting, and it is worth the whole margin

Against the trained model the planner out-shot it three to one on aim -- 0.55
against 0.16, 44% of shots on target against 10% -- and still lost 35-53.

The diagnosis was one line of instrumentation: fifteen steps before *every*
conceded goal, the planner was in attack mode, deep, with the ball already
rolling goalward past it. Lining a shot up means standing behind the ball,
which for a goalward ball means standing between it and our own goal line and
then having to be exactly on the shot line before it arrives. That is not a
save; it is a race the ball wins.

So a threatened hole now takes priority: march the ball, and if the path falls
into our own hole, get goal-side of it and never mind the shot. Blocking needs
no alignment at all, and the handle sitting goal-side sends the rebound back up
the board.

| variant | vs `exp3_24M_baseline` | p |
| --- | --- | --- |
| save + keeper (default) | 42-17 | 0.002 |
| keeper term off | 40-22 | 0.030 |
| strike tolerance 0.02 rather than 0.012 | 35-22 | 0.111 |
| save off | 35-35 | 1.000 |

The keeper term is the other half of the same blindness: `PuckTrajectory` knows
about walls and friction but not about the other handle, so a shot straight
down the middle reads as perfect, gets saved, and rebounds at our own goal.
Discounting a shot by how squarely the keeper sits on its path is what makes
the search go round them.

Defensive positioning was swept and the starting guess was already the best of
the three tried: `guard_x` -0.45 gave 31-43 where -0.62 gave 25-49 and -0.28
gave 18-55. Sitting too far forward is much worse than sitting too far back.

## What it is good for

- **The weakness the user reported.** On a probe of twelve resting-ball
  positions -- corners, back wall, edges -- it strikes all twelve and averages
  **0.76** shot-on-target, against 0.34 for the striker and 0.30 for the
  heuristic. Neither scripted expert clears a corner reliably.
- **A better behaviour-cloning expert.** `training.py` clones
  `StrikerOpponent`. The clone's ceiling is the expert, and this expert is
  much stronger, so it is the obvious next thing to try -- along with adding it
  to the self-play pool.
- **A real-time baseline.** 3.3 ms per control step of the 33 ms budget, so it
  can drive the physical board directly.

## The caveat worth stating

The planner beats the trained model *in this simulator*, using a model of the
simulator's own physics. That is exactly the thing that will not transfer: on
the real board the rolling constants, the restitution and the contact geometry
are all approximations, and `march` is only as good as they are. The trained
policy has no such dependency. Treat the planner as the strong sim-side
baseline and the better expert to clone, not as the thing to ship to hardware
unmeasured.
