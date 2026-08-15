# Slowing the board, and what broke

The board was slowed to 0.4x so the policy steers more finely and the sim sits
closer to what a real gantry can do. `control_dt` stayed at 1/30 s, so the same
rally now takes 2.5x the control steps. Everything below follows from that, and
most of it was only found by measuring.

## Scaling is not one number

`scale_arena_speeds` exists because the constants take different powers of the
scale. Under a time dilation `t -> t/s`:

| quantity | factor |
| --- | --- |
| velocities, stop thresholds | `s` |
| accelerations, forces | `s²` |
| viscous drag coefficients (reciprocal times) | `s` |
| durations counted in steps | `1/s` |
| `control_dt`, `physics_dt`, restitutions, geometry | unchanged |

Retune with the helper, not by hand. Moving one constant alone silently changes
how far the ball rolls or how hard a shot lands.

Two things that looked like constants but were not:

- The serve velocity was hardcoded at 0.25. That was 8% of the old speed cap
  and 20% of the new one, so a re-served ball came out disproportionately fast
  and never settled. Held as a fraction of the cap now.
- `gamma` was 0.985, tuned on the fast board. With 2.5x the steps per rally a
  goal 300 steps out was discounted to 0.011 instead of 0.16, roughly 15x
  weaker terminal credit. `0.985 ** 0.4 = 0.994` restores the discount per unit
  of *game* rather than per control step.

## The reward findings

Measured, not assumed:

- **`puck_distance` caused the stalemate**, not anything else. It pays for
  standing *near* the ball, so once the ball could come to rest, loitering beat
  hitting: the policy struck 2% of resting balls. Every run that kept it
  collapsed to episodes at the cap; every run without it is stable. Removing it
  is necessary *and* sufficient to avoid the stalemate.
- **`progress` is a real further gain** but is not the cure. A no-shaping run is
  also stable, but only draws with the previous leader (44-44) where a
  `progress` run beats it 88-45.
- **`aim` does not pay.** A bank-shot predictor, made accurate (68% of bank
  verdicts become goals, against a 4% base rate), still lost 89-70 to
  progress-only. The *more* accurate depth-1 variant came last of all, so
  predictor quality was not the limit -- `aim` pays for being *pointed* at the
  goal, a state reachable without ever scoring, so it competes with the terminal
  reward instead of feeding it.
- **`time_penalty` has never done anything.** Every component except `own_side`
  is differenced against the opponent's copy, and both agents receive the same
  time term, so it cancels to exactly zero in every profile.
- **Per-step costs must scale with the dilation.** `own_side` integrated to
  -10.4 an episode against a terminal of 12 -- the ratio that produced the
  earlier mutual-park failure. `slow_v1` scales it to -4.16. `progress` is
  deliberately *not* scaled: it telescopes to `final_x - initial_x`, so its
  episode total is already dilation-invariant.

## Training on the slow board

From-scratch training stalls; warm-starting from a model that already knows the
game does not. The chain, measured:

1. SB3 defaults the action std to **1.0** on a `[-1, 1]` action space while
   expert actions average **0.35** -- exploration noise about three times the
   signal, and `ent_coef 0.01` inflates it further (std reached 1.24).
2. So rollouts are effectively random. Behaviour cloning scored a cosine of
   **0.63** against the expert on expert states but **0.17** on its own, and
   touched the ball **0.1%** of steps against the expert's **18%**.
3. Cutting the noise (`log_std_init -0.9`, where the working model ended up) is
   necessary but **not sufficient**: a pilot still reached 0.14% contact, with
   actions running *opposite* to the expert on its own states (cosine -0.71).

That last point is compounding error, not a tuning problem: BC fits expert
states, the policy drifts to states it handles badly, and the reward makes
avoiding the ball locally safe. More noise tuning will not fix it.

`--spawn-distance` caps how far the ball serves from the handle in whose half it
lands, so early stages guarantee reachable contact.

## Episode budget

At `max_steps 750` only **44%** of episodes resolve before truncation, so most
never deliver a terminal reward at all. Two decent models drew 51 of 80 games.

| cap | resolves |
| --- | --- |
| 750 | 44% |
| 1125 | 53% |
| 1500 | 59% |
| 2250 | 68% |

Warm-start self-play resolves at a median of 986 steps, p75 2138. The naive
dilation answer (450 x 2.5 = 1125) lands at 53%.

**That reasoning did not survive contact with the experiment.** Two runs seeded
identically, same league, curriculum and reward, differing only in episode
budget: the 750-step run beat the 1500-step run 51-29 (p=0.02). The truncation
mechanism is real but something else outweighs it -- most likely that shorter
episodes mean more resets, so more distinct starting states per unit of
compute, and truncation carries no penalty so there is nothing to lose by not
resolving. Caveat: the 750 run used 16 envs and the 1500 run 12, so it is not a
perfectly controlled comparison.

Resolution rate is a reasonable thing to measure and a bad thing to choose on.

## Things that bite when spawns are randomised

Serving the ball and the handles anywhere, rather than from near-fixed spots,
exposed several latent bugs. Worth knowing before widening any other
distribution:

- `puck_start_hole_clearance` was fixed while `puck_capture_radius` derives from
  `goal_radius`, which the curriculum overrides. At the stage-1 radius the ball
  served *inside* the goal and 2.6% of episodes ended on step one with a random
  winner. Both clearances derive from `goal_radius` now.
- Magnets are created after the ball and handles are placed, so uniform spawns
  landed on them -- a handle within attach distance collects one in two steps,
  which the reward punishes before either agent has acted.
- The scripted experts sized their own-hole keep-out from the *default* config
  while the curriculum changed `goal_radius`, so they klasked 20% of the time at
  stage 1. Behaviour cloning would have copied it. Their geometry is per
  instance now.
- A potential field around the hole needs to cancel motion into the hazard, not
  blend against it. Blending still nets inward through the outer half of the
  bubble, and by then the acceleration limit leaves no room to brake. Behind the
  hole it also needs a sideways escape, since "directly away" there means
  driving into the back wall and every route back into play reads as inward.


## Training results, and the honest position

Nothing trained on the slow board has beaten the fast-board model yet.
`exp3_24M_baseline` remains the one to use.

| model | vs seed (sides swapped) | goals for/against | quality |
| --- | --- | --- | --- |
| seed `exp3_24M_baseline` | -- | 25 / 13 | 5.08 |
| warm start, no league | 26-74 (p<0.001) | 25 / 32 | 4.42 |
| warm start, league restored | 30-50 (p=0.03) | 22 / 30 | 4.28 |
| plus defensive shaping (`slow_v2`) | 38-56 (p=0.08) | 25 / 18 | 4.92 |
| defensive shaping doubled (`slow_v3`) | 22-60 (p<0.001) | 22 / 26 | 4.67 |

Two causes found, in order:

1. **The self-play league is not carried across a warm start.** Resuming into
   a fresh output dir restores zero checkpoint opponents, so a model forged
   against 480 snapshots practises against passive and random instead. Damage
   was measurable at +100k steps. Copying eligible snapshots across moved the
   result from 26-74 to 30-50.
2. **The reward has no defensive term.** Scoring is unchanged across all three
   models at 22-25 goals; the gap is entirely conceding, 30 against the seed's
   13. `defense` and `own_goal_danger` are zero in the whole `simple*` family,
   which was harmless while the ball always started in front of the handle and
   is not harmless now that it serves anywhere. `slow_v2` restores both, and
   conceding drops from 30 to 18 with quality up from 4.28 to 4.92. The gap to
   the seed is no longer significant, though it has not been reversed.

The three trained models are mutually non-transitive -- defensive shaping beats
league-only against the seed and on the benchmark but loses to it directly,
36-46 -- so only the comparisons against the fixed reference mean much. Cycles
like this are normal in a self-play population and are a reason not to rank
models by a single match.

### The methodological trap

Within-training reward curves are not evidence of improvement. The no-league
run's reward rose throughout while the model was getting worse -- it was
climbing against progressively weaker opposition. Only head-to-head against a
fixed reference exposed it. Benchmark aggregate scores are also measured
against third parties and can disagree with a direct match; when they conflict,
the direct match is what counts.

Episode budget shows up here too: at 750 steps, two decent models draw 39 of 60
games, so most comparisons at that budget carry very little signal.


## Positional shaping terms get farmed

Doubling the defensive weight made the model worse on *both* counts: it scored
less (25 -> 22) and conceded **more** (18 -> 26), despite the extra reward
being entirely for defending. More defensive reward bought worse defence.

That is the same failure as two earlier terms, and the pattern is worth
naming. `defense` pays for a *position* -- goal-side and aligned with the ball
-- not for an interception. Weight it heavily and the agent optimises the
position and stops actually stopping anything. Compare:

- `puck_distance` pays for standing near the ball, not for hitting it, and
  produced a policy that struck 2% of resting balls.
- `aim` pays for being pointed at the goal, a state reachable without ever
  shooting, and lost to having no aim term at all.
- `defense` pays for standing in the right place, not for saving, and doubling
  it increased goals conceded.

`progress` is the exception that shows the rule: it can only be earned by
actually moving the ball, so it cannot be collected from a standstill. When
adding shaping here, prefer terms that require the event to happen over terms
that pay for looking like it might.

`slow_v2` is the right level for the defensive terms; `slow_v3` overshoots.


## More steps will not close the remaining gap

The best configuration (`slow_v2`, league restored, 1500-step episodes) was
checked against the seed at three points through its run: 16-17 a third of the
way in, 11-17 two thirds, 16-22 at the end. Flat, if anything drifting worse.
It had plateaued rather than still climbing, so the residual gap is not a
training-duration problem and a longer run is not the answer.

What is left, in rough order of expected value:

1. **Retrain from scratch on the slow board with a long budget.** Every result
   here is an *adaptation* of a policy shaped by the fast board, and adaptation
   plateaued behind its own starting point. That is a strong hint the fast-board
   solution sits in a basin that does not contain the slow-board optimum. This
   needs the from-scratch problem solved first (see above).
2. **DAgger rather than one-shot behaviour cloning.** The measured failure is
   compounding error: the clone fits expert states and then drifts to states it
   handles badly. Re-collecting expert labels on the policy's *own* states is
   the standard fix and directly targets what was measured.
3. **Reconsider whether the slow board needs its own opponents.** The league
   carried across is made of fast-board snapshots, so the agent is practising
   against opponents that are themselves mis-adapted.
