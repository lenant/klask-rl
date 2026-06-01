# Reward Experiments

All runs use the same Pymunk aero-hockey environment, mirrored shared policy,
behavior-cloning warm start, PPO self-play snapshots, and the benchmark command:

```bash
uv run klask-benchmark --model <checkpoint> --episodes 30 --max-steps 450
```

The benchmark evaluates four matchups: passive opponent, random opponent,
heuristic opponent, and same-policy self-play. The aggregate score combines goal
margin, goals per game, defense rate, contact activity, and territory.

## Sweep Results

| Run | Steps | Quality | Goals For | Goals Against | Margin | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| aggressive | 60k | 4.26 | 13 | 15 | -2 | Active, but converted poorly and conceded too much. |
| possession | 60k | 4.67 | 19 | 16 | +3 | Best short run; much more active same-policy games. |
| balanced | 60k | 3.46 | 15 | 20 | -5 | More decisive, but weak territory and defense. |
| possession_long final | 180k | 4.75 | 35 | 30 | +5 | More decisive self-play, but not the strongest checkpoint. |
| possession_long snapshot | 150k | 4.93 | 36 | 24 | +12 | Best selected checkpoint. |

## Selected Checkpoint

Promoted locally to:

```bash
runs/klask/latest/final_model.zip
```

Source checkpoint:

```bash
runs/experiments/possession_long/latest/snapshots/policy_150000.zip
```

Benchmark details for the selected checkpoint over 40 episodes per matchup:

| Matchup | W | L | D | Goals/Game | Defense Rate | Mean Contact Steps | Mean Final Puck X |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| passive | 9 | 1 | 30 | 0.25 | 0.975 | 5.875 | 0.426 |
| random | 15 | 9 | 16 | 0.60 | 0.775 | 7.425 | 0.221 |
| heuristic | 3 | 7 | 30 | 0.25 | 0.825 | 11.075 | 0.142 |
| self-play | 9 | 7 | 24 | 0.40 | 0.825 | 13.800 | -0.029 |

## Interpretation

The possession reward produced the best game quality. It gives the model enough
incentive to keep contact and fight for the puck without making it reckless.
The 150k snapshot outperformed the 180k final checkpoint on aggregate margin and
defense, so it is the checkpoint to watch by default.

Compared with the first version, same-policy visualization is now meaningful:
both handles are controlled by the trained mirrored policy, the puck changes
possession, and games finish much more often instead of one side pushing once
and then idling.
