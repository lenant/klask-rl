"""Play two checkpoints against each other, sides swapped, with a significance test.

Aggregate benchmark scores are measured against *third parties*, so two models
can each look better than the other depending on the opponent. Playing them
directly is the measure that settles it.

    PYTHONPATH=src uv run python scripts/head_to_head.py a.zip b.zip --games 150
"""

from __future__ import annotations

import argparse
from collections import Counter
from math import comb
from pathlib import Path

from dataclasses import replace

from klask_rl.config import ArenaConfig
from klask_rl.envs import KlaskParallelEnv
from klask_rl.opponents import SB3CheckpointOpponent


def _play(left: str, right: str, games: int, seed0: int, profile: str, max_steps: int) -> Counter:
    env = KlaskParallelEnv(
        arena_config=replace(ArenaConfig(), max_steps=max_steps), reward_profile=profile
    )
    policies = {"left": SB3CheckpointOpponent(left), "right": SB3CheckpointOpponent(right)}
    results: Counter = Counter()
    for game in range(games):
        observations, _ = env.reset(seed=seed0 + game)
        winner = "draw"
        while env.agents:
            actions = {a: policies[a].act(observations[a]) for a in env.agents}
            observations, _, terminations, truncations, _ = env.step(actions)
            if any(terminations.values()) or any(truncations.values()):
                scores = env.scores
                if scores["left"] > scores["right"]:
                    winner = "left"
                elif scores["right"] > scores["left"]:
                    winner = "right"
                break
        results[winner] += 1
    env.close()
    return results


def two_sided_p(wins: int, losses: int) -> float:
    decided = wins + losses
    if decided == 0:
        return 1.0
    tail = sum(comb(decided, k) for k in range(max(wins, losses), decided + 1))
    return min(1.0, 2.0 * tail / (2**decided))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_a", type=Path)
    parser.add_argument("model_b", type=Path)
    parser.add_argument("--games", type=int, default=150, help="Total games, split across sides.")
    parser.add_argument("--seed", type=int, default=90000)
    parser.add_argument("--reward-profile", default="slow_v1")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=750,
        help="Match the training budget; the config default is far longer.",
    )
    args = parser.parse_args()

    per_side = args.games // 2
    a, b = str(args.model_a), str(args.model_b)
    forward = _play(a, b, per_side, args.seed, args.reward_profile, args.max_steps)
    reverse = _play(b, a, per_side, args.seed + 5000, args.reward_profile, args.max_steps)

    a_wins = forward["left"] + reverse["right"]
    b_wins = forward["right"] + reverse["left"]
    draws = forward["draw"] + reverse["draw"]
    p = two_sided_p(a_wins, b_wins)

    print(f"{args.model_a.name}  {a_wins}")
    print(f"{args.model_b.name}  {b_wins}")
    print(f"draws  {draws}")
    print(f"p = {p:.4f} over {a_wins + b_wins} decided games ({2 * per_side} played, sides swapped)")


if __name__ == "__main__":
    main()
