from __future__ import annotations

import argparse
import csv
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from klask_rl.cli import run_eval


POLICY_RE = re.compile(r"policy_(\d+)\.zip$")
MATCHUPS = (
    ("passive", "passive", False, 0),
    ("random", "random", False, 100),
    ("heuristic", "heuristic", False, 200),
    ("self_play", "heuristic", True, 300),
)
SUMMARY_FIELDS = (
    "quality_score",
    "wins",
    "losses",
    "draws",
    "goals_for",
    "goals_against",
    "goals_per_game",
    "draw_rate",
    "defense_rate",
    "mean_reward",
    "mean_length",
    "mean_final_puck_x",
    "mean_contact_steps",
)
FIELDNAMES = (
    "step",
    "model",
    "started_at",
    "finished_at",
    "episodes_per_matchup",
    "aggregate_quality",
    "aggregate_goals_for",
    "aggregate_goals_against",
    "aggregate_goal_margin",
    "is_leader",
    "leader_quality",
    "leader_step",
    *(
        f"{matchup}_{field}"
        for matchup, _, _, _ in MATCHUPS
        for field in SUMMARY_FIELDS
    ),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def parse_step(path: Path) -> int | None:
    match = POLICY_RE.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group(1))


def read_completed_steps(csv_path: Path) -> set[int]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="") as handle:
        return {int(row["step"]) for row in csv.DictReader(handle) if row.get("step")}


def read_best(csv_path: Path) -> tuple[float, int]:
    best_quality = float("-inf")
    best_step = 0
    if not csv_path.exists():
        return best_quality, best_step
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                quality = float(row["aggregate_quality"])
                step = int(row["step"])
            except (KeyError, TypeError, ValueError):
                continue
            if quality > best_quality:
                best_quality = quality
                best_step = step
    return best_quality, best_step


def wait_for_stable_file(path: Path, interval: float = 1.0, attempts: int = 3) -> None:
    previous_size = -1
    stable_count = 0
    while stable_count < attempts:
        size = path.stat().st_size
        if size == previous_size and size > 0:
            stable_count += 1
        else:
            stable_count = 0
        previous_size = size
        time.sleep(interval)


def snapshot_paths(snapshot_dir: Path, min_step: int, max_step: int | None) -> list[tuple[int, Path]]:
    snapshots: list[tuple[int, Path]] = []
    for path in snapshot_dir.glob("policy_*.zip"):
        step = parse_step(path)
        if step is None or step < min_step:
            continue
        if max_step is not None and step > max_step:
            continue
        snapshots.append((step, path))
    return sorted(snapshots)


def benchmark_snapshot(
    model_path: Path,
    step: int,
    episodes: int,
    seed: int,
    max_steps: int,
    deterministic: bool,
    reward_profile: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "step": step,
        "model": str(model_path),
        "started_at": utc_now(),
        "episodes_per_matchup": episodes,
    }
    aggregate_goals_for = 0
    aggregate_goals_against = 0
    qualities: list[float] = []

    for label, opponent, self_play, seed_offset in MATCHUPS:
        summary = run_eval(
            model_path=model_path,
            episodes=episodes,
            opponent_model=None,
            opponent=opponent,
            self_play=self_play,
            seed=seed + seed_offset,
            deterministic=deterministic,
            render=False,
            max_steps=max_steps,
            reward_profile=reward_profile,
        )
        summary_dict = summary.as_dict()
        qualities.append(float(summary.quality_score))
        aggregate_goals_for += summary.goals_for
        aggregate_goals_against += summary.goals_against
        for field in SUMMARY_FIELDS:
            row[f"{label}_{field}"] = summary_dict[field]

    row["finished_at"] = utc_now()
    row["aggregate_quality"] = sum(qualities) / max(1, len(qualities))
    row["aggregate_goals_for"] = aggregate_goals_for
    row["aggregate_goals_against"] = aggregate_goals_against
    row["aggregate_goal_margin"] = aggregate_goals_for - aggregate_goals_against
    return row


def append_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def copy_leader(model_path: Path, step: int, leader_dir: Path, leader_copy: Path | None) -> None:
    leader_dir.mkdir(parents=True, exist_ok=True)
    destination = leader_dir / f"leader_policy_{step}.zip"
    shutil.copy2(model_path, destination)
    if leader_copy is not None:
        leader_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, leader_copy)


def run_monitor(args: argparse.Namespace) -> None:
    csv_path = args.csv
    completed_steps = read_completed_steps(csv_path)
    best_quality, best_step = read_best(csv_path)
    print(
        {
            "csv": str(csv_path),
            "completed_steps": sorted(completed_steps),
            "best_quality": None if best_quality == float("-inf") else best_quality,
            "best_step": best_step or None,
        },
        flush=True,
    )

    while True:
        did_work = False
        for step, model_path in snapshot_paths(args.snapshot_dir, args.min_step, args.max_step):
            if step in completed_steps:
                continue
            did_work = True
            wait_for_stable_file(model_path)
            print({"event": "benchmark_start", "step": step, "model": str(model_path)}, flush=True)
            row = benchmark_snapshot(
                model_path=model_path,
                step=step,
                episodes=args.episodes,
                seed=args.seed,
                max_steps=args.max_steps,
                deterministic=not args.stochastic,
                reward_profile=args.reward_profile,
            )
            aggregate_quality = float(row["aggregate_quality"])
            if aggregate_quality > best_quality:
                best_quality = aggregate_quality
                best_step = step
                row["is_leader"] = 1
                copy_leader(model_path, step, args.leader_dir, args.leader_copy)
            else:
                row["is_leader"] = 0
            row["leader_quality"] = best_quality
            row["leader_step"] = best_step
            append_row(csv_path, row)
            completed_steps.add(step)
            print(
                {
                    "event": "benchmark_done",
                    "step": step,
                    "aggregate_quality": aggregate_quality,
                    "is_leader": bool(row["is_leader"]),
                    "leader_step": best_step,
                    "leader_quality": best_quality,
                },
                flush=True,
            )

        if args.once:
            return
        if args.max_step is not None and completed_steps:
            pending = [step for step, _ in snapshot_paths(args.snapshot_dir, args.min_step, args.max_step)]
            if pending and max(completed_steps) >= args.max_step:
                return
        if not did_work:
            time.sleep(args.poll_interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PPO snapshots as they are saved.")
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--leader-dir", type=Path, required=True)
    parser.add_argument("--leader-copy", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=310)
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--max-step", type=int)
    parser.add_argument("--reward-profile", default="simple")
    parser.add_argument("--poll-interval", type=float, default=20.0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_monitor(parse_args())
