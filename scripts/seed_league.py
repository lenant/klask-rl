"""Copy self-play opponents from a finished run into a warm-started one.

Resuming into a fresh output directory restores zero checkpoint opponents, so a
model forged against hundreds of snapshots ends up practising against only the
scripted baselines and degrades within 100k steps. Only snapshots at or below
the seed's own timestep count are eligible, since that is what the resume path
filters on.

    PYTHONPATH=src uv run python scripts/seed_league.py SOURCE_DIR DEST_DIR SEED_STEPS [--keep 64]
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

STEP = re.compile(r"(\d+)")


def snapshot_step(path: Path) -> int:
    found = STEP.findall(path.name)
    return int(found[-1]) if found else -1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument("seed_steps", type=int)
    parser.add_argument("--keep", type=int, default=64)
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        (snapshot_step(p), p) for p in args.source.glob("policy_*.zip")
    )
    eligible = [p for step, p in ranked if 0 <= step <= args.seed_steps][-args.keep :]
    for path in eligible:
        shutil.copy2(path, args.dest / path.name)

    if eligible:
        low = snapshot_step(eligible[0])
        high = snapshot_step(eligible[-1])
        print(f"copied {len(eligible)} league opponents, steps {low}..{high}")
    else:
        print(f"copied 0 league opponents (nothing at or below {args.seed_steps})")


if __name__ == "__main__":
    main()
