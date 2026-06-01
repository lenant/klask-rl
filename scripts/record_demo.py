from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from klask_rl.envs import SelfPlayKlaskEnv
from klask_rl.opponents import SB3CheckpointOpponent


def open_ffmpeg(output_path: Path, frame_shape: tuple[int, int, int], fps: int, width: int) -> subprocess.Popen:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to record a demo video")
    height_px, width_px, channels = frame_shape
    if channels != 3:
        raise ValueError(f"expected RGB frames, got shape {frame_shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scale_filter = f"scale={width}:-2:flags=lanczos" if width > 0 else "null"
    return subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width_px}x{height_px}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vf",
            scale_filter,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "24",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
    )


def write_frame(process: subprocess.Popen, frame: np.ndarray) -> None:
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin is closed")
    process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())


def record_demo(args: argparse.Namespace) -> dict[str, int | str]:
    model = PPO.load(args.model, device="cpu")
    opponent = SB3CheckpointOpponent(args.model)
    env = SelfPlayKlaskEnv(
        opponent=opponent,
        render_mode="rgb_array",
        randomize_side=False,
        max_steps=args.max_steps,
        reward_profile=args.reward_profile,
    )

    process: subprocess.Popen | None = None
    frames = 0
    completed_episodes = 0
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            frame = env.render()
            if frame is None:
                raise RuntimeError("environment did not return an RGB frame")
            if process is None:
                process = open_ffmpeg(args.output, frame.shape, args.fps, args.width)
            write_frame(process, frame)
            frames += 1

            done = False
            last_frame = frame
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                frame = env.render()
                if frame is None:
                    raise RuntimeError("environment did not return an RGB frame")
                write_frame(process, frame)
                frames += 1
                last_frame = frame
                done = terminated or truncated

            completed_episodes += 1
            for _ in range(args.pause_frames):
                write_frame(process, last_frame)
                frames += 1
    finally:
        env.close()
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg exited with status {return_code}")

    return {
        "output": str(args.output),
        "model": str(args.model),
        "episodes": completed_episodes,
        "frames": frames,
        "fps": args.fps,
        "reward_profile": args.reward_profile,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record model-vs-model Klask self-play to MP4.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960, help="Output video width; <=0 keeps source size.")
    parser.add_argument("--pause-frames", type=int, default=18)
    parser.add_argument("--reward-profile", default="simple")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(record_demo(parse_args()), indent=2))
