import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract one midpoint keyframe for each aligned scene."
    )
    parser.add_argument("video", type=Path, help="Path to input video file.")
    parser.add_argument(
        "scene_json",
        type=Path,
        help="Path to aligned scene transcript JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for extracted keyframe images.",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="ffmpeg executable path. Defaults to ffmpeg on PATH.",
    )
    args = parser.parse_args()

    with args.scene_json.open("r", encoding="utf-8") as f:
        scenes = json.load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        scene_id = int(scene["scene_id"])
        start = float(scene["start"])
        end = float(scene["end"])
        midpoint = (start + end) / 2
        output_file = args.output_dir / f"scene_{scene_id:03d}.jpg"

        command = [
            args.ffmpeg,
            "-y",
            "-ss",
            f"{midpoint:.3f}",
            "-i",
            str(args.video),
            "-frames:v",
            "1",
            str(output_file),
        ]
        subprocess.run(command, check=True)

    print(f"Saved {len(scenes)} keyframes to {args.output_dir}")


if __name__ == "__main__":
    main()
