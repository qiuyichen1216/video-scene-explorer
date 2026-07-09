import argparse
import csv
import json
from pathlib import Path


def parse_timecode(value: str) -> float:
    hours, minutes, seconds = value.strip().split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def pick_value(row: dict, *candidates: str) -> str:
    normalized = {key.strip().lower(): value for key, value in row.items()}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    raise KeyError(f"Missing any of columns: {candidates}")


def load_scenes(scene_csv: Path) -> list[dict]:
    scenes: list[dict] = []
    with scene_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    header_index = None
    for index, row in enumerate(rows):
        normalized = [cell.strip().lower() for cell in row]
        if "scene number" in normalized or "scene #" in normalized:
            header_index = index
            break

    if header_index is None:
        raise ValueError(f"Could not find scene CSV header in {scene_csv}")

    header = rows[header_index]
    data_rows = rows[header_index + 1 :]
    for raw_row in data_rows:
        if not raw_row:
            continue
        row = dict(zip(header, raw_row))
        try:
            scene_id = int(
                pick_value(
                    row,
                    "Scene Number",
                    "Scene #",
                    "Scene",
                )
            )
            start_time = parse_timecode(
                pick_value(
                    row,
                    "Start Time",
                    "Start Timecode",
                    "Start",
                )
            )
            end_time = parse_timecode(
                pick_value(
                    row,
                    "End Time",
                    "End Timecode",
                    "End",
                )
            )
        except (KeyError, ValueError):
            continue

        scenes.append(
            {
                "scene_id": scene_id,
                "start": start_time,
                "end": end_time,
            }
        )
    return scenes


def load_segments(transcript_json: Path) -> list[dict]:
    with transcript_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("segments", [])


def segment_overlaps_scene(segment: dict, scene: dict) -> bool:
    return float(segment["end"]) > scene["start"] and float(segment["start"]) < scene["end"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align Whisper transcript segments to PySceneDetect scenes."
    )
    parser.add_argument("scene_csv", type=Path, help="Path to PySceneDetect scene CSV.")
    parser.add_argument(
        "transcript_json",
        type=Path,
        help="Path to Whisper transcript JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output aligned scene transcript JSON.",
    )
    args = parser.parse_args()

    scenes = load_scenes(args.scene_csv)
    segments = load_segments(args.transcript_json)

    aligned = []
    for scene in scenes:
        matched_segments = [
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg["text"].strip(),
            }
            for seg in segments
            if segment_overlaps_scene(seg, scene)
        ]
        aligned.append(
            {
                "scene_id": scene["scene_id"],
                "start": scene["start"],
                "end": scene["end"],
                "transcript": " ".join(seg["text"] for seg in matched_segments).strip(),
                "segments": matched_segments,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(aligned, f, ensure_ascii=False, indent=2)

    print(f"Saved aligned scene transcripts to {args.output}")


if __name__ == "__main__":
    main()
