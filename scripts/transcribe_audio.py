import argparse
import json
from pathlib import Path

import whisper


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file with Whisper and save JSON output."
    )
    parser.add_argument("audio", type=Path, help="Path to input audio file.")
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model name, e.g. tiny, base, small.",
    )
    parser.add_argument(
        "--language",
        default="English",
        help="Language hint for Whisper.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output JSON file.",
    )
    args = parser.parse_args()

    model = whisper.load_model(args.model)
    result = model.transcribe(str(args.audio), language=args.language)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved transcript JSON to {args.output}")


if __name__ == "__main__":
    main()
