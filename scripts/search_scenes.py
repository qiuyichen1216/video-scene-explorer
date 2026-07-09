import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def load_scenes(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        scenes = json.load(f)
    return [scene for scene in scenes if scene.get("transcript", "").strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search aligned video scenes by natural-language query."
    )
    parser.add_argument(
        "scene_json",
        type=Path,
        help="Path to aligned scene transcript JSON.",
    )
    parser.add_argument("query", help="Natural-language search query.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of scenes to return.",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    args = parser.parse_args()

    scenes = load_scenes(args.scene_json)
    if not scenes:
        raise ValueError(f"No searchable transcripts found in {args.scene_json}")

    model = SentenceTransformer(args.model)
    texts = [scene["transcript"] for scene in scenes]
    scene_embeddings = model.encode(texts, normalize_embeddings=True)
    query_embedding = model.encode([args.query], normalize_embeddings=True)[0]

    scores = np.dot(scene_embeddings, query_embedding)
    top_indices = np.argsort(scores)[::-1][: args.top_k]

    print(f"Query: {args.query}")
    print()
    for rank, index in enumerate(top_indices, start=1):
        scene = scenes[int(index)]
        score = float(scores[index])
        transcript = scene["transcript"].replace("\n", " ").strip()
        preview = transcript[:260] + ("..." if len(transcript) > 260 else "")
        print(f"Top {rank} | score={score:.4f}")
        print(f"Scene: {scene['scene_id']}")
        print(f"Time: {scene['start']:.2f}s - {scene['end']:.2f}s")
        print(f"Transcript: {preview}")
        print("-" * 72)


if __name__ == "__main__":
    main()
