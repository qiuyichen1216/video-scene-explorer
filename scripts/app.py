import argparse
import json
from pathlib import Path

import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    return f"{minutes:02d}:{remaining:05.2f}"


def load_scenes(scene_json: Path) -> list[dict]:
    with scene_json.open("r", encoding="utf-8") as f:
        scenes = json.load(f)
    return [scene for scene in scenes if scene.get("transcript", "").strip()]


def build_app(scene_json: Path, keyframe_dir: Path, model_name: str) -> gr.Blocks:
    scenes = load_scenes(scene_json)
    if not scenes:
        raise ValueError(f"No searchable transcripts found in {scene_json}")

    model = SentenceTransformer(model_name)
    scene_texts = [scene["transcript"] for scene in scenes]
    scene_embeddings = model.encode(scene_texts, normalize_embeddings=True)

    def search(query: str, top_k: int):
        query = query.strip()
        if not query:
            return [], "Enter a query to search scenes."

        query_embedding = model.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(scene_embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:top_k]

        gallery_items = []
        details = [f"### Query\n{query}\n"]

        for rank, index in enumerate(top_indices, start=1):
            scene = scenes[int(index)]
            score = float(scores[index])
            scene_id = int(scene["scene_id"])
            start = float(scene["start"])
            end = float(scene["end"])
            transcript = scene["transcript"].strip()
            image_path = keyframe_dir / f"scene_{scene_id:03d}.jpg"

            caption = (
                f"Top {rank} | Scene {scene_id} | "
                f"{format_time(start)} - {format_time(end)} | score {score:.3f}"
            )
            if image_path.exists():
                gallery_items.append((str(image_path), caption))

            details.append(
                "\n".join(
                    [
                        f"### Top {rank}: Scene {scene_id}",
                        f"- Time: `{format_time(start)} - {format_time(end)}`",
                        f"- Score: `{score:.4f}`",
                        "",
                        transcript,
                    ]
                )
            )

        return gallery_items, "\n\n---\n\n".join(details)

    with gr.Blocks(title="Video Scene Explorer") as demo:
        gr.Markdown(
            "# Video Scene Explorer\n"
            "Search video scenes by natural-language query."
        )
        with gr.Row():
            query_input = gr.Textbox(
                label="Search query",
                placeholder="medical patient and disease",
                scale=4,
            )
            top_k_input = gr.Slider(
                minimum=1,
                maximum=min(10, len(scenes)),
                value=min(5, len(scenes)),
                step=1,
                label="Top K",
                scale=1,
            )
        search_button = gr.Button("Search", variant="primary")
        gallery = gr.Gallery(
            label="Matching Scene Keyframes",
            columns=3,
            height="auto",
        )
        details = gr.Markdown(label="Results")

        search_button.click(
            search,
            inputs=[query_input, top_k_input],
            outputs=[gallery, details],
        )
        query_input.submit(
            search,
            inputs=[query_input, top_k_input],
            outputs=[gallery, details],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Video Scene Explorer UI.")
    parser.add_argument("scene_json", type=Path, help="Path to scene transcript JSON.")
    parser.add_argument(
        "--keyframe-dir",
        type=Path,
        default=Path("keyframes"),
        help="Directory containing scene_001.jpg style keyframes.",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=7860, help="Server port.")
    args = parser.parse_args()

    demo = build_app(args.scene_json, args.keyframe_dir, args.model)
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
