# run_swaga_interactive.py

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from _repo_paths import repo_path, resolve_repo_path


DEFAULT_INDEX_DIR = repo_path("artifacts/indexes/swaga_index_dir")
DEFAULT_CONFIG_PATH = repo_path("configs/ablations/stable_baseline.json")


def load_config(path: Path) -> Dict[str, Any]:
    path = resolve_repo_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise TypeError("Config must be a JSON object (dict) at the top level.")
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive SWAGA-RAG runner (offline).")
    p.add_argument(
        "--index_dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Path to the built ontology index directory.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to SWAGA-RAG config JSON (e.g., configs/ablations/stable_baseline.json).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Embedding model device (e.g., cpu, cuda).",
    )
    return p.parse_args()


def run() -> None:
    args = parse_args()
    args.index_dir = resolve_repo_path(args.index_dir)
    args.config = resolve_repo_path(args.config)

    from swaga_rag.index.store import load_index
    from swaga_rag.index.embeddings import EmbeddingModel
    from swaga_rag.rag.pipeline import SWAGARAGPipeline

    print("=== Loading offline index ===")
    sections, text_nodes, graph_adj = load_index(str(args.index_dir))

    print("=== Loading config ===")
    config = load_config(args.config)

    print("=== Initializing embedding model ===")
    model = EmbeddingModel(device=args.device)

    pipeline = SWAGARAGPipeline(
        sections=sections,
        text_nodes=text_nodes,
        graph_adj=graph_adj,
        embedding_model=model,
        config=config,
    )

    while True:
        query = input("\nEnter a query (or 'exit'): ").strip()
        if query.lower() in ("exit", "quit"):
            break

        # Run the SWAGA-RAG pipeline
        result = pipeline.run_query(query)

        # Prepare an "LLM-ready" payload
        llm_input = {
            "query": result["query"],
            "section_candidates": result["section_candidates"],
            # Optionally, you may also include:
            # "text_nodes": result["text_nodes"],
            # "graph_context": result["graph_context"],
        }

        print("\n=== PIPELINE OUTPUT (LLM-ready) ===\n")
        print(json.dumps(llm_input, ensure_ascii=False, indent=2))
        print("\n=== END ===")


if __name__ == "__main__":
    run()



