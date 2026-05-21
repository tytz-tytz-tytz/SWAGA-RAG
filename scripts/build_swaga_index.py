from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from _repo_paths import repo_path, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the SWAGA-RAG ontology index.")
    parser.add_argument(
        "--nodes",
        type=Path,
        default=repo_path("data/processed/graphrag_nodes.cleaned.json"),
        help="Path to cleaned graph nodes JSON.",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=repo_path("data/processed/graphrag_edges.cleaned.json"),
        help="Path to cleaned graph edges JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_path("artifacts/indexes/swaga_index_dir"),
        help="Directory where the built index will be saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Embedding device, e.g. cpu or cuda.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from swaga_rag.data.loaders import load_ontology
    from swaga_rag.index.embeddings import EmbeddingModel
    from swaga_rag.index.section_index import SectionIndex
    from swaga_rag.index.store import save_index
    from swaga_rag.index.text_index import TextIndex
    from swaga_rag.ontology.hierarchy import build_hierarchy

    nodes_path = resolve_repo_path(args.nodes)
    edges_path = resolve_repo_path(args.edges)
    out_dir = resolve_repo_path(args.out_dir)

    print("=== 1. Load ontology ===")
    sections, text_nodes, graph_adj = load_ontology(str(nodes_path), str(edges_path))

    print("=== 2. Build hierarchy ===")
    build_hierarchy(sections, text_nodes)

    print("=== 3. Init embedding model ===")
    model = EmbeddingModel(device=args.device)

    print("=== 4. Compute section embeddings ===")
    sec_index = SectionIndex(model)
    sec_index.compute_section_embeddings(sections)

    print("=== 5. Compute text node embeddings ===")
    txt_index = TextIndex(model)
    txt_index.compute_textnode_embeddings(text_nodes)

    print("=== 6. Save index ===")
    save_index(str(out_dir), sections, text_nodes, graph_adj)

    print(f"\n=== DONE. Index saved to {out_dir} ===")


if __name__ == "__main__":
    main()
