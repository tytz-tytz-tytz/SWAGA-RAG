# src/index/builder.py

import pickle
from pathlib import Path

from ..data.loaders import load_ontology
from ..ontology.hierarchy import build_hierarchy
from .embeddings import EmbeddingModel
from .section_index import SectionIndex
from .text_index import TextIndex


def build_full_index(
    path_nodes: str,
    path_edges: str,
    output_file: str = "ontology_index.pkl",
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
):
    print("=== 1. Load ontology ===")
    sections, text_nodes, graph_adj = load_ontology(path_nodes, path_edges)

    print("=== 2. Build hierarchy ===")
    sections, text_nodes = build_hierarchy(sections, text_nodes)

    print("=== 3. Load embedding model ===")
    model = EmbeddingModel(model_name)

    print("=== 4. Compute section embeddings ===")
    sec_idx = SectionIndex(model)
    sections = sec_idx.compute_section_embeddings(sections)

    print("=== 5. Compute text embeddings ===")
    txt_idx = TextIndex(model)
    text_nodes = txt_idx.compute_textnode_embeddings(text_nodes)

    print("=== 6. Save index ===")
    data = {
        "sections": sections,
        "text_nodes": text_nodes,
        "graph_adj": graph_adj,
    }
    with open(output_file, "wb") as f:
        pickle.dump(data, f)

    print(f"=== DONE. Index saved to {output_file} ===")


if __name__ == "__main__":
    build_full_index("graphrag_nodes.json", "graphrag_edges.json")
