# src/index/store.py

import json
import pickle
from pathlib import Path

def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_index(dir_path: str, sections, text_nodes, graph_adj):
    """
    Сохраняет:
    - sections (dict)
    - text_nodes (dict)
    - graph_adj (dict)
    + размерность эмбеддингов (берём из любого узла)
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    print(f"[save_index] Saving to {dir_path}/")

    save_pickle(dir_path / "sections.pkl", sections)
    save_pickle(dir_path / "text_nodes.pkl", text_nodes)
    save_pickle(dir_path / "graph_adj.pkl", graph_adj)

    # определяем размерность эмбеддингов
    emb_dim = None
    for s in sections.values():
        if s.E_subtree is not None:
            emb_dim = len(s.E_subtree)
            break

    if emb_dim is None:
        raise RuntimeError("Cannot determine embedding dimension — no embeddings found.")

    with open(dir_path / "dim.json", "w", encoding="utf-8") as f:
        json.dump({"dim": emb_dim}, f)

    print("[save_index] Done.")


def load_index(dir_path: str):
    """
    Загружает:
    - sections
    - text_nodes
    - graph_adj
    и возвращает их как tuple
    """
    dir_path = Path(dir_path)

    sections = load_pickle(dir_path / "sections.pkl")
    text_nodes = load_pickle(dir_path / "text_nodes.pkl")
    graph_adj = load_pickle(dir_path / "graph_adj.pkl")

    return sections, text_nodes, graph_adj
