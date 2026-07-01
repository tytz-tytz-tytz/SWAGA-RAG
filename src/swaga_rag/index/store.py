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


def save_index(dir_path: str, sections, text_nodes, graph_adj, model_name=None):
    """
    Сохраняет:
    - sections (dict)
    - text_nodes (dict)
    - graph_adj (dict)
    + dim.json: размерность эмбеддингов и (опц.) имя энкодера, которым индекс
      построен — для проверки совместимости с query-моделью.

    model_name=None сохраняет dim.json в прежнем формате (только {"dim": ...}),
    чтобы старые индексы воспроизводились без изменений.
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

    meta = {"dim": int(emb_dim)}
    if model_name:
        meta["model"] = str(model_name)

    with open(dir_path / "dim.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    print(f"[save_index] Done. dim={emb_dim} model={model_name or '(unset)'}")


def index_meta(dir_path: str) -> dict:
    """Читает dim.json индекса; возвращает {} если файла нет."""
    p = Path(dir_path) / "dim.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_index_compatible(dir_path: str, embedding_model) -> None:
    """
    Громко падает, если query-модель не совпадает с энкодером индекса.

    - размерность query-модели обязана совпадать с dim из dim.json;
    - если индекс знает имя своего энкодера (новый формат) — оно обязано
      совпасть с именем query-модели;
    - старый индекс без имени модели: проверяется только dim, с предупреждением.
    """
    meta = index_meta(dir_path)
    idx_dim = meta.get("dim")
    idx_model = meta.get("model")
    q_dim = embedding_model.dim
    q_model = getattr(embedding_model, "model_name", "(unknown)")

    if idx_dim is not None and int(idx_dim) != int(q_dim):
        raise RuntimeError(
            f"Embedding dim mismatch for index '{dir_path}': index built with "
            f"dim={idx_dim}, but query model '{q_model}' has dim={q_dim}. "
            f"Rebuild the index with the matching encoder (per-model index dir)."
        )

    if idx_model is not None and str(idx_model) != str(q_model):
        raise RuntimeError(
            f"Encoder mismatch for index '{dir_path}': index built with "
            f"'{idx_model}', but query model is '{q_model}'. Pass --model "
            f"'{idx_model}' or point --index-dir at the matching per-model index."
        )

    if idx_model is None:
        print(
            f"[assert_index_compatible] WARNING: index '{dir_path}' has no model "
            f"name in dim.json (legacy); verified dim only (dim={idx_dim}, "
            f"query model '{q_model}')."
        )
    else:
        print(f"[assert_index_compatible] OK: {q_model} (dim={q_dim}) matches index.")


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
