import json
from pathlib import Path
from typing import List, Tuple


def load_chunks_id_text(path: Path) -> Tuple[List[str], List[str]]:
    """
    Loads ONLY (id, text) from graphrag_nodes.cleaned.json.
    Keeps only nodes with type == "Chunk" and non-empty text.
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    ids: List[str] = []
    texts: List[str] = []

    for obj in data:
        if obj.get("type") != "Chunk":
            continue

        _id = obj.get("id")
        txt = (obj.get("text") or "").strip()
        if not _id or not txt:
            continue

        ids.append(_id)
        texts.append(txt)

    return ids, texts