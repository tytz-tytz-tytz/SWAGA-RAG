import json
from pathlib import Path
from typing import List, Tuple, Iterable, Optional, Set


def load_id_text_pairs(
    nodes_path: Path,
    allowed_types: Optional[Iterable[str]] = None,
) -> List[Tuple[str, str]]:
    """
    Load nodes from graphrag_nodes.cleaned.json and return (id, text) pairs.
    Uses 'id' and 'text' fields.

    allowed_types:
        - None: keep all node types
        - iterable of node types: keep only matching types (e.g. {"Chunk"})
    """
    data = json.loads(nodes_path.read_text(encoding="utf-8"))
    out: List[Tuple[str, str]] = []
    allowed: Optional[Set[str]] = set(allowed_types) if allowed_types is not None else None

    for obj in data:
        ntype = obj.get("type")
        if allowed is not None and ntype not in allowed:
            continue
        cid = obj.get("id")
        text = obj.get("text", "")
        if not cid or not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        out.append((cid, text))

    return out
