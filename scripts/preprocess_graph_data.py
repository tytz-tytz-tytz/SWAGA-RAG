from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple


RAW_NODES = Path("data/raw/graphrag_nodes.json")
RAW_EDGES = Path("data/raw/graphrag_edges.json")

OUT_NODES = Path("data/processed/graphrag_nodes.cleaned.json")
OUT_EDGES = Path("data/processed/graphrag_edges.cleaned.json")


# Common punctuation-only junk that should not be indexed or retrieved
PUNCT_ONLY = {
    ".", ",", ";", ":", "-", "—", "–", "(", ")", "[", "]", "{", "}", "},", "{,", "}", "{"
}


def normalize_ws(text: str) -> str:
    """Normalize whitespace without changing semantic content."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return text.strip()


def strip_leading_page_number(text: str) -> str:
    """
    Remove leading page number artifacts like:
      '8 Каждая рассылка ...' -> 'Каждая рассылка ...'
    Only removes if a letter follows (Cyrillic/Latin).
    """
    return re.sub(r"^\s*\d{1,3}\s+(?=[A-Za-zА-Яа-яЁё])", "", text)


def alpha_ratio(text: str) -> float:
    """Compute ratio of alphabetic characters to total non-space characters."""
    s = re.sub(r"\s+", "", text)
    if not s:
        return 0.0
    alpha = sum(ch.isalpha() for ch in s)
    return alpha / len(s)


def is_noise_text(text: str) -> bool:
    """
    Heuristic filter for non-informative fragments.
    This is language-agnostic and does not depend on specific domain vocabulary.
    """
    if text is None:
        return True
    s = text.strip()
    if not s:
        return True
    if s in PUNCT_ONLY:
        return True
    # Page numbers like "6", "12", "101"
    if re.fullmatch(r"\d{1,3}", s):
        return True
    # Single symbol junk (e.g., "{", "}", ";") already covered above, but keep safe
    if len(s) <= 2 and re.fullmatch(r"[^\w\s]", s):
        return True
    # Very short non-informative fragments
    if len(s) < 10 and alpha_ratio(s) < 0.2:
        return True
    return False


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    if not RAW_NODES.exists():
        raise FileNotFoundError(f"Missing nodes file: {RAW_NODES}")
    if not RAW_EDGES.exists():
        raise FileNotFoundError(f"Missing edges file: {RAW_EDGES}")

    nodes: List[Dict[str, Any]] = load_json(RAW_NODES)
    edges: List[Dict[str, Any]] = load_json(RAW_EDGES)

    before_types = Counter(n.get("type") for n in nodes)
    before_nodes = len(nodes)
    before_edges = len(edges)

    kept_nodes: List[Dict[str, Any]] = []
    removed_nodes: List[Tuple[str, str, str]] = []  # (id, type, text)
    kept_ids = set()

    # Clean and filter nodes
    for n in nodes:
        n_id = n.get("id")
        n_type = n.get("type")
        text = n.get("text", "")

        # Keep Section nodes even if they look short: they define structure
        if n_type == "Section":
            if isinstance(text, str):
                n["text"] = normalize_ws(text)
            kept_nodes.append(n)
            kept_ids.add(n_id)
            continue

        if isinstance(text, str):
            text2 = normalize_ws(strip_leading_page_number(text))
        else:
            text2 = ""

        # Decide whether this node is pure noise
        if is_noise_text(text2):
            removed_nodes.append((str(n_id), str(n_type), str(text2)))
            continue

        n["text"] = text2
        kept_nodes.append(n)
        kept_ids.add(n_id)

    # Filter edges where source/target are missing after node removal
    kept_edges: List[Dict[str, Any]] = []
    dropped_edges = 0
    for e in edges:
        s = e.get("source")
        t = e.get("target")
        if s in kept_ids and t in kept_ids:
            kept_edges.append(e)
        else:
            dropped_edges += 1

    after_types = Counter(n.get("type") for n in kept_nodes)
    after_nodes = len(kept_nodes)
    after_edges = len(kept_edges)

    # Save outputs
    save_json(OUT_NODES, kept_nodes)
    save_json(OUT_EDGES, kept_edges)

    # Report
    removed_by_type = Counter(t for _, t, _ in removed_nodes)

    print("=== Preprocessing report ===")
    print(f"Nodes: {before_nodes} -> {after_nodes} (removed {before_nodes - after_nodes})")
    print(f"Edges: {before_edges} -> {after_edges} (dropped {dropped_edges})")
    print("\nNode types (before -> after):")
    for t in sorted(set(before_types) | set(after_types)):
        print(f"  {t}: {before_types.get(t,0)} -> {after_types.get(t,0)}")

    print("\nRemoved nodes by type:")
    for t, c in removed_by_type.most_common():
        print(f"  {t}: {c}")

    # Show a few examples (safe, short)
    print("\nExamples of removed texts:")
    for i, (_, t, txt) in enumerate(removed_nodes[:15], start=1):
        preview = txt.replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:77] + "..."
        print(f"  {i:02d}. [{t}] {preview}")


if __name__ == "__main__":
    main()
