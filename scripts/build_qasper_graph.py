from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def section_node(
    node_id: str,
    text: str,
    level: int,
    doc_id: str,
    split: str,
    order: int | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "Section",
        "text": (text or "").strip(),
        "attributes": {
            "label": "Section",
            "chunk_id": node_id,
            "level": level,
            "order": order,
            "doc_id": doc_id,
            "dataset": "qasper",
            "split": split,
        },
    }


def chunk_node(
    node_id: str,
    text: str,
    doc_id: str,
    split: str,
    section_id: str,
    order: int | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "Chunk",
        "text": (text or "").strip(),
        "attributes": {
            "label": "Chunk",
            "chunk_id": node_id,
            "type": "paragraph",
            "level": None,
            "order": order,
            "doc_id": doc_id,
            "dataset": "qasper",
            "split": split,
            "section_id": section_id,
        },
    }


def edge(source: str, target: str, rel: str) -> dict[str, str]:
    return {"source": source, "target": target, "type": rel}


def build_graph_for_doc(
    item: dict[str, Any],
    split: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, str]],
) -> None:
    doc_id = str(item["id"])
    title = (item.get("title") or "").strip() or doc_id
    abstract = (item.get("abstract") or "").strip()

    root_id = doc_id
    nodes.append(section_node(root_id, title, level=0, doc_id=doc_id, split=split, order=0))

    # Abstract is always represented as a Section + Chunk (if text exists).
    abstract_section_id = f"{doc_id}.abstract"
    nodes.append(
        section_node(
            abstract_section_id,
            "Abstract",
            level=1,
            doc_id=doc_id,
            split=split,
            order=0,
        )
    )
    edges.append(edge(root_id, abstract_section_id, "HAS_SUBSECTION"))
    if abstract:
        abstract_chunk_id = f"{doc_id}_abstract"
        nodes.append(
            chunk_node(
                abstract_chunk_id,
                abstract,
                doc_id=doc_id,
                split=split,
                section_id=abstract_section_id,
                order=0,
            )
        )
        edges.append(edge(abstract_section_id, abstract_chunk_id, "HAS_CHUNK"))

    full_text = item.get("full_text") or {}
    section_names = full_text.get("section_name") or []
    paragraphs = full_text.get("paragraphs") or []

    section_id_by_path: dict[str, str] = {}
    sec_count = min(len(section_names), len(paragraphs))

    # First pass: create section nodes with ids like <doc_id>.<section_index>
    for sec_idx in range(sec_count):
        raw_name = str(section_names[sec_idx] or "").strip()
        if not raw_name:
            raw_name = f"Section {sec_idx}"
        parts = [p.strip() for p in raw_name.split(":::") if p.strip()]
        sec_title = parts[-1] if parts else raw_name
        level = len(parts) if parts else 1
        sec_id = f"{doc_id}.{sec_idx}"
        nodes.append(
            section_node(
                sec_id,
                sec_title,
                level=level,
                doc_id=doc_id,
                split=split,
                order=sec_idx,
            )
        )
        section_id_by_path[raw_name] = sec_id

    # Second pass: section hierarchy + section chunks
    for sec_idx in range(sec_count):
        raw_name = str(section_names[sec_idx] or "").strip()
        if not raw_name:
            raw_name = f"Section {sec_idx}"
        sec_id = f"{doc_id}.{sec_idx}"

        parts = [p.strip() for p in raw_name.split(":::") if p.strip()]
        if len(parts) <= 1:
            parent_id = root_id
        else:
            parent_path = " ::: ".join(parts[:-1])
            parent_id = section_id_by_path.get(parent_path, root_id)
        edges.append(edge(parent_id, sec_id, "HAS_SUBSECTION"))

        sec_paragraphs = paragraphs[sec_idx] if sec_idx < len(paragraphs) else []
        if not isinstance(sec_paragraphs, list):
            continue
        chunk_order = 0
        for para_idx, para in enumerate(sec_paragraphs):
            text = (para or "").strip()
            if not text:
                continue
            chunk_id = f"{doc_id}.{sec_idx}.{para_idx}"
            nodes.append(
                chunk_node(
                    chunk_id,
                    text,
                    doc_id=doc_id,
                    split=split,
                    section_id=sec_id,
                    order=chunk_order,
                )
            )
            edges.append(edge(sec_id, chunk_id, "HAS_CHUNK"))
            chunk_order += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build graph nodes/edges from QASPER.")
    parser.add_argument(
        "--qasper-dir",
        type=Path,
        default=Path("datasets/qasper"),
        help="Directory with QASPER *.jsonl files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="QASPER split names to process.",
    )
    parser.add_argument(
        "--out-nodes",
        type=Path,
        default=Path("data/processed/qasper_nodes.cleaned.json"),
        help="Output nodes JSON path.",
    )
    parser.add_argument(
        "--out-edges",
        type=Path,
        default=Path("data/processed/qasper_edges.cleaned.json"),
        help="Output edges JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    total_docs = 0

    for split in args.splits:
        split_path = args.qasper_dir / f"{split}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        rows = read_jsonl(split_path)
        for item in rows:
            build_graph_for_doc(item, split=split, nodes=nodes, edges=edges)
        total_docs += len(rows)

    args.out_nodes.parent.mkdir(parents=True, exist_ok=True)
    args.out_edges.parent.mkdir(parents=True, exist_ok=True)

    with args.out_nodes.open("w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)
    with args.out_edges.open("w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)

    print(f"Processed docs: {total_docs}")
    print(f"Nodes written: {len(nodes)} -> {args.out_nodes}")
    print(f"Edges written: {len(edges)} -> {args.out_edges}")


if __name__ == "__main__":
    main()

