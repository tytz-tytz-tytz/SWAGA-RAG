"""Build graph nodes and edges for the BioASQ PMC structured corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = Path("data/artifacts/pmc_structured_chunks.jsonl")
DEFAULT_OUT_NODES = Path("data/processed/bioasq_pmc_nodes.cleaned.json")
DEFAULT_OUT_EDGES = Path("data/processed/bioasq_pmc_edges.cleaned.json")
DATASET_NAME = "bioasq_pmc"
SPLIT_NAME = "bioasq12b_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build graph nodes and edges from the BioASQ PMC structured chunk corpus."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to pmc_structured_chunks.jsonl.",
    )
    parser.add_argument(
        "--out-nodes",
        type=Path,
        default=DEFAULT_OUT_NODES,
        help="Output path for cleaned graph nodes JSON.",
    )
    parser.add_argument(
        "--out-edges",
        type=Path,
        default=DEFAULT_OUT_EDGES,
        help="Output path for cleaned graph edges JSON.",
    )
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object on line {line_number} of {path}")
            yield row


def section_node(
    node_id: str,
    text: str,
    level: int,
    doc_id: str,
    order: int | None,
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
            "dataset": DATASET_NAME,
            "split": SPLIT_NAME,
        },
    }


def chunk_node(
    node_id: str,
    text: str,
    doc_id: str,
    section_id: str,
    order: int | None,
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
            "dataset": DATASET_NAME,
            "split": SPLIT_NAME,
            "section_id": section_id,
        },
    }


def edge(source: str, target: str, rel: str) -> dict[str, str]:
    return {"source": source, "target": target, "type": rel}


class DocumentGraphBuilder:
    def __init__(self, pmcid: str, article_title: str) -> None:
        self.pmcid = pmcid
        self.article_title = article_title.strip() or pmcid
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, str]] = []
        self.section_ids: dict[tuple[str, ...], str] = {}
        self.child_counts: dict[str, int] = {}
        self.chunk_counts: dict[str, int] = {}
        self.seen_chunk_ids: set[str] = set()

        self.nodes.append(
            section_node(
                node_id=pmcid,
                text=self.article_title,
                level=0,
                doc_id=pmcid,
                order=0,
            )
        )
        self.section_ids[()] = pmcid

    def _next_child_order(self, parent_id: str) -> int:
        order = self.child_counts.get(parent_id, 0)
        self.child_counts[parent_id] = order + 1
        return order

    def _next_chunk_order(self, section_id: str) -> int:
        order = self.chunk_counts.get(section_id, 0)
        self.chunk_counts[section_id] = order + 1
        return order

    def ensure_abstract_section(self) -> str:
        path = ("Abstract",)
        if path not in self.section_ids:
            section_id = f"{self.pmcid}.abstract"
            self.section_ids[path] = section_id
            self.nodes.append(
                section_node(
                    node_id=section_id,
                    text="Abstract",
                    level=1,
                    doc_id=self.pmcid,
                    order=self._next_child_order(self.pmcid),
                )
            )
            self.edges.append(edge(self.pmcid, section_id, "HAS_SUBSECTION"))
        return self.section_ids[path]

    def ensure_body_section_path(self, section_path: list[str]) -> str:
        parent_tuple: tuple[str, ...] = ()
        parent_id = self.pmcid

        for title in section_path:
            title_text = str(title or "").strip() or "Untitled Section"
            current_tuple = parent_tuple + (title_text,)
            if current_tuple not in self.section_ids:
                child_order = self._next_child_order(parent_id)
                section_id = (
                    f"{self.pmcid}.{child_order}"
                    if not parent_tuple
                    else f"{parent_id}.{child_order}"
                )
                self.section_ids[current_tuple] = section_id
                self.nodes.append(
                    section_node(
                        node_id=section_id,
                        text=title_text,
                        level=len(current_tuple),
                        doc_id=self.pmcid,
                        order=child_order,
                    )
                )
                self.edges.append(edge(parent_id, section_id, "HAS_SUBSECTION"))
            parent_tuple = current_tuple
            parent_id = self.section_ids[current_tuple]

        return parent_id

    def add_chunk(self, row: dict[str, Any]) -> None:
        chunk_id = str(row.get("chunk_id") or "").strip()
        chunk_text = str(row.get("text") or "").strip()
        source_type = str(row.get("source_type") or "").strip()
        section_path = [str(x or "").strip() for x in (row.get("section_path") or []) if str(x or "").strip()]

        if not chunk_id or not chunk_text or chunk_id in self.seen_chunk_ids:
            return

        if source_type == "abstract":
            section_id = self.ensure_abstract_section()
        else:
            section_id = self.ensure_body_section_path(section_path)

        self.nodes.append(
            chunk_node(
                node_id=chunk_id,
                text=chunk_text,
                doc_id=self.pmcid,
                section_id=section_id,
                order=self._next_chunk_order(section_id),
            )
        )
        self.edges.append(edge(section_id, chunk_id, "HAS_CHUNK"))
        self.seen_chunk_ids.add(chunk_id)


def flush_document(
    builder: DocumentGraphBuilder | None,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, str]],
) -> None:
    if builder is None:
        return
    nodes.extend(builder.nodes)
    edges.extend(builder.edges)


def main() -> None:
    args = parse_args()

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    builder: DocumentGraphBuilder | None = None

    total_docs = 0
    total_chunks = 0

    for row in read_jsonl(args.input):
        pmcid = str(row.get("pmcid") or "").strip()
        article_title = str(row.get("article_title") or "").strip()
        if not pmcid:
            continue

        if builder is None or builder.pmcid != pmcid:
            flush_document(builder, nodes, edges)
            if builder is not None:
                total_docs += 1
                total_chunks += len(builder.seen_chunk_ids)
            builder = DocumentGraphBuilder(pmcid=pmcid, article_title=article_title)

        builder.add_chunk(row)

    flush_document(builder, nodes, edges)
    if builder is not None:
        total_docs += 1
        total_chunks += len(builder.seen_chunk_ids)

    args.out_nodes.parent.mkdir(parents=True, exist_ok=True)
    args.out_edges.parent.mkdir(parents=True, exist_ok=True)

    with args.out_nodes.open("w", encoding="utf-8") as handle:
        json.dump(nodes, handle, ensure_ascii=False, indent=2)
    with args.out_edges.open("w", encoding="utf-8") as handle:
        json.dump(edges, handle, ensure_ascii=False, indent=2)

    print(f"Processed docs: {total_docs}")
    print(f"Chunk nodes written: {total_chunks}")
    print(f"Total nodes written: {len(nodes)} -> {args.out_nodes}")
    print(f"Total edges written: {len(edges)} -> {args.out_edges}")


if __name__ == "__main__":
    main()
