# src/data/loaders.py

import json
from typing import Dict, List

from .models import Section, TextNode, Edge


# -------------------------------------------------------------
# JSON LOADER
# -------------------------------------------------------------
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------------
# FILTER: page-number chunks like "6"
# -------------------------------------------------------------
def is_page_number_chunk(node: dict) -> bool:
    text = (node.get("text") or "").strip()
    return text.isdigit() and len(text) <= 3


# -------------------------------------------------------------
# LOAD NODES
# -------------------------------------------------------------
def load_nodes(path_nodes: str):
    raw_nodes = load_json(path_nodes)

    sections: Dict[str, Section] = {}
    text_nodes: Dict[str, TextNode] = {}
    figures: Dict[str, dict] = {}

    for item in raw_nodes:
        node_id = item["id"]
        raw_type = item["type"]
        attrs = item.get("attributes", {})

        # --------------------------------------------------------
        # REAL DOCUMENT SECTIONS = type=="Section" AND attributes.level exists
        # --------------------------------------------------------
        if raw_type == "Section":
            lvl = attrs.get("level", None)
            if lvl is not None:                       # real document section
                try:
                    lvl = int(lvl)
                    sections[node_id] = Section(
                        id=node_id,
                        level=lvl
                    )
                except (ValueError, TypeError):
                    pass  # invalid level — ignore
            # ALL other Section nodes → ignore completely
            continue

        # --------------------------------------------------------
        # FIGURES
        # --------------------------------------------------------
        if raw_type == "Figure":
            figures[node_id] = item
            continue

        # --------------------------------------------------------
        # CHUNK → TEXT NODE
        # --------------------------------------------------------
        if raw_type == "Chunk":
            # skip page numbers like "7"
            if is_page_number_chunk(item):
                continue

            txt = item.get("text") or ""
            text_nodes[node_id] = TextNode(
                id=node_id,
                section_id=None,
                node_type="chunk",
                text=txt,
                order=attrs.get("order", None),
            )
            continue

        # --------------------------------------------------------
        # LIST ITEMS – keep as text nodes so they can be promoted
        # to chunk-level fragments during retrieval.
        # --------------------------------------------------------
        if raw_type == "ListItem":
            txt = item.get("text") or ""
            chunk_id = attrs.get("chunk_id")
            parent_id = f"chunk_{chunk_id}" if isinstance(chunk_id, str) and chunk_id.strip() else None
            text_nodes[node_id] = TextNode(
                id=node_id,
                section_id=None,
                node_type="list_item",
                text=txt,
                parent_id=parent_id,
                order=attrs.get("order", None),
            )
            continue

        # --------------------------------------------------------
        # URL / ReferenceTarget — not added as text nodes
        # --------------------------------------------------------
        continue

    return sections, text_nodes, figures


# -------------------------------------------------------------
# LOAD EDGES
# -------------------------------------------------------------
def load_edges(path_edges: str):
    raw_edges = load_json(path_edges)
    return [
        Edge(
            from_id=e["source"],
            to_id=e["target"],
            relation_type=e["type"],
        )
        for e in raw_edges
    ]


# -------------------------------------------------------------
# BUILD GRAPH + ASSIGN SECTION → TEXT NODES
# -------------------------------------------------------------
def build_graph(
    sections: Dict[str, Section],
    text_nodes: Dict[str, TextNode],
    figures: Dict[str, dict],
    edges: List[Edge]
):
    graph_adj: Dict[str, List[Edge]] = {}

    def add_edge(edge: Edge):
        graph_adj.setdefault(edge.from_id, []).append(edge)

    for e in edges:
        add_edge(e)

        # --------------------------------------------------------
        # HAS_SUBSECTION: parent Section → child Section
        # --------------------------------------------------------
        if e.relation_type == "HAS_SUBSECTION":
            parent = sections.get(e.from_id)
            child = sections.get(e.to_id)
            if parent and child:
                child.parent_id = parent.id
                parent.children_ids.append(child.id)
            continue

        # --------------------------------------------------------
        # HAS_CHUNK: Section → Chunk
        # --------------------------------------------------------
        if e.relation_type == "HAS_CHUNK":
            sec = sections.get(e.from_id)
            ch = text_nodes.get(e.to_id)
            if sec and ch:
                ch.section_id = sec.id
            continue

        # --------------------------------------------------------
        # HAS_ITEM: ignored (ListItem not stored)
        # --------------------------------------------------------
        if e.relation_type == "HAS_ITEM":
            parent = text_nodes.get(e.from_id)
            child = text_nodes.get(e.to_id)
            if parent and child:
                child.parent_id = parent.id
                if parent.section_id:
                    child.section_id = parent.section_id
            continue

        # --------------------------------------------------------
        # CAPTIONS: caption_chunk → figure
        # --------------------------------------------------------
        if e.relation_type == "CAPTIONS":
            cap = text_nodes.get(e.from_id)
            if cap:
                cap.node_type = "caption"
            continue

        # --------------------------------------------------------
        # LINKS_TO: just store in graph
        # --------------------------------------------------------
        if e.relation_type == "LINKS_TO":
            continue

    # Propagate section ids from parent chunks to list items after all edges
    # have been processed, so list items can participate in section-level
    # embeddings and downstream grouping.
    for node in text_nodes.values():
        if node.node_type == "list_item" and node.section_id is None and node.parent_id:
            parent = text_nodes.get(node.parent_id)
            if parent and parent.section_id:
                node.section_id = parent.section_id

    return sections, text_nodes, graph_adj


# -------------------------------------------------------------
# COMPLETE PIPELINE
# -------------------------------------------------------------
def load_ontology(path_nodes: str, path_edges: str):
    sections, text_nodes, figures = load_nodes(path_nodes)
    edges = load_edges(path_edges)

    sections, text_nodes, graph_adj = build_graph(
        sections,
        text_nodes,
        figures,
        edges
    )

    return sections, text_nodes, graph_adj
