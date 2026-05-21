"""Run a two-stage hybrid retrieval pipeline for QASPER: BM25 for document recall, swaga-rag for in-document reranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from bm25_rag.index.builder import load_index as load_bm25_index
from bm25_rag.rag.retrieve import retrieve_with_scores
from swaga_rag.data.models import Edge, Section, TextNode
from swaga_rag.index.embeddings import EmbeddingModel
from swaga_rag.index.store import load_index as load_ontology_index
from swaga_rag.rag.drill import DrillSelector
from swaga_rag.rag.expand import GraphExpander
from swaga_rag.rag.pipeline import SWAGARAGPipeline
from swaga_rag.rag.score import NodeScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-stage hybrid retrieval for QASPER: BM25 candidate generation + swaga-rag refinement."
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/eval/qasper_validation_queries.jsonl"),
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("data/eval/qasper_validation_gold.jsonl"),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("artifacts/indexes/qasper"),
    )
    parser.add_argument(
        "--bm25-index",
        type=Path,
        default=Path("artifacts/indexes/qasper_bm25_index.pkl"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ablations/stable_baseline.json"),
    )
    parser.add_argument("--bm25-top-n-chunks", type=int, default=50)
    parser.add_argument("--bm25-top-m-docs", type=int, default=5)
    parser.add_argument("--final-top-k", type=int, default=10)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/hybrid_rag_results/qasper_validation"),
    )
    parser.add_argument(
        "--debug-output",
        type=Path,
        default=Path("artifacts/hybrid_rag_results/qasper_validation_debug.jsonl"),
    )
    return parser.parse_args()


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON on line {line_number} of {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object on line {line_number} of {path}")
            yield row


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError("Config must be a JSON object")
    return config


def infer_doc_id_from_chunk_id(chunk_id: str) -> str:
    cid = str(chunk_id or "").strip()
    if not cid:
        return ""
    if cid.endswith("_abstract"):
        return cid[: -len("_abstract")]
    parts = cid.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:-2])
    if len(parts) >= 2:
        return ".".join(parts[:-1])
    return cid


def infer_doc_id_from_section_id(section_id: str) -> str:
    sid = str(section_id or "").strip()
    if not sid:
        return ""
    if sid.endswith(".abstract"):
        return sid[: -len(".abstract")]
    parts = sid.split(".")
    if len(parts) >= 2 and parts[-1].isdigit():
        return ".".join(parts[:-1])
    return sid


def ordered_unique(items: list[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def build_doc_maps(
    sections: dict[str, Section],
    text_nodes: dict[str, TextNode],
    graph_adj: dict[str, list[Edge]],
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str]]:
    doc_to_sections: dict[str, set[str]] = {}
    doc_to_text_nodes: dict[str, set[str]] = {}
    section_to_doc: dict[str, str] = {}

    for section_id in sections:
        doc_id = infer_doc_id_from_section_id(section_id)
        if not doc_id:
            continue
        section_to_doc[section_id] = doc_id
        doc_to_sections.setdefault(doc_id, set()).add(section_id)

    for node_id, text_node in text_nodes.items():
        doc_id = ""
        if text_node.section_id and text_node.section_id in section_to_doc:
            doc_id = section_to_doc[text_node.section_id]
        else:
            doc_id = infer_doc_id_from_chunk_id(node_id)
        if not doc_id:
            continue
        doc_to_text_nodes.setdefault(doc_id, set()).add(node_id)

    for from_id in graph_adj:
        if from_id in section_to_doc:
            continue
        doc_id = infer_doc_id_from_section_id(from_id)
        if doc_id and from_id in sections:
            section_to_doc[from_id] = doc_id
            doc_to_sections.setdefault(doc_id, set()).add(from_id)

    return doc_to_sections, doc_to_text_nodes, section_to_doc


def build_filtered_views(
    candidate_doc_ids: list[str],
    sections: dict[str, Section],
    text_nodes: dict[str, TextNode],
    graph_adj: dict[str, list[Edge]],
    doc_to_sections: dict[str, set[str]],
    doc_to_text_nodes: dict[str, set[str]],
) -> tuple[dict[str, Section], dict[str, TextNode], dict[str, list[Edge]], set[str]]:
    allowed_section_ids: set[str] = set()
    allowed_text_node_ids: set[str] = set()

    for doc_id in candidate_doc_ids:
        allowed_section_ids.update(doc_to_sections.get(doc_id, set()))
        allowed_text_node_ids.update(doc_to_text_nodes.get(doc_id, set()))

    allowed_node_ids = allowed_section_ids | allowed_text_node_ids

    filtered_sections: dict[str, Section] = {}
    for section_id in allowed_section_ids:
        section = sections[section_id]
        filtered_sections[section_id] = Section(
            id=section.id,
            level=section.level,
            parent_id=section.parent_id if section.parent_id in allowed_section_ids else None,
            children_ids=[child_id for child_id in section.children_ids if child_id in allowed_section_ids],
            local_text=section.local_text,
            subtree_text=section.subtree_text,
            E_local=section.E_local,
            E_subtree=section.E_subtree,
        )

    filtered_text_nodes: dict[str, TextNode] = {}
    for node_id in allowed_text_node_ids:
        node = text_nodes[node_id]
        filtered_text_nodes[node_id] = TextNode(
            id=node.id,
            section_id=node.section_id if node.section_id in allowed_section_ids else None,
            node_type=node.node_type,
            text=node.text,
            embedding=node.embedding,
        )

    filtered_graph_adj: dict[str, list[Edge]] = {}
    for node_id in allowed_node_ids:
        kept_edges = [
            edge
            for edge in graph_adj.get(node_id, [])
            if edge.to_id in allowed_node_ids
        ]
        if kept_edges:
            filtered_graph_adj[node_id] = kept_edges

    return filtered_sections, filtered_text_nodes, filtered_graph_adj, allowed_node_ids


def run_restricted_ontology_query(
    query: str,
    candidate_doc_ids: list[str],
    sections: dict[str, Section],
    text_nodes: dict[str, TextNode],
    graph_adj: dict[str, list[Edge]],
    doc_to_sections: dict[str, set[str]],
    doc_to_text_nodes: dict[str, set[str]],
    embedding_model: EmbeddingModel,
    config: dict[str, Any],
    final_top_k: int,
) -> list[dict[str, Any]]:
    if not candidate_doc_ids:
        return []

    filtered_sections, filtered_text_nodes, filtered_graph_adj, _ = build_filtered_views(
        candidate_doc_ids=candidate_doc_ids,
        sections=sections,
        text_nodes=text_nodes,
        graph_adj=graph_adj,
        doc_to_sections=doc_to_sections,
        doc_to_text_nodes=doc_to_text_nodes,
    )
    if not filtered_sections or not filtered_text_nodes:
        return []

    query_emb = embedding_model.encode(query)

    pipeline = SWAGARAGPipeline(
        sections=filtered_sections,
        text_nodes=filtered_text_nodes,
        graph_adj=filtered_graph_adj,
        embedding_model=embedding_model,
        config=config,
    )

    selector = DrillSelector(filtered_sections, pipeline.drill_cfg)
    seed_ids = selector.select_seeds(query_emb)
    if not seed_ids:
        ranked_l1 = selector.rank_l1_sections(query_emb)
        seed_ids = [section.id for section in ranked_l1[: max(pipeline.drill_cfg.top_r, 1)]]
    if not seed_ids:
        seed_ids = [doc_id for doc_id in candidate_doc_ids if doc_id in filtered_sections]
    if not seed_ids:
        return []

    expander = GraphExpander(
        graph_adj=filtered_graph_adj,
        max_depth=pipeline.max_graph_depth,
        max_nodes=pipeline.max_graph_nodes,
        allowed_relations=pipeline.allowed_relations,
    )
    all_nodes, _all_edges, dist_to_seed = expander.expand(seed_ids)

    scorer = NodeScorer(
        sections=filtered_sections,
        text_nodes=filtered_text_nodes,
        score_cfg=pipeline.score_cfg,
    )
    ranked = scorer.score_all(
        query_emb=query_emb,
        dist_to_seed=dist_to_seed,
        candidate_node_ids=list(all_nodes),
        top_k=final_top_k,
    )

    out: list[dict[str, Any]] = []
    for node_id, score in ranked:
        text_node = filtered_text_nodes[node_id]
        out.append(
            {
                "node_id": node_id,
                "chunk_id": node_id,
                "section_id": text_node.section_id,
                "text": text_node.text,
                "score": float(score),
            }
        )
    return out


def load_gold_qids(path: Path) -> list[str]:
    qids: list[str] = []
    for row in load_jsonl(path):
        qid = str(row.get("id") or "").strip()
        if qid:
            qids.append(qid)
    return qids


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    bm25_index = load_bm25_index(args.bm25_index)
    sections, text_nodes, graph_adj = load_ontology_index(str(args.index_dir))
    embedding_model = EmbeddingModel(device=args.device)
    doc_to_sections, doc_to_text_nodes, _section_to_doc = build_doc_maps(
        sections=sections,
        text_nodes=text_nodes,
        graph_adj=graph_adj,
    )

    queries = list(load_jsonl(args.queries))
    qids_with_gold = set(load_gold_qids(args.gold))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.debug_output.parent.mkdir(parents=True, exist_ok=True)

    candidate_doc_counts: list[int] = []
    final_prediction_counts: list[int] = []
    empty_final_predictions = 0
    processed_queries = 0

    with args.debug_output.open("w", encoding="utf-8") as debug_handle:
        for query_row in queries:
            qid = str(query_row.get("id") or "").strip()
            query = str(query_row.get("query") or "").strip()
            if not qid or not query:
                continue
            if qids_with_gold and qid not in qids_with_gold:
                continue

            processed_queries += 1

            bm25_hits = retrieve_with_scores(
                bm25_index,
                query,
                top_k=args.bm25_top_n_chunks,
            )
            bm25_top_chunk_ids = [chunk_id for chunk_id, _text, _score in bm25_hits]
            bm25_top_doc_ids = ordered_unique(
                [infer_doc_id_from_chunk_id(chunk_id) for chunk_id in bm25_top_chunk_ids],
                args.bm25_top_m_docs,
            )

            final_items = run_restricted_ontology_query(
                query=query,
                candidate_doc_ids=bm25_top_doc_ids,
                sections=sections,
                text_nodes=text_nodes,
                graph_adj=graph_adj,
                doc_to_sections=doc_to_sections,
                doc_to_text_nodes=doc_to_text_nodes,
                embedding_model=embedding_model,
                config=config,
                final_top_k=args.final_top_k,
            )

            predicted_chunk_ids = [item["chunk_id"] for item in final_items]
            predicted_node_ids = [item["node_id"] for item in final_items]

            candidate_doc_counts.append(len(bm25_top_doc_ids))
            final_prediction_counts.append(len(predicted_chunk_ids))
            if not predicted_chunk_ids:
                empty_final_predictions += 1

            out_row = {
                "id": qid,
                "query": query,
                "candidate_doc_ids": bm25_top_doc_ids,
                "predicted_chunk_ids": predicted_chunk_ids,
                "predicted_node_ids": predicted_node_ids,
                "output_ids": predicted_chunk_ids,
                "output_items": [
                    {
                        "chunk_id": item["chunk_id"],
                        "text": item["text"],
                        "score": item["score"],
                    }
                    for item in final_items
                ],
            }
            (args.out_dir / f"{qid}.json").write_text(
                json.dumps(out_row, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            debug_row = {
                "question_id": qid,
                "bm25_top_chunk_ids": bm25_top_chunk_ids,
                "bm25_top_doc_ids": bm25_top_doc_ids,
                "final_predicted_chunk_ids": predicted_chunk_ids,
            }
            debug_handle.write(json.dumps(debug_row, ensure_ascii=False) + "\n")

    avg_candidate_docs = mean(candidate_doc_counts) if candidate_doc_counts else 0.0
    avg_final_predictions = mean(final_prediction_counts) if final_prediction_counts else 0.0

    print(f"Queries processed: {processed_queries}")
    print(f"Average number of BM25 candidate docs per query: {avg_candidate_docs:.4f}")
    print(f"Average number of final predictions per query: {avg_final_predictions:.4f}")
    print(f"Queries with empty final predictions: {empty_final_predictions}")


if __name__ == "__main__":
    main()


