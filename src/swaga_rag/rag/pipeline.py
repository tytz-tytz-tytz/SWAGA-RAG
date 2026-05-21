from typing import Dict, List, Any, Optional
from collections import Counter

from ..data.models import TextNode, Section, Edge
from ..index.embeddings import EmbeddingModel
from .drill import DrillSelector, DrillConfig, cosine_sim
from .expand import GraphExpander
from .score import NodeScorer, ScoreConfig


class OntologyRAGPipeline:
    """
    Ontology-aware Retrieval-Augmented Generation (OntologyRAG) pipeline.

    This pipeline operates on a pre-built offline index:
    - section hierarchy
    - text nodes
    - graph adjacency

    It performs:
    1) semantic drill-down to select seed sections
    2) graph expansion from seeds
    3) scoring of text nodes
    4) aggregation into full section candidates (LLM-ready)
    """

    def __init__(
        self,
        sections: Dict[str, Section],
        text_nodes: Dict[str, TextNode],
        graph_adj: Dict[str, List[Edge]],
        embedding_model: EmbeddingModel,
        config: Dict[str, Any],
    ):
        self.sections = sections
        self.text_nodes = text_nodes
        self.graph_adj = graph_adj
        self.model = embedding_model

        # -----------------------------
        # Parse configuration
        # -----------------------------

        # Drill configuration
        drill_cfg_dict = config.get("drill", {})
        self.drill_cfg = DrillConfig(**drill_cfg_dict)

        # Expansion configuration
        expand_cfg = config.get("expand", {})
        self.max_graph_depth: int = int(expand_cfg.get("max_depth", 3))
        self.max_graph_nodes: int = int(expand_cfg.get("max_nodes", 200))
        self.allowed_relations = set(expand_cfg.get("allowed_relations", []))

        # Scoring configuration
        score_cfg_dict = config.get("score", {})
        self.score_cfg = ScoreConfig(**score_cfg_dict)

        # Output / runtime options
        output_cfg = config.get("output", {})
        self.top_k_text: int = int(output_cfg.get("top_k_text", 20))
        self.save_debug: bool = bool(output_cfg.get("save_debug", False))

        # Debug options (optional)
        debug_cfg = output_cfg.get("debug", {}) if isinstance(output_cfg.get("debug", {}), dict) else {}
        self.debug_top_k_nodes: int = int(debug_cfg.get("top_k_nodes", 20))
        self.debug_top_k_seeds: int = int(debug_cfg.get("top_k_seeds", 10))

    # =============================================================
    # FULL SECTION MODE (LLM-ready output)
    # =============================================================
    def build_full_sections(self, ranked_nodes: List[dict]) -> List[dict]:
        """
        Convert ranked text nodes into full section candidates.

        Steps:
        - Group ranked text nodes by section_id
        - Reconstruct full section text from all its chunks
        - Aggregate section score (max over node scores)
        - Track node_ids that contributed to the section score

        Output format:
        {
            "section_id": str,
            "title": str,
            "text": str,
            "score": float,
            "node_ids": List[str],
        }
        """

        # 1) Group scored nodes by section
        section_to_nodes: Dict[str, List[dict]] = {}
        for item in ranked_nodes:
            sid = item.get("section_id")
            if sid is None:
                continue
            section_to_nodes.setdefault(sid, []).append(item)

        output: List[dict] = []

        # 2) Build full section text and aggregate scores
        for sid, nodes in section_to_nodes.items():
            sec = self.sections[sid]

            # Section title (first line of local_text, if present)
            title = ""
            if sec.local_text:
                title = sec.local_text.split("\n")[0].strip()

            # Collect all text chunks belonging to this section
            chunks = []
            for nid, tn in self.text_nodes.items():
                if tn.section_id == sid:
                    chunks.append((nid, tn.text))

            # Sort chunks by their original order (heuristic: chunk id)
            def sort_key(x):
                nid = x[0]
                try:
                    return int(nid.split("ch")[1])
                except Exception:
                    return 999_999

            chunks_sorted = sorted(chunks, key=sort_key)
            full_text = "\n".join(t for _, t in chunks_sorted).strip()

            # Aggregate section score (max over node scores)
            section_score = float(max(node["score"] for node in nodes))
            node_ids = [node["node_id"] for node in nodes]

            output.append({
                "section_id": sid,
                "title": title,
                "text": full_text,
                "score": section_score,
                "node_ids": node_ids,
            })

        # 3) Stable sorting:
        #    - primary: descending score
        #    - secondary: section order (if parseable)
        def section_sort_key(item: dict):
            sid = item["section_id"]
            try:
                order = int(sid.split("ch")[1])
            except Exception:
                order = 999_999
            return (-item["score"], order)

        return sorted(output, key=section_sort_key)

    # =============================================================
    # DEBUG HELPERS
    # =============================================================
    def _build_debug_payload(
        self,
        query: str,
        query_emb,
        seed_ids: List[str],
        all_nodes,
        all_edges,
        dist_to_seed: Dict[str, int],
        ranked_node_pairs,
        text_context: List[dict],
    ) -> Dict[str, Any]:
        """
        Build a structured debug payload for diagnosing parameter effects.

        This payload is not meant for blind evaluation and should be stored separately.
        """
        # Seed diagnostics
        seed_levels = []
        seed_local_sims = []
        seed_subtree_sims = []

        for sid in seed_ids:
            sec = self.sections.get(sid)
            if sec is None:
                continue
            seed_levels.append(int(getattr(sec, "level", 0) or 0))
            seed_local_sims.append(cosine_sim(query_emb, getattr(sec, "E_local", None)))
            seed_subtree_sims.append(cosine_sim(query_emb, getattr(sec, "E_subtree", None)))

        # Ranked node diagnostics
        top_nodes = []
        for nid, score in ranked_node_pairs[: self.debug_top_k_nodes]:
            tn = self.text_nodes.get(nid)
            if tn is None:
                continue
            sim = cosine_sim(query_emb, getattr(tn, "embedding", None))
            top_nodes.append({
                "node_id": nid,
                "section_id": tn.section_id,
                "type": tn.node_type,
                "dist": int(dist_to_seed.get(nid, 999)),
                "sim": float(sim),
                "score": float(score),
                "text_preview": (tn.text[:200] + "…") if isinstance(tn.text, str) and len(tn.text) > 200 else tn.text,
            })

        # Type distribution in final ranked list
        type_counts = Counter([item.get("type") for item in text_context if isinstance(item, dict)])

        # Graph diagnostics
        dists = list(dist_to_seed.values())
        dist_stats = {
            "min": int(min(dists)) if dists else None,
            "max": int(max(dists)) if dists else None,
            "mean": float(sum(dists) / len(dists)) if dists else None,
        }

        # Edge type distribution inside expanded graph
        edge_type_counts = Counter([e.relation_type for e in all_edges])

        # Cosine similarity summary for ranked nodes
        ranked_sims = []
        for nid, _ in ranked_node_pairs[: self.debug_top_k_nodes]:
            tn = self.text_nodes.get(nid)
            if tn is None:
                continue
            ranked_sims.append(cosine_sim(query_emb, getattr(tn, "embedding", None)))

        sim_stats = {
            "min": float(min(ranked_sims)) if ranked_sims else None,
            "max": float(max(ranked_sims)) if ranked_sims else None,
            "mean": float(sum(ranked_sims) / len(ranked_sims)) if ranked_sims else None,
        }

        return {
            "query": query,
            "drill": {
                "config": {
                    "top_r": self.drill_cfg.top_r,
                    "top_k": self.drill_cfg.top_k,
                    "tau_local": self.drill_cfg.tau_local,
                    "tau_child": self.drill_cfg.tau_child,
                    "margin": self.drill_cfg.margin,
                },
                "seed_count": len(seed_ids),
                "seed_ids": seed_ids[: self.debug_top_k_seeds],
                "seed_levels": seed_levels,
                "seed_local_sim_stats": {
                    "min": float(min(seed_local_sims)) if seed_local_sims else None,
                    "max": float(max(seed_local_sims)) if seed_local_sims else None,
                    "mean": float(sum(seed_local_sims) / len(seed_local_sims)) if seed_local_sims else None,
                },
                "seed_subtree_sim_stats": {
                    "min": float(min(seed_subtree_sims)) if seed_subtree_sims else None,
                    "max": float(max(seed_subtree_sims)) if seed_subtree_sims else None,
                    "mean": float(sum(seed_subtree_sims) / len(seed_subtree_sims)) if seed_subtree_sims else None,
                },
            },
            "expand": {
                "config": {
                    "max_depth": self.max_graph_depth,
                    "max_nodes": self.max_graph_nodes,
                    "allowed_relations": sorted(list(self.allowed_relations)),
                },
                "expanded_node_count": int(len(all_nodes)),
                "expanded_edge_count": int(len(all_edges)),
                "dist_stats": dist_stats,
                "edge_type_counts": dict(edge_type_counts),
            },
            "score": {
                "config": {
                    "mode": self.score_cfg.mode,
                    "w_text": self.score_cfg.w_text,
                    "w_type": self.score_cfg.w_type,
                    "w_level": self.score_cfg.w_level,
                    "w_dist": self.score_cfg.w_dist,
                    "type_bonus": dict(self.score_cfg.type_bonus),
                    "level_bonus": dict(self.score_cfg.level_bonus),
                },
                "top_nodes": top_nodes,
                "ranked_sim_stats": sim_stats,
                "type_counts_in_topk": dict(type_counts),
            },
        }

    # =============================================================
    # MAIN PIPELINE METHOD
    # =============================================================
    def run_query(self, query: str) -> Dict[str, Any]:
        """
        Execute the full OntologyRAG pipeline for a single query.

        Returns a dictionary containing:
        - section_candidates (LLM-ready)
        - ranked text nodes
        - graph context (expanded subgraph)

        If save_debug is enabled in config, the output also includes a "debug" field.
        """

        # 1) Embed query
        query_emb = self.model.encode(query)

        # 2) Drill-down: select seed sections
        selector = DrillSelector(self.sections, self.drill_cfg)
        seed_ids = selector.select_seeds(query_emb)

        # 3) Graph expansion (BFS from seed sections)
        expander = GraphExpander(
            graph_adj=self.graph_adj,
            max_depth=self.max_graph_depth,
            max_nodes=self.max_graph_nodes,
            allowed_relations=self.allowed_relations,
        )
        all_nodes, all_edges, dist_to_seed = expander.expand(seed_ids)

        # 4) Score candidate text nodes
        scorer = NodeScorer(
            sections=self.sections,
            text_nodes=self.text_nodes,
            score_cfg=self.score_cfg,
        )
        ranked = scorer.score_all(
            query_emb=query_emb,
            dist_to_seed=dist_to_seed,
            candidate_node_ids=list(all_nodes),
            top_k=self.top_k_text,
        )

        # Keep raw ranked pairs for debug
        ranked_pairs = list(ranked)

        # 5) Build detailed ranked text node context
        text_context: List[dict] = []
        for nid, score in ranked:
            tn = self.text_nodes[nid]
            text_context.append({
                "node_id": nid,
                "section_id": tn.section_id,
                "type": tn.node_type,
                "text": tn.text,
                "score": float(score),
            })

        # 6) Build full section candidates (LLM-ready)
        section_candidates = self.build_full_sections(text_context)

        # 7) Build graph context (for analysis / visualization)
        graph_nodes = list(all_nodes)
        graph_edges = [
            {"from": e.from_id, "to": e.to_id, "type": e.relation_type}
            for e in all_edges
            if e.from_id in all_nodes and e.to_id in all_nodes
        ]

        # 8) Final output format
        out: Dict[str, Any] = {
            "query": query,
            "section_candidates": section_candidates,
            "text_nodes": text_context,
            "graph_context": {
                "nodes": graph_nodes,
                "edges": graph_edges,
            },
        }

        # 9) Optional debug payload (stored separately by the runner script)
        if self.save_debug:
            out["debug"] = self._build_debug_payload(
                query=query,
                query_emb=query_emb,
                seed_ids=seed_ids,
                all_nodes=all_nodes,
                all_edges=all_edges,
                dist_to_seed=dist_to_seed,
                ranked_node_pairs=ranked_pairs,
                text_context=text_context,
            )

        return out


SWAGARAGPipeline = OntologyRAGPipeline
