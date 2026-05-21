from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from ..data.models import TextNode, Section
from .drill import cosine_sim


class ScoreConfig:
    """
    Configuration for final node scoring.

    The final score is a weighted combination of:
    - semantic similarity between query and node embedding
    - node type bonus
    - section level bonus
    - graph distance penalty

    Supported modes:
    - "full": use the full scoring formula (default)
    - "text_only": score = cosine_sim(query, node_embedding)
    """

    def __init__(
        self,
        mode: str = "full",
        w_text: float = 1.0,
        w_type: float = 0.3,
        w_level: float = 0.15,
        w_dist: float = 0.2,
        type_bonus: Optional[Dict[str, float]] = None,
        level_bonus: Optional[Dict[Any, float]] = None,
        aggregation: str = "max",
        **_: Any,
    ):
        self.mode = str(mode)

        self.w_text = float(w_text)
        self.w_type = float(w_type)
        self.w_level = float(w_level)
        self.w_dist = float(w_dist)

        # Bonuses by node type (can be overridden by config)
        self.type_bonus: Dict[str, float] = dict(type_bonus) if type_bonus is not None else {
            "list_item": 1.0,
            "chunk": 0.6,
            "section_title": 0.8,
            "caption": 0.2,
        }

        # Bonuses by section depth/level (can be overridden by config)
        # NOTE: JSON keys may be strings, so we normalize to int when possible.
        self.level_bonus: Dict[int, float] = {}
        if level_bonus is None:
            self.level_bonus = {
                1: 0.1,
                2: 0.2,
                3: 0.3,
            }
        else:
            for k, v in dict(level_bonus).items():
                try:
                    kk = int(k)
                except Exception:
                    continue
                self.level_bonus[kk] = float(v)

        # Stored for downstream use (e.g., section aggregation strategy),
        # even if not applied inside this module.
        self.aggregation = str(aggregation)


class NodeScorer:
    """
    Computes a final score for each text node among candidate_node_ids
    (typically nodes reached by graph expansion).
    """

    def __init__(
        self,
        sections: Dict[str, Section],
        text_nodes: Dict[str, TextNode],
        score_cfg: ScoreConfig,
    ):
        self.sections = sections
        self.text_nodes = text_nodes
        self.cfg = score_cfg

    def score_one(
        self,
        node_id: str,
        query_emb: np.ndarray,
        dist_to_seed: Dict[str, int],
    ) -> float:
        """
        Score a single text node.

        Returns a very negative score for non-text nodes or missing embeddings.
        """
        tn = self.text_nodes.get(node_id)
        if tn is None:
            return -999.0  # Non-text node

        sim = cosine_sim(query_emb, tn.embedding)

        # Mode: text-only ablation
        if self.cfg.mode == "text_only":
            return sim

        # Type bonus
        bonus_type = self.cfg.type_bonus.get(tn.node_type, 0.0)

        # Section level bonus
        sec = self.sections.get(tn.section_id)
        lvl = sec.level if sec else 1
        bonus_level = self.cfg.level_bonus.get(int(lvl), 0.0)

        # Graph distance (unreachable nodes should not happen if candidates come from BFS,
        # but we keep a safe default penalty anyway).
        dist = dist_to_seed.get(node_id, 999)

        # Final score
        score = (
            self.cfg.w_text * sim
            + self.cfg.w_type * bonus_type
            + self.cfg.w_level * bonus_level
            - self.cfg.w_dist * dist
        )
        return float(score)

    def score_all(
        self,
        query_emb: np.ndarray,
        dist_to_seed: Dict[str, int],
        candidate_node_ids: List[str],
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Score all candidate text nodes and return the top-K by descending score.
        """
        scored: List[Tuple[str, float]] = []

        for nid in candidate_node_ids:
            # Only rank text nodes
            if nid not in self.text_nodes:
                continue

            s = self.score_one(nid, query_emb, dist_to_seed)
            scored.append((nid, float(s)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
