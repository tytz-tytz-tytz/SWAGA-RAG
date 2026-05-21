from typing import Dict, List, Set
import numpy as np

from ..data.models import Section


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Safe cosine similarity computation.

    Returns -1.0 if any vector is missing or has zero norm.
    """
    if a is None or b is None:
        return -1.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


class DrillConfig:
    """
    Configuration parameters for the drill-down procedure.

    Parameters:
    - tau_local:
        Minimum similarity threshold for a section's local text
        to be considered directly relevant.
    - tau_child:
        Minimum similarity threshold for a child subtree to continue drilling.
    - margin:
        Allowed margin by which local similarity may be worse than
        the best child subtree similarity and still be selected as a seed.
    - top_k:
        Number of top-scoring child sections to explore recursively.
    - top_r:
        Number of top-level (level=1) sections to consider as roots.
    """

    def __init__(
        self,
        tau_local: float = 0.35,
        tau_child: float = 0.45,
        margin: float = 0.05,
        top_k: int = 2,
        top_r: int = 3,
    ):
        self.tau_local = tau_local
        self.tau_child = tau_child
        self.margin = margin
        self.top_k = top_k
        self.top_r = top_r


class DrillSelector:
    """
    Implements recursive semantic drill-down for selecting seed sections.

    The algorithm compares query similarity against:
    - local section text
    - aggregated subtree embeddings

    and decides whether to:
    - select a section as a seed
    - or continue drilling into its children.
    """

    def __init__(self, sections: Dict[str, Section], config: DrillConfig):
        self.sections = sections
        self.cfg = config

    # -------------------------------------------------------------
    # STEP 1 — Rank top-level sections by subtree similarity
    # -------------------------------------------------------------
    def rank_l1_sections(self, query_emb: np.ndarray) -> List[Section]:
        """
        Rank level-1 sections by cosine similarity between
        the query embedding and section subtree embeddings.
        """
        lvl1 = [s for s in self.sections.values() if s.level == 1]
        scored = [
            (s, cosine_sim(query_emb, s.E_subtree))
            for s in lvl1
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored]

    # -------------------------------------------------------------
    # STEP 2 — Recursive drill-down
    # -------------------------------------------------------------
    def drill_section(
        self,
        sec: Section,
        query_emb: np.ndarray,
        seeds: Set[str],
    ) -> None:
        """
        Recursively traverse the section hierarchy to select seed sections.

        A section is selected as a seed if:
        - its local text similarity exceeds tau_local
        - and is not significantly worse than the best child subtree
          (within the specified margin).

        Otherwise, drilling continues into the most relevant children,
        if their subtree similarity exceeds tau_child.
        """

        cfg = self.cfg
        children = [self.sections[cid] for cid in sec.children_ids]

        score_local = cosine_sim(query_emb, sec.E_local)
        child_scores = [(c, cosine_sim(query_emb, c.E_subtree)) for c in children]

        score_best_child = max((sc for _, sc in child_scores), default=-1.0)

        # ---------------------------------------------------------
        # CASE 1 — Section has no meaningful local text
        # ---------------------------------------------------------
        if not sec.local_text or not sec.local_text.strip():
            # If no child subtree is relevant, stop drilling
            if score_best_child < cfg.tau_child:
                return

            # Otherwise, continue drilling into top-k children
            child_scores.sort(key=lambda x: x[1], reverse=True)
            for c, _ in child_scores[: cfg.top_k]:
                self.drill_section(c, query_emb, seeds)
            return

        # ---------------------------------------------------------
        # CASE 2 — Section has local text
        # ---------------------------------------------------------
        is_seed = (
            score_local >= cfg.tau_local
            and score_local >= score_best_child - cfg.margin
        )

        if is_seed:
            seeds.add(sec.id)
            return

        # If local text is not selected, but subtree is relevant,
        # continue drilling into children
        if score_best_child >= cfg.tau_child:
            child_scores.sort(key=lambda x: x[1], reverse=True)
            for c, _ in child_scores[: cfg.top_k]:
                self.drill_section(c, query_emb, seeds)

        # Otherwise, stop drilling this branch
        return

    # -------------------------------------------------------------
    # TOP-LEVEL ENTRY POINT
    # -------------------------------------------------------------
    def select_seeds(self, query_emb: np.ndarray) -> List[str]:
        """
        Full seed selection procedure:

        1) Rank level-1 sections by subtree similarity
        2) Select top-R roots
        3) Perform recursive drill-down from each root
        4) Return the collected set of seed section IDs
        """

        lvl1_ranked = self.rank_l1_sections(query_emb)
        roots = lvl1_ranked[: self.cfg.top_r]

        seeds: Set[str] = set()
        for root in roots:
            self.drill_section(root, query_emb, seeds)

        return list(seeds)
