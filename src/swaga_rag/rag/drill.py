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
        threshold_mode: str = "absolute",
        rank_top_p: float = 0.5,
        **_: object,
    ):
        self.tau_local = tau_local
        self.tau_child = tau_child
        self.margin = margin
        self.top_k = top_k
        self.top_r = top_r
        # threshold_mode:
        #   "absolute"   — compare raw cosine to tau_* (default; unchanged behaviour)
        #   "percentile" — per-query min-max normalize section scores before tau_*
        #   "rank"       — descend into the top rank_top_p fraction of children
        self.threshold_mode = str(threshold_mode)
        self.rank_top_p = float(rank_top_p)


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
        # per-query min-max bounds for percentile mode (set in select_seeds)
        self._qmin = 0.0
        self._qmax = 1.0

    def _norm(self, x: float) -> float:
        """Percentile mode: rescale a raw cosine to the per-query [min,max]
        range before threshold comparison. Absolute/rank: identity."""
        if self.cfg.threshold_mode == "percentile" and self._qmax > self._qmin:
            return (x - self._qmin) / (self._qmax - self._qmin)
        return x

    def _children_to_descend(self, child_scores):
        """Mode-aware selection of children to recurse into.

        absolute/percentile: gate on the best child's (normalized) subtree
          score vs tau_child, then take top_k.
        rank: take the top ``rank_top_p`` fraction of children by subtree score
          (no tau gate).
        """
        if not child_scores:
            return []
        ordered = sorted(child_scores, key=lambda x: x[1], reverse=True)
        if self.cfg.threshold_mode == "rank":
            import math
            k = max(1, math.ceil(self.cfg.rank_top_p * len(ordered)))
            return [c for c, _ in ordered[:k]]
        if self._norm(ordered[0][1]) < self.cfg.tau_child:
            return []
        return [c for c, _ in ordered[: self.cfg.top_k]]

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
            for c in self._children_to_descend(child_scores):
                self.drill_section(c, query_emb, seeds)
            return

        # ---------------------------------------------------------
        # CASE 2 — Section has local text
        # ---------------------------------------------------------
        # Seed if the (mode-normalized) local score clears tau_local and is not
        # much worse than the best child (margin comparison stays on raw scores
        # — the δ margin is deliberately unchanged across threshold modes).
        is_seed = (
            self._norm(score_local) >= cfg.tau_local
            and score_local >= score_best_child - cfg.margin
        )

        if is_seed:
            seeds.add(sec.id)
            return

        # Otherwise continue drilling into the mode-selected children.
        for c in self._children_to_descend(child_scores):
            self.drill_section(c, query_emb, seeds)
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

        if self.cfg.threshold_mode == "percentile":
            vals = []
            for s in self.sections.values():
                vl = cosine_sim(query_emb, s.E_local)
                vs = cosine_sim(query_emb, s.E_subtree)
                if vl > -1.0:
                    vals.append(vl)
                if vs > -1.0:
                    vals.append(vs)
            if vals:
                self._qmin, self._qmax = min(vals), max(vals)

        lvl1_ranked = self.rank_l1_sections(query_emb)
        roots = lvl1_ranked[: self.cfg.top_r]

        seeds: Set[str] = set()
        for root in roots:
            self.drill_section(root, query_emb, seeds)

        return list(seeds)
