from typing import Dict, Set, List, Optional
import collections

from ..data.models import Edge


class GraphExpander:
    """
    Bounded BFS expansion over the ontology/graph structure starting from seed nodes.

    The expander:
    - performs BFS from seed_ids
    - respects maximum depth and maximum number of visited nodes
    - optionally filters edges by allowed relation types
    """

    def __init__(
        self,
        graph_adj: Dict[str, List[Edge]],
        max_depth: int = 4,
        max_nodes: int = 500,
        allowed_relations: Optional[Set[str]] = None,
    ):
        self.graph_adj = graph_adj
        self.max_depth = max_depth
        self.max_nodes = max_nodes

        # If None or empty, treat it as "no filtering" (allow all relations).
        self.allowed_relations = set(allowed_relations) if allowed_relations else None

    # -------------------------------------------------------------
    # Main method
    # -------------------------------------------------------------
    def expand(self, seed_ids: List[str]):
        """
        Run BFS expansion from the provided seed node IDs.

        Returns:
            all_nodes: Set[str]
                All visited node IDs (including seeds).
            all_edges: List[Edge]
                All traversed edges included during expansion.
            dist_to_seed: Dict[str, int]
                Shortest BFS distance from any seed node to each visited node.
        """

        all_nodes: Set[str] = set()
        all_edges: List[Edge] = []
        dist_to_seed: Dict[str, int] = {}

        q = collections.deque()

        # Initialize BFS queue with seeds
        for sid in seed_ids:
            all_nodes.add(sid)
            dist_to_seed[sid] = 0
            q.append((sid, 0))

        # BFS loop
        while q and len(all_nodes) < self.max_nodes:
            node, depth = q.popleft()

            if depth >= self.max_depth:
                continue

            for e in self.graph_adj.get(node, []):
                # Relation filtering (if enabled)
                if self.allowed_relations is not None and e.relation_type not in self.allowed_relations:
                    continue

                tgt = e.to_id

                # Always keep the edge if we traverse it.
                # If you want to deduplicate edges, use a set of (from,to,type),
                # but for retrieval this is usually not critical.
                all_edges.append(e)

                # Add newly discovered node
                if tgt not in all_nodes:
                    all_nodes.add(tgt)
                    dist_to_seed[tgt] = depth + 1

                    if len(all_nodes) >= self.max_nodes:
                        break

                    q.append((tgt, depth + 1))

        return all_nodes, all_edges, dist_to_seed
