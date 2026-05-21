"""Unit + integration tests for the swaga_rag retrieval core.

These tests build a tiny synthetic ontology with hand-crafted unit-vector
embeddings, so they exercise the real drill/expand/score/pipeline logic
without loading any embedding model or touching the network or disk.
"""

import numpy as np
import pytest

from swaga_rag.data.models import Section, TextNode, Edge
from swaga_rag.rag.drill import cosine_sim, DrillConfig, DrillSelector
from swaga_rag.rag.expand import GraphExpander
from swaga_rag.rag.score import ScoreConfig, NodeScorer
from swaga_rag.rag.pipeline import OntologyRAGPipeline, SWAGARAGPipeline


# Orthonormal axes in R^3 — cosine_sim is 1.0 with itself and 0.0 across axes.
E0 = np.array([1.0, 0.0, 0.0])
E1 = np.array([0.0, 1.0, 0.0])
E2 = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------
# cosine_sim
# --------------------------------------------------------------------------
def test_cosine_sim_identical_orthogonal_and_missing():
    assert cosine_sim(E0, E0) == pytest.approx(1.0)
    assert cosine_sim(E0, E1) == pytest.approx(0.0)
    assert cosine_sim(E0, None) == -1.0
    assert cosine_sim(None, E0) == -1.0
    assert cosine_sim(E0, np.zeros(3)) == -1.0
    # Scale invariance.
    assert cosine_sim(E0, 5.0 * E0) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# DrillSelector
# --------------------------------------------------------------------------
def test_drill_selects_relevant_leaf_and_skips_irrelevant():
    sections = {
        "A": Section(id="A", level=1, children_ids=[], local_text="a", E_local=E0, E_subtree=E0),
        "B": Section(id="B", level=1, children_ids=[], local_text="b", E_local=E1, E_subtree=E1),
    }
    selector = DrillSelector(sections, DrillConfig())
    seeds = selector.select_seeds(query_emb=E0)
    assert seeds == ["A"]


def test_drill_descends_into_children_when_parent_has_no_local_text():
    sections = {
        "P": Section(id="P", level=1, children_ids=["C"], local_text="", E_local=E1, E_subtree=E0),
        "C": Section(id="C", level=2, children_ids=[], local_text="c", E_local=E0, E_subtree=E0),
    }
    selector = DrillSelector(sections, DrillConfig())
    seeds = selector.select_seeds(query_emb=E0)
    assert seeds == ["C"]


def test_drill_rank_l1_sections_orders_by_subtree_similarity():
    sections = {
        "X": Section(id="X", level=1, E_subtree=E1),
        "Y": Section(id="Y", level=1, E_subtree=E0),
        "Z": Section(id="Z", level=2, E_subtree=E0),  # not level 1 -> excluded
    }
    ranked = DrillSelector(sections, DrillConfig()).rank_l1_sections(E0)
    assert [s.id for s in ranked] == ["Y", "X"]


# --------------------------------------------------------------------------
# GraphExpander
# --------------------------------------------------------------------------
def _chain_graph():
    # s -> a -> b -> c, plus a typed side edge s -> x
    return {
        "s": [Edge("s", "a", "contains"), Edge("s", "x", "see_also")],
        "a": [Edge("a", "b", "contains")],
        "b": [Edge("b", "c", "contains")],
    }


def test_expand_respects_max_depth():
    nodes, edges, dist = GraphExpander(_chain_graph(), max_depth=2).expand(["s"])
    assert dist["s"] == 0 and dist["a"] == 1 and dist["b"] == 2
    assert "c" not in nodes  # c is at depth 3, beyond max_depth=2


def test_expand_respects_max_nodes():
    nodes, _edges, _dist = GraphExpander(_chain_graph(), max_depth=10, max_nodes=2).expand(["s"])
    assert len(nodes) == 2


def test_expand_relation_filter():
    nodes, _edges, _dist = GraphExpander(
        _chain_graph(), max_depth=10, allowed_relations={"contains"}
    ).expand(["s"])
    assert "x" not in nodes  # reached only via see_also, which is filtered out
    assert {"s", "a", "b", "c"} <= nodes


# --------------------------------------------------------------------------
# NodeScorer
# --------------------------------------------------------------------------
def _scorer(mode="full", **cfg):
    sections = {"S": Section(id="S", level=2)}
    text_nodes = {
        "n_chunk": TextNode(id="n_chunk", section_id="S", node_type="chunk", text="x", embedding=E0),
        "n_caption": TextNode(id="n_caption", section_id="S", node_type="caption", text="y", embedding=E1),
    }
    return NodeScorer(sections, text_nodes, ScoreConfig(mode=mode, **cfg))


def test_score_one_text_only_mode_is_pure_cosine():
    s = _scorer(mode="text_only")
    assert s.score_one("n_chunk", E0, {"n_chunk": 0}) == pytest.approx(1.0)
    assert s.score_one("n_caption", E0, {"n_caption": 0}) == pytest.approx(0.0)


def test_score_one_full_mode_matches_formula():
    s = _scorer()  # defaults: w_text=1, w_type=.3, w_level=.15, w_dist=.2
    # chunk: sim=1, type_bonus(chunk)=0.6, level_bonus(2)=0.2, dist=1
    expected = 1.0 * 1.0 + 0.3 * 0.6 + 0.15 * 0.2 - 0.2 * 1
    assert s.score_one("n_chunk", E0, {"n_chunk": 1}) == pytest.approx(expected)


def test_score_one_non_text_node_is_strongly_negative():
    s = _scorer()
    assert s.score_one("does_not_exist", E0, {}) == -999.0


def test_score_all_ranks_and_truncates_and_skips_non_text():
    s = _scorer()
    ranked = s.score_all(E0, {"n_chunk": 0, "n_caption": 0}, ["n_chunk", "n_caption", "ghost"], top_k=1)
    assert len(ranked) == 1
    assert ranked[0][0] == "n_chunk"  # higher sim than caption


# --------------------------------------------------------------------------
# Pipeline integration (with a stub embedding model)
# --------------------------------------------------------------------------
class _StubModel:
    """Stand-in for EmbeddingModel: returns a fixed query embedding."""

    def __init__(self, vec):
        self._vec = vec

    def encode(self, query):
        return self._vec


def _toy_index():
    sections = {
        "root": Section(id="root", level=1, children_ids=["secA"], local_text="", E_local=E1, E_subtree=E0),
        "secA": Section(id="secA", level=2, children_ids=[], local_text="Section A title\nbody", E_local=E0, E_subtree=E0),
    }
    text_nodes = {
        "secA_t0": TextNode(id="secA_t0", section_id="secA", node_type="chunk", text="first chunk", embedding=E0),
        "secA_t1": TextNode(id="secA_t1", section_id="secA", node_type="chunk", text="second chunk", embedding=E0),
    }
    graph_adj = {
        "secA": [Edge("secA", "secA_t0", "contains"), Edge("secA", "secA_t1", "contains")],
    }
    return sections, text_nodes, graph_adj


def test_run_query_returns_expected_structure_and_finds_section():
    sections, text_nodes, graph_adj = _toy_index()
    pipe = OntologyRAGPipeline(
        sections, text_nodes, graph_adj,
        embedding_model=_StubModel(E0),
        config={"expand": {"max_depth": 3}, "output": {"top_k_text": 10}},
    )
    out = pipe.run_query("anything")

    assert set(out) == {"query", "section_candidates", "text_nodes", "graph_context"}
    assert out["query"] == "anything"

    # Both chunks of secA should be retrieved and ranked.
    ranked_ids = {n["node_id"] for n in out["text_nodes"]}
    assert ranked_ids == {"secA_t0", "secA_t1"}

    # They aggregate into a single section candidate carrying the full text.
    assert len(out["section_candidates"]) == 1
    cand = out["section_candidates"][0]
    assert cand["section_id"] == "secA"
    assert cand["title"] == "Section A title"
    assert "first chunk" in cand["text"] and "second chunk" in cand["text"]
    assert cand["score"] == pytest.approx(max(n["score"] for n in out["text_nodes"]))


def test_run_query_includes_debug_payload_when_enabled():
    sections, text_nodes, graph_adj = _toy_index()
    pipe = OntologyRAGPipeline(
        sections, text_nodes, graph_adj,
        embedding_model=_StubModel(E0),
        config={"output": {"save_debug": True}},
    )
    out = pipe.run_query("q")
    assert "debug" in out
    assert out["debug"]["drill"]["seed_count"] >= 1


def test_swaga_alias_is_the_pipeline_class():
    assert SWAGARAGPipeline is OntologyRAGPipeline
