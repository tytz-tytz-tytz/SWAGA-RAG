"""Tests for the shared retrieval helpers in scripts/_common.py.

The key guarantee these lock in: the result JSON written by RetrievalRunner
is byte-for-byte identical to the inline format the run_queries_*/run_bioasq_*
scripts used before the refactor, so existing experiment outputs stay
comparable.
"""

import importlib.util
import json
import sys
from pathlib import Path

_COMMON_PATH = Path(__file__).resolve().parents[1] / "scripts" / "_common.py"
_spec = importlib.util.spec_from_file_location("_common_under_test", _COMMON_PATH)
_common = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _common  # dataclass needs the module registered
_spec.loader.exec_module(_common)


def _legacy_json(qid, query, hits):
    """Reproduces the exact dict + serialization the scripts used inline."""
    out = {
        "id": qid,
        "query": query,
        "output_items": [
            {"chunk_id": cid, "text": text, "score": score}
            for cid, text, score in hits
        ],
        "output_ids": [cid for cid, _text, _score in hits],
    }
    return json.dumps(out, ensure_ascii=False, indent=2)


HITS = [("c1", "first", 0.9), ("c2", "вторая строка", 0.5)]


def test_build_result_matches_legacy_dict():
    result = _common.build_result("q1", "hello", HITS)
    assert result == {
        "id": "q1",
        "query": "hello",
        "output_items": [
            {"chunk_id": "c1", "text": "first", "score": 0.9},
            {"chunk_id": "c2", "text": "вторая строка", "score": 0.5},
        ],
        "output_ids": ["c1", "c2"],
    }


def test_build_result_consumes_generators_safely():
    # A generator can only be iterated once; build_result must materialize it
    # so output_items and output_ids stay consistent.
    result = _common.build_result("q", "x", (h for h in HITS))
    assert result["output_ids"] == ["c1", "c2"]
    assert len(result["output_items"]) == 2


def test_runner_writes_files_byte_identical_to_legacy(tmp_path):
    queries = [
        {"id": "q1", "query": "alpha"},
        {"id": "q2", "query": "бета"},
    ]
    hits_by_query = {"alpha": HITS, "бета": [("c9", "x", 0.1)]}

    runner = _common.RetrievalRunner(tag="BM25", out_dir=tmp_path, log_every=1)
    runner.run(queries, lambda query: hits_by_query[query])

    for q in queries:
        written = (tmp_path / f"{q['id']}.json").read_text(encoding="utf-8")
        expected = _legacy_json(q["id"], q["query"], hits_by_query[q["query"]])
        assert written == expected


def test_runner_returns_elapsed_and_creates_one_file_per_query(tmp_path):
    queries = [{"id": f"q{i}", "query": f"text{i}"} for i in range(3)]
    runner = _common.RetrievalRunner(tag="Dense", out_dir=tmp_path, speed_decimals=2)
    elapsed = runner.run(queries, lambda query: [("c", query, 1.0)])
    assert elapsed >= 0.0
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["q0.json", "q1.json", "q2.json"]
