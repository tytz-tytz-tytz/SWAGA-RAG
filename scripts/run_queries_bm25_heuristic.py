from __future__ import annotations

import argparse
from pathlib import Path

from _common import iter_queries, RetrievalRunner

from bm25_rag.index.builder import load_index
from bm25_rag.rag.heuristic import BM25HeuristicConfig, retrieve_heuristic_with_scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BM25 heuristic retrieval for query jsonl.")
    p.add_argument("--index-path", type=Path, default=Path("artifacts/indexes/bm25_index.pkl"))
    p.add_argument("--queries-path", type=Path, default=Path("data/eval/queries.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/bm25_rag_heuristic_results"))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--candidate-multiplier", type=int, default=12)
    p.add_argument("--min-chars", type=int, default=80)
    p.add_argument("--fallback-min-chars", type=int, default=40)
    p.add_argument("--rare-terms-top-n", type=int, default=2)
    p.add_argument("--relax-threshold", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--label", type=str, default="default")
    return p.parse_args()



def build_cfg(args: argparse.Namespace) -> BM25HeuristicConfig:
    return BM25HeuristicConfig(
        top_k=args.top_k,
        candidate_multiplier=args.candidate_multiplier,
        min_chars=args.min_chars,
        fallback_min_chars=args.fallback_min_chars,
        rare_terms_top_n=args.rare_terms_top_n,
        relax_threshold=args.relax_threshold,
    )


def main() -> None:
    args = parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = load_index(args.index_path)
    cfg = build_cfg(args)
    queries = list(iter_queries(args.queries_path))

    print(f"[BM25-Heur] label={args.label}")
    print(f"[BM25-Heur] index={args.index_path}")
    print(f"[BM25-Heur] queries={args.queries_path}")
    print(f"[BM25-Heur] out={args.out_dir}")
    print(f"[BM25-Heur] loaded queries={len(queries)}")

    RetrievalRunner(tag="BM25-Heur", out_dir=args.out_dir, log_every=args.log_every).run(
        queries,
        lambda query: retrieve_heuristic_with_scores(index, query, cfg),
    )


if __name__ == "__main__":
    main()
