from __future__ import annotations

import argparse
from pathlib import Path

from _common import iter_queries, RetrievalRunner

from bm25_rag.index.builder import load_index
from bm25_rag.rag.retrieve import retrieve_with_scores



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BM25 retrieval for BioASQ query jsonl.")
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("artifacts/indexes/bioasq_bm25_index.pkl"),
    )
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=Path("data/eval/bioasq_eval_queries.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/bm25_rag_results/bioasq_eval"),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="CLI parity only: BM25 runs on CPU.",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--label", type=str, default="bioasq_pmc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "cuda":
        print("[BM25] device=cuda requested, but BM25 is CPU-only. Using CPU.")
    print(f"[BM25] label={args.label}")
    print(f"[BM25] index={args.index_path}")
    print(f"[BM25] queries={args.queries_path}")
    print(f"[BM25] out={args.out_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = load_index(args.index_path)
    queries = list(iter_queries(args.queries_path))
    print(f"[BM25] loaded queries={len(queries)}")

    RetrievalRunner(tag="BM25", out_dir=args.out_dir, log_every=args.log_every).run(
        queries,
        lambda query: retrieve_with_scores(index, query, top_k=args.top_k),
    )


if __name__ == "__main__":
    main()
