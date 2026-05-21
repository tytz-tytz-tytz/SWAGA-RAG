from __future__ import annotations

import argparse
import time
from pathlib import Path

from bm25_rag.data.loaders import load_id_text_pairs
from bm25_rag.index.builder import build_bm25_index, save_index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build BM25 index from graph nodes.")
    p.add_argument(
        "--nodes-path",
        type=Path,
        default=Path("data/processed/graphrag_nodes.cleaned.json"),
        help="Input nodes json path.",
    )
    p.add_argument(
        "--out-path",
        type=Path,
        default=Path("artifacts/indexes/bm25_index.pkl"),
        help="Output BM25 index pickle path.",
    )
    p.add_argument("--k1", type=float, default=1.5, help="BM25 k1 parameter.")
    p.add_argument("--b", type=float, default=0.75, help="BM25 b parameter.")
    p.add_argument(
        "--node-types",
        nargs="+",
        default=["Chunk"],
        help="Node types to include in BM25 corpus (default: Chunk).",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=10000,
        help="Progress print interval during tokenization.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help=(
            "For CLI parity with other pipelines. BM25 is CPU-only; "
            "if cuda is passed, script will log fallback to CPU."
        ),
    )
    p.add_argument(
        "--label",
        type=str,
        default="default",
        help="Label for logs (e.g. qasper).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    if args.device == "cuda":
        print("[BM25] device=cuda requested, but BM25 is CPU-only. Using CPU.")
    print(f"[BM25] label={args.label}")
    print(f"[BM25] nodes={args.nodes_path}")
    print(f"[BM25] out={args.out_path}")
    print(f"[BM25] node_types={args.node_types}")

    t_load = time.perf_counter()
    docs = load_id_text_pairs(args.nodes_path, allowed_types=args.node_types)
    print(f"[BM25] loaded docs={len(docs)} in {time.perf_counter()-t_load:.1f}s")

    t_build = time.perf_counter()
    index = build_bm25_index(
        docs,
        k1=args.k1,
        b=args.b,
        verbose=True,
        log_every=max(args.log_every, 1),
    )
    print(f"[BM25] built index in {time.perf_counter()-t_build:.1f}s")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    t_save = time.perf_counter()
    save_index(index, args.out_path)
    print(f"[BM25] saved index in {time.perf_counter()-t_save:.1f}s")

    print(f"[BM25] index saved to: {args.out_path}")
    print(f"[BM25] docs={len(index.ids)} | avgdl={index.avgdl:.2f} | vocab={len(index.idf)}")
    print(f"[BM25] total elapsed: {(time.perf_counter()-t0):.1f}s")


if __name__ == "__main__":
    main()
