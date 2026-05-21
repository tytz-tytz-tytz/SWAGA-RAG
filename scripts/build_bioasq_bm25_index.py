from __future__ import annotations

import argparse
import time
from pathlib import Path

from bm25_rag.data.loaders import load_id_text_pairs
from bm25_rag.index.builder import build_bm25_index, save_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build BM25 index for the BioASQ PMC chunk corpus."
    )
    parser.add_argument(
        "--nodes-path",
        type=Path,
        default=Path("data/processed/bioasq_pmc_nodes.cleaned.json"),
        help="Input nodes json path.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("artifacts/indexes/bioasq_bm25_index.pkl"),
        help="Output BM25 index pickle path.",
    )
    parser.add_argument("--k1", type=float, default=1.5, help="BM25 k1 parameter.")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter.")
    parser.add_argument(
        "--node-types",
        nargs="+",
        default=["Chunk"],
        help="Node types to include in BM25 corpus (default: Chunk).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10000,
        help="Progress print interval during tokenization.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="CLI parity only: BM25 is CPU-only.",
    )
    parser.add_argument("--label", type=str, default="bioasq_pmc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()

    if args.device == "cuda":
        print("[BM25] device=cuda requested, but BM25 is CPU-only. Using CPU.")
    print(f"[BM25] label={args.label}")
    print(f"[BM25] nodes={args.nodes_path}")
    print(f"[BM25] out={args.out_path}")
    print(f"[BM25] node_types={args.node_types}")

    loaded_at = time.perf_counter()
    docs = load_id_text_pairs(args.nodes_path, allowed_types=args.node_types)
    print(f"[BM25] loaded docs={len(docs)} in {time.perf_counter() - loaded_at:.1f}s")

    built_at = time.perf_counter()
    index = build_bm25_index(
        docs,
        k1=args.k1,
        b=args.b,
        verbose=True,
        log_every=max(args.log_every, 1),
    )
    print(f"[BM25] built index in {time.perf_counter() - built_at:.1f}s")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    saved_at = time.perf_counter()
    save_index(index, args.out_path)
    print(f"[BM25] saved index in {time.perf_counter() - saved_at:.1f}s")

    print(f"[BM25] index saved to: {args.out_path}")
    print(f"[BM25] docs={len(index.ids)} | avgdl={index.avgdl:.2f} | vocab={len(index.idf)}")
    print(f"[BM25] total elapsed: {(time.perf_counter() - started_at):.1f}s")


if __name__ == "__main__":
    main()
