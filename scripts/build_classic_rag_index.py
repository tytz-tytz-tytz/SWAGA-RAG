from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from classic_rag.index.builder import build_index, save_index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build dense (classic RAG) index from graph chunks.")
    p.add_argument("--nodes", type=Path, default=Path("data/processed/graphrag_nodes.cleaned.json"))
    p.add_argument("--out", type=Path, default=Path("artifacts/indexes/classic_rag_index.pkl"))
    p.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache/hf"))
    p.add_argument("--label", type=str, default="default")
    return p.parse_args()


def resolve_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        print("[DenseIndex] CUDA requested but unavailable. Fallback to CPU.")
        return "cpu"
    except Exception:
        print("[DenseIndex] torch check failed. Fallback to CPU.")
        return "cpu"


def setup_cache_env(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "hub").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    setup_cache_env(args.cache_dir)
    device = resolve_device(args.device)

    print(f"[DenseIndex] label={args.label}")
    print(f"[DenseIndex] nodes={args.nodes}")
    print(f"[DenseIndex] out={args.out}")
    print(f"[DenseIndex] cache={args.cache_dir}")

    index = build_index(
        nodes_path=args.nodes,
        model_name=args.model_name,
        device=device,
        batch_size=args.batch_size,
        show_progress_bar=True,
        cache_folder=str(args.cache_dir),
        verbose=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_index(index, args.out)

    print(f"[DenseIndex] saved: {args.out}")
    print(f"[DenseIndex] docs={len(index.ids)} dim={index.embeddings.shape[1]}")
    print(f"[DenseIndex] total elapsed={(time.perf_counter()-t0):.1f}s")


if __name__ == "__main__":
    main()
