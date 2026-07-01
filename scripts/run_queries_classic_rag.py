from __future__ import annotations

import argparse
import os
from pathlib import Path

from _common import iter_queries, RetrievalRunner

import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

from classic_rag.index.builder import load_index
from rag_common.encoder_spec import encoder_spec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run dense retrieval for query jsonl.")
    p.add_argument("--index-path", type=Path, default=Path("artifacts/indexes/classic_rag_index.pkl"))
    p.add_argument("--queries-path", type=Path, default=Path("data/eval/queries.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/classic_rag_results"))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache/hf"))
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--label", type=str, default="default")
    return p.parse_args()


def resolve_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        print("[Dense] CUDA requested but unavailable. Fallback to CPU.")
        return "cpu"
    except Exception:
        print("[Dense] torch check failed. Fallback to CPU.")
        return "cpu"


def setup_cache_env(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "hub").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def resolve_model_path(model_name: str, cache_dir: Path) -> str:
    if os.name != "nt":
        return model_name
    local_model_dir = cache_dir / "models" / model_name.replace("/", "__")
    local_model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_model_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return str(local_model_dir)



def retrieve_from_query_vec(index, qvec: np.ndarray, top_k: int) -> list[str]:
    scores = (index.embeddings @ qvec.T).reshape(-1)
    top_idx = np.argsort(-scores)[:top_k]
    return [index.texts[i] for i in top_idx]


def retrieve_items_from_query_vec(index, qvec: np.ndarray, top_k: int) -> list[dict]:
    scores = (index.embeddings @ qvec.T).reshape(-1)
    top_idx = np.argsort(-scores)[:top_k]
    return [
        {
            "chunk_id": index.ids[i],
            "text": index.texts[i],
            "score": float(scores[i]),
        }
        for i in top_idx
    ]


def main() -> None:
    args = parse_args()

    setup_cache_env(args.cache_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    print(f"[Dense] label={args.label}")
    print(f"[Dense] index={args.index_path}")
    print(f"[Dense] queries={args.queries_path}")
    print(f"[Dense] out={args.out_dir}")
    print(f"[Dense] device={device}")

    index = load_index(args.index_path)
    model_path = resolve_model_path(index.model_name, args.cache_dir)
    if model_path != index.model_name:
        print(f"[Dense] model resolved to local dir: {model_path}")
    model = SentenceTransformer(model_path, device=device, cache_folder=str(args.cache_dir))

    # Per-model query prefix (e5/bge); empty for mpnet/MiniLM -> unchanged.
    query_prefix = encoder_spec(index.model_name).query_prefix
    if query_prefix:
        print(f"[Dense] query_prefix={query_prefix!r}")

    queries = list(iter_queries(args.queries_path))
    print(f"[Dense] loaded queries={len(queries)}")

    def retrieve_fn(qtext: str):
        qvec = model.encode([query_prefix + qtext], normalize_embeddings=True, show_progress_bar=False)
        qvec = np.asarray(qvec, dtype=np.float32)
        items = retrieve_items_from_query_vec(index, qvec, args.top_k)
        return [(it["chunk_id"], it["text"], it["score"]) for it in items]

    RetrievalRunner(
        tag="Dense",
        out_dir=args.out_dir,
        log_every=args.log_every,
        speed_decimals=2,
        done_message="done. Results:",
    ).run(queries, retrieve_fn)


if __name__ == "__main__":
    main()
