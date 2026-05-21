from __future__ import annotations

import argparse
import os
from pathlib import Path

from _common import iter_queries, RetrievalRunner

import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

from classic_rag.index.builder import load_index
from classic_rag.rag.heuristic import HeuristicRAGConfig, select_heuristic_from_candidates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run dense heuristic retrieval for query jsonl.")
    p.add_argument("--index-path", type=Path, default=Path("artifacts/indexes/classic_rag_index.pkl"))
    p.add_argument("--queries-path", type=Path, default=Path("data/eval/queries.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/classic_rag_heuristic_results"))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--candidate-multiplier", type=int, default=6)
    p.add_argument("--min-chars", type=int, default=80)
    p.add_argument("--fallback-min-chars", type=int, default=40)
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
        print("[Dense-Heur] CUDA requested but unavailable. Fallback to CPU.")
        return "cpu"
    except Exception:
        print("[Dense-Heur] torch check failed. Fallback to CPU.")
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



def build_cfg(args: argparse.Namespace) -> HeuristicRAGConfig:
    return HeuristicRAGConfig(
        top_k=args.top_k,
        candidate_multiplier=args.candidate_multiplier,
        min_chars=args.min_chars,
        fallback_min_chars=args.fallback_min_chars,
    )


def dense_candidates(index, qvec: np.ndarray, top_k: int) -> list[tuple[str, str, float]]:
    scores = (index.embeddings @ qvec.T).reshape(-1)
    top_idx = np.argsort(-scores)[:top_k]
    return [(index.ids[i], index.texts[i], float(scores[i])) for i in top_idx]


def main() -> None:
    args = parse_args()

    setup_cache_env(args.cache_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    print(f"[Dense-Heur] label={args.label}")
    print(f"[Dense-Heur] index={args.index_path}")
    print(f"[Dense-Heur] queries={args.queries_path}")
    print(f"[Dense-Heur] out={args.out_dir}")
    print(f"[Dense-Heur] device={device}")

    index = load_index(args.index_path)
    model_path = resolve_model_path(index.model_name, args.cache_dir)
    if model_path != index.model_name:
        print(f"[Dense-Heur] model resolved to local dir: {model_path}")
    model = SentenceTransformer(model_path, device=device, cache_folder=str(args.cache_dir))

    cfg = build_cfg(args)
    queries = list(iter_queries(args.queries_path))
    cand_k = max(cfg.top_k * cfg.candidate_multiplier, cfg.top_k)
    print(f"[Dense-Heur] loaded queries={len(queries)}")

    def retrieve_fn(qtext: str):
        qvec = model.encode([qtext], normalize_embeddings=True, show_progress_bar=False)
        qvec = np.asarray(qvec, dtype=np.float32)
        candidates = dense_candidates(index, qvec, cand_k)
        return select_heuristic_from_candidates(candidates, cfg)

    RetrievalRunner(
        tag="Dense-Heur",
        out_dir=args.out_dir,
        log_every=args.log_every,
        speed_decimals=2,
        done_message="done. Results:",
    ).run(queries, retrieve_fn)


if __name__ == "__main__":
    main()
