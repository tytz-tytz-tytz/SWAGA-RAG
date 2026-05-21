from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from swaga_rag.data.loaders import load_ontology
from swaga_rag.index.store import save_index
from swaga_rag.ontology.hierarchy import build_hierarchy


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build swaga-rag index for the BioASQ PMC corpus with batching, "
            "checkpoint/resume, and local HF cache support."
        )
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=Path("data/processed/bioasq_pmc_nodes.cleaned.json"),
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=Path("data/processed/bioasq_pmc_edges.cleaned.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/indexes/bioasq_pmc"),
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save checkpoint every N batches.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if present.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=Path("artifacts/cache/hf"),
    )
    parser.add_argument("--tmp-dir", type=Path, default=Path(".tmp"))
    parser.add_argument(
        "--strict-cuda",
        action="store_true",
        help="Fail if CUDA is unavailable.",
    )
    parser.add_argument(
        "--max-sections",
        type=int,
        default=0,
        help="For smoke-run: embed at most this many sections.",
    )
    parser.add_argument(
        "--max-text-nodes",
        type=int,
        default=0,
        help="For smoke-run: embed at most this many text nodes.",
    )
    return parser.parse_args()


def _save_pickle(path: Path, obj: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def set_runtime_dirs(hf_cache_dir: Path, tmp_dir: Path) -> None:
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    (hf_cache_dir / "transformers").mkdir(parents=True, exist_ok=True)
    (hf_cache_dir / "datasets").mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache_dir / "transformers")
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache_dir / "datasets")
    os.environ["TORCH_HOME"] = str(hf_cache_dir / "torch")
    os.environ["TMP"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def check_device(device: str, strict_cuda: bool) -> str:
    if device == "cpu":
        return "cpu"
    try:
        import torch
    except Exception as exc:
        if strict_cuda:
            raise RuntimeError(f"Cannot import torch for CUDA check: {exc}") from exc
        print("[WARN] torch import failed, fallback to CPU.")
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if strict_cuda:
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    print("[WARN] CUDA unavailable, fallback to CPU.")
    return "cpu"


def resolve_model_path(model_name: str, hf_cache_dir: Path) -> str:
    if os.name != "nt":
        return model_name

    local_model_dir = hf_cache_dir / "models" / model_name.replace("/", "__")
    if local_model_dir.exists() and any(local_model_dir.iterdir()):
        print(f"[HF] Using cached model at {local_model_dir}")
        return str(local_model_dir)

    local_model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[HF] snapshot_download -> {local_model_dir}")
    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_model_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return str(local_model_dir)


def encode_batch(model: SentenceTransformer, texts: list[str], batch_size: int):
    safe = [text if (isinstance(text, str) and text.strip()) else " " for text in texts]
    return model.encode(
        safe,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=batch_size,
    )


def checkpoint_save(
    ckpt_dir: Path,
    sections: dict[str, Any],
    text_nodes: dict[str, Any],
    graph_adj: dict[str, Any],
    progress: dict[str, Any],
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _save_pickle(ckpt_dir / "sections.pkl", sections)
    _save_pickle(ckpt_dir / "text_nodes.pkl", text_nodes)
    _save_pickle(ckpt_dir / "graph_adj.pkl", graph_adj)
    (ckpt_dir / "progress.json").write_text(
        json.dumps(progress, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[CKPT] phase={progress['phase']} index={progress['index']}")


def checkpoint_load(ckpt_dir: Path):
    sections = _load_pickle(ckpt_dir / "sections.pkl")
    text_nodes = _load_pickle(ckpt_dir / "text_nodes.pkl")
    graph_adj = _load_pickle(ckpt_dir / "graph_adj.pkl")
    progress = json.loads((ckpt_dir / "progress.json").read_text(encoding="utf-8"))
    return sections, text_nodes, graph_adj, progress


def run_phase_sections(
    model: SentenceTransformer,
    sections: dict[str, Any],
    section_ids: list[str],
    batch_size: int,
    checkpoint_every: int,
    ckpt_dir: Path,
    graph_adj: dict[str, Any],
    text_nodes: dict[str, Any],
    phase: str,
    start_index: int = 0,
) -> None:
    total = len(section_ids)
    batch_counter = 0
    started_at = time.perf_counter()

    for index in range(start_index, total, batch_size):
        ids = section_ids[index : index + batch_size]
        texts = [sections[sid].local_text for sid in ids] if phase == "sections_local" else [
            sections[sid].subtree_text for sid in ids
        ]
        vectors = encode_batch(model, texts, batch_size=batch_size)
        for sid, vector in zip(ids, vectors):
            if phase == "sections_local":
                sections[sid].E_local = vector
            else:
                sections[sid].E_subtree = vector

        batch_counter += 1
        done = min(index + batch_size, total)
        if done % (batch_size * 5) == 0 or done == total:
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            speed = done / elapsed
            eta_sec = max(total - done, 0) / max(speed, 1e-9)
            pct = (done / total) * 100 if total else 100.0
            print(
                f"[{phase}] {done}/{total} ({pct:.1f}%) | "
                f"{speed:.1f} items/s | ETA {eta_sec/60:.1f} min"
            )

        if batch_counter % checkpoint_every == 0:
            checkpoint_save(
                ckpt_dir,
                sections,
                text_nodes,
                graph_adj,
                {"phase": phase, "index": done},
            )

    elapsed = max(time.perf_counter() - started_at, 1e-9)
    print(f"[{phase}] done in {elapsed/60:.1f} min")


def run_phase_text_nodes(
    model: SentenceTransformer,
    text_nodes: dict[str, Any],
    node_ids: list[str],
    batch_size: int,
    checkpoint_every: int,
    ckpt_dir: Path,
    graph_adj: dict[str, Any],
    sections: dict[str, Any],
    start_index: int = 0,
) -> None:
    total = len(node_ids)
    batch_counter = 0
    started_at = time.perf_counter()

    for index in range(start_index, total, batch_size):
        ids = node_ids[index : index + batch_size]
        texts = [text_nodes[nid].text for nid in ids]
        vectors = encode_batch(model, texts, batch_size=batch_size)
        for nid, vector in zip(ids, vectors):
            text_nodes[nid].embedding = vector

        batch_counter += 1
        done = min(index + batch_size, total)
        if done % (batch_size * 10) == 0 or done == total:
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            speed = done / elapsed
            eta_sec = max(total - done, 0) / max(speed, 1e-9)
            pct = (done / total) * 100 if total else 100.0
            print(
                f"[text_nodes] {done}/{total} ({pct:.1f}%) | "
                f"{speed:.1f} items/s | ETA {eta_sec/60:.1f} min"
            )

        if batch_counter % checkpoint_every == 0:
            checkpoint_save(
                ckpt_dir,
                sections,
                text_nodes,
                graph_adj,
                {"phase": "text_nodes", "index": done},
            )

    elapsed = max(time.perf_counter() - started_at, 1e-9)
    print(f"[text_nodes] done in {elapsed/60:.1f} min")


def main() -> None:
    overall_started_at = time.perf_counter()
    args = parse_args()
    set_runtime_dirs(args.hf_cache_dir, args.tmp_dir)

    device = check_device(args.device, args.strict_cuda)
    print(f"[ENV] device={device}")
    print(f"[ENV] HF_HOME={os.environ.get('HF_HOME')}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "_checkpoint"

    if args.resume and (ckpt_dir / "progress.json").exists():
        print("[RESUME] Loading checkpoint...")
        sections, text_nodes, graph_adj, progress = checkpoint_load(ckpt_dir)
        phase = progress.get("phase", "sections_local")
        start_index = int(progress.get("index", 0))
        print(f"[RESUME] phase={phase} index={start_index}")
    else:
        print("[LOAD] Reading ontology graph...")
        sections, text_nodes, graph_adj = load_ontology(str(args.nodes), str(args.edges))
        print(f"[LOAD] sections={len(sections)} text_nodes={len(text_nodes)}")
        print("[HIER] Building hierarchy texts...")
        build_hierarchy(sections, text_nodes)
        phase = "sections_local"
        start_index = 0

    section_ids = list(sections.keys())
    node_ids = list(text_nodes.keys())

    if args.max_sections > 0:
        section_ids = section_ids[: args.max_sections]
        print(f"[SMOKE] limiting sections to {len(section_ids)}")
    if args.max_text_nodes > 0:
        node_ids = node_ids[: args.max_text_nodes]
        print(f"[SMOKE] limiting text_nodes to {len(node_ids)}")

    model_path = resolve_model_path(MODEL_NAME, args.hf_cache_dir)
    print(f"[MODEL] Loading {model_path} on {device}")
    model = SentenceTransformer(model_path, device=device, cache_folder=str(args.hf_cache_dir))

    if phase == "sections_local":
        run_phase_sections(
            model=model,
            sections=sections,
            section_ids=section_ids,
            batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
            ckpt_dir=ckpt_dir,
            graph_adj=graph_adj,
            text_nodes=text_nodes,
            phase="sections_local",
            start_index=start_index,
        )
        phase = "sections_subtree"
        start_index = 0

    if phase == "sections_subtree":
        run_phase_sections(
            model=model,
            sections=sections,
            section_ids=section_ids,
            batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
            ckpt_dir=ckpt_dir,
            graph_adj=graph_adj,
            text_nodes=text_nodes,
            phase="sections_subtree",
            start_index=start_index,
        )
        phase = "text_nodes"
        start_index = 0

    if phase == "text_nodes":
        run_phase_text_nodes(
            model=model,
            text_nodes=text_nodes,
            node_ids=node_ids,
            batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
            ckpt_dir=ckpt_dir,
            graph_adj=graph_adj,
            sections=sections,
            start_index=start_index,
        )

    print("[SAVE] Saving final index...")
    save_index(str(out_dir), sections, text_nodes, graph_adj)
    checkpoint_save(
        ckpt_dir,
        sections,
        text_nodes,
        graph_adj,
        {"phase": "done", "index": 0},
    )

    elapsed = max(time.perf_counter() - overall_started_at, 1e-9)
    print(f"[DONE] Index saved to {out_dir}")
    print(f"[DONE] Total elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()

