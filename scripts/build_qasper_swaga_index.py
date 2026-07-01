from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

from swaga_rag.data.loaders import load_ontology
from swaga_rag.ontology.hierarchy import build_hierarchy
from swaga_rag.index.store import save_index
from rag_common.encoder_spec import DEFAULT_EMBED_MODEL, encoder_spec, model_slug


MODEL_NAME = DEFAULT_EMBED_MODEL  # backward-compatible default (mpnet)


def default_out_dir(model_name: str) -> Path:
    """mpnet -> legacy path; other encoders -> per-model directory."""
    if model_name == DEFAULT_EMBED_MODEL:
        return Path("artifacts/indexes/qasper")
    return Path(f"artifacts/indexes/qasper__{model_slug(model_name)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build swaga-rag index for QASPER with CUDA batching, cache folder, "
            "and checkpoint/resume."
        )
    )
    p.add_argument("--nodes", type=Path, default=Path("data/processed/qasper_nodes.cleaned.json"))
    p.add_argument("--edges", type=Path, default=Path("data/processed/qasper_edges.cleaned.json"))
    p.add_argument("--model-name", type=str, default=MODEL_NAME,
                   help="Encoder HF id (default: mpnet). e5/bge get query/passage prefixes.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Index dir. Default: artifacts/indexes/qasper for mpnet, "
                        "artifacts/indexes/qasper__<slug> otherwise.")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--subtree-batch-size", type=int, default=0,
                   help="Batch size for the long sections_subtree phase (<=512 tok). "
                        "0 => use --batch-size. Lower it to avoid OOM at large --batch-size.")
    p.add_argument("--checkpoint-every", type=int, default=50, help="Save checkpoint every N batches.")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint if present.")
    p.add_argument("--hf-cache-dir", type=Path, default=Path("artifacts/cache/hf"))
    p.add_argument("--tmp-dir", type=Path, default=Path(".tmp"))
    p.add_argument("--strict-cuda", action="store_true", help="Fail if CUDA is unavailable.")
    p.add_argument("--max-sections", type=int, default=0, help="For smoke-run: embed at most this many sections.")
    p.add_argument("--max-text-nodes", type=int, default=0, help="For smoke-run: embed at most this many text nodes.")
    return p.parse_args()


def _save_pickle(path: Path, obj: Any) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


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
    except Exception as e:
        if strict_cuda:
            raise RuntimeError(f"Cannot import torch for CUDA check: {e}") from e
        print("[WARN] torch import failed, fallback to CPU.")
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if strict_cuda:
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    print("[WARN] CUDA unavailable, fallback to CPU.")
    return "cpu"


def resolve_model_path(model_name: str, hf_cache_dir: Path) -> str:
    """
    On Windows, avoid symlink-based HF cache layout that can require elevated rights.
    We pre-download into a plain local directory and load from there.
    """
    if os.name != "nt":
        return model_name

    local_model_dir = hf_cache_dir / "models" / model_name.replace("/", "__")
    local_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"[HF] snapshot_download -> {local_model_dir}")
    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_model_dir),
        local_dir_use_symlinks=False,  # deprecated but harmless for compatibility
        resume_download=True,
    )
    return str(local_model_dir)


def encode_batch(model: SentenceTransformer, texts: list[str], batch_size: int, passage_prefix: str = ""):
    safe = [passage_prefix + (t if (isinstance(t, str) and t.strip()) else " ") for t in texts]
    return model.encode(
        safe,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=batch_size,
    )


def log_subtree_truncation(model: SentenceTransformer, sections: dict, limit: int | None = None, sample: int = 2500) -> None:
    """Log the fraction of sections whose subtree_text exceeds the encoder's
    max token length (drill-down quality proxy for the new encoder).

    Estimated on a fixed-seed random sample of up to `sample` sections to keep
    the tokenization pass cheap (full-corpus tokenization is slow)."""
    import random
    limit = int(limit or getattr(model, "max_seq_length", 0) or 512)
    tok = model.tokenizer
    texts = [t for s in sections.values() if (t := (getattr(s, "subtree_text", "") or "")).strip()]
    population = len(texts)
    if sample and population > sample:
        texts = random.Random(0).sample(texts, sample)
    total = len(texts)
    over = []
    for txt in texts:
        n = len(tok(txt, add_special_tokens=True, truncation=False)["input_ids"])
        if n > limit:
            over.append(n)
    frac = (len(over) / total) if total else 0.0
    mean_over = (sum(over) / len(over)) if over else 0.0
    print(
        f"[TRUNC] subtree_text > {limit} tok: {len(over)}/{total} sampled "
        f"(of {population}) -> {frac*100:.1f}%; mean tokens among truncated={mean_over:.0f}"
    )


def _overall_suffix(overall_offset, overall_total, overall_t0, done) -> str:
    """Whole-index progress across all 3 embedding phases."""
    if overall_offset is None or not overall_total or overall_t0 is None:
        return ""
    ovr = overall_offset + done
    elapsed = max(time.perf_counter() - overall_t0, 1e-9)
    rate = ovr / elapsed
    eta = (overall_total - ovr) / max(rate, 1e-9)
    return f"  ||  [OVERALL {ovr}/{overall_total} {ovr/overall_total*100:.0f}% | {rate:.1f} it/s | ETA {eta/60:.0f} min]"


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
    (ckpt_dir / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
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
    passage_prefix: str = "",
    overall_offset=None,
    overall_total=None,
    overall_t0=None,
) -> None:
    total = len(section_ids)
    batch_counter = 0
    t0 = time.perf_counter()
    for i in range(start_index, total, batch_size):
        ids = section_ids[i : i + batch_size]
        if phase == "sections_local":
            texts = [sections[sid].local_text for sid in ids]
        else:
            texts = [sections[sid].subtree_text for sid in ids]
        vecs = encode_batch(model, texts, batch_size=batch_size, passage_prefix=passage_prefix)
        for sid, vec in zip(ids, vecs):
            if phase == "sections_local":
                sections[sid].E_local = vec
            else:
                sections[sid].E_subtree = vec
        batch_counter += 1
        done = min(i + batch_size, total)
        if done % (batch_size * 5) == 0 or done == total:
            elapsed = max(time.perf_counter() - t0, 1e-9)
            speed = done / elapsed
            left = max(total - done, 0)
            eta_sec = left / max(speed, 1e-9)
            pct = (done / total) * 100 if total else 100.0
            print(
                f"[{phase}] {done}/{total} ({pct:.1f}%) | "
                f"{speed:.1f} items/s | ETA {eta_sec/60:.1f} min"
                + _overall_suffix(overall_offset, overall_total, overall_t0, done)
            )
        if batch_counter % checkpoint_every == 0:
            checkpoint_save(
                ckpt_dir,
                sections,
                text_nodes,
                graph_adj,
                {"phase": phase, "index": done},
            )
    elapsed = max(time.perf_counter() - t0, 1e-9)
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
    passage_prefix: str = "",
    overall_offset=None,
    overall_total=None,
    overall_t0=None,
) -> None:
    total = len(node_ids)
    batch_counter = 0
    t0 = time.perf_counter()
    for i in range(start_index, total, batch_size):
        ids = node_ids[i : i + batch_size]
        texts = [text_nodes[nid].text for nid in ids]
        vecs = encode_batch(model, texts, batch_size=batch_size, passage_prefix=passage_prefix)
        for nid, vec in zip(ids, vecs):
            text_nodes[nid].embedding = vec
        batch_counter += 1
        done = min(i + batch_size, total)
        if done % (batch_size * 10) == 0 or done == total:
            elapsed = max(time.perf_counter() - t0, 1e-9)
            speed = done / elapsed
            left = max(total - done, 0)
            eta_sec = left / max(speed, 1e-9)
            pct = (done / total) * 100 if total else 100.0
            print(
                f"[text_nodes] {done}/{total} ({pct:.1f}%) | "
                f"{speed:.1f} items/s | ETA {eta_sec/60:.1f} min"
                + _overall_suffix(overall_offset, overall_total, overall_t0, done)
            )
        if batch_counter % checkpoint_every == 0:
            checkpoint_save(
                ckpt_dir,
                sections,
                text_nodes,
                graph_adj,
                {"phase": "text_nodes", "index": done},
            )
    elapsed = max(time.perf_counter() - t0, 1e-9)
    print(f"[text_nodes] done in {elapsed/60:.1f} min")


def main() -> None:
    overall_t0 = time.perf_counter()
    args = parse_args()
    set_runtime_dirs(args.hf_cache_dir, args.tmp_dir)

    device = check_device(args.device, args.strict_cuda)
    print(f"[ENV] device={device}")
    print(f"[ENV] HF_HOME={os.environ.get('HF_HOME')}")

    spec = encoder_spec(args.model_name)
    passage_prefix = spec.passage_prefix
    print(f"[MODEL] name={args.model_name} passage_prefix={passage_prefix!r} query_prefix={spec.query_prefix!r}")

    out_dir = args.out_dir or default_out_dir(args.model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUT] index dir = {out_dir}")
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

    model_path = resolve_model_path(args.model_name, args.hf_cache_dir)
    print(f"[MODEL] Loading {model_path} on {device}")
    model = SentenceTransformer(model_path, device=device, cache_folder=str(args.hf_cache_dir))

    # Drill-down quality proxy: how much subtree text the encoder truncates.
    log_subtree_truncation(model, sections)

    # Whole-index progress accounting (3 phases: local, subtree, text_nodes).
    n_sec = len(section_ids)
    overall_total = 2 * n_sec + len(node_ids)
    overall_t0 = time.perf_counter()
    print(f"[OVERALL] total embeddings to compute: {overall_total} (sections x2 + text_nodes)")

    # Continue from checkpoint phase.
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
            passage_prefix=passage_prefix,
            overall_offset=0,
            overall_total=overall_total,
            overall_t0=overall_t0,
        )
        phase = "sections_subtree"
        start_index = 0

    if phase == "sections_subtree":
        subtree_bs = args.subtree_batch_size or args.batch_size
        run_phase_sections(
            model=model,
            sections=sections,
            section_ids=section_ids,
            batch_size=subtree_bs,
            checkpoint_every=args.checkpoint_every,
            ckpt_dir=ckpt_dir,
            graph_adj=graph_adj,
            text_nodes=text_nodes,
            phase="sections_subtree",
            start_index=start_index,
            passage_prefix=passage_prefix,
            overall_offset=n_sec,
            overall_total=overall_total,
            overall_t0=overall_t0,
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
            passage_prefix=passage_prefix,
            overall_offset=2 * n_sec,
            overall_total=overall_total,
            overall_t0=overall_t0,
        )

    # Final full save in standard format.
    print("[SAVE] Saving final index...")
    save_index(str(out_dir), sections, text_nodes, graph_adj, model_name=args.model_name)
    checkpoint_save(
        ckpt_dir,
        sections,
        text_nodes,
        graph_adj,
        {"phase": "done", "index": 0},
    )
    total_elapsed = max(time.perf_counter() - overall_t0, 1e-9)
    print(f"[DONE] Index saved to {out_dir}")
    print(f"[DONE] Total elapsed: {total_elapsed/60:.1f} min")


if __name__ == "__main__":
    main()

