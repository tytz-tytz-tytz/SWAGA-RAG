from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
SWAGA_REPO_ROOT = SCRIPT_DIR.parent
BUGSY_SRC_DIR = SWAGA_REPO_ROOT.parent / "bugsy_pipeline" / "src"

# Put bugsy_pipeline/src FIRST so `swaga_rag` resolves to the modified package
# (with run_query_subgraphs and SubgraphAssembler).
if not BUGSY_SRC_DIR.exists():
    raise FileNotFoundError(
        f"bugsy_pipeline src not found at: {BUGSY_SRC_DIR}. "
        "Expected sibling repo layout: parent/{SWAGA-RAG, bugsy_pipeline}."
    )
if str(BUGSY_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(BUGSY_SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _repo_paths import repo_path, resolve_repo_path  # noqa: E402
from _common import iter_validated_queries

DEFAULT_QUERIES_PATH = repo_path("data/eval/queries_5rag.jsonl")
DEFAULT_INDEX_DIR = repo_path("artifacts/indexes/swaga_index_dir")
DEFAULT_CONFIG_PATH = repo_path("configs/judge_prep/rag/swaga-rag-subgraphs.json")



def extract_ranked_items_from_subgraphs(result: dict) -> List[Dict[str, Any]]:
    """
    Build output_items from the `subgraphs` field produced by
    SWAGARAGPipeline.run_query_subgraphs (bugsy_pipeline version).

    Each subgraph window becomes a single judge-visible chunk. The chunk_id is
    a stable composite of the parent section and the window range so judge
    payloads have deterministic identifiers.
    """
    subgraphs = result.get("subgraphs", [])
    if not isinstance(subgraphs, list):
        return []

    out: List[Dict[str, Any]] = []
    for sg in subgraphs:
        if not isinstance(sg, dict):
            continue
        text = sg.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        section_id = sg.get("section_id")
        window_node_ids = sg.get("window_node_ids") or []
        if isinstance(section_id, str) and window_node_ids:
            first = window_node_ids[0]
            last = window_node_ids[-1]
            chunk_id = f"{section_id}::{first}..{last}"
        elif isinstance(section_id, str):
            chunk_id = section_id
        else:
            anchors = sg.get("anchor_node_ids") or []
            chunk_id = anchors[0] if anchors else ""
        if not chunk_id:
            continue

        payload: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "text": text.strip(),
        }
        score = sg.get("score")
        if isinstance(score, (int, float)):
            payload["score"] = float(score)
        for key in ("section_id", "parent_section_id", "title",
                    "anchor_node_ids", "window_node_ids", "from_start"):
            value = sg.get(key)
            if value is not None:
                payload[key] = value
        out.append(payload)

    return out


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise TypeError("Config must be a JSON object (dict) at the top level.")
    return cfg


def sanitize_run_name(name: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.,")
    cleaned = "".join(ch if ch in allowed else "_" for ch in name).strip("._-")
    return cleaned or "run"


def compute_run_id(config: Dict[str, Any], cli_run_id: str | None) -> str:
    if cli_run_id:
        return sanitize_run_name(cli_run_id)

    run_cfg = config.get("run", {})
    name = run_cfg.get("name")
    append_ts = bool(run_cfg.get("append_timestamp", True))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if isinstance(name, str) and name.strip():
        base = sanitize_run_name(name.strip())
        return f"{base}__{ts}" if append_ts else base

    return ts


def get_out_dir_from_config(config: Dict[str, Any]) -> Path:
    run_cfg = config.get("run", {})
    out_dir = run_cfg.get("out_dir")
    if not isinstance(out_dir, str) or not out_dir.strip():
        raise ValueError(
            "Missing run.out_dir in config. "
            "Please set config['run']['out_dir'], e.g. 'artifacts/swaga_rag_results'."
        )
    return Path(out_dir)


def is_debug_enabled(config: Dict[str, Any]) -> bool:
    output_cfg = config.get("output", {})
    return bool(output_cfg.get("save_debug", False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run modified SWAGA-RAG (bugsy_pipeline variant) with subgraph "
            "(chunk-window) assembly. Output_items are taken from the "
            "`subgraphs` field of run_query_subgraphs()."
        )
    )
    p.add_argument("--queries_path", type=Path, default=DEFAULT_QUERIES_PATH)
    p.add_argument("--index_dir", type=Path, default=DEFAULT_INDEX_DIR)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    queries_path = resolve_repo_path(args.queries_path)
    index_dir = resolve_repo_path(args.index_dir)
    config_path = resolve_repo_path(args.config)

    from swaga_rag.index.store import load_index
    from swaga_rag.index.embeddings import EmbeddingModel
    from swaga_rag.rag.pipeline import SWAGARAGPipeline

    pipeline_module = sys.modules[SWAGARAGPipeline.__module__]
    if not hasattr(SWAGARAGPipeline, "run_query_subgraphs"):
        raise RuntimeError(
            f"Loaded SWAGARAGPipeline from {pipeline_module.__file__} does not "
            "expose run_query_subgraphs. Check that bugsy_pipeline/src is on "
            "sys.path BEFORE SWAGA-RAG/src."
        )

    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")

    config = load_config(config_path)
    debug_enabled = is_debug_enabled(config)

    out_dir = resolve_repo_path(get_out_dir_from_config(config))
    run_id = compute_run_id(config, args.run_id)
    if args.run_id == ".":
        run_dir = out_dir
    else:
        run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = run_dir / "config.json"
    try:
        shutil.copyfile(config_path, config_snapshot_path)
    except Exception:
        with config_snapshot_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    sections, text_nodes, graph_adj = load_index(str(index_dir))
    model = EmbeddingModel(device=args.device)

    pipeline = SWAGARAGPipeline(
        sections=sections,
        text_nodes=text_nodes,
        graph_adj=graph_adj,
        embedding_model=model,
        config=config,
    )

    print(f"[INFO] Pipeline module: {pipeline_module.__file__}")
    print(f"[INFO] Output dir: {run_dir}")

    for item in iter_validated_queries(queries_path):
        qid = str(item["id"])
        query = str(item["query"])

        result = pipeline.run_query_subgraphs(query)
        if not isinstance(result, dict):
            raise TypeError("run_query_subgraphs(query) must return a dict")

        output_items = extract_ranked_items_from_subgraphs(result)
        out_obj = {
            "id": qid,
            "query": query,
            "output_items": output_items,
            "output_ids": [x["chunk_id"] for x in output_items],
        }

        out_path = run_dir / f"{qid}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)

        if debug_enabled:
            debug_payload = result.get("debug")
            if isinstance(debug_payload, dict):
                debug_path = run_dir / f"{qid}.debug.json"
                with debug_path.open("w", encoding="utf-8") as f:
                    json.dump(debug_payload, f, ensure_ascii=False, indent=2)

        print(f"[OK] {qid} -> {out_path} ({len(output_items)} subgraphs)")

    print(f"[DONE] Results written to: {run_dir}")
    print(f"[DONE] Config snapshot: {config_snapshot_path}")


if __name__ == "__main__":
    main()
