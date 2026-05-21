from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from _repo_paths import repo_path, resolve_repo_path
from _common import iter_validated_queries

DEFAULT_QUERIES_PATH = repo_path("data/eval/queries.jsonl")
DEFAULT_INDEX_DIR = repo_path("artifacts/indexes/swaga_index_dir")
DEFAULT_CONFIG_PATH = repo_path("configs/ablations/stable_baseline.json")



def extract_ranked_items(result: dict) -> List[Dict[str, Any]]:
    """
    Extract ranked evidence blocks from SWAGA-RAG output.
    Preference is given to `text_nodes` because it already contains block IDs.
    """
    items = result.get("text_nodes", [])
    if not isinstance(items, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        chunk_id = item.get("node_id")
        text = item.get("text")
        score = item.get("score")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            continue
        if not isinstance(text, str) or not text.strip():
            continue

        payload: Dict[str, Any] = {
            "chunk_id": chunk_id.strip(),
            "text": text.strip(),
        }
        label = item.get("label")
        if isinstance(label, str) and label.strip():
            payload["label"] = label.strip()
        if isinstance(score, (int, float)):
            payload["score"] = float(score)
        for key in ("type", "chunk_id", "related_chunk_id", "item_ids", "node_ids", "source_node_ids"):
            value = item.get(key)
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
    """
    Make a run name safe for filesystem paths.
    Keeps letters/digits/_-., replaces everything else with underscore.
    """
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.,")
    cleaned = "".join(ch if ch in allowed else "_" for ch in name).strip("._-")
    return cleaned or "run"


def compute_run_id(config: Dict[str, Any], cli_run_id: str | None) -> str:
    """
    Determine run_id folder name.

    Priority:
    1) CLI --run_id (explicit override)
    2) config["run"]["name"] (human label)
    3) fallback to timestamp only
    """
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
    """
    Read output directory from config.

    Expected:
      config["run"]["out_dir"] = "artifacts/swaga_rag_results"
    """
    run_cfg = config.get("run", {})
    out_dir = run_cfg.get("out_dir")
    if not isinstance(out_dir, str) or not out_dir.strip():
        raise ValueError(
            "Missing run.out_dir in config. "
            "Please set config['run']['out_dir'], e.g. 'artifacts/swaga_rag_results'."
        )
    return Path(out_dir)


def is_debug_enabled(config: Dict[str, Any]) -> bool:
    """
    Whether to save debug files for each query.
    Controlled by config["output"]["save_debug"].
    """
    output_cfg = config.get("output", {})
    return bool(output_cfg.get("save_debug", False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SWAGA-RAG retrieval for all queries (offline).")
    p.add_argument("--queries_path", type=Path, default=DEFAULT_QUERIES_PATH)
    p.add_argument("--index_dir", type=Path, default=DEFAULT_INDEX_DIR)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional run id folder name. Overrides config['run']['name'] if provided.",
    )
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

    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")

    config = load_config(config_path)
    debug_enabled = is_debug_enabled(config)

    # Compute run directory from config
    out_dir = resolve_repo_path(get_out_dir_from_config(config))
    run_id = compute_run_id(config, args.run_id)
    # Special: if --run_id "." then write directly into out_dir (flat layout)
    if args.run_id == ".":
        run_dir = out_dir
    else:
        run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)


    # Save an exact copy of the config used for this run (reproducibility)
    config_snapshot_path = run_dir / "config.json"
    try:
        shutil.copyfile(config_path, config_snapshot_path)
    except Exception:
        with config_snapshot_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    # Load index + initialize embedding model
    sections, text_nodes, graph_adj = load_index(str(index_dir))
    model = EmbeddingModel(device=args.device)

    pipeline = SWAGARAGPipeline(
        sections=sections,
        text_nodes=text_nodes,
        graph_adj=graph_adj,
        embedding_model=model,
        config=config,
    )

    for item in iter_validated_queries(queries_path):
        qid = str(item["id"])
        query = str(item["query"])

        result = pipeline.run_query(query)
        if not isinstance(result, dict):
            raise TypeError("run_query(query) must return a dict to extract text chunks")

        output_items = extract_ranked_items(result)
        out_obj = {
            "id": qid,
            "query": query,
            "output_items": output_items,
            "output_ids": [x["chunk_id"] for x in output_items],
        }

        out_path = run_dir / f"{qid}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)

        # --- Debug output (stored separately, never used for blind judging) ---
        if debug_enabled:
            debug_payload = result.get("debug")
            if isinstance(debug_payload, dict):
                debug_path = run_dir / f"{qid}.debug.json"
                with debug_path.open("w", encoding="utf-8") as f:
                    json.dump(debug_payload, f, ensure_ascii=False, indent=2)

        print(f"[OK] {qid} -> {out_path}")

    print(f"[DONE] Results written to: {run_dir}")
    print(f"[DONE] Config snapshot: {config_snapshot_path}")
    if debug_enabled:
        print("[DONE] Debug files saved as: Qxxx.debug.json")


if __name__ == "__main__":
    main()



