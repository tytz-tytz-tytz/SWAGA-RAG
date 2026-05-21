"""
Step 1: Unify retrieval outputs to a fixed token budget.

For each (method, query) pair, walk output_items in their original ranking
order and accumulate items while the cumulative token count stays <= budget.
An item that does not fit as a whole is skipped (no mid-item truncation).

Outputs:
  artifacts/judge_v2/unified/{method}/{qid}.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _repo_paths import repo_path, resolve_repo_path  # noqa: E402

import tiktoken  # noqa: E402


DEFAULT_METHODS_CFG = repo_path("configs/judge_v2/methods.json")
DEFAULT_OUT_DIR = repo_path("artifacts/judge_v2/unified")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def unify_one(
    method: str,
    qid: str,
    query: str,
    items: List[Dict[str, Any]],
    encoder: "tiktoken.Encoding",
    budget: int,
) -> Dict[str, Any]:
    used: List[Dict[str, Any]] = []
    total = 0
    truncated = 0
    rank = 0
    for item in items:
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            truncated += 1
            continue
        tcount = len(encoder.encode(text))
        if total + tcount > budget:
            truncated += 1
            continue
        rank += 1
        used.append({
            "chunk_id": item.get("chunk_id", ""),
            "text": text,
            "rank": rank,
            "tokens": tcount,
        })
        total += tcount

    return {
        "query_id": qid,
        "query": query,
        "method": method,
        "items": used,
        "total_tokens": total,
        "items_count": len(used),
        "items_truncated_count": truncated,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unify retrieval outputs to a token budget.")
    p.add_argument("--methods_config", type=Path, default=DEFAULT_METHODS_CFG)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = resolve_repo_path(args.methods_config)
    out_dir = resolve_repo_path(args.out_dir)

    cfg = _read_json(cfg_path)
    budget = int(cfg["token_budget"])
    encoder = tiktoken.get_encoding(cfg["encoding_name"])

    queries_path = resolve_repo_path(cfg["queries_file"])
    queries = _read_jsonl(queries_path)
    qid_to_query: Dict[str, str] = {str(q["id"]): str(q["query"]) for q in queries}

    summary: Dict[str, Dict[str, Any]] = {}

    for m in cfg["methods"]:
        method_name = m["name"]
        method_dir = resolve_repo_path(m["dir"])
        method_out = out_dir / method_name

        total_items = 0
        total_truncated = 0
        total_tokens_sum = 0
        n_files = 0

        for qid, query in qid_to_query.items():
            src_path = method_dir / f"{qid}.json"
            if not src_path.exists():
                raise FileNotFoundError(f"Missing retrieval output: {src_path}")
            src = _read_json(src_path)
            items = src.get("output_items") or []
            if not isinstance(items, list):
                items = []

            unified = unify_one(method_name, qid, query, items, encoder, budget)
            out_path = method_out / f"{qid}.json"
            _write_json(out_path, unified)

            n_files += 1
            total_items += unified["items_count"]
            total_truncated += unified["items_truncated_count"]
            total_tokens_sum += unified["total_tokens"]

        summary[method_name] = {
            "files": n_files,
            "avg_items": round(total_items / max(n_files, 1), 2),
            "avg_truncated": round(total_truncated / max(n_files, 1), 2),
            "avg_tokens": round(total_tokens_sum / max(n_files, 1), 1),
        }
        print(f"[OK] {method_name}: {summary[method_name]}")

    summary_path = out_dir / "summary.json"
    _write_json(summary_path, {"budget": budget, "methods": summary})
    print(f"[DONE] Summary: {summary_path}")


if __name__ == "__main__":
    main()
