"""
Step 2: Build 540 pairs JSONL for pairwise LLM-judge evaluation.

9 method comparisons x 30 queries x 2 permutations (AB, BA) = 540 entries.

Each entry contains the fully assembled contexts for the judge prompt — no
further postprocessing should be required at judging time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _repo_paths import repo_path, resolve_repo_path  # noqa: E402


DEFAULT_METHODS_CFG = repo_path("configs/judge_v2/methods.json")
DEFAULT_UNIFIED_DIR = repo_path("artifacts/judge_v2/unified")
DEFAULT_OUT_PATH = repo_path("artifacts/judge_v2/pairs.jsonl")

CONTEXT_JOINER = "\n\n"

# Order is significant — it defines comparison_id below and pair_id numbering.
COMPARISONS: List[Tuple[str, str]] = [
    ("swaga_chunks",  "bm25"),
    ("swaga_chunks",  "bm25_heuristic"),
    ("swaga_chunks",  "dense"),
    ("swaga_chunks",  "dense_heuristic"),
    ("swaga_windows", "bm25"),
    ("swaga_windows", "bm25_heuristic"),
    ("swaga_windows", "dense"),
    ("swaga_windows", "dense_heuristic"),
    ("swaga_windows", "swaga_chunks"),
]


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


def _assemble_context(unified: Dict[str, Any]) -> str:
    items = unified.get("items") or []
    parts = [it["text"].strip() for it in items if isinstance(it.get("text"), str) and it["text"].strip()]
    return CONTEXT_JOINER.join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build pairs JSONL for judge evaluation.")
    p.add_argument("--methods_config", type=Path, default=DEFAULT_METHODS_CFG)
    p.add_argument("--unified_dir", type=Path, default=DEFAULT_UNIFIED_DIR)
    p.add_argument("--out_path", type=Path, default=DEFAULT_OUT_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _read_json(resolve_repo_path(args.methods_config))
    unified_dir = resolve_repo_path(args.unified_dir)
    out_path = resolve_repo_path(args.out_path)

    queries = _read_jsonl(resolve_repo_path(cfg["queries_file"]))
    qids = [str(q["id"]) for q in queries]
    qid_to_query = {str(q["id"]): str(q["query"]) for q in queries}

    methods = {m["name"] for m in cfg["methods"]}
    needed = {m for pair in COMPARISONS for m in pair}
    missing = needed - methods
    if missing:
        raise ValueError(f"Comparisons reference methods missing from methods.json: {missing}")

    cache: Dict[Tuple[str, str], str] = {}

    def _ctx(method: str, qid: str) -> str:
        key = (method, qid)
        if key not in cache:
            u = _read_json(unified_dir / method / f"{qid}.json")
            cache[key] = _assemble_context(u)
        return cache[key]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    counter = 0
    with out_path.open("w", encoding="utf-8") as f:
        for method_a, method_b in COMPARISONS:
            comparison_id = f"{method_a}_vs_{method_b}"
            for qid in qids:
                ctx_first = _ctx(method_a, qid)
                ctx_second = _ctx(method_b, qid)
                for perm in ("AB", "BA"):
                    counter += 1
                    if perm == "AB":
                        ma, mb, ca, cb = method_a, method_b, ctx_first, ctx_second
                    else:
                        ma, mb, ca, cb = method_b, method_a, ctx_second, ctx_first
                    rec = {
                        "pair_id": f"p{counter:03d}_{qid}_{perm}",
                        "comparison_id": comparison_id,
                        "query_id": qid,
                        "query": qid_to_query[qid],
                        "perm": perm,
                        "method_A": ma,
                        "method_B": mb,
                        "context_A": ca,
                        "context_B": cb,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_written += 1

    print(f"[DONE] Wrote {n_written} pairs to {out_path}")
    print(f"       Comparisons: {len(COMPARISONS)}, queries: {len(qids)}, perms: 2")


if __name__ == "__main__":
    main()
