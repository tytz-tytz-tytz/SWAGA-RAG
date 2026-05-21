"""
Print per-method token statistics for the unified retrieval outputs.

Reports mean / median / min / max / share-of-files-with-fill>=80% of budget,
and counts of items per file (raw vs after unification).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _repo_paths import repo_path, resolve_repo_path  # noqa: E402

import tiktoken  # noqa: E402


DEFAULT_METHODS_CFG = repo_path("configs/judge_v2/methods.json")
DEFAULT_UNIFIED_DIR = repo_path("artifacts/judge_v2/unified")


def _read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methods_config", type=Path, default=DEFAULT_METHODS_CFG)
    p.add_argument("--unified_dir", type=Path, default=DEFAULT_UNIFIED_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = _read_json(resolve_repo_path(args.methods_config))
    budget = int(cfg["token_budget"])
    encoder = tiktoken.get_encoding(cfg["encoding_name"])
    unified_dir = resolve_repo_path(args.unified_dir)

    print(f"Budget: {budget} tokens (encoding {cfg['encoding_name']})\n")
    print(f"{'method':22s} {'mean':>6s} {'med':>6s} {'min':>6s} {'max':>6s} "
          f"{'>=80% fill':>10s} {'items_after':>11s} {'raw_items':>9s} "
          f"{'raw_tok_max':>11s}")
    for m in cfg["methods"]:
        name = m["name"]
        src_dir = resolve_repo_path(m["dir"])
        method_unified = unified_dir / name

        totals = []
        items_after = []
        raw_item_counts = []
        raw_total_tokens = []
        n_full = 0

        for path in sorted(method_unified.glob("*.json")):
            unified = _read_json(path)
            tot = int(unified["total_tokens"])
            totals.append(tot)
            items_after.append(int(unified["items_count"]))
            if tot >= int(0.8 * budget):
                n_full += 1

            # raw items (before unification)
            qid = unified["query_id"]
            raw = _read_json(src_dir / f"{qid}.json")
            raw_items = raw.get("output_items") or []
            raw_item_counts.append(len(raw_items))
            raw_total = sum(
                len(encoder.encode(it.get("text", "")))
                for it in raw_items
                if isinstance(it.get("text"), str)
            )
            raw_total_tokens.append(raw_total)

        if not totals:
            print(f"{name:22s} (no files)")
            continue
        share_full = n_full / len(totals)
        print(
            f"{name:22s} "
            f"{int(statistics.mean(totals)):>6d} "
            f"{int(statistics.median(totals)):>6d} "
            f"{min(totals):>6d} "
            f"{max(totals):>6d} "
            f"{share_full*100:>9.1f}% "
            f"{statistics.mean(items_after):>11.1f} "
            f"{statistics.mean(raw_item_counts):>9.1f} "
            f"{max(raw_total_tokens):>11d}"
        )

    print()
    print("Columns:")
    print("  mean/med/min/max  — total_tokens per query (after unification, <= budget)")
    print("  >=80% fill        — share of queries whose unified total reaches 0.8 * budget")
    print("  items_after       — avg items kept after unification")
    print("  raw_items         — avg items in the raw retrieval output (before unification)")
    print("  raw_tok_max       — single largest raw total tokens across all 30 queries")
    print("                       (if this is well below budget, baseline cannot fill budget")
    print("                        even with unlimited top_k unless it returns longer chunks)")


if __name__ == "__main__":
    main()
