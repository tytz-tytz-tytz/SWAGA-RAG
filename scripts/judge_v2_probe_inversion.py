"""
Verification probe for the BA-inversion logic.

Picks one concrete (comparison, query) — defaults to swaga_chunks_vs_bm25
on Q5R001 — fabricates a small set of 6 judge decisions (3 judges x 2 perms),
applies normalize_label() to each, then runs the >=4-of-6 voting rule used in
judge_v2_aggregate.py. Prints the full trace so the math can be eyeballed.

Not an LLM call — no API key needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _repo_paths import repo_path, resolve_repo_path  # noqa: E402

from judge_v2_metrics_common import normalize_label, split_comparison_id  # noqa: E402

PAIRS_PATH = repo_path("artifacts/judge_v2/pairs.jsonl")


def _load_two(comparison_id: str, qid: str):
    pair_ab = None
    pair_ba = None
    with PAIRS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["comparison_id"] != comparison_id or r["query_id"] != qid:
                continue
            if r["perm"] == "AB":
                pair_ab = r
            elif r["perm"] == "BA":
                pair_ba = r
    if pair_ab is None or pair_ba is None:
        raise RuntimeError(f"Could not find both perms for {comparison_id} / {qid}")
    return pair_ab, pair_ba


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe BA inversion + majority voting.")
    p.add_argument("--comparison", default="swaga_chunks_vs_bm25")
    p.add_argument("--qid", default="Q5R001")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    first, second = split_comparison_id(args.comparison)
    print(f"Comparison: {args.comparison}")
    print(f"  method_first  (= method_A in pair_aggregates output) = {first}")
    print(f"  method_second (= method_B in pair_aggregates output) = {second}")
    print(f"Query: {args.qid}")
    pair_ab, pair_ba = _load_two(args.comparison, args.qid)
    print("\nPair AB header:")
    print(f"  pair_id={pair_ab['pair_id']}  method_A={pair_ab['method_A']}  method_B={pair_ab['method_B']}")
    print("Pair BA header:")
    print(f"  pair_id={pair_ba['pair_id']}  method_A={pair_ba['method_A']}  method_B={pair_ba['method_B']}")

    # Fabricated decisions: a deliberately mixed pattern so we can see what
    # normalize_label produces in every code path. The fabricated pattern
    # corresponds to: judge_1 prefers `first` in both perms, judge_2 has
    # position bias (always says A), judge_3 says equal on AB and `first`-A on BA.
    fabricated = [
        # (judge, perm, label_seen)
        ("judge_1", "AB", "A"),    # → first
        ("judge_1", "BA", "B"),    # → first (B in BA == method_first)
        ("judge_2", "AB", "A"),    # → first
        ("judge_2", "BA", "A"),    # → second  (A in BA == method_second; raw position bias)
        ("judge_3", "AB", "equal"),
        ("judge_3", "BA", "B"),    # → first
    ]

    votes = []
    print("\n--- Per-decision normalization (axis: relevance) ---")
    for jname, perm, label in fabricated:
        norm = normalize_label(label, perm)
        votes.append(norm)
        meaning_a = pair_ab["method_A"] if perm == "AB" else pair_ba["method_A"]
        meaning_b = pair_ab["method_B"] if perm == "AB" else pair_ba["method_B"]
        target = first if norm == "first" else (second if norm == "second" else "equal")
        print(
            f"  {jname:8s}  perm={perm}  label={label:5s}  "
            f"(A={meaning_a}, B={meaning_b})  "
            f"-> normalized={norm:7s}  -> vote for {target}"
        )

    n_first = sum(1 for v in votes if v == "first")
    n_second = sum(1 for v in votes if v == "second")
    n_equal = sum(1 for v in votes if v == "equal")
    print(f"\nVote counts: first={n_first}  second={n_second}  equal={n_equal}  (total={len(votes)})")

    if n_first >= 4:
        outcome = f"WIN: {first}"
    elif n_second >= 4:
        outcome = f"WIN: {second}"
    else:
        outcome = "TIE"
    print(f"≥4-of-6 rule outcome: {outcome}")

    print("\n--- Sanity: position-bias-only pattern (all 6 say 'A') ---")
    biased = [normalize_label("A", perm) for perm in ("AB", "BA") for _ in range(3)]
    bf = sum(1 for v in biased if v == "first")
    bs = sum(1 for v in biased if v == "second")
    print(f"  3 AB 'A' + 3 BA 'A' -> first={bf}  second={bs}  -> "
          f"{'tie' if bf < 4 and bs < 4 else 'win'} (correctly diluted by inversion)")


if __name__ == "__main__":
    main()
