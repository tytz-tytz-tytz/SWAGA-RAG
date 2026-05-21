"""
Per-comparison match between manual labels and aggregated LLM verdict.

For each of the 9 comparisons:
  - All 5 calibration queries x 3 axes = 15 cells.
  - Manual verdict: AB-label normalized to first/second/equal.
  - LLM aggregate: 6-vote outcome (3 judges x 2 perms) with >=4-of-6 rule.
  - Match if verdicts coincide.
Also breaks down per-axis within each comparison and reports the direction-
only match (counts only cells where manual chose a side).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _repo_paths import repo_path  # noqa: E402

from judge_v2_metrics_common import (  # noqa: E402
    AXES, axis_label, load_decisions, load_pairs_index, normalize_label,
)

FULL_DIR = repo_path("artifacts/judge_v2/full_run")
PAIRS = repo_path("artifacts/judge_v2/pairs.jsonl")
MANUAL = repo_path("artifacts/judge_v2/calibration/manual.jsonl")
CFG = repo_path("configs/judge_v2/judges.json")

cfg = json.load(CFG.open(encoding="utf-8"))
calib = set(cfg["calibration"]["query_ids"])
judges = [j["name"] for j in cfg["judges"]]
pairs = load_pairs_index(PAIRS)
decisions = {j: load_decisions(FULL_DIR / f"llm_{j}.jsonl") for j in judges}

manual = {}
for line in MANUAL.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    manual[r["pair_id"]] = r.get("manual_labels") or {}


def vote_outcome(votes):
    f = sum(1 for v in votes if v == "first")
    s = sum(1 for v in votes if v == "second")
    if f >= 4: return "first"
    if s >= 4: return "second"
    return "tie"


# Aggregate per (comparison_id, axis)
per_cmp = {}
for pid, pair in pairs.items():
    if pair["query_id"] not in calib or pair["perm"] != "AB":
        continue
    cmp_id = pair["comparison_id"]
    bucket = per_cmp.setdefault(cmp_id, {a: {"match": 0, "total": 0,
                                              "manual_decisive_match": 0,
                                              "manual_decisive_total": 0,
                                              "manual_tie_count": 0,
                                              "agg_tie_count": 0} for a in AXES})

    ba_pid = next((p for p, pr in pairs.items()
                   if (pr["comparison_id"], pr["query_id"], pr["perm"])
                   == (cmp_id, pair["query_id"], "BA")), None)

    for axis in AXES:
        man_lab = manual.get(pid, {}).get(axis)
        if not man_lab:
            continue
        man_norm = normalize_label(man_lab, "AB")
        man_verdict = "tie" if man_norm == "equal" else man_norm

        votes = []
        for jn in judges:
            for ppid, perm in ((pid, "AB"), (ba_pid, "BA")):
                if ppid is None: continue
                d = decisions[jn].get(ppid)
                if d is None: continue
                lab = axis_label(d, axis)
                if lab is None: continue
                n = normalize_label(lab, perm)
                if n is not None:
                    votes.append(n)
        if len(votes) < 4:
            continue
        agg = vote_outcome(votes)

        bucket[axis]["total"] += 1
        if man_verdict == agg:
            bucket[axis]["match"] += 1
        if man_verdict == "tie":
            bucket[axis]["manual_tie_count"] += 1
        else:
            bucket[axis]["manual_decisive_total"] += 1
            if man_verdict == agg:
                bucket[axis]["manual_decisive_match"] += 1
        if agg == "tie":
            bucket[axis]["agg_tie_count"] += 1

# Print
order = [
    "swaga_chunks_vs_bm25",
    "swaga_chunks_vs_bm25_heuristic",
    "swaga_chunks_vs_dense",
    "swaga_chunks_vs_dense_heuristic",
    "swaga_windows_vs_bm25",
    "swaga_windows_vs_bm25_heuristic",
    "swaga_windows_vs_dense",
    "swaga_windows_vs_dense_heuristic",
    "swaga_windows_vs_swaga_chunks",
]

print(f"{'comparison':38s} {'rel':>6s} {'clean':>6s} {'suf':>6s} {'mean':>6s} | {'decisive':>10s} | {'man_tie':>7s} {'agg_tie':>7s}")
print("-" * 110)
all_match, all_total = 0, 0
for cmp_id in order:
    b = per_cmp.get(cmp_id, {})
    rates = []
    decisive_match = 0
    decisive_total = 0
    man_tie = 0
    agg_tie = 0
    for axis in AXES:
        d = b.get(axis, {"match":0,"total":0,"manual_decisive_match":0,"manual_decisive_total":0,"manual_tie_count":0,"agg_tie_count":0})
        r = (d["match"] / d["total"]) if d["total"] else 0
        rates.append(r)
        all_match += d["match"]; all_total += d["total"]
        decisive_match += d["manual_decisive_match"]
        decisive_total += d["manual_decisive_total"]
        man_tie += d["manual_tie_count"]
        agg_tie += d["agg_tie_count"]
    dec_rate = (decisive_match / decisive_total) if decisive_total else 0
    mean = sum(rates)/3
    print(f"{cmp_id:38s} {rates[0]:.2f}   {rates[1]:.2f}   {rates[2]:.2f}   {mean:.2f}   | {decisive_match:>3d}/{decisive_total:<3d} {dec_rate:.0%} | {man_tie:>4d}    {agg_tie:>4d}")

print("-" * 110)
print(f"{'OVERALL':38s} {'':>6s} {'':>6s} {'':>6s} {all_match/all_total:.2f}")
