"""
Compare aggregated 6-vote LLM verdicts against manual labels on the
calibration subset.

For each (comparison, calibration_query, axis):
  - Manual verdict: take the AB pair's manual label, normalize to
    first/second/equal.
  - LLM verdict: take all 6 decisions (3 judges x 2 perms), apply
    normalize_label, then >=4-of-6 rule -> first/second/tie.
  - Match if verdicts coincide.

Reports per-axis and overall match rate. Also tabulates the
{manual} x {agg} confusion matrix per axis.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
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
calib_qids = set(cfg["calibration"]["query_ids"])
judges = [j["name"] for j in cfg["judges"]]
pairs_index = load_pairs_index(PAIRS)
decisions = {j: load_decisions(FULL_DIR / f"llm_{j}.jsonl") for j in judges}

manual = {}
for line in MANUAL.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    pid = r["pair_id"]
    manual[pid] = r.get("manual_labels") or {}


def vote_outcome(votes):
    f = sum(1 for v in votes if v == "first")
    s = sum(1 for v in votes if v == "second")
    if f >= 4: return "first"
    if s >= 4: return "second"
    return "tie"


per_axis = {a: {"match": 0, "total": 0,
                "confusion": Counter()}  # (manual, agg) -> count
            for a in AXES}

for pid, pair in pairs_index.items():
    if pair["query_id"] not in calib_qids: continue
    if pair["perm"] != "AB":  # iterate once per (comparison, query)
        continue

    for axis in AXES:
        # manual canonical verdict
        man_lab = manual.get(pid, {}).get(axis)
        if man_lab is None: continue
        man_norm = normalize_label(man_lab, "AB")  # first/second/equal
        man_verdict = "tie" if man_norm == "equal" else man_norm

        # collect 6 LLM votes (AB + BA, 3 judges)
        votes = []
        # find BA counterpart
        ba_pid = None
        for opid, op in pairs_index.items():
            if (op["comparison_id"], op["query_id"], op["perm"]) == (pair["comparison_id"], pair["query_id"], "BA"):
                ba_pid = opid; break
        for jn in judges:
            for ppid, perm in ((pid, "AB"), (ba_pid, "BA")):
                if ppid is None: continue
                dec = decisions[jn].get(ppid)
                if dec is None: continue
                lab = axis_label(dec, axis)
                if lab is None: continue
                norm = normalize_label(lab, perm)
                if norm is not None:
                    votes.append(norm)
        if len(votes) < 4: continue
        agg = vote_outcome(votes)

        per_axis[axis]["total"] += 1
        if man_verdict == agg:
            per_axis[axis]["match"] += 1
        per_axis[axis]["confusion"][(man_verdict, agg)] += 1

print(f"{'axis':14s} match/total  rate")
print("-" * 40)
mean_num = 0; mean_den = 0
for axis in AXES:
    d = per_axis[axis]
    rate = d["match"] / d["total"] if d["total"] else 0
    mean_num += d["match"]; mean_den += d["total"]
    print(f"{axis:14s} {d['match']:3d}/{d['total']:<3d}     {rate:.3f}")
print("-" * 40)
print(f"{'mean':14s} {mean_num:3d}/{mean_den:<3d}     {mean_num/mean_den:.3f}")

print("\nConfusion matrices (manual_verdict -> agg_verdict):")
for axis in AXES:
    print(f"\n  {axis}")
    print(f"    {'man\\agg':10s} {'first':>6s} {'second':>6s} {'tie':>6s}")
    for m in ("first", "second", "tie"):
        row = " ".join(f"{per_axis[axis]['confusion'].get((m,a),0):6d}" for a in ("first","second","tie"))
        print(f"    {m:10s} {row}")
