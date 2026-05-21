"""
Step 10: aggregate the full-run LLM-judge decisions into reportable shape.

Outputs (all under artifacts/judge_v2/aggregates/):
  - pair_aggregates.json    — per-comparison, per-axis win/tie/loss rates over 30 queries
  - agreement_metrics.json  — inter-judge agreement + per-judge permutation consistency
  - operational_metrics.json — totals: calls, retries, failures, latency, usage
  - csv/pair_axis.csv       — long-form table for import into Word/Excel
  - csv/operational.csv     — per-judge operational summary

Voting rule (per comparison_id, per query_id, per axis):
  Collect 6 normalized votes = 3 judges x 2 perms (with perm inversion).
  Count how many vote "first" vs "second" vs "equal".
  >=4 votes for one side -> that side wins.
  Otherwise -> tie. (`equal` votes never count toward either side.)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _repo_paths import repo_path, resolve_repo_path  # noqa: E402

from judge_v2_metrics_common import (  # noqa: E402
    AXES,
    axis_label,
    invert_label_ab,
    load_decisions,
    load_pairs_index,
    normalize_label,
    split_comparison_id,
)


DEFAULT_PAIRS = repo_path("artifacts/judge_v2/pairs.jsonl")
DEFAULT_FULL_DIR = repo_path("artifacts/judge_v2/full_run")
DEFAULT_JUDGES_CFG = repo_path("configs/judge_v2/judges.json")
DEFAULT_OUT_DIR = repo_path("artifacts/judge_v2/aggregates")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _vote_outcome(votes: List[str]) -> str:
    """Apply the >=4-of-6 rule. Returns "first" | "second" | "tie"."""
    f = sum(1 for v in votes if v == "first")
    s = sum(1 for v in votes if v == "second")
    if f >= 4:
        return "first"
    if s >= 4:
        return "second"
    return "tie"


def _aggregate_pairs(
    pairs_index: Dict[str, Dict[str, Any]],
    decisions_by_judge: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    judges = list(decisions_by_judge.keys())
    by_cmp: Dict[str, Dict[str, Dict[str, List[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for pid, pair in pairs_index.items():
        cmp_id = pair["comparison_id"]
        qid = pair["query_id"]
        perm = pair["perm"]
        for jn in judges:
            dec = decisions_by_judge[jn].get(pid)
            if dec is None or dec.get("status") != "ok":
                continue
            for axis in AXES:
                lab = axis_label(dec, axis)
                if lab is None:
                    continue
                norm = normalize_label(lab, perm)
                if norm is None:
                    continue
                by_cmp[cmp_id][qid][axis].append(norm)

    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cmp_id, q_map in by_cmp.items():
        first, second = split_comparison_id(cmp_id)
        per_axis: Dict[str, Dict[str, float]] = {}
        for axis in AXES:
            n_first = n_second = n_tie = 0
            n_q = 0
            for qid, axes_votes in q_map.items():
                votes = axes_votes.get(axis) or []
                if not votes:
                    continue
                n_q += 1
                outcome = _vote_outcome(votes)
                if outcome == "first":
                    n_first += 1
                elif outcome == "second":
                    n_second += 1
                else:
                    n_tie += 1
            denom = max(n_q, 1)
            per_axis[axis] = {
                "win_A": n_first / denom,
                "tie": n_tie / denom,
                "win_B": n_second / denom,
                "n_queries": n_q,
            }
        valid_axes = [per_axis[a] for a in AXES if per_axis[a]["n_queries"] > 0]
        if valid_axes:
            overall = {
                "win_A": sum(a["win_A"] for a in valid_axes) / len(valid_axes),
                "tie": sum(a["tie"] for a in valid_axes) / len(valid_axes),
                "win_B": sum(a["win_B"] for a in valid_axes) / len(valid_axes),
            }
        else:
            overall = {"win_A": 0.0, "tie": 0.0, "win_B": 0.0}
        result[cmp_id] = {
            "method_A": first,
            "method_B": second,
            **{a: per_axis[a] for a in AXES},
            "overall": overall,
        }
    return result


def _agreement_metrics(
    pairs_index: Dict[str, Dict[str, Any]],
    decisions_by_judge: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    judges = list(decisions_by_judge.keys())

    # Inter-judge agreement per axis: share of pair_ids where all judges agree.
    per_axis_inter: Dict[str, Dict[str, int]] = {a: {"all_agree": 0, "total": 0} for a in AXES}
    for pid in pairs_index:
        for axis in AXES:
            vals: List[str] = []
            for jn in judges:
                lab = axis_label(decisions_by_judge[jn].get(pid, {}), axis)
                if lab is not None:
                    vals.append(lab)
            if len(vals) != len(judges):
                continue
            per_axis_inter[axis]["total"] += 1
            if all(v == vals[0] for v in vals):
                per_axis_inter[axis]["all_agree"] += 1
    inter_rates = {
        a: (per_axis_inter[a]["all_agree"] / per_axis_inter[a]["total"]
            if per_axis_inter[a]["total"] else None)
        for a in AXES
    }
    inter_valid = [v for v in inter_rates.values() if v is not None]
    inter = {
        "per_axis_rate": inter_rates,
        "per_axis_counts": per_axis_inter,
        "mean_rate": sum(inter_valid) / len(inter_valid) if inter_valid else None,
    }

    # Permutation consistency per judge.
    perm_consistency: Dict[str, Dict[str, Any]] = {}
    for jn in judges:
        per_axis: Dict[str, Dict[str, int]] = {a: {"consistent": 0, "total": 0} for a in AXES}
        by_cqa: Dict[Tuple[str, str, str], Dict[str, str]] = defaultdict(dict)
        for pid, dec in decisions_by_judge[jn].items():
            pair = pairs_index.get(pid)
            if pair is None:
                continue
            for axis in AXES:
                lab = axis_label(dec, axis)
                if lab is None:
                    continue
                by_cqa[(pair["comparison_id"], pair["query_id"], axis)][pair["perm"]] = lab
        for (_, _, axis), per_perm in by_cqa.items():
            ab = per_perm.get("AB")
            ba = per_perm.get("BA")
            if ab is None or ba is None:
                continue
            per_axis[axis]["total"] += 1
            if ba == invert_label_ab(ab):
                per_axis[axis]["consistent"] += 1
        rates = {
            a: (per_axis[a]["consistent"] / per_axis[a]["total"] if per_axis[a]["total"] else None)
            for a in AXES
        }
        valid = [v for v in rates.values() if v is not None]
        perm_consistency[jn] = {
            "per_axis_rate": rates,
            "per_axis_counts": per_axis,
            "mean_rate": sum(valid) / len(valid) if valid else None,
        }

    return {
        "inter_judge_agreement": inter,
        "permutation_consistency_per_judge": perm_consistency,
    }


def _operational_metrics(
    decisions_by_judge: Dict[str, Dict[str, Dict[str, Any]]],
    pricing_by_judge: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """Per-judge and total: call counts, token usage, latency, USD cost.

    Cost is computed against ``pricing_per_1m_tokens`` declared in
    configs/judge_v2/judges.json. For judges accessed via CometAPI the actual
    billed amount may differ from the recorded "public" estimate because of
    the gateway markup — recorded as a disclaimer in the output JSON.
    """
    zeros = lambda: {
        "total_calls": 0, "ok": 0, "failed": 0, "retry": 0,
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "latency_ms": 0, "cost_usd_estimated": 0.0,
    }
    totals: Dict[str, Any] = zeros()
    per_judge: Dict[str, Dict[str, Any]] = {}
    for jn, decs in decisions_by_judge.items():
        agg = zeros()
        price = pricing_by_judge.get(jn, {"input_usd": 0.0, "output_usd": 0.0})
        for d in decs.values():
            agg["total_calls"] += 1
            if d.get("status") == "ok":
                agg["ok"] += 1
            else:
                agg["failed"] += 1
            if int(d.get("attempts") or 1) > 1:
                agg["retry"] += 1
            usage = d.get("usage") or {}
            in_tok = int(usage.get("input_tokens") or 0)
            out_tok = int(usage.get("output_tokens") or 0)
            tot_tok = usage.get("total_tokens")
            tot_tok = int(tot_tok) if isinstance(tot_tok, (int, float)) else in_tok + out_tok
            agg["input_tokens"] += in_tok
            agg["output_tokens"] += out_tok
            agg["total_tokens"] += tot_tok
            agg["cost_usd_estimated"] += (
                in_tok * float(price["input_usd"]) / 1_000_000
                + out_tok * float(price["output_usd"]) / 1_000_000
            )
            lat = d.get("latency_ms")
            if isinstance(lat, (int, float)):
                agg["latency_ms"] += int(lat)
        per_judge[jn] = agg
        for k in totals:
            totals[k] += agg[k]
    return {
        "per_judge": per_judge,
        "totals": totals,
        "cost_disclaimer": (
            "cost_usd_estimated is computed from public per-provider pricing "
            "declared in configs/judge_v2/judges.json. For judges routed "
            "through CometAPI (anthropic_haiku, gemini_2_5_flash) the actual "
            "billed amount may differ from this estimate due to the gateway markup."
        ),
    }


def _export_csv(
    pair_aggregates: Dict[str, Any],
    operational: Dict[str, Any],
    out_dir: Path,
) -> None:
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    pair_axis_path = csv_dir / "pair_axis.csv"
    with pair_axis_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison_id", "method_A", "method_B", "axis",
                    "win_A", "tie", "win_B", "n_queries"])
        for cmp_id, payload in pair_aggregates.items():
            for axis in AXES:
                row = payload[axis]
                w.writerow([cmp_id, payload["method_A"], payload["method_B"], axis,
                            f"{row['win_A']:.4f}", f"{row['tie']:.4f}",
                            f"{row['win_B']:.4f}", row["n_queries"]])
            o = payload["overall"]
            w.writerow([cmp_id, payload["method_A"], payload["method_B"], "overall",
                        f"{o['win_A']:.4f}", f"{o['tie']:.4f}", f"{o['win_B']:.4f}", ""])

    op_path = csv_dir / "operational.csv"
    with op_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["judge", "total_calls", "ok", "failed", "retry",
                    "input_tokens", "output_tokens", "total_tokens",
                    "latency_ms", "cost_usd_estimated"])
        for jn, agg in operational["per_judge"].items():
            w.writerow([jn, agg["total_calls"], agg["ok"], agg["failed"], agg["retry"],
                        agg["input_tokens"], agg["output_tokens"], agg["total_tokens"],
                        agg["latency_ms"], f"{agg['cost_usd_estimated']:.4f}"])
        t = operational["totals"]
        w.writerow(["TOTAL", t["total_calls"], t["ok"], t["failed"], t["retry"],
                    t["input_tokens"], t["output_tokens"], t["total_tokens"],
                    t["latency_ms"], f"{t['cost_usd_estimated']:.4f}"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate full-run judge decisions.")
    p.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    p.add_argument("--full_dir", type=Path, default=DEFAULT_FULL_DIR)
    p.add_argument("--judges_config", type=Path, default=DEFAULT_JUDGES_CFG)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pairs_index = load_pairs_index(resolve_repo_path(args.pairs))
    cfg = _read_json(resolve_repo_path(args.judges_config))
    judge_names = [j["name"] for j in cfg["judges"]]
    full_dir = resolve_repo_path(args.full_dir)

    decisions_by_judge: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for jn in judge_names:
        decisions_by_judge[jn] = load_decisions(full_dir / f"llm_{jn}.jsonl")

    pricing_by_judge: Dict[str, Dict[str, float]] = {}
    for j in cfg["judges"]:
        pricing = j.get("pricing_per_1m_tokens")
        if isinstance(pricing, dict):
            pricing_by_judge[j["name"]] = {
                "input_usd": float(pricing.get("input_usd", 0.0)),
                "output_usd": float(pricing.get("output_usd", 0.0)),
            }

    pair_aggregates = _aggregate_pairs(pairs_index, decisions_by_judge)
    agreement = _agreement_metrics(pairs_index, decisions_by_judge)
    operational = _operational_metrics(decisions_by_judge, pricing_by_judge)

    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "pair_aggregates.json").open("w", encoding="utf-8") as f:
        json.dump(pair_aggregates, f, ensure_ascii=False, indent=2)
    with (out_dir / "agreement_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(agreement, f, ensure_ascii=False, indent=2)
    with (out_dir / "operational_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(operational, f, ensure_ascii=False, indent=2)
    _export_csv(pair_aggregates, operational, out_dir)

    print(f"[DONE] Aggregates in {out_dir}")
    for cmp_id, payload in pair_aggregates.items():
        o = payload["overall"]
        print(
            f"  {cmp_id}: overall  win_A={o['win_A']:.2f}  "
            f"tie={o['tie']:.2f}  win_B={o['win_B']:.2f}"
        )
    inter_mean = agreement["inter_judge_agreement"]["mean_rate"]
    print(f"  inter-judge agreement (mean across axes): "
          f"{inter_mean if inter_mean is None else round(inter_mean, 3)}")
    print(f"  operational totals: {operational['totals']}")


if __name__ == "__main__":
    main()
