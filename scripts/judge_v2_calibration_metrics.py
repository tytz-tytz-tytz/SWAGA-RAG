"""
Step 8: calibration metrics.

Reads:
  - artifacts/judge_v2/pairs.jsonl
  - artifacts/judge_v2/calibration/manual.jsonl
  - artifacts/judge_v2/calibration/llm_{judge}.jsonl  (per-judge)

Produces:
  - artifacts/judge_v2/calibration/metrics.json

Metrics
-------
8.1 Per-judge agreement with manual labelling (per axis + mean).
8.2 Inter-judge agreement: share of pairs where all three judges agree
    (per axis + mean).
8.3 Permutation consistency per judge: for each (comparison, query, axis),
    pair the judge's AB and BA labels and check that BA equals invert(AB).
"""
from __future__ import annotations

import argparse
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
    read_jsonl,
)


DEFAULT_PAIRS = repo_path("artifacts/judge_v2/pairs.jsonl")
DEFAULT_MANUAL = repo_path("artifacts/judge_v2/calibration/manual.jsonl")
DEFAULT_LLM_DIR = repo_path("artifacts/judge_v2/calibration")
DEFAULT_JUDGES_CFG = repo_path("configs/judge_v2/judges.json")
DEFAULT_OUT = repo_path("artifacts/judge_v2/calibration/metrics.json")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _agreement_with_manual(
    judge_decisions: Dict[str, Dict[str, Any]],
    manual_by_pid: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    per_axis: Dict[str, Dict[str, int]] = {a: {"match": 0, "total": 0} for a in AXES}
    for pid, manual_labels in manual_by_pid.items():
        dec = judge_decisions.get(pid)
        if dec is None:
            continue
        for axis in AXES:
            man = manual_labels.get(axis)
            llm = axis_label(dec, axis)
            if man is None or llm is None:
                continue
            per_axis[axis]["total"] += 1
            if man == llm:
                per_axis[axis]["match"] += 1

    rates: Dict[str, Optional[float]] = {}
    for axis in AXES:
        t = per_axis[axis]["total"]
        rates[axis] = (per_axis[axis]["match"] / t) if t else None
    valid = [r for r in rates.values() if r is not None]
    mean = sum(valid) / len(valid) if valid else None
    return {
        "per_axis_rate": rates,
        "per_axis_counts": per_axis,
        "mean_rate": mean,
    }


def _inter_judge_agreement(
    decisions_by_judge: Dict[str, Dict[str, Dict[str, Any]]],
    pair_ids: List[str],
) -> Dict[str, Any]:
    judges = list(decisions_by_judge.keys())
    per_axis: Dict[str, Dict[str, int]] = {a: {"all_agree": 0, "total": 0} for a in AXES}
    for pid in pair_ids:
        for axis in AXES:
            vals: List[str] = []
            for j in judges:
                lab = axis_label(decisions_by_judge[j].get(pid, {}), axis)
                if lab is not None:
                    vals.append(lab)
            if len(vals) != len(judges):
                continue
            per_axis[axis]["total"] += 1
            if all(v == vals[0] for v in vals):
                per_axis[axis]["all_agree"] += 1
    rates = {a: (per_axis[a]["all_agree"] / per_axis[a]["total"] if per_axis[a]["total"] else None)
             for a in AXES}
    valid = [r for r in rates.values() if r is not None]
    return {
        "per_axis_rate": rates,
        "per_axis_counts": per_axis,
        "mean_rate": sum(valid) / len(valid) if valid else None,
    }


def _permutation_consistency(
    decisions: Dict[str, Dict[str, Any]],
    pairs_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """For each (comparison, qid, axis): get AB label and BA label from one
    judge, declare consistent iff BA == invert(AB) (treating equal == equal)."""
    by_cqa: Dict[Tuple[str, str, str], Dict[str, str]] = defaultdict(dict)
    for pid, dec in decisions.items():
        pair = pairs_index.get(pid)
        if pair is None:
            continue
        cmp_id = pair["comparison_id"]
        qid = pair["query_id"]
        perm = pair["perm"]
        for axis in AXES:
            lab = axis_label(dec, axis)
            if lab is None:
                continue
            by_cqa[(cmp_id, qid, axis)][perm] = lab

    per_axis: Dict[str, Dict[str, int]] = {a: {"consistent": 0, "total": 0} for a in AXES}
    for (cmp_id, qid, axis), per_perm in by_cqa.items():
        ab = per_perm.get("AB")
        ba = per_perm.get("BA")
        if ab is None or ba is None:
            continue
        per_axis[axis]["total"] += 1
        if ba == invert_label_ab(ab):
            per_axis[axis]["consistent"] += 1
    rates = {a: (per_axis[a]["consistent"] / per_axis[a]["total"] if per_axis[a]["total"] else None)
             for a in AXES}
    valid = [r for r in rates.values() if r is not None]
    return {
        "per_axis_rate": rates,
        "per_axis_counts": per_axis,
        "mean_rate": sum(valid) / len(valid) if valid else None,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute calibration metrics.")
    p.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    p.add_argument("--manual", type=Path, default=DEFAULT_MANUAL)
    p.add_argument("--llm_dir", type=Path, default=DEFAULT_LLM_DIR)
    p.add_argument("--judges_config", type=Path, default=DEFAULT_JUDGES_CFG)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Acceptance threshold for mean agreement with manual labels.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pairs_index = load_pairs_index(resolve_repo_path(args.pairs))
    manual_rows = read_jsonl(resolve_repo_path(args.manual))
    manual_by_pid: Dict[str, Dict[str, str]] = {}
    for r in manual_rows:
        pid = r.get("pair_id")
        labels = r.get("manual_labels") or {}
        if isinstance(pid, str) and isinstance(labels, dict):
            manual_by_pid[pid] = {k: v for k, v in labels.items() if k in AXES}

    cfg = _read_json(resolve_repo_path(args.judges_config))
    judge_names = [j["name"] for j in cfg["judges"]]
    calib_qids = set(cfg["calibration"]["query_ids"])
    calib_pids = [pid for pid, p in pairs_index.items() if p["query_id"] in calib_qids]

    llm_dir = resolve_repo_path(args.llm_dir)
    decisions_by_judge: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for jn in judge_names:
        path = llm_dir / f"llm_{jn}.jsonl"
        decisions_by_judge[jn] = load_decisions(path)

    judges_metrics: Dict[str, Any] = {}
    for jn in judge_names:
        agr = _agreement_with_manual(decisions_by_judge[jn], manual_by_pid)
        perm = _permutation_consistency(decisions_by_judge[jn], pairs_index)
        judges_metrics[jn] = {
            "agreement_with_manual": agr,
            "permutation_consistency": perm,
        }

    inter = _inter_judge_agreement(decisions_by_judge, calib_pids)

    n_pass = sum(
        1 for jn in judge_names
        if (judges_metrics[jn]["agreement_with_manual"]["mean_rate"] or 0.0) >= args.threshold
    )
    acceptance = {
        "threshold": args.threshold,
        "passing_judges": n_pass,
        "required": 2,
        "passed": n_pass >= 2,
    }

    out = {
        "calibration_query_ids": sorted(calib_qids),
        "n_calibration_pairs": len(calib_pids),
        "n_manual_labelled_pairs": sum(
            1 for v in manual_by_pid.values() if len(v) == len(AXES)
        ),
        "judges": judges_metrics,
        "inter_judge_agreement": inter,
        "acceptance": acceptance,
    }

    out_path = resolve_repo_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Metrics: {out_path}")
    print(f"  manual fully labelled: {out['n_manual_labelled_pairs']} / {out['n_calibration_pairs']}")
    for jn, jm in judges_metrics.items():
        mean = jm["agreement_with_manual"]["mean_rate"]
        perm_mean = jm["permutation_consistency"]["mean_rate"]
        print(
            f"  [{jn}] agreement-with-manual mean={mean if mean is None else round(mean, 3)} "
            f"perm-consistency mean={perm_mean if perm_mean is None else round(perm_mean, 3)}"
        )
    inter_mean = inter["mean_rate"]
    print(f"  inter-judge agreement mean: {inter_mean if inter_mean is None else round(inter_mean, 3)}")
    print(f"  ACCEPTANCE: {acceptance}")


if __name__ == "__main__":
    main()
