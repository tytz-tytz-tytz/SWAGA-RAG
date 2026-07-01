"""Stage D: calibration agreement vs expert labels. No API calls.

After you run the 3 judges over pairs_calib.jsonl (writing
artifacts/judge_v2/calib_run/llm_{judge}.jsonl), this computes — like ВКР
Tables 5-6 — :

  (1) per-judge per-axis agreement with the expert (manual_calib.jsonl);
  (2) majority-aggregation (>=4 of 6 = 3 judges x 2 perms) agreement with the
      expert, per axis;
  (3) an explicit cleanliness-axis inversion check per judge (direct vs inverted
      agreement) — catches a judge that systematically flips the "cleaner" side.

Expert labels were re-keyed onto the current pair_ids by judge_v2_stage.py.
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
    AXES, axis_label, invert_label_ab, load_decisions, load_pairs_index, normalize_label,
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibration agreement vs expert (Tables 5-6).")
    p.add_argument("--pairs", type=Path, default=repo_path("artifacts/judge_v2/pairs_calib.jsonl"))
    p.add_argument("--manual", type=Path, default=repo_path("artifacts/judge_v2/calibration/manual_calib.jsonl"))
    p.add_argument("--calib_dir", type=Path, default=repo_path("artifacts/judge_v2/calib_run"))
    p.add_argument("--judges_config", type=Path, default=repo_path("configs/judge_v2/judges.json"))
    p.add_argument("--out", type=Path, default=repo_path("artifacts/judge_v2/calibration/agreement_calib.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pairs_index = load_pairs_index(resolve_repo_path(args.pairs))
    manual = {r["pair_id"]: r["manual_labels"] for r in _read_jsonl(resolve_repo_path(args.manual))}
    cfg = _read_json(resolve_repo_path(args.judges_config))
    judge_names = [j["name"] for j in cfg["judges"]]
    calib_dir = resolve_repo_path(args.calib_dir)

    decisions: Dict[str, Dict[str, Dict[str, Any]]] = {}
    missing_files = []
    for jn in judge_names:
        fp = calib_dir / f"llm_{jn}.jsonl"
        if not fp.exists():
            missing_files.append(str(fp))
            decisions[jn] = {}
        else:
            decisions[jn] = load_decisions(fp)
    if missing_files:
        print("[WARN] judge outputs not found yet (run pairs_calib first):")
        for m in missing_files:
            print("   -", m)

    # (1) per-judge per-axis agreement with expert (raw labels, same perm frame)
    per_judge: Dict[str, Any] = {}
    for jn in judge_names:
        per_axis = {a: {"match": 0, "total": 0} for a in AXES}
        # cleanliness inversion bookkeeping
        clean_inv = {"direct": 0, "inverted": 0, "total": 0}
        for pid, man in manual.items():
            dec = decisions[jn].get(pid)
            if dec is None or dec.get("status") != "ok":
                continue
            for axis in AXES:
                exp = man.get(axis)
                got = axis_label(dec, axis)
                if exp is None or got is None:
                    continue
                per_axis[axis]["total"] += 1
                if got == exp:
                    per_axis[axis]["match"] += 1
                if axis == "cleanliness":
                    clean_inv["total"] += 1
                    if got == exp:
                        clean_inv["direct"] += 1
                    if got == invert_label_ab(exp):
                        clean_inv["inverted"] += 1
        rates = {a: (per_axis[a]["match"] / per_axis[a]["total"] if per_axis[a]["total"] else None) for a in AXES}
        valid = [v for v in rates.values() if v is not None]
        direct = clean_inv["direct"] / clean_inv["total"] if clean_inv["total"] else None
        inverted = clean_inv["inverted"] / clean_inv["total"] if clean_inv["total"] else None
        flag = bool(direct is not None and inverted is not None and inverted > direct and direct < 0.5)
        per_judge[jn] = {
            "per_axis_agreement": rates,
            "mean_agreement": sum(valid) / len(valid) if valid else None,
            "cleanliness_inversion": {
                "direct_rate": direct, "inverted_rate": inverted, "inverted_suspected": flag,
            },
        }

    # (2) majority (>=4/6) agreement with expert, per axis
    # group judge votes + expert label by (comparison, query, axis)
    votes: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    expert_dec: Dict[Tuple[str, str, str], str] = {}
    for pid, pair in pairs_index.items():
        cmp_id, qid, perm = pair["comparison_id"], pair["query_id"], pair["perm"]
        for jn in judge_names:
            dec = decisions[jn].get(pid)
            if dec is None or dec.get("status") != "ok":
                continue
            for axis in AXES:
                lab = axis_label(dec, axis)
                norm = normalize_label(lab, perm) if lab else None
                if norm is not None:
                    votes[(cmp_id, qid, axis)].append(norm)
        # expert: take the AB-perm label as canonical decision
        man = manual.get(pid)
        if man and perm == "AB":
            for axis in AXES:
                e = normalize_label(man.get(axis), "AB") if man.get(axis) else None
                if e is not None:
                    expert_dec[(cmp_id, qid, axis)] = e

    def _majority(vs: List[str]) -> str:
        f = sum(1 for v in vs if v == "first"); s = sum(1 for v in vs if v == "second")
        if f >= 4:
            return "first"
        if s >= 4:
            return "second"
        return "tie"

    maj_axis = {a: {"match": 0, "total": 0} for a in AXES}
    for (cmp_id, qid, axis), vs in votes.items():
        exp = expert_dec.get((cmp_id, qid, axis))
        if exp is None or len(vs) < 6:  # need full panel (3 judges x 2 perms)
            continue
        maj_axis[axis]["total"] += 1
        if _majority(vs) == exp:
            maj_axis[axis]["match"] += 1
    maj_rates = {a: (maj_axis[a]["match"] / maj_axis[a]["total"] if maj_axis[a]["total"] else None) for a in AXES}
    maj_valid = [v for v in maj_rates.values() if v is not None]

    out = {
        "n_pairs_with_expert": len(manual),
        "judges": judge_names,
        "per_judge_vs_expert": per_judge,
        "majority_vs_expert": {
            "per_axis_agreement": maj_rates,
            "per_axis_counts": maj_axis,
            "mean_agreement": sum(maj_valid) / len(maj_valid) if maj_valid else None,
        },
    }
    out_path = resolve_repo_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n=== Calibration agreement (vs expert) ===")
    for jn in judge_names:
        pj = per_judge[jn]
        ci = pj["cleanliness_inversion"]
        print(f"  {jn}: mean={pj['mean_agreement']} per_axis={pj['per_axis_agreement']}")
        print(f"      cleanliness inversion: direct={ci['direct_rate']} inverted={ci['inverted_rate']} "
              f"SUSPECTED={ci['inverted_suspected']}")
    print(f"  MAJORITY vs expert: mean={out['majority_vs_expert']['mean_agreement']} "
          f"per_axis={maj_rates}")
    print(f"[DONE] {out_path}")


if __name__ == "__main__":
    main()
