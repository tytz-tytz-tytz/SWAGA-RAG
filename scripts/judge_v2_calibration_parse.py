"""
Step 6 (parse): read the filled calibration markdown and emit manual.jsonl.

Each pair section is identified by the `## [i/N] pair_id` header. For each
axis (relevance / noise / sufficiency) the parser looks for the line that
contains a checked box (`[x]` or `[X]`) and extracts the choice (A / B /
equal). If exactly one box per axis is checked — label is recorded. If zero
or more than one — that pair is reported and skipped.

Output: artifacts/judge_v2/calibration/manual.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _repo_paths import repo_path, resolve_repo_path  # noqa: E402

from judge_v2_metrics_common import invert_label_ab  # noqa: E402


DEFAULT_IN = repo_path("artifacts/judge_v2/calibration/labeling.md")
DEFAULT_OUT = repo_path("artifacts/judge_v2/calibration/manual.jsonl")
DEFAULT_PAIRS = repo_path("artifacts/judge_v2/pairs.jsonl")

AXES = ("relevance", "cleanliness", "sufficiency")
CHOICES = ("A", "B", "equal")

PAIR_HEADER_RE = re.compile(r"^##\s+\[\d+/\d+\]\s+(?P<pid>\S+)\s*$")
AXIS_HEADER_RE = re.compile(r"^-\s+\*\*(?P<axis>relevance|cleanliness|sufficiency)\*\*")
# A "checked" line: a bullet under an axis that has [x] / [X].
CHECK_LINE_RE = re.compile(r"^\s*-\s*\[(?P<mark>[xX ])\]\s*(?P<choice>A|B|equal)\s*$")


def _parse_section(lines: List[str]) -> Dict[str, List[str]]:
    """Within one pair section, collect checked choices per axis."""
    result: Dict[str, List[str]] = {a: [] for a in AXES}
    current_axis: Optional[str] = None
    for ln in lines:
        m = AXIS_HEADER_RE.match(ln)
        if m:
            current_axis = m.group("axis")
            continue
        if current_axis is None:
            continue
        m2 = CHECK_LINE_RE.match(ln)
        if not m2:
            continue
        if m2.group("mark").lower() == "x":
            result[current_axis].append(m2.group("choice"))
    return result


def parse_markdown(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        text = f.read()
    lines = text.splitlines()

    # Find pair sections by header lines.
    sections: List[Dict[str, Any]] = []
    cur_pid: Optional[str] = None
    cur_lines: List[str] = []
    for ln in lines:
        m = PAIR_HEADER_RE.match(ln)
        if m:
            if cur_pid is not None:
                sections.append({"pair_id": cur_pid, "lines": cur_lines})
            cur_pid = m.group("pid")
            cur_lines = []
            continue
        if cur_pid is not None:
            cur_lines.append(ln)
    if cur_pid is not None:
        sections.append({"pair_id": cur_pid, "lines": cur_lines})

    results: List[Dict[str, Any]] = []
    for sec in sections:
        marks = _parse_section(sec["lines"])
        labels: Dict[str, str] = {}
        problems: List[str] = []
        for axis in AXES:
            chosen = marks[axis]
            if len(chosen) == 1:
                labels[axis] = chosen[0]
            elif len(chosen) == 0:
                problems.append(f"{axis}: no check")
            else:
                problems.append(f"{axis}: multiple checks {chosen}")
        results.append({
            "pair_id": sec["pair_id"],
            "manual_labels": labels,
            "problems": problems,
        })
    return results


def _load_pairs_index(pairs_path: Path):
    """Return {pair_id: pair_record} for fast lookup."""
    out = {}
    with pairs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["pair_id"]] = rec
    return out


def _find_counterpart_ba(pairs_index, ab_record):
    """For an AB pair, return the BA pair_id with matching comparison/query."""
    target = (ab_record["comparison_id"], ab_record["query_id"])
    for pid, rec in pairs_index.items():
        if rec["perm"] == "BA" and (rec["comparison_id"], rec["query_id"]) == target:
            return pid
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse calibration markdown to JSONL.")
    p.add_argument("--in_path", type=Path, default=DEFAULT_IN)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS,
                   help="pairs.jsonl — used to derive BA counterparts.")
    p.add_argument(
        "--no_auto_invert_ba",
        action="store_true",
        help="Disable auto-inversion of AB labels into BA records.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail (non-zero exit) if any pair has missing/duplicate marks.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = resolve_repo_path(args.in_path)
    out_path = resolve_repo_path(args.out)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    rows = parse_markdown(in_path)
    n_full = sum(1 for r in rows if len(r["manual_labels"]) == 3 and not r["problems"])
    n_partial = sum(1 for r in rows if r["problems"])

    pairs_index = None
    if not args.no_auto_invert_ba:
        pairs_path = resolve_repo_path(args.pairs)
        pairs_index = _load_pairs_index(pairs_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_inverted = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            out_rec = {"pair_id": r["pair_id"], "manual_labels": r["manual_labels"]}
            if r["problems"]:
                out_rec["problems"] = r["problems"]
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n_written += 1

            if pairs_index is None:
                continue
            ab_rec = pairs_index.get(r["pair_id"])
            if not ab_rec or ab_rec.get("perm") != "AB":
                continue
            if len(r["manual_labels"]) != 3 or r["problems"]:
                continue
            ba_pid = _find_counterpart_ba(pairs_index, ab_rec)
            if ba_pid is None:
                continue
            inverted = {ax: invert_label_ab(v) for ax, v in r["manual_labels"].items()}
            ba_rec = {
                "pair_id": ba_pid,
                "manual_labels": inverted,
                "derived_from": r["pair_id"],
                "derivation": "ab_inverted",
            }
            f.write(json.dumps(ba_rec, ensure_ascii=False) + "\n")
            n_written += 1
            n_inverted += 1

    print(f"[DONE] {len(rows)} markdown pairs parsed: {n_full} fully labelled, {n_partial} with problems")
    if pairs_index is not None:
        print(f"       Auto-derived {n_inverted} BA records by inversion")
    print(f"       Total records in {out_path}: {n_written}")
    if n_partial:
        print("       Pairs with problems:")
        for r in rows:
            if r["problems"]:
                print(f"         {r['pair_id']}: {'; '.join(r['problems'])}")
        if args.strict:
            sys.exit(1)


if __name__ == "__main__":
    main()
