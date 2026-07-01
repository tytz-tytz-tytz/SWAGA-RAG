"""Stage G (report): render the aggregate JSONs into a markdown report.

Reads artifacts/judge_v2/aggregates/{pair_aggregates,agreement_metrics,
operational_metrics}.json (produced by judge_v2_aggregate.py) and writes
report.md with ВКР-style Tables 7-9: per-comparison win/tie/loss by axis and
aggregated, inter-judge agreement, per-judge permutation consistency, and
operational/cost summary. No API calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _repo_paths import repo_path, resolve_repo_path  # noqa: E402

AXES = ("relevance", "cleanliness", "sufficiency")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pct(x: Any) -> str:
    return "—" if x is None else f"{float(x)*100:.0f}%"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render judge aggregates to markdown.")
    p.add_argument("--agg_dir", type=Path, default=repo_path("artifacts/judge_v2/aggregates"))
    p.add_argument("--out", type=Path, default=repo_path("artifacts/judge_v2/aggregates/report.md"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    agg_dir = resolve_repo_path(args.agg_dir)
    pair_agg = _read_json(agg_dir / "pair_aggregates.json")
    agree = _read_json(agg_dir / "agreement_metrics.json")
    oper = _read_json(agg_dir / "operational_metrics.json")

    L = []
    L.append("# LLM-as-judge — сводный отчёт (judge_v2)\n")

    # Table 7-8: per-comparison win/tie/loss by axis + aggregated
    L.append("## Таблица 7–8. Парные сравнения (win A / tie / win B)\n")
    L.append("| Сравнение (A vs B) | Ось | win A | tie | win B | n |")
    L.append("|---|---|---:|---:|---:|---:|")
    for cmp_id, p in pair_agg.items():
        label = f"{p['method_A']} vs {p['method_B']}"
        for axis in AXES:
            r = p[axis]
            L.append(f"| {label} | {axis} | {_pct(r['win_A'])} | {_pct(r['tie'])} | {_pct(r['win_B'])} | {r['n_queries']} |")
        o = p["overall"]
        L.append(f"| **{label}** | **overall** | **{_pct(o['win_A'])}** | **{_pct(o['tie'])}** | **{_pct(o['win_B'])}** | |")
    L.append("")

    # Table 9a: inter-judge agreement
    inter = agree.get("inter_judge_agreement", {})
    L.append("## Таблица 9. Согласованность судей\n")
    L.append("### Inter-judge agreement (доля пар, где все судьи согласны)\n")
    L.append("| Ось | rate |")
    L.append("|---|---:|")
    for axis in AXES:
        L.append(f"| {axis} | {_pct(inter.get('per_axis_rate', {}).get(axis))} |")
    L.append(f"| **mean** | **{_pct(inter.get('mean_rate'))}** |")
    L.append("")

    # Table 9b: permutation consistency per judge
    L.append("### Permutation consistency (BA == invert(AB)) по судьям\n")
    L.append("| Судья | relevance | cleanliness | sufficiency | mean |")
    L.append("|---|---:|---:|---:|---:|")
    for jn, pc in agree.get("permutation_consistency_per_judge", {}).items():
        r = pc.get("per_axis_rate", {})
        L.append(f"| {jn} | {_pct(r.get('relevance'))} | {_pct(r.get('cleanliness'))} | "
                 f"{_pct(r.get('sufficiency'))} | {_pct(pc.get('mean_rate'))} |")
    L.append("")

    # Operational / cost
    L.append("## Операционная сводка / стоимость\n")
    L.append("| Судья | calls | ok | failed | retry | in_tok | out_tok | $ est |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for jn, a in oper.get("per_judge", {}).items():
        L.append(f"| {jn} | {a['total_calls']} | {a['ok']} | {a['failed']} | {a['retry']} | "
                 f"{a['input_tokens']} | {a['output_tokens']} | ${a['cost_usd_estimated']:.3f} |")
    t = oper.get("totals", {})
    if t:
        L.append(f"| **TOTAL** | {t['total_calls']} | {t['ok']} | {t['failed']} | {t['retry']} | "
                 f"{t['input_tokens']} | {t['output_tokens']} | ${t['cost_usd_estimated']:.3f} |")
    L.append("")
    L.append(f"> {oper.get('cost_disclaimer', '')}\n")

    out_path = resolve_repo_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"[DONE] markdown report -> {out_path}")


if __name__ == "__main__":
    main()
