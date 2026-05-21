from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare swaga-rag runs vs a chosen baseline by overlap of retrieved outputs "
            "(blind-safe Qxxx.json). Designed for param_experiments/<run_name>/ layout."
        )
    )
    p.add_argument(
        "--root_dir",
        type=Path,
        required=True,
        help="Root directory containing run subfolders, e.g. artifacts/swaga_rag_results/param_experiments",
    )
    p.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Baseline run folder name under root_dir, e.g. stable_baseline",
    )
    p.add_argument(
        "--k",
        type=int,
        default=20,
        help="Compute overlap@k and Jaccard@k using first k output chunks.",
    )
    p.add_argument(
        "--include_pairwise",
        action="store_true",
        help="Additionally compute pairwise overlaps between all runs (writes extra CSV/MD).",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Directory to write reports. Default: root_dir (reports are placed in root).",
    )
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="Optional comma-separated list of run folder names to compare (baseline always included).",
    )
    return p.parse_args()


def load_run_outputs(run_dir: Path) -> Dict[str, List[str]]:
    """
    Load {qid -> output_chunks} from a run directory.

    Expects files like Q001.json with {"id": ..., "output": [...] }.
    """
    outputs: Dict[str, List[str]] = {}
    for fp in sorted(run_dir.glob("Q*.json")):
        if fp.name.endswith(".debug.json"):
            continue
        try:
            with fp.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        qid = str(obj.get("id") or fp.stem)
        out = obj.get("output", [])
        if not isinstance(out, list):
            continue

        norm = [str(x).strip() for x in out if isinstance(x, str) and x.strip()]
        outputs[qid] = norm
    return outputs


def overlap_metrics(a: List[str], b: List[str], k: int) -> Tuple[float, float, int, int, int]:
    """
    overlap@k = |intersection(setA,setB)| / k
    jaccard@k = |intersection| / |union|
    Exact string matching. Uses top-k slices.
    Returns (overlap_at_k, jaccard_at_k, inter, union, k_eff)
    """
    if k <= 0:
        return 0.0, 0.0, 0, 0, 0
    A = a[:k]
    B = b[:k]
    setA = set(A)
    setB = set(B)
    inter = len(setA.intersection(setB))
    union = len(setA.union(setB))
    overlap_at_k = inter / k
    jaccard_at_k = (inter / union) if union else 0.0
    return overlap_at_k, jaccard_at_k, inter, union, k


def mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if isinstance(x, (int, float))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def median(xs: List[float]) -> Optional[float]:
    xs = sorted([x for x in xs if isinstance(x, (int, float))])
    if not xs:
        return None
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return float(xs[mid])
    return float((xs[mid - 1] + xs[mid]) / 2)


def discover_run_dirs(root_dir: Path) -> List[Path]:
    """
    A run dir is a direct child of root_dir that contains at least one Q*.json file.
    """
    runs: List[Path] = []
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("Q*.json")):
            runs.append(child)
    return runs


def write_markdown_vs_baseline(
    out_path: Path,
    root_dir: Path,
    baseline_name: str,
    k: int,
    run_summaries: List[Dict[str, Any]],
) -> None:
    def fmt(x: Any) -> str:
        if x is None:
            return "n/a"
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)

    lines: List[str] = []
    lines.append("# Overlap Report (vs baseline)\n\n")
    lines.append(f"Root: `{root_dir}`  \n")
    lines.append(f"Baseline: `{baseline_name}`  \n")
    lines.append(f"Metrics: overlap@{k}, Jaccard@{k} (exact string match on retrieved chunks)\n\n")

    lines.append("## Summary (each run compared to baseline)\n\n")
    lines.append("| Run | Common queries | Mean overlap@k | Median overlap@k | Mean Jaccard@k | Median Jaccard@k |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")

    # baseline row first if present
    run_summaries_sorted = sorted(run_summaries, key=lambda r: (r["run"] != baseline_name, r["run"]))

    for s in run_summaries_sorted:
        lines.append(
            f"| `{s['run']}` | {s['n_common_queries']} | "
            f"{fmt(s['mean_overlap_at_k'])} | {fmt(s['median_overlap_at_k'])} | "
            f"{fmt(s['mean_jaccard_at_k'])} | {fmt(s['median_jaccard_at_k'])} |\n"
        )

    lines.append("\n## Interpretation (diagnostic, not quality)\n")
    lines.append(
        f"- overlap@{k} close to **1.0** vs baseline в‡’ retrieval output is largely unchanged by that config.\n"
        f"- overlap@{k} noticeably lower в‡’ config changes what gets retrieved (structure/text weights likely matter).\n"
        f"- Jaccard@{k} helps distinguish whether differences are small reorderings vs truly different sets.\n"
    )

    out_path.write_text("".join(lines), encoding="utf-8")


def write_pairwise_markdown(
    out_path: Path,
    root_dir: Path,
    k: int,
    pair_summaries: List[Dict[str, Any]],
) -> None:
    def fmt(x: Any) -> str:
        if x is None:
            return "n/a"
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)

    lines: List[str] = []
    lines.append("# Overlap Report (pairwise)\n\n")
    lines.append(f"Root: `{root_dir}`  \n")
    lines.append(f"Metrics: overlap@{k}, Jaccard@{k}\n\n")

    lines.append("| Run A | Run B | Common queries | Mean overlap@k | Mean Jaccard@k |\n")
    lines.append("|---|---|---:|---:|---:|\n")
    for s in pair_summaries:
        lines.append(
            f"| `{s['run_a']}` | `{s['run_b']}` | {s['n_common_queries']} | "
            f"{fmt(s['mean_overlap_at_k'])} | {fmt(s['mean_jaccard_at_k'])} |\n"
        )

    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root_dir: Path = args.root_dir
    baseline_name: str = args.baseline
    k: int = args.k

    if not root_dir.exists():
        raise FileNotFoundError(f"Root dir not found: {root_dir}")

    out_dir = args.out_dir or root_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = root_dir / baseline_name
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline run dir not found: {baseline_dir}")

    # Discover runs (direct children)
    run_dirs = discover_run_dirs(root_dir)
    if not run_dirs:
        raise RuntimeError(f"No run dirs with Q*.json found under: {root_dir}")

    # Apply --only filter if provided (baseline always included)
    if args.only:
        allowed = {x.strip() for x in args.only.split(",") if x.strip()}
        allowed.add(baseline_name)
        run_dirs = [rd for rd in run_dirs if rd.name in allowed]

    # Load baseline
    baseline_outputs = load_run_outputs(baseline_dir)
    if not baseline_outputs:
        raise RuntimeError(f"No Qxxx.json outputs found in baseline: {baseline_dir}")

    # Compare each run vs baseline
    vs_rows: List[Dict[str, Any]] = []
    run_summaries: List[Dict[str, Any]] = []

    for rd in run_dirs:
        run_name = rd.name
        run_outputs = load_run_outputs(rd)
        if not run_outputs:
            print(f"[SKIP] No outputs in: {rd}")
            continue

        common_qids = sorted(set(baseline_outputs.keys()).intersection(run_outputs.keys()))
        if not common_qids:
            print(f"[SKIP] No common query IDs vs baseline for: {run_name}")
            continue

        overlaps: List[float] = []
        jaccs: List[float] = []

        for qid in common_qids:
            a = baseline_outputs.get(qid, [])
            b = run_outputs.get(qid, [])
            ov, jc, inter, union, k_eff = overlap_metrics(a, b, k)
            overlaps.append(ov)
            jaccs.append(jc)

            vs_rows.append({
                "qid": qid,
                "baseline": baseline_name,
                "run": run_name,
                "k": k_eff,
                "overlap_at_k": ov,
                "jaccard_at_k": jc,
                "intersect_size": inter,
                "union_size": union,
                "baseline_len": len(a),
                "run_len": len(b),
            })

        run_summaries.append({
            "run": run_name,
            "n_common_queries": len(common_qids),
            "mean_overlap_at_k": mean(overlaps),
            "median_overlap_at_k": median(overlaps),
            "mean_jaccard_at_k": mean(jaccs),
            "median_jaccard_at_k": median(jaccs),
        })

    if not vs_rows:
        raise RuntimeError("No comparable runs produced any rows (check baseline name and presence of Qxxx.json).")

    # Write VS baseline CSV/MD into out_dir (root param_experiments)
    csv_path = out_dir / f"overlap_vs_{baseline_name}_k{k}.csv"
    md_path = out_dir / f"overlap_vs_{baseline_name}_k{k}.md"

    cols = [
        "qid", "baseline", "run", "k",
        "overlap_at_k", "jaccard_at_k",
        "intersect_size", "union_size",
        "baseline_len", "run_len",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(vs_rows)

    write_markdown_vs_baseline(md_path, root_dir, baseline_name, k, run_summaries)

    # Optional: pairwise all runs
    if args.include_pairwise:
        # Load all runs once
        runs_loaded: List[Tuple[str, Dict[str, List[str]]]] = []
        for rd in run_dirs:
            data = load_run_outputs(rd)
            if data:
                runs_loaded.append((rd.name, data))

        pair_rows: List[Dict[str, Any]] = []
        pair_summaries: List[Dict[str, Any]] = []

        for i in range(len(runs_loaded)):
            for j in range(i + 1, len(runs_loaded)):
                run_a, data_a = runs_loaded[i]
                run_b, data_b = runs_loaded[j]
                common_qids = sorted(set(data_a.keys()).intersection(data_b.keys()))
                if not common_qids:
                    continue
                ovs: List[float] = []
                jcs: List[float] = []
                for qid in common_qids:
                    ov, jc, inter, union, k_eff = overlap_metrics(data_a.get(qid, []), data_b.get(qid, []), k)
                    ovs.append(ov)
                    jcs.append(jc)
                    pair_rows.append({
                        "qid": qid,
                        "run_a": run_a,
                        "run_b": run_b,
                        "k": k_eff,
                        "overlap_at_k": ov,
                        "jaccard_at_k": jc,
                    })
                pair_summaries.append({
                    "run_a": run_a,
                    "run_b": run_b,
                    "n_common_queries": len(common_qids),
                    "mean_overlap_at_k": mean(ovs),
                    "mean_jaccard_at_k": mean(jcs),
                })

        pair_csv = out_dir / f"overlap_pairwise_k{k}.csv"
        pair_md = out_dir / f"overlap_pairwise_k{k}.md"

        with pair_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["qid", "run_a", "run_b", "k", "overlap_at_k", "jaccard_at_k"])
            w.writeheader()
            w.writerows(pair_rows)

        write_pairwise_markdown(pair_md, root_dir, k, pair_summaries)

    # Console summary
    print(f"[OK] Baseline: {baseline_name}")
    print(f"[OK] Wrote: {csv_path}")
    print(f"[OK] Wrote: {md_path}")
    for s in sorted(run_summaries, key=lambda x: (x["run"] != baseline_name, x["run"])):
        mo = s["mean_overlap_at_k"]
        mj = s["mean_jaccard_at_k"]
        print(
            f"[RUN] {s['run']} vs {baseline_name}: "
            f"common_q={s['n_common_queries']}, "
            f"mean overlap@{k}={(mo if mo is not None else 0.0):.4f}, "
            f"mean jaccard@{k}={(mj if mj is not None else 0.0):.4f}"
        )


if __name__ == "__main__":
    main()

