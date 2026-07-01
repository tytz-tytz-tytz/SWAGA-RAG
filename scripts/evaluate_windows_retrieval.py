"""Window-level retrieval evaluation (variant C) for SWAGA-RAG windowed runs.

Unlike the chunk-level evaluators (evaluate_retrieval_metrics.py /
evaluate_bioasq_retrieval.py), here the unit of retrieval is the *window*:

  - a window is relevant if any of its member chunks (window_node_ids) is in gold;
  - Recall@k = fraction of gold chunks covered by the union of the top-k windows;
  - MRR      = 1 / rank of the first relevant window;
  - nDCG@k   = windows as binary-relevant ranked items;
  - context_noise@k = share of top-k windows that do not overlap gold.

These numbers are NOT directly comparable with the chunk-level baselines
(BM25 / Dense / swaga_chunks): the unit of measurement differs. Use this as a
window-level diagnostic, alongside variant A (chunk-id expansion via the
existing chunk-level evaluators) for the comparable tables.

Predictions are read from per-query JSON files whose output_items carry the
window metadata produced by the windowed runners (window_node_ids).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Set


@dataclass
class QueryEval:
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    context_noise: float
    retrieved_windows: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Window-level (variant C) evaluation of SWAGA-RAG windowed runs."
    )
    p.add_argument(
        "--gold-path",
        type=Path,
        required=True,
        help="JSONL gold. Fields: (id | question_id) and gold_chunk_ids.",
    )
    p.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec NAME=DIR (per-query JSON with windowed output_items). Repeatable.",
    )
    p.add_argument("--k", type=int, default=10, help="Cutoff for nDCG and context noise.")
    p.add_argument(
        "--match-mode",
        type=str,
        choices=["strict", "same_section", "both"],
        default="strict",
        help="Chunk-id matching criterion (same_section is Qasper-specific).",
    )
    p.add_argument(
        "--skip-empty-gold",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts/reports/windows_retrieval_metrics.json"),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("artifacts/reports/windows_retrieval_metrics.csv"),
    )
    return p.parse_args()


def parse_runs(items: Sequence[str]) -> Dict[str, Path]:
    if not items:
        raise ValueError("At least one --run NAME=DIR is required.")
    out: Dict[str, Path] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid --run '{raw}'. Expected NAME=DIR")
        name, path = raw.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --run '{raw}'. Expected NAME=DIR")
        out[name] = Path(path)
    return out


def load_gold(path: Path) -> Dict[str, List[str]]:
    gold: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("id") or row.get("question_id") or "").strip()
            if not qid:
                continue
            ids = row.get("gold_chunk_ids") or []
            gold[qid] = [str(x) for x in ids if str(x).strip()]
    return gold


def load_window_predictions(run_dir: Path) -> Dict[str, List[List[str]]]:
    """Map qid -> ranked list of windows, each window a list of member chunk ids."""
    pred: Dict[str, List[List[str]]] = {}
    files = [p for p in run_dir.glob("*.json") if p.name.lower() != "config.json"]
    for fp in files:
        row = json.loads(fp.read_text(encoding="utf-8"))
        qid = str(row.get("id", fp.stem))
        items = row.get("output_items") or []
        windows: List[List[str]] = []
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                members = it.get("window_node_ids")
                if isinstance(members, list) and members:
                    win = [str(x) for x in members if str(x).strip()]
                else:
                    # Fall back to the single chunk id when window metadata is absent.
                    cid = it.get("chunk_id")
                    win = [str(cid)] if isinstance(cid, str) and cid.strip() else []
                if win:
                    windows.append(win)
        pred[qid] = windows
    return pred


def section_key(chunk_id: str) -> str:
    cid = str(chunk_id)
    if cid.endswith("_abstract"):
        return cid[: -len("_abstract")] + ".abstract"
    m = re.match(r"^(.*)\.(\d+)\.(\d+)$", cid)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return cid


def convert_units(ids: Sequence[str], match_mode: str) -> List[str]:
    if match_mode == "strict":
        return list(ids)
    if match_mode == "same_section":
        return [section_key(x) for x in ids]
    raise ValueError(f"Unknown match_mode: {match_mode}")


def window_units(window: List[str], match_mode: str) -> Set[str]:
    return set(convert_units(window, match_mode))


def is_relevant(window: List[str], gold_units: Set[str], match_mode: str) -> bool:
    return bool(window_units(window, match_mode) & gold_units)


def recall_at_k(windows: List[List[str]], gold_units: Set[str], k: int, match_mode: str) -> float:
    if not gold_units:
        return 0.0
    covered: Set[str] = set()
    for window in windows[:k]:
        covered |= window_units(window, match_mode) & gold_units
    return len(covered) / len(gold_units)


def mrr_score(windows: List[List[str]], gold_units: Set[str], match_mode: str) -> float:
    if not gold_units:
        return 0.0
    for rank, window in enumerate(windows, start=1):
        if is_relevant(window, gold_units, match_mode):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(windows: List[List[str]], gold_units: Set[str], k: int, match_mode: str) -> float:
    if not gold_units:
        return 0.0
    dcg = 0.0
    for rank, window in enumerate(windows[:k], start=1):
        if is_relevant(window, gold_units, match_mode):
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(gold_units), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def context_noise(windows: List[List[str]], gold_units: Set[str], k: int, match_mode: str) -> float:
    topk = windows[:k]
    if not topk:
        return 0.0
    relevant = sum(1 for w in topk if is_relevant(w, gold_units, match_mode))
    return (len(topk) - relevant) / len(topk)


def eval_query(windows: List[List[str]], gold_ids: List[str], k: int, match_mode: str) -> QueryEval:
    gold_units = set(convert_units(gold_ids, match_mode))
    return QueryEval(
        recall_at_5=recall_at_k(windows, gold_units, 5, match_mode),
        recall_at_10=recall_at_k(windows, gold_units, 10, match_mode),
        mrr=mrr_score(windows, gold_units, match_mode),
        ndcg_at_10=ndcg_at_k(windows, gold_units, 10, match_mode),
        context_noise=context_noise(windows, gold_units, k, match_mode),
        retrieved_windows=len(windows[:k]),
    )


def aggregate(scores: List[QueryEval]) -> Dict[str, float]:
    if not scores:
        return {
            "Recall@5": 0.0,
            "Recall@10": 0.0,
            "MRR": 0.0,
            "nDCG@10": 0.0,
            "context_noise": 0.0,
            "avg_retrieved_windows": 0.0,
        }
    return {
        "Recall@5": mean(x.recall_at_5 for x in scores),
        "Recall@10": mean(x.recall_at_10 for x in scores),
        "MRR": mean(x.mrr for x in scores),
        "nDCG@10": mean(x.ndcg_at_10 for x in scores),
        "context_noise": mean(x.context_noise for x in scores),
        "avg_retrieved_windows": mean(x.retrieved_windows for x in scores),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    headers = [
        "method",
        "match_mode",
        "queries_evaluated",
        "Recall@5",
        "Recall@10",
        "MRR",
        "nDCG@10",
        "context_noise",
        "avg_retrieved_windows",
        "missing_predictions",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(str(r.get(h, "")) for h in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    runs = parse_runs(args.run)
    gold = load_gold(args.gold_path)
    all_qids = list(gold.keys())
    eval_qids = (
        [q for q in all_qids if gold.get(q)] if args.skip_empty_gold else all_qids
    )
    skipped = len(all_qids) - len(eval_qids)

    match_modes = ["strict", "same_section"] if args.match_mode == "both" else [args.match_mode]

    summary_rows: List[Dict[str, object]] = []
    by_method: Dict[str, Dict[str, Dict[str, object]]] = {}

    for name, run_dir in runs.items():
        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {run_dir}")
        pred = load_window_predictions(run_dir)
        by_method[name] = {}
        for mode in match_modes:
            missing = 0
            per_query: List[QueryEval] = []
            for qid in eval_qids:
                windows = pred.get(qid, [])
                if qid not in pred:
                    missing += 1
                per_query.append(eval_query(windows, gold[qid], args.k, mode))
            agg = aggregate(per_query)
            row = {
                "method": name,
                "match_mode": mode,
                "queries_evaluated": len(eval_qids),
                **{k: round(v, 6) for k, v in agg.items()},
                "missing_predictions": missing,
            }
            summary_rows.append(row)
            by_method[name][mode] = row

    out = {
        "gold_path": str(args.gold_path),
        "level": "window",
        "k_context_noise": args.k,
        "skip_empty_gold": args.skip_empty_gold,
        "skipped_empty_gold": skipped,
        "queries_total_in_gold": len(all_qids),
        "queries_evaluated": len(eval_qids),
        "match_mode": args.match_mode,
        "methods": by_method
        if args.match_mode == "both"
        else {m: by_method[m][match_modes[0]] for m in by_method},
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out_csv, summary_rows)

    print(f"[window-level] Gold total: {len(all_qids)}  Evaluated: {len(eval_qids)}  Skipped empty: {skipped}")
    for row in summary_rows:
        print(
            f"[{row['method']}/{row['match_mode']}] "
            f"R@5={row['Recall@5']:.4f} R@10={row['Recall@10']:.4f} "
            f"MRR={row['MRR']:.4f} nDCG@10={row['nDCG@10']:.4f} "
            f"noise={row['context_noise']:.4f} avg_windows={row['avg_retrieved_windows']:.2f} "
            f"missing={row['missing_predictions']}"
        )
    print(f"JSON: {args.out_json}")
    print(f"CSV:  {args.out_csv}")


if __name__ == "__main__":
    main()
