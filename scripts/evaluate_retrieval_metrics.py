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
    retrieved_nodes: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate retrieval runs against gold chunk ids.")
    p.add_argument(
        "--gold-path",
        type=Path,
        default=Path("data/eval/qasper_validation_gold.jsonl"),
        help="JSONL with fields: id, gold_chunk_ids",
    )
    p.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec NAME=DIR. Can be repeated. If omitted, 5 default runs are used.",
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="Cutoff for nDCG and context noise.",
    )
    p.add_argument(
        "--match-mode",
        type=str,
        choices=["strict", "same_section", "both"],
        default="strict",
        help="Matching criterion for relevance.",
    )
    p.add_argument(
        "--skip-empty-gold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip queries with empty gold_chunk_ids (use --no-skip-empty-gold to include them).",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts/reports/qasper_retrieval_metrics.json"),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("artifacts/reports/qasper_retrieval_metrics.csv"),
    )
    p.add_argument(
        "--diagnostic-sample-out",
        type=Path,
        default=None,
        help="Optional path to save small diagnostic sample jsonl.",
    )
    p.add_argument(
        "--diagnostic-sample-size",
        type=int,
        default=30,
        help="How many examples to save in diagnostic sample.",
    )
    return p.parse_args()


def default_runs() -> Dict[str, Path]:
    return {
        "bm25": Path("artifacts/bm25_rag_results/qasper_validation"),
        "bm25_heur": Path("artifacts/bm25_rag_heuristic_results/qasper_validation"),
        "dense": Path("artifacts/classic_rag_results/qasper_validation"),
        "dense_heur": Path("artifacts/classic_rag_heuristic_results/qasper_validation"),
        "ontology": Path("artifacts/swaga_rag_results/param_experiments/stable_baseline/qasper_validation"),
    }


def parse_runs(items: Sequence[str]) -> Dict[str, Path]:
    if not items:
        return default_runs()
    out: Dict[str, Path] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid --run '{raw}'. Expected NAME=DIR")
        name, path = raw.split("=", 1)
        name = name.strip()
        path = path.strip()
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
            qid = str(row["id"])
            ids = row.get("gold_chunk_ids") or []
            gold[qid] = [str(x) for x in ids if str(x).strip()]
    return gold


def load_predictions(run_dir: Path) -> Dict[str, List[str]]:
    pred: Dict[str, List[str]] = {}
    files = [p for p in run_dir.glob("*.json") if p.name.lower() != "config.json"]
    for fp in files:
        row = json.loads(fp.read_text(encoding="utf-8"))
        qid = str(row.get("id", fp.stem))

        output_ids = row.get("output_ids")
        if isinstance(output_ids, list):
            ids = [str(x) for x in output_ids if str(x).strip()]
        else:
            # Fallback: derive ids from output_items if needed
            items = row.get("output_items") or []
            ids = []
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict):
                        cid = it.get("chunk_id")
                        if isinstance(cid, str) and cid.strip():
                            ids.append(cid)
        pred[qid] = ids
    return pred


def section_key(chunk_id: str) -> str:
    cid = str(chunk_id)
    if cid.endswith("_abstract"):
        return cid[: -len("_abstract")] + ".abstract"
    m = re.match(r"^(.*)\.(\d+)\.(\d+)$", cid)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return cid


def convert_units(ids: List[str], match_mode: str) -> List[str]:
    if match_mode == "strict":
        return list(ids)
    if match_mode == "same_section":
        return [section_key(x) for x in ids]
    raise ValueError(f"Unknown match_mode: {match_mode}")


def matched_units(retrieved: List[str], gold_units: Set[str], k: int, match_mode: str) -> Set[str]:
    r_units = convert_units(retrieved[:k], match_mode)
    return set(r_units) & gold_units


def recall_at_k(retrieved: List[str], gold_units: Set[str], k: int, match_mode: str) -> float:
    if not gold_units:
        return 0.0
    hits = len(matched_units(retrieved, gold_units, k, match_mode))
    return hits / len(gold_units)


def mrr_score(retrieved: List[str], gold_units: Set[str], match_mode: str) -> float:
    if not gold_units:
        return 0.0
    for i, unit in enumerate(convert_units(retrieved, match_mode), start=1):
        if unit in gold_units:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: List[str], gold_units: Set[str], k: int, match_mode: str) -> float:
    if not gold_units:
        return 0.0
    dcg = 0.0
    for i, unit in enumerate(convert_units(retrieved, match_mode)[:k], start=1):
        rel = 1.0 if unit in gold_units else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 1)

    ideal_hits = min(len(gold_units), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def context_noise(retrieved: List[str], gold_units: Set[str], k: int, match_mode: str) -> float:
    rk = retrieved[:k]
    if not rk:
        return 0.0
    hits = len(matched_units(retrieved, gold_units, k, match_mode))
    return (len(rk) - hits) / len(rk)


def eval_query(retrieved: List[str], gold_ids: List[str], k: int, match_mode: str) -> QueryEval:
    gold_units = set(convert_units(gold_ids, match_mode))
    return QueryEval(
        recall_at_5=recall_at_k(retrieved, gold_units, 5, match_mode),
        recall_at_10=recall_at_k(retrieved, gold_units, 10, match_mode),
        mrr=mrr_score(retrieved, gold_units, match_mode),
        ndcg_at_10=ndcg_at_k(retrieved, gold_units, 10, match_mode),
        context_noise=context_noise(retrieved, gold_units, k, match_mode),
        retrieved_nodes=len(retrieved[:k]),
    )


def aggregate(scores: List[QueryEval]) -> Dict[str, float]:
    if not scores:
        return {
            "Recall@5": 0.0,
            "Recall@10": 0.0,
            "MRR": 0.0,
            "nDCG@10": 0.0,
            "context_noise": 0.0,
            "avg_retrieved_nodes": 0.0,
        }
    return {
        "Recall@5": mean(x.recall_at_5 for x in scores),
        "Recall@10": mean(x.recall_at_10 for x in scores),
        "MRR": mean(x.mrr for x in scores),
        "nDCG@10": mean(x.ndcg_at_10 for x in scores),
        "context_noise": mean(x.context_noise for x in scores),
        "avg_retrieved_nodes": mean(x.retrieved_nodes for x in scores),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    headers = [
        "method",
        "queries_evaluated",
        "Recall@5",
        "Recall@10",
        "MRR",
        "nDCG@10",
        "context_noise",
        "avg_retrieved_nodes",
        "missing_predictions",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(headers)]
    for r in rows:
        vals = [str(r.get(h, "")) for h in headers]
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    runs = parse_runs(args.run)
    gold = load_gold(args.gold_path)
    all_qids = list(gold.keys())

    if args.skip_empty_gold:
        eval_qids = [qid for qid in all_qids if len(gold.get(qid, [])) > 0]
    else:
        eval_qids = all_qids
    skipped_empty_gold = len(all_qids) - len(eval_qids)

    if args.match_mode == "both":
        match_modes = ["strict", "same_section"]
    else:
        match_modes = [args.match_mode]

    summary_rows: List[Dict[str, object]] = []
    by_method: Dict[str, Dict[str, Dict[str, object]]] = {}
    diag_rows: List[dict] = []

    for name, run_dir in runs.items():
        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {run_dir}")
        pred = load_predictions(run_dir)
        by_method[name] = {}

        for mode in match_modes:
            missing = 0
            per_query_scores: List[QueryEval] = []

            for qid in eval_qids:
                retrieved = pred.get(qid, [])
                if qid not in pred:
                    missing += 1
                per_query_scores.append(eval_query(retrieved, gold[qid], args.k, mode))

                if (
                    args.diagnostic_sample_out is not None
                    and len(diag_rows) < args.diagnostic_sample_size
                    and mode == "strict"
                ):
                    strict_hits = sorted(list(matched_units(retrieved, set(convert_units(gold[qid], "strict")), args.k, "strict")))
                    section_hits = sorted(
                        list(matched_units(retrieved, set(convert_units(gold[qid], "same_section")), args.k, "same_section"))
                    )
                    diag_rows.append(
                        {
                            "method": name,
                            "id": qid,
                            "gold_chunk_ids": gold[qid],
                            "retrieved_topk_ids": retrieved[: args.k],
                            "strict_hits": strict_hits,
                            "same_section_hits": section_hits,
                        }
                    )

            agg = aggregate(per_query_scores)
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
        "k_context_noise": args.k,
        "skip_empty_gold": args.skip_empty_gold,
        "skipped_empty_gold": skipped_empty_gold,
        "queries_total_in_gold": len(all_qids),
        "queries_evaluated": len(eval_qids),
        "match_mode": args.match_mode,
        "methods": by_method if args.match_mode == "both" else {m: by_method[m][match_modes[0]] for m in by_method},
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out_csv, summary_rows)
    if args.diagnostic_sample_out is not None:
        args.diagnostic_sample_out.parent.mkdir(parents=True, exist_ok=True)
        with args.diagnostic_sample_out.open("w", encoding="utf-8") as f:
            for row in diag_rows[: args.diagnostic_sample_size]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Gold total: {len(all_qids)}")
    print(f"Evaluated: {len(eval_qids)}")
    print(f"Skipped empty-gold: {skipped_empty_gold}")
    print(f"Match mode: {args.match_mode}")
    print("")
    for row in summary_rows:
        print(
            f"[{row['method']}/{row['match_mode']}] "
            f"R@5={row['Recall@5']:.4f} "
            f"R@10={row['Recall@10']:.4f} "
            f"MRR={row['MRR']:.4f} "
            f"nDCG@10={row['nDCG@10']:.4f} "
            f"noise={row['context_noise']:.4f} "
            f"avg_nodes={row['avg_retrieved_nodes']:.2f} "
            f"missing={row['missing_predictions']}"
        )
    print("")
    print(f"JSON: {args.out_json}")
    print(f"CSV:  {args.out_csv}")
    if args.diagnostic_sample_out is not None:
        print(f"Diagnostic sample: {args.diagnostic_sample_out}")


if __name__ == "__main__":
    main()

