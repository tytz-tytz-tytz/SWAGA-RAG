from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List


@dataclass
class QueryEval:
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    context_noise: float
    retrieved_nodes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BioASQ retrieval runs against chunk-level gold annotations."
    )
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=Path("data/artifacts/bioasq_retrieval_eval.jsonl"),
        help="JSONL with fields: question_id, gold_chunk_ids",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/swaga_rag_results/bioasq_eval"),
        help="Directory with per-query retrieval outputs.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Cutoff for nDCG and context noise.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts/reports/bioasq_retrieval_metrics.json"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("artifacts/reports/bioasq_retrieval_metrics.csv"),
    )
    return parser.parse_args()


def load_gold(path: Path) -> Dict[str, List[str]]:
    gold: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_number} of {path}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object on line {line_number} of {path}")
            qid = str(row.get("question_id") or "").strip()
            ids = row.get("gold_chunk_ids") or []
            if not qid:
                continue
            gold[qid] = [str(x) for x in ids if str(x).strip()]
    return gold


def load_predictions(run_dir: Path) -> Dict[str, List[str]]:
    predictions: Dict[str, List[str]] = {}
    files = [path for path in run_dir.glob("*.json") if path.name.lower() != "config.json"]
    for path in files:
        row = json.loads(path.read_text(encoding="utf-8"))
        qid = str(row.get("id", path.stem))

        output_ids = row.get("output_ids")
        if isinstance(output_ids, list):
            ids = [str(x) for x in output_ids if str(x).strip()]
        else:
            items = row.get("output_items") or []
            ids = []
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        chunk_id = item.get("chunk_id")
                        if isinstance(chunk_id, str) and chunk_id.strip():
                            ids.append(chunk_id)
        predictions[qid] = ids
    return predictions


def matched_units(retrieved: List[str], gold_units: set[str], k: int) -> set[str]:
    return set(retrieved[:k]) & gold_units


def recall_at_k(retrieved: List[str], gold_units: set[str], k: int) -> float:
    if not gold_units:
        return 0.0
    return len(matched_units(retrieved, gold_units, k)) / len(gold_units)


def mrr_score(retrieved: List[str], gold_units: set[str]) -> float:
    if not gold_units:
        return 0.0
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold_units:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[str], gold_units: set[str], k: int) -> float:
    if not gold_units:
        return 0.0

    dcg = 0.0
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in gold_units:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(gold_units), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def context_noise(retrieved: List[str], gold_units: set[str], k: int) -> float:
    retrieved_topk = retrieved[:k]
    if not retrieved_topk:
        return 0.0
    hits = len(matched_units(retrieved, gold_units, k))
    return (len(retrieved_topk) - hits) / len(retrieved_topk)


def eval_query(retrieved: List[str], gold_ids: List[str], k: int) -> QueryEval:
    gold_units = set(gold_ids)
    return QueryEval(
        recall_at_5=recall_at_k(retrieved, gold_units, 5),
        recall_at_10=recall_at_k(retrieved, gold_units, 10),
        mrr=mrr_score(retrieved, gold_units),
        ndcg_at_10=ndcg_at_k(retrieved, gold_units, 10),
        context_noise=context_noise(retrieved, gold_units, k),
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
        "Recall@5": mean(score.recall_at_5 for score in scores),
        "Recall@10": mean(score.recall_at_10 for score in scores),
        "MRR": mean(score.mrr for score in scores),
        "nDCG@10": mean(score.ndcg_at_10 for score in scores),
        "context_noise": mean(score.context_noise for score in scores),
        "avg_retrieved_nodes": mean(score.retrieved_nodes for score in scores),
    }


def write_csv(path: Path, row: Dict[str, object]) -> None:
    headers = [
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
    lines = [",".join(headers), ",".join(str(row.get(header, "")) for header in headers)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not args.run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {args.run_dir}")

    gold = load_gold(args.gold_path)
    eval_qids = [qid for qid, gold_ids in gold.items() if gold_ids]
    predictions = load_predictions(args.run_dir)

    missing_predictions = 0
    per_query_scores: List[QueryEval] = []
    for qid in eval_qids:
        retrieved = predictions.get(qid, [])
        if qid not in predictions:
            missing_predictions += 1
        per_query_scores.append(eval_query(retrieved, gold[qid], args.k))

    aggregated = aggregate(per_query_scores)
    summary_row: Dict[str, object] = {
        "queries_evaluated": len(eval_qids),
        **aggregated,
        "missing_predictions": missing_predictions,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary_row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.out_csv, summary_row)

    print(f"Queries evaluated: {summary_row['queries_evaluated']}")
    print(f"Missing predictions: {summary_row['missing_predictions']}")
    print(f"Recall@5: {summary_row['Recall@5']:.4f}")
    print(f"Recall@10: {summary_row['Recall@10']:.4f}")
    print(f"MRR: {summary_row['MRR']:.4f}")
    print(f"nDCG@10: {summary_row['nDCG@10']:.4f}")
    print(f"Context noise: {summary_row['context_noise']:.4f}")
    print(f"Avg retrieved nodes: {summary_row['avg_retrieved_nodes']:.4f}")
    print(f"JSON report: {args.out_json}")
    print(f"CSV report: {args.out_csv}")


if __name__ == "__main__":
    main()

