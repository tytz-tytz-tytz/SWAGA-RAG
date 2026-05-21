from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Set, Tuple


WS_RE = re.compile(r"\s+")
REF_TOKEN_RE = re.compile(r"\b(?:BIBREF|FIGREF|TABREF)\d+\b", flags=re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9Р°-СЏС‘]+", flags=re.IGNORECASE)


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = REF_TOKEN_RE.sub(" ", text)
    text = text.lower()
    text = WS_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(normalize_text(text))


def section_key(chunk_id: str) -> str:
    cid = str(chunk_id)
    if cid.endswith("_abstract"):
        return cid[: -len("_abstract")] + ".abstract"
    m = re.match(r"^(.*)\.(\d+)\.(\d+)$", cid)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return cid


def doc_key(chunk_id: str) -> str:
    cid = str(chunk_id)
    if cid.endswith("_abstract"):
        return cid[: -len("_abstract")]
    m = re.match(r"^(.*)\.(\d+)\.(\d+)$", cid)
    if m:
        return m.group(1)
    return cid


@dataclass
class QueryContext:
    qid: str
    query: str
    doc_id: str
    gold_chunk_ids: List[str]
    evidence_texts: List[str]
    gold_chunk_texts: Dict[str, str]


@dataclass
class MetricRow:
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    context_noise: float
    avg_retrieved_nodes: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnostic audit for QASPER retrieval evaluation.")
    p.add_argument("--gold", type=Path, default=Path("data/eval/qasper_validation_gold.jsonl"))
    p.add_argument("--queries", type=Path, default=Path("data/eval/qasper_validation_queries.jsonl"))
    p.add_argument("--nodes", type=Path, default=Path("data/processed/qasper_nodes.cleaned.json"))
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--sample-size", type=int, default=25)
    p.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec NAME=DIR. Can repeat. If omitted, defaults to five methods.",
    )
    p.add_argument(
        "--sample-method",
        type=str,
        default="bm25",
        help="Method name to build detailed false-negative sample from.",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts/reports/qasper_eval_diagnostic.json"),
    )
    p.add_argument(
        "--out-sample-jsonl",
        type=Path,
        default=Path("artifacts/reports/qasper_eval_diagnostic_sample.jsonl"),
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


def parse_runs(items: List[str]) -> Dict[str, Path]:
    if not items:
        return default_runs()
    out: Dict[str, Path] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid --run '{raw}', expected NAME=DIR")
        name, d = raw.split("=", 1)
        out[name.strip()] = Path(d.strip())
    return out


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_qcontexts(gold_path: Path, queries_path: Path, nodes_path: Path) -> Dict[str, QueryContext]:
    q_map = {str(r["id"]): str(r["query"]) for r in load_jsonl(queries_path)}
    gold_rows = load_jsonl(gold_path)
    nodes = json.loads(nodes_path.read_text(encoding="utf-8"))
    chunk_text = {
        str(n.get("id")): str(n.get("text") or "")
        for n in nodes
        if isinstance(n, dict) and n.get("type") == "Chunk"
    }

    out: Dict[str, QueryContext] = {}
    for r in gold_rows:
        qid = str(r["id"])
        gcids = [str(x) for x in (r.get("gold_chunk_ids") or []) if str(x).strip()]
        if not gcids:
            continue
        gtexts = {cid: chunk_text.get(cid, "") for cid in gcids}
        out[qid] = QueryContext(
            qid=qid,
            query=q_map.get(qid, ""),
            doc_id=str(r.get("doc_id") or ""),
            gold_chunk_ids=gcids,
            evidence_texts=[str(x) for x in (r.get("evidence_texts") or []) if str(x).strip()],
            gold_chunk_texts=gtexts,
        )
    return out


def load_preds(run_dir: Path) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for fp in run_dir.glob("*.json"):
        if fp.name.lower() == "config.json":
            continue
        obj = json.loads(fp.read_text(encoding="utf-8"))
        qid = str(obj.get("id", fp.stem))
        items = obj.get("output_items") or []
        normalized_items: List[dict] = []
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                cid = str(it.get("chunk_id") or "").strip()
                if not cid:
                    continue
                normalized_items.append(
                    {
                        "chunk_id": cid,
                        "text": str(it.get("text") or ""),
                        "score": float(it.get("score", 0.0)),
                    }
                )
        out[qid] = normalized_items
    return out


def overlap_ratio(retr_text: str, ev_text: str) -> float:
    rt = set(tokenize(retr_text))
    et = set(tokenize(ev_text))
    if not rt or not et:
        return 0.0
    return len(rt & et) / len(et)


def build_matcher_units(ctx: QueryContext, criterion: str) -> List[str]:
    if criterion in {"strict_chunk", "source_id"}:
        return list(dict.fromkeys(ctx.gold_chunk_ids))
    if criterion == "same_section":
        return list(dict.fromkeys(section_key(x) for x in ctx.gold_chunk_ids))
    if criterion in {"contains_evidence", "text_overlap_30"}:
        norm_evs = [normalize_text(x) for x in ctx.evidence_texts if normalize_text(x)]
        return list(dict.fromkeys(norm_evs))
    if criterion == "same_doc":
        return [ctx.doc_id]
    raise ValueError(f"Unknown criterion: {criterion}")


def matched_units_for_item(ctx: QueryContext, item: dict, criterion: str) -> Set[str]:
    cid = str(item.get("chunk_id") or "")
    txt = str(item.get("text") or "")
    if criterion in {"strict_chunk", "source_id"}:
        return {cid} if cid in set(ctx.gold_chunk_ids) else set()
    if criterion == "same_section":
        ck = section_key(cid)
        gold_sec = {section_key(x) for x in ctx.gold_chunk_ids}
        return {ck} if ck in gold_sec else set()
    if criterion == "contains_evidence":
        n_txt = normalize_text(txt)
        out: Set[str] = set()
        for e in ctx.evidence_texts:
            n_e = normalize_text(e)
            if not n_e:
                continue
            if n_e in n_txt or n_txt in n_e:
                out.add(n_e)
        return out
    if criterion == "text_overlap_30":
        out: Set[str] = set()
        for e in ctx.evidence_texts:
            n_e = normalize_text(e)
            if not n_e:
                continue
            if overlap_ratio(txt, e) >= 0.30:
                out.add(n_e)
        return out
    if criterion == "same_doc":
        return {ctx.doc_id} if doc_key(cid) == ctx.doc_id else set()
    raise ValueError(f"Unknown criterion: {criterion}")


def eval_query(items: List[dict], ctx: QueryContext, criterion: str, k: int) -> Tuple[float, float, float, float, float, int]:
    gold_units = set(build_matcher_units(ctx, criterion))
    denom = len(gold_units)
    if denom == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0

    topk = items[:k]

    def recall_at(n: int) -> float:
        matched: Set[str] = set()
        for it in items[:n]:
            matched |= matched_units_for_item(ctx, it, criterion)
        return len(matched) / denom

    matched_seen: Set[str] = set()
    rr = 0.0
    gains: List[float] = []
    for rank, it in enumerate(topk, start=1):
        new_units = matched_units_for_item(ctx, it, criterion) - matched_seen
        if new_units and rr == 0.0:
            rr = 1.0 / rank
        gains.append(1.0 if new_units else 0.0)
        matched_seen |= new_units

    dcg = sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))
    ideal_len = min(denom, k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_len + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    rel_items = sum(1 for g in gains if g > 0.0)
    noise = (len(topk) - rel_items) / len(topk) if topk else 0.0
    return recall_at(5), recall_at(10), rr, ndcg, noise, len(topk)


def aggregate(rows: List[Tuple[float, float, float, float, float, int]]) -> MetricRow:
    return MetricRow(
        recall_at_5=mean(x[0] for x in rows) if rows else 0.0,
        recall_at_10=mean(x[1] for x in rows) if rows else 0.0,
        mrr=mean(x[2] for x in rows) if rows else 0.0,
        ndcg_at_10=mean(x[3] for x in rows) if rows else 0.0,
        context_noise=mean(x[4] for x in rows) if rows else 0.0,
        avg_retrieved_nodes=mean(x[5] for x in rows) if rows else 0.0,
    )


def main() -> None:
    args = parse_args()
    runs = parse_runs(args.run)
    qctx = load_qcontexts(args.gold, args.queries, args.nodes)
    qids = sorted(qctx.keys())

    criteria = [
        "strict_chunk",
        "source_id",
        "same_section",
        "contains_evidence",
        "text_overlap_30",
        "same_doc",
    ]

    results: Dict[str, Dict[str, dict]] = {}
    preds_by_method: Dict[str, Dict[str, List[dict]]] = {}

    for m, d in runs.items():
        if not d.exists():
            raise FileNotFoundError(f"Run dir not found: {d}")
        preds = load_preds(d)
        preds_by_method[m] = preds
        results[m] = {}
        for c in criteria:
            rows = [eval_query(preds.get(qid, []), qctx[qid], c, args.k) for qid in qids]
            agg = aggregate(rows)
            results[m][c] = {
                "Recall@5": round(agg.recall_at_5, 6),
                "Recall@10": round(agg.recall_at_10, 6),
                "MRR": round(agg.mrr, 6),
                "nDCG@10": round(agg.ndcg_at_10, 6),
                "context_noise": round(agg.context_noise, 6),
                "avg_retrieved_nodes": round(agg.avg_retrieved_nodes, 6),
            }

    # Build sample: strict false negatives that become positive under softer criteria.
    sample_method = args.sample_method
    if sample_method not in preds_by_method:
        raise ValueError(f"--sample-method '{sample_method}' is not in runs: {sorted(preds_by_method.keys())}")
    sample_preds = preds_by_method[sample_method]

    sample_rows: List[dict] = []
    for qid in qids:
        ctx = qctx[qid]
        items = sample_preds.get(qid, [])[:10]
        strict = eval_query(items, ctx, "strict_chunk", args.k)
        same_sec = eval_query(items, ctx, "same_section", args.k)
        overlap30 = eval_query(items, ctx, "text_overlap_30", args.k)
        if strict[1] == 0.0 and (same_sec[1] > 0.0 or overlap30[1] > 0.0):
            detailed_items = []
            for rank, it in enumerate(items, start=1):
                detailed_items.append(
                    {
                        "rank": rank,
                        "chunk_id": it["chunk_id"],
                        "score": it["score"],
                        "text": it["text"],
                        "strict_match": bool(matched_units_for_item(ctx, it, "strict_chunk")),
                        "same_section_match": bool(matched_units_for_item(ctx, it, "same_section")),
                        "contains_evidence_match": bool(matched_units_for_item(ctx, it, "contains_evidence")),
                        "overlap30_match": bool(matched_units_for_item(ctx, it, "text_overlap_30")),
                    }
                )

            sample_rows.append(
                {
                    "id": qid,
                    "query": ctx.query,
                    "doc_id": ctx.doc_id,
                    "gold_chunk_ids": ctx.gold_chunk_ids,
                    "gold_chunk_texts": ctx.gold_chunk_texts,
                    "evidence_texts": ctx.evidence_texts,
                    "metrics": {
                        "strict_chunk": {
                            "Recall@10": strict[1],
                            "MRR": strict[2],
                        },
                        "same_section": {
                            "Recall@10": same_sec[1],
                            "MRR": same_sec[2],
                        },
                        "text_overlap_30": {
                            "Recall@10": overlap30[1],
                            "MRR": overlap30[2],
                        },
                    },
                    "retrieved_top10": detailed_items,
                }
            )
            if len(sample_rows) >= args.sample_size:
                break

    out = {
        "gold_path": str(args.gold),
        "queries_path": str(args.queries),
        "nodes_path": str(args.nodes),
        "k": args.k,
        "queries_evaluated": len(qids),
        "runs": {k: str(v) for k, v in runs.items()},
        "criteria": criteria,
        "metrics": results,
        "sample_method": sample_method,
        "sample_size": len(sample_rows),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    args.out_sample_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_sample_jsonl.open("w", encoding="utf-8") as f:
        for r in sample_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Queries evaluated: {len(qids)}")
    print(f"Diagnostic JSON: {args.out_json}")
    print(f"Sample JSONL: {args.out_sample_jsonl} (rows={len(sample_rows)})")
    print("")
    for method, mdata in results.items():
        s = mdata["strict_chunk"]
        sec = mdata["same_section"]
        ov = mdata["text_overlap_30"]
        print(
            f"[{method}] strict R@10={s['Recall@10']:.4f}, "
            f"same_section R@10={sec['Recall@10']:.4f}, "
            f"overlap30 R@10={ov['Recall@10']:.4f}"
        )


if __name__ == "__main__":
    main()

