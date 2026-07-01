"""Stage 4: consolidated encoder/first-stage/threshold summary tables.

Re-evaluates every run produced in the encoder-benchmark matrix (no retrieval
re-run) and writes one markdown table per corpus:
  Qasper  — strict ‖ same_section (Recall@5/10, MRR, nDCG@10)
  BioASQ  — snippet-level (Recall@5/10, MRR, nDCG@10, context noise)
Each row is labelled with encoder / first-stage / threshold / variant.
Missing run dirs are skipped with a note. No API calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _repo_paths import repo_path  # noqa: E402
import evaluate_retrieval_metrics as qe  # noqa: E402
import evaluate_bioasq_retrieval as be  # noqa: E402

MAIN = Path("C:/Users/alexs/spbu/SWAGA-RAG")
HR = MAIN / "artifacts/hybrid_rag_results"
CR = MAIN / "artifacts/classic_rag_results"
BR = MAIN / "artifacts/bm25_rag_results"
SR = MAIN / "artifacts/swaga_rag_results"

QGOLD = MAIN / "data/eval/qasper_validation_gold.jsonl"
BGOLD = MAIN / "data/artifacts/bioasq_retrieval_eval.jsonl"

# label, run-dir
QASPER = [
    ("BM25 (lexical)", BR / "qasper_bm25_sync"),
    ("dense · e5", CR / "qasper_e5_dense"),
    ("hybrid chunks · mpnet · bm25-first m5", HR / "qasper_chunks"),
    ("hybrid chunks · bge · bm25-first m5", HR / "qasper_bge_chunks"),
    ("hybrid chunks · e5 · bm25-first m5", HR / "qasper_e5_chunks"),
    ("hybrid chunks · e5 · bm25-first m10", HR / "qasper_e5_bm25_m10"),
    ("hybrid chunks · e5 · bm25-first m20", HR / "qasper_e5_bm25_m20"),
    ("hybrid chunks · e5 · dense-first m5", HR / "qasper_e5_dense_m5"),
    ("hybrid chunks · e5 · dense-first m10", HR / "qasper_e5_dense_m10"),
    ("hybrid chunks · e5 · dense-first m20", HR / "qasper_e5_dense_m20"),
    ("hybrid windows · e5 · bm25-first m5", HR / "qasper_e5_windows"),
]

BIOASQ = [
    ("BM25 (lexical)", BR / "bioasq_bm25_sync"),
    ("dense · bge", CR / "bioasq_bge_dense"),
    ("hybrid chunks · mpnet · bm25-first m5", SR / "bioasq_chunks"),
    ("hybrid chunks · pubmedbert · bm25-first m5", SR / "bioasq_pubmedbert"),
    ("hybrid chunks · bge · bm25-first m5", SR / "bioasq_bge"),
    ("hybrid chunks · bge · bm25-first m10", SR / "bioasq_bge_bm25_m10"),
    ("hybrid chunks · bge · bm25-first m20", SR / "bioasq_bge_bm25_m20"),
    ("hybrid chunks · bge · dense-first m5", SR / "bioasq_bge_dense_m5"),
    ("hybrid chunks · bge · dense-first m10", SR / "bioasq_bge_dense_m10"),
    ("hybrid chunks · bge · dense-first m20", SR / "bioasq_bge_dense_m20"),
    ("hybrid windows · bge · dense-first m5", SR / "bioasq_bge_dense_m5_windows"),
]


def qasper_rows():
    gold = qe.load_gold(QGOLD)
    qids = [q for q in gold if gold.get(q)]
    rows = []
    for label, d in QASPER:
        if not d.exists():
            rows.append((label, None)); continue
        pred = qe.load_predictions(d)
        out = {}
        for mode in ("strict", "same_section"):
            agg = qe.aggregate([qe.eval_query(pred.get(q, []), gold[q], 10, mode) for q in qids])
            out[mode] = agg
        rows.append((label, out))
    return rows


def bioasq_rows():
    gold = be.load_gold(BGOLD)
    qids = [q for q, g in gold.items() if g]
    rows = []
    for label, d in BIOASQ:
        if not d.exists():
            rows.append((label, None)); continue
        pred = be.load_predictions(d)
        agg = be.aggregate([be.eval_query(pred.get(q, []), gold[q], 10) for q in qids])
        rows.append((label, agg))
    return rows


def f(x):
    return "—" if x is None else f"{x:.3f}"


def main():
    L = ["# Этап 3–4 — сводная таблица retrieval (encoder / first-stage / threshold)\n"]

    L.append("## Qasper (888 запросов; strict ‖ same_section)\n")
    L.append("| Метод | s R@5 | s R@10 | s MRR | s nDCG | ss R@5 | ss R@10 | ss MRR | ss nDCG |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for label, out in qasper_rows():
        if out is None:
            L.append(f"| {label} | — | — | — | — | — | — | — | — |"); continue
        s, ss = out["strict"], out["same_section"]
        L.append(f"| {label} | {f(s['Recall@5'])} | {f(s['Recall@10'])} | {f(s['MRR'])} | {f(s['nDCG@10'])} "
                 f"| {f(ss['Recall@5'])} | {f(ss['Recall@10'])} | {f(ss['MRR'])} | {f(ss['nDCG@10'])} |")
    L.append("")

    L.append("## BioASQ (280 вопросов; snippet-level strict)\n")
    L.append("| Метод | R@5 | R@10 | MRR | nDCG@10 | noise |")
    L.append("|---|--:|--:|--:|--:|--:|")
    for label, agg in bioasq_rows():
        if agg is None:
            L.append(f"| {label} | — | — | — | — | — |"); continue
        L.append(f"| {label} | {f(agg['Recall@5'])} | {f(agg['Recall@10'])} | {f(agg['MRR'])} "
                 f"| {f(agg['nDCG@10'])} | {f(agg['context_noise'])} |")
    L.append("")

    L.append("_Примечания: dense = classic_rag (тот же энкодер, что SWAGA-индекс). "
             "hybrid = BM25/dense first-stage doc-recall + SWAGA in-doc localization. "
             "windows разворачиваются в chunk-id (вариант A). threshold_mode инертен на "
             "канонической конфигурации (см. отдельную сводку)._\n")

    out = repo_path("artifacts/reports/stage3_summary.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"[DONE] {out}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
