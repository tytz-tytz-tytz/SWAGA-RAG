from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REF_TOKEN_RE = re.compile(r"\b(?:BIBREF|FIGREF|TABREF)\d+\b", flags=re.IGNORECASE)
WS_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9а-яё]+", flags=re.IGNORECASE)

# Evidence too short often produces noisy substring matches (e.g., "Languages").
# Keep exact normalized matching for all lengths, but gate substring fallback.
MIN_SUBSTRING_EVIDENCE_CHARS = 60
MIN_SUBSTRING_EVIDENCE_TOKENS = 5


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = REF_TOKEN_RE.sub(" ", text)
    text = text.lower()
    text = WS_RE.sub(" ", text).strip()
    return text


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text or ""))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build queries/gold files for QASPER validation retrieval evaluation."
    )
    p.add_argument(
        "--qasper-path",
        type=Path,
        default=Path("datasets/qasper/validation.jsonl"),
        help="Path to QASPER split jsonl.",
    )
    p.add_argument(
        "--nodes-path",
        type=Path,
        default=Path("data/processed/qasper_nodes.cleaned.json"),
        help="Optional path to qasper graph nodes for gold-id validation.",
    )
    p.add_argument(
        "--out-queries",
        type=Path,
        default=Path("data/eval/qasper_validation_queries.jsonl"),
        help="Output queries jsonl: id, query",
    )
    p.add_argument(
        "--out-gold",
        type=Path,
        default=Path("data/eval/qasper_validation_gold.jsonl"),
        help="Output gold jsonl with mapped chunk ids.",
    )
    p.add_argument(
        "--out-joined",
        type=Path,
        default=Path("data/eval/qasper_validation_joined.jsonl"),
        help="Output joined jsonl with query + gold in one line.",
    )
    return p.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_doc_chunks(doc: dict[str, Any]) -> list[tuple[str, str, str]]:
    """
    Returns list of tuples: (chunk_id, raw_text, normalized_text)
    Chunk IDs are built exactly as in scripts/build_qasper_graph.py.
    """
    doc_id = str(doc.get("id", "")).strip()
    chunks: list[tuple[str, str, str]] = []

    abstract = (doc.get("abstract") or "").strip()
    if abstract:
        cid = f"{doc_id}_abstract"
        chunks.append((cid, abstract, normalize_text(abstract)))

    full_text = doc.get("full_text") or {}
    section_names = full_text.get("section_name") or []
    paragraphs = full_text.get("paragraphs") or []
    sec_count = min(len(section_names), len(paragraphs))

    for sec_idx in range(sec_count):
        sec_paragraphs = paragraphs[sec_idx] if sec_idx < len(paragraphs) else []
        if not isinstance(sec_paragraphs, list):
            continue
        for para_idx, para in enumerate(sec_paragraphs):
            txt = (para or "").strip()
            if not txt:
                continue
            cid = f"{doc_id}.{sec_idx}.{para_idx}"
            chunks.append((cid, txt, normalize_text(txt)))

    return chunks


def collect_question_evidence_texts(q_answers_item: Any) -> list[str]:
    """
    Extract evidence strings for one QASPER question from all annotator answers.
    """
    if not isinstance(q_answers_item, dict):
        return []
    answers = q_answers_item.get("answer", [])
    if not isinstance(answers, list):
        return []

    out: list[str] = []
    seen: set[str] = set()
    for ans in answers:
        if not isinstance(ans, dict):
            continue
        ev = ans.get("evidence", [])
        if not isinstance(ev, list):
            continue
        for e in ev:
            txt = (str(e or "")).strip()
            if not txt:
                continue
            key = normalize_text(txt)
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(txt)
    return out


def map_evidence_to_chunks(
    evidence_texts: list[str],
    chunks: list[tuple[str, str, str]],
) -> tuple[list[str], list[str]]:
    """
    Returns:
      mapped_chunk_ids (deduplicated, stable order)
      unmapped_evidence_texts
    """
    chunk_ids: list[str] = []
    unmapped: list[str] = []
    seen_ids: set[str] = set()

    # Pre-index by normalized text for exact matching
    exact_map: dict[str, list[str]] = {}
    for cid, _, norm in chunks:
        exact_map.setdefault(norm, []).append(cid)

    for ev in evidence_texts:
        # Skip non-span placeholders that often point to figure/table metadata.
        if str(ev).strip().upper().startswith("FLOAT SELECTED:"):
            unmapped.append(ev)
            continue

        norm_ev = normalize_text(ev)
        if not norm_ev:
            continue

        matched_ids: list[str] = []

        # 1) Exact normalized equality
        if norm_ev in exact_map:
            matched_ids.extend(exact_map[norm_ev])
        else:
            # 2) Substring matches within same document chunks.
            #    This fallback is intentionally gated to reduce false positives
            #    from short evidence fragments.
            if (
                len(norm_ev) >= MIN_SUBSTRING_EVIDENCE_CHARS
                and token_count(norm_ev) >= MIN_SUBSTRING_EVIDENCE_TOKENS
            ):
                for cid, _, norm_chunk in chunks:
                    if not norm_chunk:
                        continue
                    if norm_ev in norm_chunk or norm_chunk in norm_ev:
                        matched_ids.append(cid)

        if not matched_ids:
            unmapped.append(ev)
            continue

        for cid in matched_ids:
            if cid not in seen_ids:
                seen_ids.add(cid)
                chunk_ids.append(cid)

    return chunk_ids, unmapped


def main() -> None:
    args = parse_args()
    if not args.qasper_path.exists():
        raise FileNotFoundError(f"QASPER file not found: {args.qasper_path}")

    valid_chunk_ids: set[str] | None = None
    if args.nodes_path.exists():
        with args.nodes_path.open("r", encoding="utf-8") as f:
            nodes = json.load(f)
        valid_chunk_ids = {
            str(n.get("id"))
            for n in nodes
            if isinstance(n, dict) and n.get("type") == "Chunk"
        }

    query_lines: list[dict[str, Any]] = []
    gold_lines: list[dict[str, Any]] = []
    joined_lines: list[dict[str, Any]] = []

    docs_scanned = 0
    questions_scanned = 0
    questions_with_gold = 0
    total_evidence_texts = 0
    total_unmapped_evidence_texts = 0
    missing_ids_vs_nodes = 0

    for doc in iter_jsonl(args.qasper_path):
        docs_scanned += 1
        doc_id = str(doc.get("id", "")).strip()
        chunks = build_doc_chunks(doc)

        qas = doc.get("qas", {}) if isinstance(doc.get("qas"), dict) else {}
        q_texts = qas.get("question", [])
        q_ids = qas.get("question_id", [])
        q_answers = qas.get("answers", [])

        q_count = min(len(q_texts), len(q_ids), len(q_answers))
        for i in range(q_count):
            questions_scanned += 1
            q_text = str(q_texts[i] or "").strip()
            qid_raw = str(q_ids[i] or "").strip()
            if not q_text or not qid_raw:
                continue

            ex_id = f"{doc_id}__{qid_raw}"
            evidence_texts = collect_question_evidence_texts(q_answers[i])
            total_evidence_texts += len(evidence_texts)
            gold_chunk_ids, unmapped = map_evidence_to_chunks(evidence_texts, chunks)
            total_unmapped_evidence_texts += len(unmapped)

            if gold_chunk_ids:
                questions_with_gold += 1

            if valid_chunk_ids is not None:
                miss = [cid for cid in gold_chunk_ids if cid not in valid_chunk_ids]
                missing_ids_vs_nodes += len(miss)

            query_obj = {"id": ex_id, "query": q_text}
            gold_obj = {
                "id": ex_id,
                "doc_id": doc_id,
                "question_id": qid_raw,
                "gold_chunk_ids": gold_chunk_ids,
                "evidence_texts": evidence_texts,
                "unmapped_evidence_texts": unmapped,
            }
            joined_obj = {**query_obj, **gold_obj}

            query_lines.append(query_obj)
            gold_lines.append(gold_obj)
            joined_lines.append(joined_obj)

    args.out_queries.parent.mkdir(parents=True, exist_ok=True)
    args.out_gold.parent.mkdir(parents=True, exist_ok=True)
    args.out_joined.parent.mkdir(parents=True, exist_ok=True)

    with args.out_queries.open("w", encoding="utf-8") as f:
        for row in query_lines:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with args.out_gold.open("w", encoding="utf-8") as f:
        for row in gold_lines:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with args.out_joined.open("w", encoding="utf-8") as f:
        for row in joined_lines:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Docs scanned: {docs_scanned}")
    print(f"Questions scanned: {questions_scanned}")
    print(f"Questions with >=1 mapped gold chunk: {questions_with_gold}")
    print(f"Total evidence strings: {total_evidence_texts}")
    print(f"Unmapped evidence strings: {total_unmapped_evidence_texts}")
    print(f"Queries file: {args.out_queries}")
    print(f"Gold file: {args.out_gold}")
    print(f"Joined file: {args.out_joined}")
    if valid_chunk_ids is not None:
        print(f"Missing mapped chunk ids vs nodes: {missing_ids_vs_nodes}")


if __name__ == "__main__":
    main()
