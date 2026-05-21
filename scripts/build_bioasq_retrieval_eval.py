"""Build a question-level BioASQ retrieval evaluation dataset from chunk-level gold matches."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


DEFAULT_BIOASQ_PATH = Path("data/datasets/bioasq/bioasq12b_eval.jsonl")
FALLBACK_BIOASQ_PATH = Path("datasets/bioasq/bioasq12b_eval.jsonl")
MATCHES_PATH = Path("data/artifacts/bioasq_gold_chunk_matches.jsonl")
OUTPUT_PATH = Path("data/artifacts/bioasq_retrieval_eval.jsonl")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_bioasq_path(path: Path) -> Path:
    if path.exists():
        return path
    if path == DEFAULT_BIOASQ_PATH and FALLBACK_BIOASQ_PATH.exists():
        return FALLBACK_BIOASQ_PATH
    raise FileNotFoundError(f"BioASQ input file not found: {path}")


def load_gold_by_question(path: Path) -> dict[str, dict[str, set[str]]]:
    gold_by_question: dict[str, dict[str, set[str]]] = {}

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

            question_id = str(row.get("question_id") or "").strip()
            chunk_id = str(row.get("matched_chunk_id") or "").strip()
            pmcid = str(row.get("pmcid") or "").strip()

            if not question_id or not chunk_id:
                continue

            if question_id not in gold_by_question:
                gold_by_question[question_id] = {
                    "gold_chunk_ids": set(),
                    "gold_pmcids": set(),
                }

            gold_by_question[question_id]["gold_chunk_ids"].add(chunk_id)
            if pmcid:
                gold_by_question[question_id]["gold_pmcids"].add(pmcid)

    return gold_by_question


def main() -> None:
    bioasq_path = resolve_bioasq_path(DEFAULT_BIOASQ_PATH)
    gold_by_question = load_gold_by_question(MATCHES_PATH)

    ensure_parent_dir(OUTPUT_PATH)

    total_questions = 0
    kept_questions = 0
    total_gold_chunks = 0
    total_gold_pmcids = 0
    kept_type_counts: Counter[str] = Counter()

    with (
        bioasq_path.open("r", encoding="utf-8") as bioasq_handle,
        OUTPUT_PATH.open("w", encoding="utf-8") as output_handle,
    ):
        for line_number, line in enumerate(bioasq_handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue

            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_number} of {bioasq_path}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object on line {line_number} of {bioasq_path}"
                )

            total_questions += 1

            question_id = str(row.get("question_id") or "").strip()
            question = str(row.get("question") or "").strip()
            question_type = str(row.get("type") or "").strip()

            gold = gold_by_question.get(question_id)
            if not gold or not gold["gold_chunk_ids"]:
                continue

            gold_chunk_ids = sorted(gold["gold_chunk_ids"])
            gold_pmcids = sorted(gold["gold_pmcids"])

            output_row = {
                "question_id": question_id,
                "question": question,
                "type": question_type,
                "gold_chunk_ids": gold_chunk_ids,
                "gold_pmcids": gold_pmcids,
            }
            output_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")

            kept_questions += 1
            total_gold_chunks += len(gold_chunk_ids)
            total_gold_pmcids += len(gold_pmcids)
            kept_type_counts[question_type] += 1

    avg_gold_chunks = (total_gold_chunks / kept_questions) if kept_questions else 0.0
    avg_gold_pmcids = (total_gold_pmcids / kept_questions) if kept_questions else 0.0

    print(f"Total questions in BioASQ eval: {total_questions}")
    print(f"Questions with at least one gold chunk: {kept_questions}")
    print(f"Average number of gold chunks per kept question: {avg_gold_chunks:.4f}")
    print(f"Average number of gold PMCID per kept question: {avg_gold_pmcids:.4f}")
    print("Count by question type among kept questions:")
    for question_type in sorted(kept_type_counts):
        print(f"{question_type}: {kept_type_counts[question_type]}")


if __name__ == "__main__":
    main()
