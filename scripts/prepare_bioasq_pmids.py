"""Extract BioASQ evaluation question-to-PMID mappings and summary statistics."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path


DEFAULT_INPUT_PATH = Path("data/datasets/bioasq/bioasq12b_eval.jsonl")
FALLBACK_INPUT_PATH = Path("datasets/bioasq/bioasq12b_eval.jsonl")
DEFAULT_LONG_OUTPUT_PATH = Path("data/artifacts/bioasq_question_pmids.jsonl")
DEFAULT_STATS_OUTPUT_PATH = Path("data/artifacts/bioasq_question_stats.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare BioASQ evaluation question-to-PMID mappings."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the BioASQ evaluation JSONL file.",
    )
    parser.add_argument(
        "--long-output",
        type=Path,
        default=DEFAULT_LONG_OUTPUT_PATH,
        help="Path to write question-to-PMID rows as JSONL.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=DEFAULT_STATS_OUTPUT_PATH,
        help="Path to write per-question PMID counts as JSONL.",
    )
    return parser.parse_args()


def resolve_input_path(path: Path) -> Path:
    if path.exists():
        return path
    if path == DEFAULT_INPUT_PATH and FALLBACK_INPUT_PATH.exists():
        return FALLBACK_INPUT_PATH
    raise FileNotFoundError(f"Input file not found: {path}")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_pmids(raw_pmids: object) -> list[str]:
    if not isinstance(raw_pmids, list):
        return []

    unique_pmids: OrderedDict[str, None] = OrderedDict()
    for value in raw_pmids:
        pmid = str(value).strip()
        if pmid:
            unique_pmids[pmid] = None
    return list(unique_pmids)


def iter_question_rows(path: Path):
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
                raise ValueError(
                    f"Expected a JSON object on line {line_number} of {path}"
                )
            yield row


def write_outputs(
    input_path: Path,
    long_output_path: Path,
    stats_output_path: Path,
) -> tuple[int, int, int]:
    ensure_parent_dir(long_output_path)
    ensure_parent_dir(stats_output_path)

    total_questions = 0
    total_pairs = 0
    unique_pmids: set[str] = set()

    with (
        long_output_path.open("w", encoding="utf-8") as long_handle,
        stats_output_path.open("w", encoding="utf-8") as stats_handle,
    ):
        for row in iter_question_rows(input_path):
            question_id = str(row.get("question_id") or "").strip()
            question_type = str(row.get("type") or "").strip()
            pmids = normalize_pmids(row.get("relevant_passage_ids"))

            total_questions += 1
            total_pairs += len(pmids)
            unique_pmids.update(pmids)

            for pmid in pmids:
                long_row = {
                    "question_id": question_id,
                    "type": question_type,
                    "pmid": pmid,
                }
                long_handle.write(json.dumps(long_row, ensure_ascii=False) + "\n")

            stats_row = {
                "question_id": question_id,
                "type": question_type,
                "n_relevant_pmids": len(pmids),
            }
            stats_handle.write(json.dumps(stats_row, ensure_ascii=False) + "\n")

    return total_questions, total_pairs, len(unique_pmids)


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input)
    total_questions, total_pairs, unique_pmid_count = write_outputs(
        input_path=input_path,
        long_output_path=args.long_output,
        stats_output_path=args.stats_output,
    )

    print(f"Total questions: {total_questions}")
    print(f"Total (question, pmid) pairs: {total_pairs}")
    print(f"Unique PMIDs: {unique_pmid_count}")


if __name__ == "__main__":
    main()
