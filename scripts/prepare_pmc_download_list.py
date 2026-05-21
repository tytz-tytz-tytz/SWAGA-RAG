"""Build PMCID download lists for filtered BioASQ question sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ARTIFACTS_DIR = Path("data/artifacts")
QUESTION_PMIDS_PATH = ARTIFACTS_DIR / "bioasq_question_pmids.jsonl"
PMID_PMCID_PATH = ARTIFACTS_DIR / "bioasq_pmid_pmcid.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare PMCID download lists for filtered BioASQ question sets."
    )
    parser.add_argument(
        "--filter-set",
        choices=("lenient", "strict"),
        required=True,
        help="Which filtered question set to use.",
    )
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_filtered_questions(path: Path) -> dict[str, str]:
    questions: dict[str, str] = {}
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
            question_id = str(row.get("question_id") or "").strip()
            question_type = str(row.get("type") or "").strip()
            if question_id:
                questions[question_id] = question_type
    return questions


def load_question_pmids(
    path: Path,
    selected_questions: dict[str, str],
) -> dict[str, dict[str, object]]:
    question_records: dict[str, dict[str, object]] = {
        question_id: {
            "question_id": question_id,
            "type": question_type,
            "pmids": set(),
        }
        for question_id, question_type in selected_questions.items()
    }

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

            question_id = str(row.get("question_id") or "").strip()
            if question_id not in question_records:
                continue

            question_type = str(row.get("type") or "").strip()
            pmid = str(row.get("pmid") or "").strip()
            record = question_records[question_id]

            if not record["type"] and question_type:
                record["type"] = question_type
            if pmid:
                record["pmids"].add(pmid)

    return question_records


def load_pmid_pmcid_map(path: Path) -> dict[str, str]:
    pmid_to_pmcid: dict[str, str] = {}
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

            pmid = str(row.get("pmid") or "").strip()
            pmcid = str(row.get("pmcid") or "").strip()
            if pmid and pmcid:
                pmid_to_pmcid[pmid] = pmcid

    return pmid_to_pmcid


def build_question_pmcid_rows(
    question_records: dict[str, dict[str, object]],
    pmid_to_pmcid: dict[str, str],
) -> tuple[list[dict[str, object]], set[str], set[str]]:
    rows: list[dict[str, object]] = []
    unique_pmids: set[str] = set()
    unique_pmcids: set[str] = set()

    for question_id, record in question_records.items():
        pmids = sorted(str(pmid) for pmid in record["pmids"])
        pmcids = sorted({pmid_to_pmcid[pmid] for pmid in pmids if pmid in pmid_to_pmcid})

        unique_pmids.update(pmids)
        unique_pmcids.update(pmcids)

        rows.append(
            {
                "question_id": question_id,
                "type": str(record["type"]),
                "pmids": pmids,
                "pmcids": pmcids,
                "n_pmids": len(pmids),
                "n_pmcids": len(pmcids),
            }
        )

    return rows, unique_pmids, unique_pmcids


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_pmcid_list(path: Path, pmcids: set[str]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for pmcid in sorted(pmcids):
            handle.write(f"{pmcid}\n")


def main() -> None:
    args = parse_args()

    filtered_questions_path = (
        ARTIFACTS_DIR / f"bioasq_questions_{args.filter_set}.jsonl"
    )
    pmcid_list_path = ARTIFACTS_DIR / f"pmcids_to_download_{args.filter_set}.txt"
    question_pmcids_path = (
        ARTIFACTS_DIR / f"bioasq_question_pmcids_{args.filter_set}.jsonl"
    )

    selected_questions = load_filtered_questions(filtered_questions_path)
    question_records = load_question_pmids(QUESTION_PMIDS_PATH, selected_questions)
    pmid_to_pmcid = load_pmid_pmcid_map(PMID_PMCID_PATH)

    rows, unique_pmids, unique_pmcids = build_question_pmcid_rows(
        question_records=question_records,
        pmid_to_pmcid=pmid_to_pmcid,
    )

    write_pmcid_list(pmcid_list_path, unique_pmcids)
    write_jsonl(question_pmcids_path, rows)

    print(f"Total filtered questions: {len(selected_questions)}")
    print(f"Total unique PMIDs in filtered set: {len(unique_pmids)}")
    print(f"Total unique PMCID in filtered set: {len(unique_pmcids)}")


if __name__ == "__main__":
    main()
