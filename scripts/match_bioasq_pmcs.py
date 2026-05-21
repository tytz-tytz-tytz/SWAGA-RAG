"""Match BioASQ PMIDs to PMCIDs and compute question-level PMC coverage stats."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT_PATH = Path("data/artifacts/bioasq_question_pmids.jsonl")
DEFAULT_MAPPING_OUTPUT_PATH = Path("data/artifacts/bioasq_pmid_pmcid.jsonl")
DEFAULT_COVERAGE_OUTPUT_PATH = Path("data/artifacts/bioasq_question_coverage.jsonl")
DEFAULT_LENIENT_OUTPUT_PATH = Path("data/artifacts/bioasq_questions_lenient.jsonl")
DEFAULT_STRICT_OUTPUT_PATH = Path("data/artifacts/bioasq_questions_strict.jsonl")

PMID_COLUMN_CANDIDATES = {"pmid", "pubmed_id", "pubmed"}
PMCID_COLUMN_CANDIDATES = {"pmcid", "pmc", "pmc_id"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match BioASQ PMIDs to PMCIDs using a local PMC mapping file."
    )
    parser.add_argument(
        "--pmc-map",
        type=Path,
        required=True,
        help="Path to a local CSV or TSV file containing PMID and PMCID columns.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the BioASQ long-format PMID JSONL file.",
    )
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=DEFAULT_MAPPING_OUTPUT_PATH,
        help="Path to write BioASQ PMID-to-PMCID matches as JSONL.",
    )
    parser.add_argument(
        "--coverage-output",
        type=Path,
        default=DEFAULT_COVERAGE_OUTPUT_PATH,
        help="Path to write question-level PMC coverage stats as JSONL.",
    )
    parser.add_argument(
        "--lenient-output",
        type=Path,
        default=DEFAULT_LENIENT_OUTPUT_PATH,
        help="Path to write leniently filtered questions as JSONL.",
    )
    parser.add_argument(
        "--strict-output",
        type=Path,
        default=DEFAULT_STRICT_OUTPUT_PATH,
        help="Path to write strictly filtered questions as JSONL.",
    )
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def open_text_maybe_gzip(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def normalize_identifier(value: object) -> str:
    return str(value or "").strip()


def normalize_pmcid(value: object) -> str:
    raw = normalize_identifier(value)
    if not raw:
        return ""
    upper_value = raw.upper()
    if upper_value.startswith("PMC"):
        suffix = upper_value[3:].strip()
        return f"PMC{suffix}" if suffix else ""
    return f"PMC{upper_value}"


def detect_csv_dialect(path: Path) -> csv.Dialect:
    with open_text_maybe_gzip(path) as handle:
        sample = handle.read(8192)
    if not sample.strip():
        raise ValueError(f"Mapping file is empty: {path}")
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t")
    except csv.Error:
        class SimpleDialect(csv.Dialect):
            delimiter = "\t" if "\t" in sample and sample.count("\t") >= sample.count(",") else ","
            quotechar = '"'
            doublequote = True
            skipinitialspace = False
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL

        return SimpleDialect()


def normalize_fieldnames(fieldnames: Iterable[str] | None) -> list[str]:
    if fieldnames is None:
        raise ValueError("Mapping file is missing a header row.")
    normalized = [normalize_identifier(name).lower() for name in fieldnames]
    if not any(normalized):
        raise ValueError("Mapping file header row is empty.")
    return normalized


def find_required_columns(fieldnames: list[str]) -> tuple[str, str]:
    pmid_column = ""
    pmcid_column = ""

    for fieldname in fieldnames:
        if not pmid_column and fieldname in PMID_COLUMN_CANDIDATES:
            pmid_column = fieldname
        if not pmcid_column and fieldname in PMCID_COLUMN_CANDIDATES:
            pmcid_column = fieldname

    if not pmid_column or not pmcid_column:
        raise ValueError(
            "Could not find PMID/PMCID columns in mapping file header. "
            f"Found columns: {fieldnames}"
        )

    return pmid_column, pmcid_column


def load_bioasq_question_pmids(
    path: Path,
) -> tuple[dict[str, dict[str, object]], set[str]]:
    questions: dict[str, dict[str, object]] = {}
    unique_pmids: set[str] = set()

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

            question_id = normalize_identifier(row.get("question_id"))
            question_type = normalize_identifier(row.get("type"))
            pmid = normalize_identifier(row.get("pmid"))

            if question_id not in questions:
                questions[question_id] = {
                    "question_id": question_id,
                    "type": question_type,
                    "pmids": set(),
                }

            question_record = questions[question_id]
            if not question_record["type"] and question_type:
                question_record["type"] = question_type

            if pmid:
                question_record["pmids"].add(pmid)
                unique_pmids.add(pmid)

    return questions, unique_pmids


def load_pmid_to_pmcid_map(path: Path, target_pmids: set[str]) -> dict[str, str]:
    dialect = detect_csv_dialect(path)
    pmid_to_pmcid: dict[str, str] = {}

    with open_text_maybe_gzip(path) as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        normalized_fieldnames = normalize_fieldnames(reader.fieldnames)
        reader.fieldnames = normalized_fieldnames
        pmid_column, pmcid_column = find_required_columns(normalized_fieldnames)

        for row in reader:
            if not row:
                continue

            pmid = normalize_identifier(row.get(pmid_column))
            if not pmid or pmid not in target_pmids or pmid in pmid_to_pmcid:
                continue

            pmcid = normalize_pmcid(row.get(pmcid_column))
            if not pmcid:
                continue

            pmid_to_pmcid[pmid] = pmcid

    return pmid_to_pmcid


def build_coverage_row(
    question_id: str,
    question_type: str,
    pmids: set[str],
    pmid_to_pmcid: dict[str, str],
) -> dict[str, object]:
    matched_pmcs = {pmid_to_pmcid[pmid] for pmid in pmids if pmid in pmid_to_pmcid}
    n_relevant_pmids = len(pmids)
    n_pmcs = len(matched_pmcs)
    coverage_ratio = (n_pmcs / n_relevant_pmids) if n_relevant_pmids else 0.0

    return {
        "question_id": question_id,
        "type": question_type,
        "n_relevant_pmids": n_relevant_pmids,
        "n_pmcs": n_pmcs,
        "coverage_ratio": coverage_ratio,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    questions, unique_pmids = load_bioasq_question_pmids(args.input)
    pmid_to_pmcid = load_pmid_to_pmcid_map(args.pmc_map, unique_pmids)

    mapping_rows = (
        {"pmid": pmid, "pmcid": pmid_to_pmcid[pmid]}
        for pmid in sorted(pmid_to_pmcid)
    )
    write_jsonl(args.mapping_output, mapping_rows)

    coverage_rows: list[dict[str, object]] = []
    lenient_rows: list[dict[str, object]] = []
    strict_rows: list[dict[str, object]] = []

    for question_id, question_record in questions.items():
        coverage_row = build_coverage_row(
            question_id=question_id,
            question_type=str(question_record["type"]),
            pmids=set(question_record["pmids"]),
            pmid_to_pmcid=pmid_to_pmcid,
        )
        coverage_rows.append(coverage_row)

        if int(coverage_row["n_pmcs"]) >= 1:
            lenient_rows.append(coverage_row)

        if (
            int(coverage_row["n_pmcs"]) >= 2
            or float(coverage_row["coverage_ratio"]) >= 0.5
        ):
            strict_rows.append(coverage_row)

    write_jsonl(args.coverage_output, coverage_rows)
    write_jsonl(args.lenient_output, lenient_rows)
    write_jsonl(args.strict_output, strict_rows)

    pmid_coverage_ratio = (
        len(pmid_to_pmcid) / len(unique_pmids) if unique_pmids else 0.0
    )

    print(f"Total unique PMIDs in BioASQ: {len(unique_pmids)}")
    print(f"PMIDs matched to PMCID: {len(pmid_to_pmcid)}")
    print(f"PMID coverage ratio: {pmid_coverage_ratio:.4f}")
    print(f"Total questions: {len(questions)}")
    print(f"Questions kept by lenient filter: {len(lenient_rows)}")
    print(f"Questions kept by strict filter: {len(strict_rows)}")


if __name__ == "__main__":
    main()
