"""Match BioASQ evidence snippets to structured PMC chunks to build gold chunk annotations."""

from __future__ import annotations

import json
import re
from pathlib import Path


DEFAULT_BIOASQ_PATH = Path("data/datasets/bioasq/bioasq12b_eval.jsonl")
FALLBACK_BIOASQ_PATH = Path("datasets/bioasq/bioasq12b_eval.jsonl")
PMID_PMCID_PATH = Path("data/artifacts/bioasq_pmid_pmcid.jsonl")
CHUNKS_PATH = Path("data/artifacts/pmc_structured_chunks.jsonl")
MATCHES_PATH = Path("data/artifacts/bioasq_gold_chunk_matches.jsonl")
UNMATCHED_PATH = Path("data/artifacts/bioasq_unmatched_snippets.jsonl")

WHITESPACE_RE = re.compile(r"\s+")
PMID_RE = re.compile(r"(\d+)(?:[/?#].*)?$")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_bioasq_path(path: Path) -> Path:
    if path.exists():
        return path
    if path == DEFAULT_BIOASQ_PATH and FALLBACK_BIOASQ_PATH.exists():
        return FALLBACK_BIOASQ_PATH
    raise FileNotFoundError(f"BioASQ input file not found: {path}")


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", str(text or "").lower()).strip()


def tokenize(text: str) -> set[str]:
    normalized = normalize_text(text)
    if not normalized:
        return set()
    return set(normalized.split())


def extract_pmid_from_document(document_url: str) -> str:
    text = str(document_url or "").strip()
    if not text:
        return ""
    match = PMID_RE.search(text)
    return match.group(1) if match else ""


def load_pmid_to_pmcid_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
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

            pmid = str(row.get("pmid") or "").strip()
            pmcid = str(row.get("pmcid") or "").strip()
            if pmid and pmcid:
                mapping[pmid] = pmcid
    return mapping


def load_chunks_by_pmcid(path: Path) -> dict[str, list[dict[str, object]]]:
    chunks_by_pmcid: dict[str, list[dict[str, object]]] = {}

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

            pmcid = str(row.get("pmcid") or "").strip()
            text = str(row.get("text") or "")
            normalized_text = normalize_text(text)

            if not pmcid or not normalized_text:
                continue

            chunk_record = {
                "chunk_id": str(row.get("chunk_id") or "").strip(),
                "source_type": str(row.get("source_type") or "").strip(),
                "section_path": row.get("section_path") or [],
                "text": text,
                "normalized_text": normalized_text,
                "token_set": tokenize(text),
            }
            chunks_by_pmcid.setdefault(pmcid, []).append(chunk_record)

    return chunks_by_pmcid


def choose_substring_match(
    snippet_normalized: str,
    chunks: list[dict[str, object]],
) -> tuple[dict[str, object] | None, float]:
    best_chunk: dict[str, object] | None = None
    best_score = -1.0
    best_length = 10**18

    for chunk in chunks:
        chunk_normalized = str(chunk["normalized_text"])
        if snippet_normalized and snippet_normalized in chunk_normalized:
            chunk_length = len(chunk_normalized)
            score = len(snippet_normalized) / chunk_length if chunk_length else 0.0
            if score > best_score or (score == best_score and chunk_length < best_length):
                best_chunk = chunk
                best_score = score
                best_length = chunk_length

    return best_chunk, best_score


def choose_token_overlap_match(
    snippet_tokens: set[str],
    chunks: list[dict[str, object]],
) -> tuple[dict[str, object] | None, float]:
    if not snippet_tokens:
        return None, 0.0

    best_chunk: dict[str, object] | None = None
    best_score = -1.0
    best_length = 10**18

    for chunk in chunks:
        chunk_tokens = set(chunk["token_set"])
        overlap = len(snippet_tokens & chunk_tokens) / len(snippet_tokens)
        chunk_length = len(str(chunk["normalized_text"]))
        if overlap > best_score or (overlap == best_score and chunk_length < best_length):
            best_chunk = chunk
            best_score = overlap
            best_length = chunk_length

    if best_score >= 0.6:
        return best_chunk, best_score
    return None, best_score


def build_match_row(
    question_id: str,
    question_type: str,
    pmid: str,
    pmcid: str,
    snippet_text: str,
    chunk: dict[str, object],
    match_method: str,
    match_score: float,
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "type": question_type,
        "pmid": pmid,
        "pmcid": pmcid,
        "snippet_text": snippet_text,
        "matched_chunk_id": chunk["chunk_id"],
        "source_type": chunk["source_type"],
        "section_path": chunk["section_path"],
        "match_method": match_method,
        "match_score": match_score,
    }


def build_unmatched_row(
    question_id: str,
    question_type: str,
    pmid: str,
    pmcid: str | None,
    snippet_text: str,
    reason: str,
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "type": question_type,
        "pmid": pmid,
        "pmcid": pmcid,
        "snippet_text": snippet_text,
        "reason": reason,
    }


def main() -> None:
    bioasq_path = resolve_bioasq_path(DEFAULT_BIOASQ_PATH)
    pmid_to_pmcid = load_pmid_to_pmcid_map(PMID_PMCID_PATH)
    chunks_by_pmcid = load_chunks_by_pmcid(CHUNKS_PATH)

    ensure_parent_dir(MATCHES_PATH)
    ensure_parent_dir(UNMATCHED_PATH)

    total_snippets_seen = 0
    snippets_with_pmcid = 0
    matched_by_substring = 0
    matched_by_token_overlap = 0
    unmatched_no_pmcid = 0
    unmatched_no_chunk_match = 0
    matched_questions: set[str] = set()

    with (
        bioasq_path.open("r", encoding="utf-8") as bioasq_handle,
        MATCHES_PATH.open("w", encoding="utf-8") as matches_handle,
        UNMATCHED_PATH.open("w", encoding="utf-8") as unmatched_handle,
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

            question_id = str(row.get("question_id") or "").strip()
            question_type = str(row.get("type") or "").strip()
            snippets = row.get("snippets") or []

            if not isinstance(snippets, list):
                continue

            for snippet in snippets:
                if not isinstance(snippet, dict):
                    continue

                total_snippets_seen += 1

                snippet_text = str(snippet.get("text") or "").strip()
                document_url = str(snippet.get("document") or "").strip()
                pmid = extract_pmid_from_document(document_url)
                pmcid = pmid_to_pmcid.get(pmid)

                if not pmcid:
                    unmatched_no_pmcid += 1
                    unmatched_row = build_unmatched_row(
                        question_id=question_id,
                        question_type=question_type,
                        pmid=pmid,
                        pmcid=None,
                        snippet_text=snippet_text,
                        reason="no_pmcid",
                    )
                    unmatched_handle.write(json.dumps(unmatched_row, ensure_ascii=False) + "\n")
                    continue

                snippets_with_pmcid += 1
                candidate_chunks = chunks_by_pmcid.get(pmcid, [])
                snippet_normalized = normalize_text(snippet_text)

                if not candidate_chunks or not snippet_normalized:
                    unmatched_no_chunk_match += 1
                    unmatched_row = build_unmatched_row(
                        question_id=question_id,
                        question_type=question_type,
                        pmid=pmid,
                        pmcid=pmcid,
                        snippet_text=snippet_text,
                        reason="no_chunk_match",
                    )
                    unmatched_handle.write(json.dumps(unmatched_row, ensure_ascii=False) + "\n")
                    continue

                substring_chunk, substring_score = choose_substring_match(
                    snippet_normalized,
                    candidate_chunks,
                )
                if substring_chunk is not None:
                    matched_by_substring += 1
                    matched_questions.add(question_id)
                    match_row = build_match_row(
                        question_id=question_id,
                        question_type=question_type,
                        pmid=pmid,
                        pmcid=pmcid,
                        snippet_text=snippet_text,
                        chunk=substring_chunk,
                        match_method="substring",
                        match_score=substring_score,
                    )
                    matches_handle.write(json.dumps(match_row, ensure_ascii=False) + "\n")
                    continue

                token_chunk, token_score = choose_token_overlap_match(
                    tokenize(snippet_normalized),
                    candidate_chunks,
                )
                if token_chunk is not None:
                    matched_by_token_overlap += 1
                    matched_questions.add(question_id)
                    match_row = build_match_row(
                        question_id=question_id,
                        question_type=question_type,
                        pmid=pmid,
                        pmcid=pmcid,
                        snippet_text=snippet_text,
                        chunk=token_chunk,
                        match_method="token_overlap",
                        match_score=token_score,
                    )
                    matches_handle.write(json.dumps(match_row, ensure_ascii=False) + "\n")
                    continue

                unmatched_no_chunk_match += 1
                unmatched_row = build_unmatched_row(
                    question_id=question_id,
                    question_type=question_type,
                    pmid=pmid,
                    pmcid=pmcid,
                    snippet_text=snippet_text,
                    reason="no_chunk_match",
                )
                unmatched_handle.write(json.dumps(unmatched_row, ensure_ascii=False) + "\n")

    print(f"Total snippets seen: {total_snippets_seen}")
    print(f"Snippets with PMCID: {snippets_with_pmcid}")
    print(f"Matched by substring: {matched_by_substring}")
    print(f"Matched by token overlap: {matched_by_token_overlap}")
    print(f"Unmatched because no PMCID: {unmatched_no_pmcid}")
    print(f"Unmatched because no chunk match: {unmatched_no_chunk_match}")
    print(f"Total unique questions with at least one matched chunk: {len(matched_questions)}")


if __name__ == "__main__":
    main()
