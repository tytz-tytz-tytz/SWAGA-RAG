"""Build a structured chunk corpus from validated PMC XML files."""

from __future__ import annotations

import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET


XML_DIR = Path("data/pmc_xml")
VALIDATION_PATH = Path("data/artifacts/pmc_xml_validation.jsonl")
OUTPUT_PATH = Path("data/artifacts/pmc_structured_chunks.jsonl")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
WHITESPACE_RE = re.compile(r"\s+")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def iter_children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if local_name(child.tag) == name]


def find_first(element: ET.Element, name: str) -> ET.Element | None:
    for child in element.iter():
        if local_name(child.tag) == name:
            return child
    return None


def find_text_in_children(element: ET.Element, child_name: str) -> str:
    child = None
    for candidate in list(element):
        if local_name(candidate.tag) == child_name:
            child = candidate
            break
    if child is None:
        return ""
    return normalize_whitespace(" ".join(child.itertext()))


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            split_at = text.rfind(" ", start, end)
            if split_at > start + max(100, chunk_size // 3):
                end = split_at
        chunk = normalize_whitespace(text[start:end])
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = max(start + step, end - overlap)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def load_valid_pmcs(path: Path) -> set[str]:
    valid_pmcs: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object on line {line_number} of {path}")
            if str(row.get("status") or "").strip() == "valid":
                pmcid = str(row.get("pmcid") or "").strip()
                if pmcid:
                    valid_pmcs.add(pmcid)
    return valid_pmcs


def build_chunk_id(
    pmcid: str,
    source_type: str,
    section_path: list[str],
    paragraph_index: int,
    chunk_index: int,
) -> str:
    path_text = " > ".join(section_path)
    return f"{pmcid}::{source_type}::{path_text}::p{paragraph_index}::c{chunk_index}"


def make_chunk_rows(
    pmcid: str,
    article_title: str,
    source_type: str,
    section_path: list[str],
    paragraph_text: str,
    paragraph_index: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for chunk_index, chunk in enumerate(chunk_text(paragraph_text)):
        rows.append(
            {
                "pmcid": pmcid,
                "article_title": article_title,
                "source_type": source_type,
                "section_path": section_path,
                "paragraph_index": paragraph_index,
                "chunk_index": chunk_index,
                "chunk_id": build_chunk_id(
                    pmcid=pmcid,
                    source_type=source_type,
                    section_path=section_path,
                    paragraph_index=paragraph_index,
                    chunk_index=chunk_index,
                ),
                "text": chunk,
            }
        )
    return rows


def paragraph_text_from_element(element: ET.Element) -> str:
    return normalize_whitespace(" ".join(element.itertext()))


def extract_abstract_rows(
    pmcid: str,
    article_title: str,
    root: ET.Element,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    abstract_element = find_first(root, "abstract")
    if abstract_element is None:
        return rows

    paragraphs = iter_children(abstract_element, "p")
    if not paragraphs:
        abstract_text = normalize_whitespace(" ".join(abstract_element.itertext()))
        if abstract_text:
            rows.extend(
                make_chunk_rows(
                    pmcid=pmcid,
                    article_title=article_title,
                    source_type="abstract",
                    section_path=["Abstract"],
                    paragraph_text=abstract_text,
                    paragraph_index=0,
                )
            )
        return rows

    for paragraph_index, paragraph in enumerate(paragraphs):
        text = paragraph_text_from_element(paragraph)
        if not text:
            continue
        rows.extend(
            make_chunk_rows(
                pmcid=pmcid,
                article_title=article_title,
                source_type="abstract",
                section_path=["Abstract"],
                paragraph_text=text,
                paragraph_index=paragraph_index,
            )
        )
    return rows


def iter_direct_paragraphs(sec_element: ET.Element) -> list[ET.Element]:
    return [child for child in list(sec_element) if local_name(child.tag) == "p"]


def extract_section_title(sec_element: ET.Element) -> str:
    title = find_text_in_children(sec_element, "title")
    return title or "Untitled Section"


def extract_body_rows_from_section(
    sec_element: ET.Element,
    pmcid: str,
    article_title: str,
    section_path: list[str],
    paragraph_counter: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for paragraph in iter_direct_paragraphs(sec_element):
        text = paragraph_text_from_element(paragraph)
        if not text:
            continue
        paragraph_index = paragraph_counter[0]
        paragraph_counter[0] += 1
        rows.extend(
            make_chunk_rows(
                pmcid=pmcid,
                article_title=article_title,
                source_type="body",
                section_path=section_path,
                paragraph_text=text,
                paragraph_index=paragraph_index,
            )
        )

    for child in list(sec_element):
        if local_name(child.tag) != "sec":
            continue
        child_title = extract_section_title(child)
        rows.extend(
            extract_body_rows_from_section(
                sec_element=child,
                pmcid=pmcid,
                article_title=article_title,
                section_path=section_path + [child_title],
                paragraph_counter=paragraph_counter,
            )
        )

    return rows


def extract_body_rows(
    pmcid: str,
    article_title: str,
    root: ET.Element,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    body = find_first(root, "body")
    if body is None:
        return rows

    paragraph_counter = [0]
    for sec_element in iter_children(body, "sec"):
        section_title = extract_section_title(sec_element)
        rows.extend(
            extract_body_rows_from_section(
                sec_element=sec_element,
                pmcid=pmcid,
                article_title=article_title,
                section_path=[section_title],
                paragraph_counter=paragraph_counter,
            )
        )
    return rows


def extract_article_title(root: ET.Element) -> str:
    for element in root.iter():
        if local_name(element.tag) == "article-title":
            return normalize_whitespace(" ".join(element.itertext()))
    return ""


def parse_article(path: Path) -> list[dict[str, object]]:
    root = ET.parse(path).getroot()
    pmcid = path.stem
    article_title = extract_article_title(root)

    rows: list[dict[str, object]] = []
    rows.extend(extract_abstract_rows(pmcid=pmcid, article_title=article_title, root=root))
    rows.extend(extract_body_rows(pmcid=pmcid, article_title=article_title, root=root))
    return rows


def main() -> None:
    valid_pmcs = load_valid_pmcs(VALIDATION_PATH)
    xml_paths = [XML_DIR / f"{pmcid}.xml" for pmcid in sorted(valid_pmcs)]

    processed_files = 0
    skipped_files = 0
    chunk_records_written = 0

    ensure_parent_dir(OUTPUT_PATH)
    with OUTPUT_PATH.open("w", encoding="utf-8") as out_handle:
        for xml_path in xml_paths:
            if not xml_path.exists():
                skipped_files += 1
                continue
            try:
                rows = parse_article(xml_path)
            except (ET.ParseError, OSError, UnicodeDecodeError):
                skipped_files += 1
                continue

            processed_files += 1
            for row in rows:
                out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                chunk_records_written += 1

    print(f"Valid XML files processed: {processed_files}")
    print(f"Chunk records written: {chunk_records_written}")
    print(f"Files skipped due to parse issues during this stage: {skipped_files}")


if __name__ == "__main__":
    main()
