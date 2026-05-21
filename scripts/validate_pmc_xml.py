"""Validate downloaded PMC XML files before structural parsing."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET


INPUT_DIR = Path("data/pmc_xml")
REPORT_PATH = Path("data/artifacts/pmc_xml_validation.jsonl")
INVALID_PMCIDS_PATH = Path("data/artifacts/pmcids_invalid_xml.txt")
INVALID_STATUSES = {
    "invalid_xml",
    "missing_article_tag",
    "missing_body",
    "parse_error",
    "empty",
}


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def looks_like_xml_with_article(text: str) -> bool:
    return "<article" in text


def find_first_element(root: ET.Element, name: str) -> ET.Element | None:
    if local_name(root.tag) == name:
        return root
    for element in root.iter():
        if local_name(element.tag) == name:
            return element
    return None


def count_elements(root: ET.Element, name: str) -> int:
    count = 0
    for element in root.iter():
        if local_name(element.tag) == name:
            count += 1
    return count


def validate_xml_file(path: Path) -> dict[str, object]:
    pmcid = path.stem
    file_size_bytes = path.stat().st_size

    result: dict[str, object] = {
        "pmcid": pmcid,
        "path": path.as_posix(),
        "file_size_bytes": file_size_bytes,
        "status": "",
        "has_article_tag": False,
        "has_body": False,
        "n_sec_elements": 0,
        "error": None,
    }

    if file_size_bytes == 0:
        result["status"] = "empty"
        return result

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        result["status"] = "invalid_xml"
        result["error"] = str(exc)
        return result

    has_article_tag = looks_like_xml_with_article(text)
    result["has_article_tag"] = has_article_tag

    if not has_article_tag:
        result["status"] = "missing_article_tag"
        return result

    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        result["status"] = "parse_error"
        result["error"] = str(exc)
        return result

    article_element = find_first_element(root, "article")
    result["has_article_tag"] = article_element is not None

    body_element = find_first_element(root, "body")
    result["has_body"] = body_element is not None
    result["n_sec_elements"] = count_elements(root, "sec")

    if body_element is None:
        result["status"] = "missing_body"
        return result

    result["status"] = "valid"
    return result


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_invalid_pmcids(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if str(row["status"]) in INVALID_STATUSES:
                handle.write(f"{row['pmcid']}\n")


def main() -> None:
    xml_files = sorted(INPUT_DIR.glob("*.xml"))
    results = [validate_xml_file(path) for path in xml_files]

    write_jsonl(REPORT_PATH, results)
    write_invalid_pmcids(INVALID_PMCIDS_PATH, results)

    status_counts = Counter(str(row["status"]) for row in results)
    valid_count = status_counts.get("valid", 0)
    invalid_count = len(results) - valid_count

    print(f"Total files scanned: {len(results)}")
    for status in sorted(status_counts):
        print(f"{status}: {status_counts[status]}")
    print(f"Number of valid files: {valid_count}")
    print(f"Number of invalid files: {invalid_count}")


if __name__ == "__main__":
    main()
