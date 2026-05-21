"""Download PMC full-text XML files for a list of PMCID identifiers."""

from __future__ import annotations

import argparse
import http.client
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_INPUT_PATH = Path("data/artifacts/pmcids_to_download_strict.txt")
DEFAULT_OUTPUT_DIR = Path("data/pmc_xml")
DEFAULT_STATUS_LOG = Path("data/artifacts/pmc_download_status.jsonl")
EFETCH_URL_TEMPLATE = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=pmc&id={pmcid}&retmode=xml"
)
USER_AGENT = "swaga-rag/0.1 (alexsachare239@gmail.com)"
ACCEPT_HEADER = "application/xml,text/xml,*/*"
CONNECTION_HEADER = "close"
REQUEST_DELAY_SECONDS = 0.75
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 5
BACKOFF_SECONDS = (1, 2, 4, 8, 16)
RETRYABLE_EXCEPTIONS = (
    urllib.error.URLError,
    urllib.error.HTTPError,
    http.client.RemoteDisconnected,
    http.client.IncompleteRead,
    TimeoutError,
    ConnectionResetError,
    EOFError,
    OSError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download PMC XML files by PMCID.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to a text file with one PMCID per line.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where downloaded PMC XML files will be stored.",
    )
    parser.add_argument(
        "--status-log",
        type=Path,
        default=DEFAULT_STATUS_LOG,
        help="Path to write the JSONL download status log.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_pmcids(path: Path) -> list[str]:
    pmcids: list[str] = []
    seen: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            pmcid = line.strip()
            if not pmcid or pmcid in seen:
                continue
            seen.add(pmcid)
            pmcids.append(pmcid)

    return pmcids


def build_output_path(output_dir: Path, pmcid: str) -> Path:
    return output_dir / f"{pmcid}.xml"


def build_temp_output_path(output_dir: Path, pmcid: str) -> Path:
    return output_dir / f"{pmcid}.xml.tmp"


def build_request_url(pmcid: str) -> str:
    return EFETCH_URL_TEMPLATE.format(pmcid=pmcid)


def looks_like_xml(content: bytes) -> bool:
    if not content:
        return False
    prefix = content.lstrip()
    return prefix.startswith(b"<?xml") or prefix.startswith(b"<")


def cleanup_temp_file(path: Path) -> None:
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def download_xml_once(pmcid: str) -> bytes:
    request = urllib.request.Request(
        build_request_url(pmcid),
        headers={
            "User-Agent": USER_AGENT,
            "Accept": ACCEPT_HEADER,
            "Connection": CONNECTION_HEADER,
        },
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return response.read()


def download_xml_with_retries(pmcid: str) -> bytes:
    last_error: BaseException | None = None

    for attempt in range(MAX_RETRIES):
        attempt_number = attempt + 1
        try:
            log(f"[{pmcid}] download attempt {attempt_number}/{MAX_RETRIES}")
            content = download_xml_once(pmcid)
            log(f"[{pmcid}] download completed ({len(content)} bytes)")
            return content
        except RETRYABLE_EXCEPTIONS as exc:
            last_error = exc
            if attempt == MAX_RETRIES - 1:
                log(f"[{pmcid}] giving up after {attempt_number} attempts: {exc}")
                break
            backoff_seconds = BACKOFF_SECONDS[attempt]
            log(
                f"[{pmcid}] transient error on attempt {attempt_number}/{MAX_RETRIES}: "
                f"{exc}. Retrying in {backoff_seconds}s"
            )
            time.sleep(backoff_seconds)

    if last_error is None:
        raise RuntimeError(f"Download failed for {pmcid} without a captured exception")
    raise last_error


def write_status_row(handle, row: dict[str, object]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def process_pmcid(
    pmcid: str,
    output_dir: Path,
) -> dict[str, object]:
    output_path = build_output_path(output_dir, pmcid)
    temp_output_path = build_temp_output_path(output_dir, pmcid)
    relative_path = output_path.as_posix()

    try:
        cleanup_temp_file(temp_output_path)

        if output_path.exists() and output_path.stat().st_size > 0:
            log(f"[{pmcid}] skipped: existing non-empty file")
            return {
                "pmcid": pmcid,
                "status": "skipped",
                "path": relative_path,
                "error": None,
            }

        content = download_xml_with_retries(pmcid)
        if not looks_like_xml(content):
            raise ValueError("Downloaded content does not look like XML")

        temp_output_path.write_bytes(content)
        if temp_output_path.stat().st_size == 0:
            raise ValueError("Downloaded XML file is empty after write")

        temp_output_path.replace(output_path)
        log(f"[{pmcid}] saved to {relative_path}")

        return {
            "pmcid": pmcid,
            "status": "success",
            "path": relative_path,
            "error": None,
        }
    except Exception as exc:
        cleanup_temp_file(temp_output_path)
        log(f"[{pmcid}] failed: {exc}")
        return {
            "pmcid": pmcid,
            "status": "failed",
            "path": relative_path,
            "error": str(exc),
        }


def main() -> None:
    args = parse_args()

    ensure_dir(args.output_dir)
    ensure_parent_dir(args.status_log)

    pmcids = load_pmcids(args.input)
    log(f"Loaded {len(pmcids)} PMCID values from {args.input.as_posix()}")

    downloaded = 0
    skipped = 0
    failed = 0

    with args.status_log.open("w", encoding="utf-8") as status_handle:
        for index, pmcid in enumerate(pmcids, start=1):
            log(f"[{index}/{len(pmcids)}] processing {pmcid}")
            result = process_pmcid(pmcid, args.output_dir)
            write_status_row(status_handle, result)

            status = str(result["status"])
            if status == "success":
                downloaded += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1

            if index < len(pmcids):
                log(f"[{pmcid}] sleeping for {REQUEST_DELAY_SECONDS:.2f}s before next request")
                time.sleep(REQUEST_DELAY_SECONDS)

    log("Download run finished")
    print(f"Total PMCID: {len(pmcids)}")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()

