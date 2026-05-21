"""Small helpers shared across the experiment scripts."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Tuple

# A retrieval hit is (chunk_id, text, score).
Hit = Tuple[str, str, float]


def iter_queries(path: Path) -> Iterator[dict]:
    """Yield JSON objects from a JSONL file, skipping blank lines."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_validated_queries(path: Path) -> Iterator[Dict]:
    """Yield JSONL query objects, requiring an 'id' and 'query' field on each line."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" not in obj or "query" not in obj:
                raise ValueError(f"Line {line_no} must contain 'id' and 'query'")
            yield obj


def build_result(qid: str, query: str, hits: Iterable[Hit]) -> dict:
    """Assemble the canonical per-query result document shared by all retrieval runs."""
    hits = list(hits)
    return {
        "id": qid,
        "query": query,
        "output_items": [
            {"chunk_id": cid, "text": text, "score": score}
            for cid, text, score in hits
        ],
        "output_ids": [cid for cid, _text, _score in hits],
    }


def write_result(out_dir: Path, qid: str, result: dict) -> None:
    """Write a single result document to ``{out_dir}/{qid}.json`` (UTF-8, indent=2)."""
    (out_dir / f"{qid}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@dataclass
class RetrievalRunner:
    """Drives the per-query retrieval loop shared by the run_queries_*/run_bioasq_* scripts.

    The caller supplies a ``retrieve_fn(query) -> iterable of (chunk_id, text, score)``.
    The runner writes one ``{qid}.json`` per query in the canonical format and prints
    progress/ETA lines. The result files are identical across baselines; only the log
    ``tag``, the speed precision and the final ``done`` wording differ, so those are
    parameters.
    """

    tag: str
    out_dir: Path
    log_every: int = 100
    speed_decimals: int = 1
    done_message: str = "done. Results saved to:"

    def run(self, queries: List[dict], retrieve_fn: Callable[[str], Iterable[Hit]]) -> float:
        t0 = time.perf_counter()
        total = len(queries)

        for i, q in enumerate(queries, start=1):
            qid = q["id"]
            query = q["query"]

            result = build_result(qid, query, retrieve_fn(query))
            write_result(self.out_dir, qid, result)

            if i % max(self.log_every, 1) == 0 or i == total:
                elapsed = max(time.perf_counter() - t0, 1e-9)
                speed = i / elapsed
                left = max(total - i, 0)
                eta = left / max(speed, 1e-9)
                pct = (i / total) * 100 if total else 100.0
                print(
                    f"[{self.tag}] {i}/{total} ({pct:.1f}%) | "
                    f"{speed:.{self.speed_decimals}f} q/s | ETA {eta/60:.1f} min"
                )

        elapsed_total = time.perf_counter() - t0
        print(f"[{self.tag}] {self.done_message} {self.out_dir}")
        print(f"[{self.tag}] total elapsed: {elapsed_total:.1f}s")
        return elapsed_total
