"""Build BioASQ retrieval query JSONL in the format expected by run scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_INPUT_PATH = Path("data/artifacts/bioasq_retrieval_eval.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/eval/bioasq_eval_queries.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build BioASQ query JSONL from retrieval eval annotations."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with (
        args.input.open("r", encoding="utf-8") as input_handle,
        args.output.open("w", encoding="utf-8") as output_handle,
    ):
        for line_number, line in enumerate(input_handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_number} of {args.input}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object on line {line_number} of {args.input}"
                )

            question_id = str(row.get("question_id") or "").strip()
            question = str(row.get("question") or "").strip()
            if not question_id or not question:
                continue

            out_row = {"id": question_id, "query": question}
            output_handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            total += 1

    print(f"Queries written: {total}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
