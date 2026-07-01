from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt BioASQ hybrid JSONL predictions into per-query files for evaluation."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/artifacts/bioasq_hybrid_predictions.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/artifacts/hybrid_eval_adapter"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
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

            qid = str(row.get("question_id") or row.get("id") or "").strip()
            query = str(row.get("question") or row.get("query") or "").strip()
            output_ids = row.get("predicted_chunk_ids")
            if not isinstance(output_ids, list):
                output_ids = row.get("output_ids") or []

            normalized_output_ids = [str(x) for x in output_ids if str(x).strip()]

            # Preserve rich output_items (e.g. window metadata: window_node_ids,
            # anchor_node_ids) when the producer emitted them, so window-level
            # evaluation (variant C) keeps working. Fall back to bare chunk ids.
            source_items = row.get("output_items")
            if isinstance(source_items, list) and source_items:
                output_items = source_items
            else:
                output_items = [{"chunk_id": chunk_id} for chunk_id in normalized_output_ids]

            out_row = {
                "id": qid,
                "query": query,
                "output_ids": normalized_output_ids,
                "output_items": output_items,
            }

            out_path = args.out_dir / f"{qid}.json"
            out_path.write_text(
                json.dumps(out_row, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            written += 1

    print(f"Files written: {written}")
    print(f"Output dir: {args.out_dir}")


if __name__ == "__main__":
    main()
