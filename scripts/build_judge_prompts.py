from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


from judge_prep.prompt_builder import (
    build_prompts,
    prompts_to_markdown,
    prompts_to_messages,
)

LETTERS = ["A", "B", "C", "D", "E"]


def read_json(path: Path) -> Dict[str, Any]:
    """Read a UTF-8 encoded JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, text: str) -> None:
    """Write text to a file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write a JSON file with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _extract_contexts(payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract candidate contexts in a robust way, matching prompt_builder priority:
    1) contexts_for_judge
    2) retrieved_contexts
    3) contexts
    4) candidates
    """
    raw = payload.get("contexts_for_judge")
    if isinstance(raw, dict):
        return {k: _as_str(raw.get(k, "")) for k in LETTERS}

    for key in ("retrieved_contexts", "contexts", "candidates"):
        raw2 = payload.get(key)
        if isinstance(raw2, dict):
            return {k: _as_str(raw2.get(k, "")) for k in LETTERS}

    return {k: "" for k in LETTERS}


def _is_pairwise_payload(payload: Dict[str, Any]) -> bool:
    """
    Heuristic: treat payload as pairwise if A and B are present (non-empty after strip)
    and C/D/E are empty or missing.
    """
    ctx = _extract_contexts(payload)
    a = ctx.get("A", "").strip()
    b = ctx.get("B", "").strip()
    if not a or not b:
        return False

    # If C/D/E contain anything non-whitespace, it's not pairwise
    for k in ("C", "D", "E"):
        if ctx.get(k, "").strip():
            return False

    return True


def _are_identical_pairwise(payload: Dict[str, Any]) -> bool:
    """
    Only checks equality for pairwise payloads. Uses the exact strings the judge would see.
    """
    if not _is_pairwise_payload(payload):
        return False

    ctx = _extract_contexts(payload)
    a = ctx.get("A", "").strip()
    b = ctx.get("B", "").strip()
    return a == b


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build LLM-as-judge prompts from artifacts/judge_payloads/*.json"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="artifacts/judge_payloads",
        help="Directory containing Qxxx.json judge payloads",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/judge_prompts",
        help="Directory to write generated judge prompts",
    )
    parser.add_argument(
        "--write_md",
        action="store_true",
        help="Write one Markdown file per query (Qxxx.md)",
    )
    parser.add_argument(
        "--write_messages_json",
        action="store_true",
        help="Write one messages JSON per query (Qxxx.messages.json)",
    )
    parser.add_argument(
        "--write_jsonl",
        action="store_true",
        help="Write prompts.jsonl with one {id, messages} per line",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already existing per-query prompt files and keep the rest.",
    )
    parser.add_argument(
        "--prompt_variant",
        type=str,
        default="legacy",
        choices=["legacy", "reasoning"],
        help="Choose judge prompt template variant.",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    if not in_dir.exists():
        raise SystemExit(f"Input directory does not exist: {in_dir}")

    payload_paths = sorted(in_dir.glob("Q*.json"))
    if not payload_paths:
        raise SystemExit(f"No payloads found in {in_dir} (expected Q*.json)")

    jsonl_lines: List[str] = []
    skipped_identical: List[str] = []

    for payload_path in payload_paths:
        payload = read_json(payload_path)

        # Determine query id early (for logging / skipping)
        qid = payload.get("id") if isinstance(payload, dict) else payload_path.stem
        if not qid:
            qid = payload_path.stem
        qid = str(qid)

        # Budget saver: skip identical A/B in pairwise payloads
        if isinstance(payload, dict) and _are_identical_pairwise(payload):
            skipped_identical.append(qid)
            continue

        # Build prompts (usually returns a list of length 1)
        prompts = build_prompts(payload, prompt_variant=args.prompt_variant)

        # Write Markdown prompt (human-readable)
        if args.write_md:
            md = prompts_to_markdown(prompts, title=qid, prompt_variant=args.prompt_variant)
            md_path = out_dir / f"{qid}.md"
            if not (args.resume and md_path.exists()):
                write_text(md_path, md)

        # Write per-query messages JSON
        if args.write_messages_json:
            msg_obj = {
                "id": qid,
                "messages": prompts_to_messages(prompts, prompt_variant=args.prompt_variant),
            }
            msg_path = out_dir / f"{qid}.messages.json"
            if not (args.resume and msg_path.exists()):
                write_json(msg_path, msg_obj)

        # Accumulate JSONL lines
        if args.write_jsonl:
            msg_obj = {
                "id": qid,
                "messages": prompts_to_messages(prompts, prompt_variant=args.prompt_variant),
            }
            jsonl_lines.append(json.dumps(msg_obj, ensure_ascii=False))

    # Write aggregated JSONL file
    if args.write_jsonl:
        write_text(
            out_dir / "prompts.jsonl",
            "\n".join(jsonl_lines) + ("\n" if jsonl_lines else ""),
        )

    processed = len(payload_paths) - len(skipped_identical)
    print(
        f"Done. Processed {processed} payload(s). "
        f"Skipped {len(skipped_identical)} identical pairwise payload(s). "
        f"Output written to {out_dir}"
    )
    if skipped_identical:
        # Keep it short but informative
        preview = ", ".join(skipped_identical[:20])
        suffix = "" if len(skipped_identical) <= 20 else f" ... (+{len(skipped_identical) - 20} more)"
        print(f"Skipped identical: {preview}{suffix}")


if __name__ == "__main__":
    main()
