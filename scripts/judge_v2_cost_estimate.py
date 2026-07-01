"""Stage E: dry-run cost estimator. No API calls.

Counts, per staged pair file (calib / bridge30 / full150), the real input tokens
of every judge call (system_prompt + filled user_template over the actual
unified contexts already embedded in the pairs), times the 3 judges, and prints
USD per stage/judge using the pricing in configs/judge_v2/judges.json.

Output tokens are estimated (compact JSON verdict ~ a few dozen tokens); override
with --output-tokens-est. Input dominates cost here, so the input side is exact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _repo_paths import repo_path, resolve_repo_path  # noqa: E402

import tiktoken


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dry-run cost estimate for the judge stages.")
    p.add_argument("--judges_config", type=Path, default=repo_path("configs/judge_v2/judges.json"))
    p.add_argument("--stage_dir", type=Path, default=repo_path("artifacts/judge_v2"))
    p.add_argument("--encoding", type=str, default="cl100k_base")
    p.add_argument("--output-tokens-est", type=int, default=40,
                   help="Estimated output tokens per verdict (compact JSON).")
    p.add_argument("--stages", nargs="+", default=["calib", "bridge30", "full150"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _read_json(resolve_repo_path(args.judges_config))
    enc = tiktoken.get_encoding(args.encoding)
    sys_tok = len(enc.encode(cfg["system_prompt"]))
    tmpl = cfg["user_template"]
    judges = cfg["judges"]
    out_est = int(args.output_tokens_est)
    stage_dir = resolve_repo_path(args.stage_dir)

    stage_files = {
        "calib": stage_dir / "pairs_calib.jsonl",
        "bridge30": stage_dir / "pairs_bridge30.jsonl",
        "full150": stage_dir / "pairs_full150.jsonl",
    }

    print(f"system_prompt tokens={sys_tok} | output_tokens_est={out_est}/call | encoding={args.encoding}")
    print(f"judges: {', '.join(j['name'] + ' [' + j['model'] + ']' for j in judges)}\n")

    grand = 0.0
    for stage in args.stages:
        path = stage_files[stage]
        if not path.exists():
            print(f"[skip] {stage}: {path} not found"); continue
        pairs = _read_jsonl(path)
        # input tokens per pair (same across judges): system + filled user template
        in_tok_total = 0
        for p in pairs:
            user = tmpl.format(query=p["query"], context_A=p["context_A"], context_B=p["context_B"])
            in_tok_total += sys_tok + len(enc.encode(user))
        n = len(pairs)
        print(f"=== stage {stage}: {n} pairs/judge, {n*len(judges)} total calls ===")
        print(f"    avg input tokens/call = {in_tok_total/max(n,1):.0f}")
        stage_cost = 0.0
        for j in judges:
            pr = j["pricing_per_1m_tokens"]
            in_cost = in_tok_total * float(pr["input_usd"]) / 1_000_000
            out_cost = n * out_est * float(pr["output_usd"]) / 1_000_000
            jc = in_cost + out_cost
            stage_cost += jc
            print(f"    {j['name']:22s} in={in_tok_total:>9d} out~={n*out_est:>7d}  "
                  f"${in_cost:6.3f}+${out_cost:5.3f} = ${jc:6.3f}")
        print(f"    STAGE TOTAL (3 judges): ${stage_cost:.3f}\n")
        grand += stage_cost

    print(f"GRAND TOTAL across stages {args.stages}: ${grand:.3f}")
    print("NB: CometAPI gateway markup may differ from listed per-model prices.")


if __name__ == "__main__":
    main()
