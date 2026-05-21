"""
Smoke: send one pair through all three judges and print parsed labels +
raw response + token usage. Cost is dominated by a single judge call per
provider so total should be well under $0.01.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _repo_paths import repo_path, resolve_repo_path  # noqa: E402

from judge_v2_client import JudgeClient, JudgeConfig  # noqa: E402


DEFAULT_PAIRS = repo_path("artifacts/judge_v2/pairs.jsonl")
DEFAULT_JUDGES_CFG = repo_path("configs/judge_v2/judges.json")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_pair(path: Path, pair_id: str) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["pair_id"] == pair_id:
                return rec
    raise SystemExit(f"pair_id not found: {pair_id}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pair_id", type=str, default="p001_Q5R001_AB")
    p.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    p.add_argument("--judges_config", type=Path, default=DEFAULT_JUDGES_CFG)
    return p.parse_args()


async def _amain() -> None:
    args = parse_args()
    pair = _find_pair(resolve_repo_path(args.pairs), args.pair_id)
    cfg = _read_json(resolve_repo_path(args.judges_config))

    print(f"Pair: {pair['pair_id']}")
    print(f"  comparison: {pair['comparison_id']}  perm: {pair['perm']}")
    print(f"  query: {pair['query']}")
    print(f"  method_A={pair['method_A']}  method_B={pair['method_B']}")
    print(f"  |ctx_A|={len(pair['context_A'])} chars  |ctx_B|={len(pair['context_B'])} chars")
    print()

    req = cfg.get("request", {})
    for jraw in cfg["judges"]:
        jcfg = JudgeConfig.from_dict(jraw)
        print(f"=== {jcfg.name} (backend={jcfg.backend}, model={jcfg.model}) ===")
        if jcfg.base_url:
            print(f"  base_url: {jcfg.base_url}")
        client = JudgeClient(
            jcfg,
            system_prompt=cfg["system_prompt"],
            user_template=cfg["user_template"],
            temperature=float(req.get("temperature", 0.0)),
            max_tokens=int(req.get("max_tokens", 200)),
            timeout_seconds=float(req.get("timeout_seconds", 120)),
        )
        decision = await client.judge_pair(pair)
        print(f"  status={decision.status}  attempts={decision.attempts}  latency_ms={decision.latency_ms}")
        if decision.error:
            print(f"  error: {decision.error}")
        print(f"  labels: {decision.labels}")
        print(f"  usage: {decision.usage}")
        raw_short = (decision.raw or "")[:300].replace("\n", "\\n")
        print(f"  raw[:300]: {raw_short}")
        print()


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
