"""
Step 9: full LLM-judge run on all 540 pairs across three judges.

For each judge:
  - shuffle the 540 pairs with that judge's own seed (from configs/judge_v2/judges.json)
  - run through the concurrent client (semaphore=concurrency from config)
  - append each decision immediately to artifacts/judge_v2/full_run/llm_{judge}.jsonl

The script is resumable: pair_ids already present (status=ok) in a per-judge
output are skipped on re-run. A per-judge ``run_meta.json`` records start
time, completion, totals.

No batching of writes — every decision is fsynced on its own line. If the
process dies, at most ``concurrency`` items are lost.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _repo_paths import repo_path, resolve_repo_path  # noqa: E402

from judge_v2_client import JudgeConfig, run_judge_on_pairs  # noqa: E402


DEFAULT_PAIRS = repo_path("artifacts/judge_v2/pairs.jsonl")
DEFAULT_JUDGES_CFG = repo_path("configs/judge_v2/judges.json")
DEFAULT_OUT_DIR = repo_path("artifacts/judge_v2/full_run")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _already_done(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    done: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = obj.get("pair_id")
            status = obj.get("status")
            if isinstance(pid, str) and status == "ok":
                done.add(pid)
    return done


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full pairwise LLM-judge evaluation.")
    p.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    p.add_argument("--judges_config", type=Path, default=DEFAULT_JUDGES_CFG)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--judge",
        type=str,
        default=None,
        help="Run only this judge by name (default: all from config).",
    )
    p.add_argument(
        "--require_calibration",
        action="store_true",
        help=(
            "Refuse to run unless artifacts/judge_v2/calibration/metrics.json "
            "exists and acceptance.passed is true."
        ),
    )
    return p.parse_args()


async def _amain() -> None:
    args = parse_args()
    pairs_all = _read_jsonl(resolve_repo_path(args.pairs))
    cfg = _read_json(resolve_repo_path(args.judges_config))
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.require_calibration:
        metrics_path = repo_path("artifacts/judge_v2/calibration/metrics.json")
        if not metrics_path.exists():
            raise SystemExit(f"Calibration metrics missing: {metrics_path}")
        m = _read_json(metrics_path)
        if not (m.get("acceptance") or {}).get("passed"):
            raise SystemExit(f"Calibration not passed: {m.get('acceptance')}")

    judges_cfg = cfg["judges"]
    if args.judge:
        judges_cfg = [j for j in judges_cfg if j["name"] == args.judge]
        if not judges_cfg:
            raise ValueError(f"Unknown judge: {args.judge}")

    req = cfg.get("request", {})
    concurrency = int(cfg.get("concurrency", 5))

    summary: List[Dict[str, Any]] = []
    for jraw in judges_cfg:
        jcfg = JudgeConfig.from_dict(jraw)
        out_path = out_dir / f"llm_{jcfg.name}.jsonl"
        done = _already_done(out_path)

        rng = random.Random(jcfg.shuffle_seed)
        shuffled = list(pairs_all)
        rng.shuffle(shuffled)
        todo = [p for p in shuffled if p["pair_id"] not in done]

        if not todo:
            print(f"[{jcfg.name}] all {len(shuffled)} done — skipping")
            summary.append({"judge": jcfg.name, "total": len(shuffled), "skipped": True})
            continue
        print(
            f"[{jcfg.name}] {len(shuffled)} total, {len(done)} already done, "
            f"{len(todo)} to do, seed={jcfg.shuffle_seed}, concurrency={concurrency}"
        )
        stats = await run_judge_on_pairs(
            jcfg,
            system_prompt=cfg["system_prompt"],
            user_template=cfg["user_template"],
            pairs=todo,
            out_path=out_path,
            concurrency=concurrency,
            temperature=float(req.get("temperature", 0.0)),
            max_tokens=int(req.get("max_tokens", 200)),
            timeout_seconds=float(req.get("timeout_seconds", 120)),
        )
        summary.append(stats)
        print(f"[{jcfg.name}] {stats}")

    summary_path = out_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Summary: {summary_path}")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
