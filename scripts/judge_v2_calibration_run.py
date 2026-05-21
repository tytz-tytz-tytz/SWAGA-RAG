"""
Step 7: run the 90 calibration pairs through all three LLM judges.

Each judge writes to artifacts/judge_v2/calibration/llm_{judge_name}.jsonl,
appending one decision per line as it is returned. Re-running the script
SKIPs pair_ids that are already present in the per-judge file — so it is
safe to interrupt and resume.
"""
from __future__ import annotations

import argparse
import asyncio
import json
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
DEFAULT_OUT_DIR = repo_path("artifacts/judge_v2/calibration")


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
    p = argparse.ArgumentParser(description="Run calibration LLM-judge pass.")
    p.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    p.add_argument("--judges_config", type=Path, default=DEFAULT_JUDGES_CFG)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--judge",
        type=str,
        default=None,
        help="Run only this judge by name (default: all three).",
    )
    return p.parse_args()


async def _amain() -> None:
    args = parse_args()
    pairs_all = _read_jsonl(resolve_repo_path(args.pairs))
    cfg = _read_json(resolve_repo_path(args.judges_config))
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calib_qids = set(cfg["calibration"]["query_ids"])
    subset = [p for p in pairs_all if p["query_id"] in calib_qids]
    print(f"[INFO] Calibration subset: {len(subset)} pairs")

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
        todo = [p for p in subset if p["pair_id"] not in done]
        if not todo:
            print(f"[{jcfg.name}] all {len(subset)} pairs already done — skipping")
            summary.append({"judge": jcfg.name, "total": len(subset), "skipped": True})
            continue
        if done:
            print(f"[{jcfg.name}] resuming: {len(done)} done, {len(todo)} to do")
        else:
            print(f"[{jcfg.name}] running: {len(todo)} pairs")
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
