from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _repo_paths import repo_path, resolve_repo_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run SWAGA-RAG for each config in configs/ablations and store outputs in:\n"
            "  artifacts/swaga_rag_results/param_experiments/<config_stem>/Qxxx.json\n"
            "\n"
            "For each experiment folder we save exactly ONE config copy: <out_dir>/config.json\n"
            "No meta files, no nested config folders.\n"
            "\n"
            "Requires entrypoint to support: --run_id .  (write directly into run.out_dir).\n"
        )
    )

    p.add_argument("--configs_dir", type=Path, default=repo_path("configs/ablations"))
    p.add_argument("--out_root", type=Path, default=repo_path("artifacts/swaga_rag_results/param_experiments"))

    # baseline is just for your manual workflow; script doesn't treat it specially
    p.add_argument("--baseline_config", type=Path, default=repo_path("configs/ablations/stable_baseline.json"))

    p.add_argument("--entrypoint", type=Path, default=repo_path("scripts/run_queries_swaga.py"))

    p.add_argument("--queries_path", type=Path, default=None)
    p.add_argument("--index_dir", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--only", type=str, default=None, help="Comma-separated config stems to run.")
    p.add_argument("--overwrite", action="store_true", help="Delete existing per-config dirs before running.")
    p.add_argument("--dry_run", action="store_true")

    p.add_argument("--build_debug_report", action="store_true")

    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Config must be a dict at top-level: {path}")
    return obj


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def list_configs(configs_dir: Path) -> List[Path]:
    if not configs_dir.exists():
        raise FileNotFoundError(f"configs_dir not found: {configs_dir}")
    cfgs = sorted(configs_dir.glob("*.json"))
    if not cfgs:
        raise FileNotFoundError(f"No *.json found in: {configs_dir}")
    return cfgs


def filter_only(cfgs: List[Path], only: Optional[str]) -> List[Path]:
    if not only:
        return cfgs
    allow = {x.strip() for x in only.split(",") if x.strip()}
    return [c for c in cfgs if c.stem in allow]


def dir_non_empty(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def build_resolved_config(src_cfg_path: Path, out_dir: Path) -> Dict[str, Any]:
    """
    Resolved config for a specific ablation:
    - force run.out_dir = out_dir
    - force run.append_timestamp = False
    """
    cfg = load_json(src_cfg_path)

    run_cfg = cfg.get("run")
    if not isinstance(run_cfg, dict):
        run_cfg = {}
        cfg["run"] = run_cfg

    run_cfg["out_dir"] = str(out_dir.as_posix())
    run_cfg["append_timestamp"] = False

    # name can stay; it won't matter if entrypoint writes flat with --run_id "."
    if "name" not in run_cfg or not isinstance(run_cfg.get("name"), str):
        run_cfg["name"] = src_cfg_path.stem

    return cfg


def run_entrypoint(
    entrypoint: Path,
    config_path: Path,
    queries_path: Optional[Path],
    index_dir: Optional[Path],
    device: Optional[str],
    dry_run: bool,
) -> None:
    entrypoint = resolve_repo_path(entrypoint)
    if not entrypoint.exists():
        raise FileNotFoundError(f"Entrypoint not found: {entrypoint}")

    cmd = [sys.executable, str(entrypoint), "--config", str(config_path), "--run_id", "."]

    if queries_path is not None:
        cmd += ["--queries_path", str(queries_path)]
    if index_dir is not None:
        cmd += ["--index_dir", str(index_dir)]
    if device is not None:
        cmd += ["--device", str(device)]

    print("cmd:", " ".join(cmd))
    if dry_run:
        print("[DRY RUN] Skipping execution.")
        return

    proc = subprocess.run(cmd, cwd=repo_path("."), capture_output=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Entrypoint failed (exit={proc.returncode}). "
            f"This requires entrypoint to support '--run_id .' as 'write directly into run.out_dir'."
        )


def maybe_build_debug_report(out_dir: Path, dry_run: bool) -> None:
    dbg = repo_path("scripts/build_swaga_debug_report.py")
    if not dbg.exists():
        print("[WARN] scripts/build_swaga_debug_report.py not found; skipping debug report.")
        return

    cmd = [sys.executable, str(dbg), "--run_dir", str(out_dir)]
    print("post:", " ".join(cmd))
    if dry_run:
        print("[DRY RUN] Skipping debug report.")
        return

    proc = subprocess.run(cmd, cwd=repo_path("."), capture_output=False)
    if proc.returncode != 0:
        print(f"[WARN] Debug report returned non-zero ({proc.returncode}). Continuing.")


def main() -> None:
    args = parse_args()
    args.configs_dir = resolve_repo_path(args.configs_dir)
    args.out_root = resolve_repo_path(args.out_root)
    args.baseline_config = resolve_repo_path(args.baseline_config)
    args.entrypoint = resolve_repo_path(args.entrypoint)

    cfgs = filter_only(list_configs(args.configs_dir), args.only)
    if not cfgs:
        print("[NO CONFIGS] Nothing to run.")
        sys.exit(0)

    args.out_root.mkdir(parents=True, exist_ok=True)

    print(f"[PLAN] configs_dir: {args.configs_dir}")
    print(f"[PLAN] out_root:    {args.out_root}")
    print(f"[PLAN] baseline:    {args.baseline_config} (recorded manually, not used by script)")
    print(f"[PLAN] entrypoint:  {args.entrypoint}")
    print(f"[PLAN] overwrite:   {args.overwrite}")
    print(f"[PLAN] dry_run:     {args.dry_run}")

    for cfg_path in cfgs:
        stem = cfg_path.stem
        out_dir = args.out_root / stem

        if dir_non_empty(out_dir) and not args.overwrite:
            print(f"\n[SKIP] Exists and non-empty (use --overwrite to rerun): {out_dir}")
            continue

        print(f"\n=== RUN: {stem} ===")
        print(f"out_dir: {out_dir}")

        prepare_out_dir(out_dir, overwrite=args.overwrite)

        # 1) write resolved config used by the run
        resolved_cfg = build_resolved_config(cfg_path, out_dir)
        resolved_path = out_dir / "config.json"   # single canonical copy inside the run folder
        write_json(resolved_path, resolved_cfg)

        # 2) run
        run_entrypoint(
            entrypoint=args.entrypoint,
            config_path=resolved_path,
            queries_path=args.queries_path,
            index_dir=args.index_dir,
            device=args.device,
            dry_run=args.dry_run,
        )

        # 3) optional debug report
        if args.build_debug_report:
            maybe_build_debug_report(out_dir, dry_run=args.dry_run)

        if not args.dry_run and not any(out_dir.glob("Q*.json")):
            print("[WARN] No Q*.json found in out_dir after run. Check entrypoint outputs/paths.")
        else:
            print("[OK] Outputs present.")

    print("\n[DONE]")


if __name__ == "__main__":
    main()


