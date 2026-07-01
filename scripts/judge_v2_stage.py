"""Stage B: integrity asserts + pair staging + calibration re-keying.

1) Assert (via json.loads, not wc -l) that exactly the N unique query ids in the
   queries file flow through every method run dir -> unified -> pairs, with none
   lost or extra.
2) Split pairs.jsonl into three staged files by query id:
     pairs_calib.jsonl    — the 5 calibration queries (from judges.json)
     pairs_bridge30.jsonl — Q5R001..Q5R030
     pairs_full150.jsonl  — all queries
3) Re-key the existing expert calibration (manual.jsonl, keyed by OLD pair_ids
   from the previous 540-pair run) onto the NEW pair_ids, matching by the stable
   semantic key (comparison_id, query_id, perm). Retrieval is unchanged for the
   calibration queries, so the labels remain valid; only pair_id numbering moved.

Pure data wrangling — no API calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _repo_paths import repo_path, resolve_repo_path  # noqa: E402

KEY = Tuple[str, str, str]  # (comparison_id, query_id, perm)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln} bad JSON: {e}") from e
    return rows


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _query_ids(path: Path) -> List[str]:
    ids = [str(r["id"]).strip() for r in _read_jsonl(path)]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"[FAIL] duplicate query ids in {path}")
    return ids


def _dir_ids(d: Path) -> Set[str]:
    return {p.stem for p in d.glob("*.json") if p.name.lower() != "config.json"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Integrity asserts + pair staging + calib re-keying.")
    p.add_argument("--queries", type=Path, default=repo_path("data/eval/queries_5rag.jsonl"))
    p.add_argument("--methods_config", type=Path, default=repo_path("configs/judge_v2/methods.json"))
    p.add_argument("--unified_dir", type=Path, default=repo_path("artifacts/judge_v2/unified"))
    p.add_argument("--pairs", type=Path, default=repo_path("artifacts/judge_v2/pairs.jsonl"))
    p.add_argument("--out_dir", type=Path, default=repo_path("artifacts/judge_v2"))
    # Old run (for calibration label re-keying):
    p.add_argument("--old_pairs", type=Path,
                   default=Path("C:/Users/alexs/spbu/SWAGA-RAG/artifacts/judge_v2/pairs.jsonl"),
                   help="Previous pairs.jsonl that the expert manual labels were keyed against.")
    p.add_argument("--manual", type=Path,
                   default=Path("C:/Users/alexs/spbu/SWAGA-RAG/artifacts/judge_v2/calibration/manual.jsonl"),
                   help="Expert calibration labels (keyed by OLD pair_ids).")
    p.add_argument("--bridge_n", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    queries_path = resolve_repo_path(args.queries)
    pairs_path = resolve_repo_path(args.pairs)
    unified_dir = resolve_repo_path(args.unified_dir)
    cfg = _read_json(resolve_repo_path(args.methods_config))
    out_dir = resolve_repo_path(args.out_dir)

    qids = _query_ids(queries_path)
    qid_set = set(qids)
    N = len(qids)
    print(f"[queries] {N} unique ids (e.g. {qids[0]}..{qids[-1]})")

    # ---- 1) integrity: retrieval dirs -> unified -> pairs ----
    fails: List[str] = []
    base = resolve_repo_path(repo_path("."))
    for m in cfg["methods"]:
        name, mdir = m["name"], resolve_repo_path(Path(m["dir"]))
        got = _dir_ids(mdir)
        if got != qid_set:
            fails.append(f"retrieval[{name}] {mdir}: missing={sorted(qid_set-got)[:5]} extra={sorted(got-qid_set)[:5]}")
        ug = _dir_ids(unified_dir / name)
        if ug != qid_set:
            fails.append(f"unified[{name}]: missing={sorted(qid_set-ug)[:5]} extra={sorted(ug-qid_set)[:5]}")

    pairs = _read_jsonl(pairs_path)
    pair_qids = {p["query_id"] for p in pairs}
    if pair_qids != qid_set:
        fails.append(f"pairs: missing={sorted(qid_set-pair_qids)[:5]} extra={sorted(pair_qids-qid_set)[:5]}")
    # expected pair count = comparisons * N * 2 perms
    n_cmp = len({p["comparison_id"] for p in pairs})
    exp = n_cmp * N * 2
    if len(pairs) != exp:
        fails.append(f"pairs count {len(pairs)} != {n_cmp} comparisons x {N} queries x 2 perms = {exp}")

    if fails:
        print("[FAIL] integrity:")
        for f in fails:
            print("   -", f)
        raise SystemExit(1)
    print(f"[OK] integrity: {N} ids consistent across {len(cfg['methods'])} methods -> unified -> {len(pairs)} pairs "
          f"({n_cmp} comparisons x {N} x 2)")

    # ---- 2) split staged pair files ----
    calib_ids = set(cfg.get("calibration", {}).get("query_ids", []))
    if not calib_ids:
        # fall back to judges.json calibration if methods.json lacks it
        jcfg = _read_json(resolve_repo_path(repo_path("configs/judge_v2/judges.json")))
        calib_ids = set(jcfg.get("calibration", {}).get("query_ids", []))
    bridge_ids = {f"Q5R{ i:03d}" for i in range(1, args.bridge_n + 1)}

    def _write(path: Path, rows: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stages = {
        "pairs_calib.jsonl": [p for p in pairs if p["query_id"] in calib_ids],
        "pairs_bridge30.jsonl": [p for p in pairs if p["query_id"] in bridge_ids],
        "pairs_full150.jsonl": list(pairs),
    }
    for fname, rows in stages.items():
        _write(out_dir / fname, rows)
        print(f"[stage] {fname}: {len(rows)} pairs ({len({r['query_id'] for r in rows})} queries)")

    # ---- 3) re-key expert calibration onto new pair_ids ----
    old_pairs = _read_jsonl(resolve_repo_path(args.old_pairs))
    manual = _read_jsonl(resolve_repo_path(args.manual))
    old_pid_to_key: Dict[str, KEY] = {
        p["pair_id"]: (p["comparison_id"], p["query_id"], p["perm"]) for p in old_pairs
    }
    new_key_to_pid: Dict[KEY, str] = {
        (p["comparison_id"], p["query_id"], p["perm"]): p["pair_id"] for p in pairs
    }
    rekeyed: List[Dict[str, Any]] = []
    missing: List[str] = []
    for row in manual:
        old_pid = row["pair_id"]
        key = old_pid_to_key.get(old_pid)
        new_pid = new_key_to_pid.get(key) if key else None
        if new_pid is None:
            missing.append(old_pid)
            continue
        rekeyed.append({
            "pair_id": new_pid,
            "manual_labels": row["manual_labels"],
            "rekeyed_from": old_pid,
            "semantic_key": {"comparison_id": key[0], "query_id": key[1], "perm": key[2]},
        })
    if missing:
        raise SystemExit(f"[FAIL] re-key: {len(missing)} manual labels could not be mapped (e.g. {missing[:3]})")
    manual_out = resolve_repo_path(repo_path("artifacts/judge_v2/calibration/manual_calib.jsonl"))
    _write(manual_out, rekeyed)
    cov_qids = {r["semantic_key"]["query_id"] for r in rekeyed}
    print(f"[OK] re-keyed {len(rekeyed)} expert labels -> {manual_out} "
          f"(queries {sorted(cov_qids)}, all mapped)")
    print("[DONE] staging complete.")


if __name__ == "__main__":
    main()
