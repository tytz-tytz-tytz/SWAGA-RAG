# scripts/build_judge_payloads.py
"""
Build offline blind judge payloads with deterministic shuffling.

Output format (KEPT AS-IS for backward compatibility):
{
  "id": "Q001",
  "query": "...",
  "contexts_for_judge": { "A": "...", "B": "...", "C": "...", "D": "...", "E": "..." },
  "private_mapping": { "A": "bm25", "B": "ontology", ... }
}

Notes:
- This script intentionally keeps the legacy payload structure above.
- Debug payloads (optional) are written separately and are not intended for judges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from judge_prep.clean_cap import CleanCapConfig, clean_and_cap


# =========================
# Config models
# =========================

@dataclass(frozen=True)
class MethodSpec:
    method: str
    dir: Path


@dataclass(frozen=True)
class ShuffleConfig:
    enabled: bool
    seed: int
    keys: List[str]
    mode: str  # "seed_plus_queryid_hash" | "seed_plus_index"


@dataclass(frozen=True)
class DebugConfig:
    enabled: bool
    output_dir: Path
    include_meta: bool


@dataclass(frozen=True)
class AppConfig:
    queries_file: Optional[Path]
    output_dir: Path
    methods: List[MethodSpec]
    clean_cap: CleanCapConfig
    shuffle: ShuffleConfig
    debug: DebugConfig


# =========================
# JSON IO
# =========================

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# Deterministic RNG helpers
# =========================

def _stable_u32_from_str(s: str) -> int:
    """Stable hash across runs and platforms (unlike Python's built-in hash())."""
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _make_rng_for_query(app_cfg: AppConfig, qid: str, index: int) -> random.Random:
    """Deterministic shuffling per query."""
    if not app_cfg.shuffle.enabled:
        # Fixed mapping: keep methods order as provided by config, keys order as provided.
        return random.Random(0)

    base_seed = int(app_cfg.shuffle.seed)

    if app_cfg.shuffle.mode == "seed_plus_queryid_hash":
        per_q = _stable_u32_from_str(qid)
        seed = (base_seed + per_q) & 0xFFFFFFFF
        return random.Random(seed)

    # seed_plus_index
    seed = (base_seed + int(index)) & 0xFFFFFFFF
    return random.Random(seed)


# =========================
# Config loading
# =========================

def _load_config(config_path: Path) -> AppConfig:
    cfg_raw = _read_json(config_path)
    base = config_path.parent

    # Optional canonical queries source (used to override query text from method outputs).
    queries_file_raw = cfg_raw.get("queries_file")
    queries_file = (base / queries_file_raw).resolve() if queries_file_raw else None

    # Output dir (legacy judge payload structure is preserved).
    output_dir = (base / cfg_raw.get("output_dir", "artifacts/judge_payloads")).resolve()

    # Methods
    methods_raw = cfg_raw.get("methods", [])
    if not isinstance(methods_raw, list) or len(methods_raw) not in (2, 3, 4, 5):
        raise ValueError("Config must contain 2..5 methods under 'methods'.")
    n_methods = len(methods_raw)


    methods: List[MethodSpec] = []
    for m in methods_raw:
        if "method" not in m or "dir" not in m:
            raise ValueError("Each methods[] entry must have 'method' and 'dir'.")
        methods.append(
            MethodSpec(
                method=str(m["method"]),
                dir=(base / str(m["dir"])).resolve(),
            )
        )

    # Clean+cap config
    cc = cfg_raw.get("clean_cap", {}) if isinstance(cfg_raw.get("clean_cap", {}), dict) else {}
    clean_cfg = CleanCapConfig(
        min_chars=int(cc.get("min_chars", 20)),
        token_budget_per_method=int(cc.get("token_budget_per_method", 350)),
        encoding_name=str(cc.get("encoding_name", "cl100k_base")),
        top_k_chunks=cc.get("top_k_chunks", 7),
        joiner=str(cc.get("joiner", "\n\n---\n\n")),
        min_chunk_chars=int(cc.get("min_chunk_chars", 5)),
        allow_first_chunk_trim=bool(cc.get("allow_first_chunk_trim", True)),
    )

    # Shuffle config
    sh = cfg_raw.get("shuffle", {}) if isinstance(cfg_raw.get("shuffle", {}), dict) else {}
    shuffle_cfg = ShuffleConfig(
        enabled=bool(sh.get("enabled", True)),
        seed=int(sh.get("seed", 42)),
        keys=[str(k) for k in sh.get("keys", ["A", "B", "C", "D", "E"])],
        mode=str(sh.get("mode", "seed_plus_queryid_hash")),
    )
    if len(shuffle_cfg.keys) != n_methods:
        raise ValueError("shuffle.keys length must match number of methods.")
    if len(set(shuffle_cfg.keys)) != len(shuffle_cfg.keys):
        raise ValueError("shuffle.keys must be unique.")
    if shuffle_cfg.mode not in ("seed_plus_queryid_hash", "seed_plus_index"):
        raise ValueError("shuffle.mode must be 'seed_plus_queryid_hash' or 'seed_plus_index'.")

    # Debug config (optional)
    dbg = cfg_raw.get("debug", {}) if isinstance(cfg_raw.get("debug", {}), dict) else {}
    debug_cfg = DebugConfig(
        enabled=bool(dbg.get("enabled", False)),
        output_dir=(base / str(dbg.get("output_dir", "artifacts/judge_payloads_debug"))).resolve(),
        include_meta=bool(dbg.get("include_meta", True)),
    )

    return AppConfig(
        queries_file=queries_file,
        output_dir=output_dir,
        methods=methods,
        clean_cap=clean_cfg,
        shuffle=shuffle_cfg,
        debug=debug_cfg,
    )


# =========================
# Data loading helpers
# =========================

def _list_query_ids_from_methods(methods: List[MethodSpec]) -> List[str]:
    """
    Use the intersection of filenames across methods to ensure every method has Qxxx.json.
    """
    sets: List[set[str]] = []
    for m in methods:
        files = {p.stem for p in m.dir.glob("Q*.json") if p.is_file()}
        sets.append(files)
    common = set.intersection(*sets) if sets else set()
    return sorted(common)


def _read_query_text(queries_file: Optional[Path], qid: str, fallback_query: str) -> str:
    """
    Load query text from a canonical queries file, if provided.
    Otherwise, keep query text from method outputs (fallback_query).
    """
    if queries_file is None:
        return fallback_query

    path = queries_file
    if not path.exists():
        return fallback_query

    # queries.jsonl format: {"id": "...", "query": "..."}
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if str(obj.get("id")) == qid:
                    return str(obj.get("query", fallback_query))
        return fallback_query

    # queries.json format: list or dict
    obj = _read_json(path)
    if isinstance(obj, list):
        for it in obj:
            if str(it.get("id")) == qid:
                return str(it.get("query", fallback_query))
        return fallback_query
    if isinstance(obj, dict):
        if qid in obj:
            return str(obj[qid])
    return fallback_query


def _load_method_result(spec: MethodSpec, qid: str) -> Tuple[str, List[str]]:
    """
    Load one method's output for one query.
    Expected per-method file format:
    {
      "id": "Q001",
      "query": "...",
      "output": ["chunk1", "chunk2", ...]
    }
    """
    in_path = spec.dir / f"{qid}.json"
    data = _read_json(in_path)

    file_qid = str(data.get("id", "")).strip()
    if file_qid and file_qid != qid:
        raise ValueError(f"ID mismatch in {in_path}: expected {qid}, got {file_qid}")

    query = str(data.get("query", "")).strip()

    output = data.get("output")
    if isinstance(output, list):
        chunks = [str(x) for x in output]
        return query, chunks

    output_items = data.get("output_items")
    if isinstance(output_items, list):
        chunks: List[str] = []
        for item in output_items:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text)
        return query, chunks

    raise ValueError(
        f"Invalid output format in {in_path} (expected 'output' list or 'output_items' list)."
    )


# =========================
# Payload builder
# =========================

def build_payload_for_query(app_cfg: AppConfig, qid: str, index: int) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Build a single legacy-format judge payload + optional debug payload.
    """
    cleaned_variants: List[Tuple[str, str]] = []  # (method_name, cleaned_text)
    debug_variants: List[Dict[str, Any]] = []

    query_text_fallback: Optional[str] = None

    # 1) Load and clean all methods (identical clean+cap for fairness).
    for spec in app_cfg.methods:
        query_from_file, chunks = _load_method_result(spec, qid)

        if query_text_fallback is None:
            query_text_fallback = query_from_file

        # Use the same query text for all methods (avoid method-specific query drift).
        q_for_clean = (query_text_fallback or query_from_file or "").strip()

        res = clean_and_cap(chunks, app_cfg.clean_cap, query=q_for_clean)
        cleaned_variants.append((spec.method, res.text))

        if app_cfg.debug.enabled:
            debug_variants.append(
                {
                    "method": spec.method,
                    "tokens": res.tokens,
                    "stats": {
                        "in_chunks": res.stats.in_chunks,
                        "kept_chunks": res.stats.kept_chunks,
                        "dropped_empty": res.stats.dropped_empty,
                        "dropped_dedup": res.stats.dropped_dedup,
                        "dropped_caption": res.stats.dropped_caption,
                        "dropped_heading": res.stats.dropped_heading,
                        "dropped_table_header": res.stats.dropped_table_header,
                        "dropped_trailing_colon": res.stats.dropped_trailing_colon,
                        "truncated": res.stats.truncated,
                        "keyword_filter_applied": getattr(res.stats, "keyword_filter_applied", False),
                        "keyword_filter_fallback_used": getattr(res.stats, "keyword_filter_fallback_used", False),
                    },
                }
            )

    if query_text_fallback is None:
        query_text_fallback = ""

    # Canonicalize the query text if a queries file is provided.
    query_text = _read_query_text(app_cfg.queries_file, qid, query_text_fallback)

    # 2) Shuffle and assign to keys (A..E) deterministically per query.
    keys = list(app_cfg.shuffle.keys)
    variants = list(cleaned_variants)

    if app_cfg.shuffle.enabled:
        rng = _make_rng_for_query(app_cfg, qid, index)
        rng.shuffle(variants)

    contexts_for_judge: Dict[str, str] = {}
    private_mapping: Dict[str, str] = {}

    for k, (method_name, text) in zip(keys, variants, strict=True):
        contexts_for_judge[k] = text
        private_mapping[k] = method_name

    # 3) Legacy payload structure (kept as requested).
    payload: Dict[str, Any] = {
        "id": qid,
        "query": query_text,
        "contexts_for_judge": contexts_for_judge,
        "private_mapping": private_mapping,
    }

    # 4) Optional debug payload (not for judges).
    debug_payload: Optional[Dict[str, Any]] = None
    if app_cfg.debug.enabled:
        debug_payload = {
            "id": qid,
            "query": query_text,
            "shuffle": {
                "enabled": app_cfg.shuffle.enabled,
                "seed": app_cfg.shuffle.seed,
                "mode": app_cfg.shuffle.mode,
                "keys": keys,
                "private_mapping": private_mapping,
            },
            "clean_cap": {
                "token_budget_per_method": app_cfg.clean_cap.token_budget_per_method,
                "encoding_name": app_cfg.clean_cap.encoding_name,
                "min_chars": app_cfg.clean_cap.min_chars,
                "min_paragraph_chars": getattr(app_cfg.clean_cap, "min_paragraph_chars", None),
                "joiner": getattr(app_cfg.clean_cap, "joiner", None),
            },
            "variants": debug_variants,
        }
        if app_cfg.debug.include_meta:
            debug_payload["meta"] = {
                "note": "Debug file is not intended for judges.",
            }

    return payload, debug_payload


# =========================
# CLI
# =========================

def main() -> int:
    parser = argparse.ArgumentParser(description="Build offline blind judge payloads with deterministic shuffling.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/judge_prep.json",
        help="Path to JSON config (default: configs/judge_prep.json).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Optional: build only one query id, e.g. Q010.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: build only first N queries (after sorting).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already existing payload/debug files and keep the rest.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    app_cfg = _load_config(cfg_path)

    qids = _list_query_ids_from_methods(app_cfg.methods)
    if args.only:
        if args.only not in qids:
            raise ValueError(f"Query id {args.only} not found in all method folders.")
        qids = [args.only]

    if args.limit is not None:
        qids = qids[: max(0, int(args.limit))]

    if not qids:
        print("No common Q*.json files found across all method folders.")
        return 1

    app_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if app_cfg.debug.enabled:
        app_cfg.debug.output_dir.mkdir(parents=True, exist_ok=True)

    built = 0
    for idx, qid in enumerate(qids):
        payload, debug_payload = build_payload_for_query(app_cfg, qid, idx)

        out_path = app_cfg.output_dir / f"{qid}.json"
        if not (args.resume and out_path.exists()):
            _write_json(out_path, payload)

        if app_cfg.debug.enabled and debug_payload is not None:
            dbg_path = app_cfg.debug.output_dir / f"{qid}.debug.json"
            if not (args.resume and dbg_path.exists()):
                _write_json(dbg_path, debug_payload)

        built += 1
        print(f"[OK] {qid} -> {out_path}")

    print(f"Done. Built {built} payload(s) into: {app_cfg.output_dir}")
    if app_cfg.debug.enabled:
        print(f"Debug files saved into: {app_cfg.debug.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
