#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build aggregated LLM-as-judge reports for two modes:

1) Multi-way (A–E) "rag_5way":
   Strict schema per run (inside parsed_json / judge_response / flat):
   {
     "relevance": {"A":0..5,...,"E":0..5},
     "usefulness_for_logic": {...},
     "noise": {...},
     "overall": {...},
     "winner": "A".."E" or "",
     "ranking": ["A","B","C","D","E"],
     "failure_letters": ["A"...],
     "confidence": 0..5,
     "rationales": {"A":"...",...,"E":"..."}
   }

2) Pairwise (A/B) "ablation_pairs":
   Strict schema per run:
   {
     "decision": "A"|"B"|"" ,
     "reason": "..."
   }

Key requirement (per user request):
- The "full list of queries" (expected QIDs) is taken from judge_payloads folder,
  not inferred from outputs. If a QID exists in payloads but has no outputs, it is
  treated as a deterministic skipped tie for pairwise (ablation_pairs).

Input directories:
- --judge_outputs_dir: recursively scanned for Q###_N.json
- --judge_payloads_dir: recursively scanned for Q###.json payloads

Outputs (CSV):
- <prefix>_runs.csv:
    per (qid, model, replica) with decoded winner/decision methods
- <prefix>_long.csv:
    only for fiveway: per (qid, model, replica, letter) metrics
- <prefix>_summary.csv:
    only for fiveway: per (method) aggregated metrics pooled over runs
- <prefix>_winners_fiveway.csv:
    per qid winners distribution across runs
- <prefix>_winners_pairwise.csv:
    per qid decision distribution across runs; includes SKIPPED_IDENTICAL if missing outputs

Outputs (Markdown):
- <prefix>_report.md:
    compact summary with win-rates and tie/skip rates.

Notes:
- Comments are in English (as requested).
- No changes required to build_judge_payloads.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LETTERS_5 = ["A", "B", "C", "D", "E"]
LETTERS_2 = ["A", "B"]


# ---------------------------
# Helpers
# ---------------------------

def safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None or x == "":
            return default
        return int(x)
    except Exception:
        return default


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def safe_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    return str(x)


def mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return None
    return sum(xs) / len(xs)


def std(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if len(xs) < 2:
        return 0.0 if xs else None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)

def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first top-level JSON object from a text blob.
    Works even if the text contains markdown fences or extra text.
    """
    if not text:
        return None

    # Fast path: try direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Slow path: find first {...} by brace counting
    s = text
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start:i+1]
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def dump_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def dump_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def load_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def list_json_files_recursive(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*.json") if p.is_file()])


def parse_qid_replica_from_filename(p: Path) -> Tuple[Optional[str], Optional[int]]:
    """
    Supports:
      Q002_2.json -> ("Q002", 2)
      Q002.json   -> ("Q002", None)
    """
    m = re.match(r"^(Q[0-9A-Za-z]+)(?:_(\d+))?\.json$", p.name)
    if not m:
        return None, None
    qid = m.group(1)
    replica = safe_int(m.group(2), None)
    return qid, replica


def _md_table(rows: List[Dict[str, Any]], headers: List[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def _relpath(root: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except Exception:
        return str(p)


# ---------------------------
# Payloads (expected QIDs come from here)
# ---------------------------

@dataclass
class PayloadInfo:
    qid: str
    query: str
    mapping: Dict[str, str]   # letter -> method
    payload_path: str         # relative to payloads root


def load_payloads_recursive(payloads_root: Path) -> Dict[str, PayloadInfo]:
    """
    Load payloads recursively. Assumes payload filename is Q###.json.
    If the same QID appears multiple times (should not for your per-pair folders),
    the last one wins, but we keep path for reference.
    """
    payloads: Dict[str, PayloadInfo] = {}
    for p in list_json_files_recursive(payloads_root):
        if not re.match(r"^Q[0-9A-Za-z]+\.json$", p.name):
            continue
        raw, err = load_json(p)
        if raw is None:
            continue

        qid = safe_str(raw.get("id"))
        if not re.match(r"^Q[0-9A-Za-z]+$", qid):
            continue

        query = safe_str(raw.get("query"))
        mapping_raw = raw.get("private_mapping") or {}
        mapping: Dict[str, str] = {}
        if isinstance(mapping_raw, dict):
            for k, v in mapping_raw.items():
                if isinstance(k, str) and k in ("A", "B", "C", "D", "E"):
                    mapping[k] = safe_str(v)

        payloads[qid] = PayloadInfo(
            qid=qid,
            query=query,
            mapping=mapping,
            payload_path=_relpath(payloads_root, p),
        )
    return payloads


def expected_qids_from_payloads(payloads_root: Path) -> List[str]:
    """
    Build the expected list of qids strictly from payload files.
    We sort by numeric Q index.
    """
    def _qid_sort_key(qid: str) -> tuple[int, str]:
        m = re.match(r"^Q(?:[A-Za-z]+)?(\d+)$", qid)
        if m:
            return (int(m.group(1)), qid)
        m = re.match(r"^Q\d+[A-Za-z]+(\d+)$", qid)
        if m:
            return (int(m.group(1)), qid)
        digits = re.findall(r"\d+", qid)
        return (int(digits[-1]) if digits else 10**9, qid)

    qids: List[str] = []
    for p in list_json_files_recursive(payloads_root):
        m = re.match(r"^(Q[0-9A-Za-z]+)\.json$", p.name)
        if m:
            qids.append(m.group(1))
    # unique + numeric sort
    qids = sorted(set(qids), key=_qid_sort_key)
    return qids


# ---------------------------
# Judge output parsing
# ---------------------------

def _get_judge_block(raw: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns (parsed_judge_dict, error_string)

    Supported runner output:
      {"status":"OK", "parsed_json": {...}} or status ERROR/DRY_RUN

    Legacy:
      {"judge_response": {...}}

    Flat schemas:
      - pairwise: {"decision":..., "reason":...}
      - fiveway:  has all required keys
    """
    if isinstance(raw.get("parsed_json"), dict):
        return raw["parsed_json"], None

    if isinstance(raw.get("judge_response"), dict):
        return raw["judge_response"], None

    # Flat fallback
    if isinstance(raw.get("decision"), str) and isinstance(raw.get("reason"), str):
        return raw, None

    fiveway_required_legacy = [
        "relevance", "usefulness_for_logic", "noise", "overall",
        "winner", "ranking", "failure_letters", "confidence", "rationales"
    ]
    fiveway_required_reasoning = [
        "relevance", "sufficiency_for_reasoning", "coherence", "noise", "overall",
        "winner", "ranking", "failure_letters", "confidence", "rationales"
    ]
    if all(k in raw for k in fiveway_required_legacy) or all(k in raw for k in fiveway_required_reasoning):
        return raw, None

    return None, "missing parsed_json/judge_response (and not a supported flat schema)"


def _detect_mode_from_mapping(mapping: Dict[str, str]) -> str:
    """
    Determine evaluation mode from payload mapping.
    - If mapping includes any of C/D/E -> fiveway
    - Else -> pairwise (your ablations always A/B)
    """
    keys = set(mapping.keys())
    if any(k in keys for k in ("C", "D", "E")):
        return "fiveway"
    return "pairwise"


def _is_tie_run(r: RunRecord) -> bool:
    """
    Treat a parsed OK run as a tie when the judge did not emit a winner/decision.
    Parse errors are never treated as ties.
    """
    if not r.parse_ok:
        return False
    if r.mode == "pairwise":
        return r.decision_letter == ""
    return r.winner_letter == ""


def _normalize_metric_map(d: Any, letters: List[str]) -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {L: None for L in letters}
    if not isinstance(d, dict):
        return out
    for L in letters:
        out[L] = safe_int(d.get(L), None)
    return out


def _validate_fiveway(obj: Dict[str, Any]) -> Optional[str]:
    if "sufficiency_for_reasoning" in obj or "coherence" in obj:
        required = [
            "relevance", "sufficiency_for_reasoning", "coherence", "noise", "overall",
            "winner", "ranking", "failure_letters", "confidence", "rationales"
        ]
    else:
        required = [
            "relevance", "usefulness_for_logic", "noise", "overall",
            "winner", "ranking", "failure_letters", "confidence", "rationales"
        ]
    missing = [k for k in required if k not in obj]
    if missing:
        return f"missing keys: {missing}"
    return None


def _validate_pairwise(obj: Dict[str, Any]) -> Optional[str]:
    required = ["decision", "reason"]
    missing = [k for k in required if k not in obj]
    if missing:
        return f"missing keys: {missing}"
    return None


@dataclass
class RunRecord:
    qid: str
    replica: Optional[int]
    model: str
    file_path: str
    parse_ok: bool
    parse_error: str

    query: str
    mode: str  # "fiveway" | "pairwise"

    # fiveway fields
    relevance: Dict[str, Optional[int]]
    usefulness_for_logic: Dict[str, Optional[int]]
    coherence: Dict[str, Optional[int]]
    noise: Dict[str, Optional[int]]
    overall: Dict[str, Optional[int]]
    winner_letter: str
    ranking_letters: List[str]
    failure_letters: List[str]
    confidence: Optional[int]
    rationales: Dict[str, str]

    # pairwise fields
    decision_letter: str
    reason: str

    # mapping
    letter_to_method: Dict[str, str]


def load_runs(judge_outputs_dir: Path, payloads_by_qid: Dict[str, PayloadInfo]) -> List[RunRecord]:
    runs: List[RunRecord] = []

    empty_metrics = {L: None for L in LETTERS_5}
    empty_rationales = {L: "" for L in LETTERS_5}

    def _blank_record(
        *,
        qid: str,
        replica: Optional[int],
        model: str,
        file_path: str,
        parse_ok: bool,
        parse_error: str,
        query: str,
        mode: str,
        decision_letter: str = "",
        reason: str = "",
        winner_letter: str = "",
        ranking_letters: Optional[List[str]] = None,
        failure_letters: Optional[List[str]] = None,
        confidence: Optional[int] = None,
        relevance: Optional[Dict[str, Optional[int]]] = None,
        usefulness_for_logic: Optional[Dict[str, Optional[int]]] = None,
        coherence: Optional[Dict[str, Optional[int]]] = None,
        noise: Optional[Dict[str, Optional[int]]] = None,
        overall: Optional[Dict[str, Optional[int]]] = None,
        rationales: Optional[Dict[str, str]] = None,
        letter_to_method: Optional[Dict[str, str]] = None,
    ) -> RunRecord:
        return RunRecord(
            qid=qid,
            replica=replica,
            model=model,
            file_path=file_path,
            parse_ok=parse_ok,
            parse_error=parse_error,
            query=query,
            mode=mode,
            relevance=relevance or dict(empty_metrics),
            usefulness_for_logic=usefulness_for_logic or dict(empty_metrics),
            coherence=coherence or dict(empty_metrics),
            noise=noise or dict(empty_metrics),
            overall=overall or dict(empty_metrics),
            winner_letter=winner_letter,
            ranking_letters=ranking_letters or [],
            failure_letters=failure_letters or [],
            confidence=confidence,
            rationales=rationales or dict(empty_rationales),
            decision_letter=decision_letter,
            reason=reason,
            letter_to_method=letter_to_method or {},
        )

    for p in list_json_files_recursive(judge_outputs_dir):
        qid, replica = parse_qid_replica_from_filename(p)
        raw, err = load_json(p)

        if qid is None:
            if raw and isinstance(raw.get("qid"), str) and re.match(r"^Q[0-9A-Za-z]+$", raw["qid"]):
                qid = raw["qid"]
            elif raw and isinstance(raw.get("id"), str) and re.match(r"^Q[0-9A-Za-z]+$", raw["id"]):
                qid = raw["id"]
            else:
                continue

        payload = payloads_by_qid.get(qid)
        mapping = payload.mapping if payload else {}
        mode = _detect_mode_from_mapping(mapping)
        query = payload.query if payload else ""

        model = ""
        if raw and isinstance(raw.get("judge"), dict):
            model = safe_str(raw["judge"].get("name"), "")
        if not model and raw:
            model = safe_str(raw.get("model") or raw.get("judge_model"), "")

        if raw is None:
            runs.append(
                _blank_record(
                    qid=qid,
                    replica=replica,
                    model=model,
                    file_path=str(p),
                    parse_ok=False,
                    parse_error=err or "json load error",
                    query=query,
                    mode=mode,
                    letter_to_method=mapping,
                )
            )
            continue

        status = safe_str(raw.get("status"), "")
        if status and status != "OK":
            raw_text = safe_str(raw.get("raw_response_text"), "")
            salvaged = extract_first_json_object(raw_text)
            if mode == "pairwise" and isinstance(salvaged, dict):
                decision = safe_str(salvaged.get("decision"), "").strip()
                reason = safe_str(salvaged.get("reason"), "")
                if decision.lower() in {"tie", "equal", "same", "draw"}:
                    decision = ""
                if decision in ("A", "B", "") and reason:
                    runs.append(
                        _blank_record(
                            qid=qid,
                            replica=replica,
                            model=model,
                            file_path=str(p),
                            parse_ok=True,
                            parse_error="SALVAGED_FROM_ERROR",
                            query=query,
                            mode=mode,
                            decision_letter=decision,
                            reason=reason,
                            letter_to_method=mapping,
                        )
                    )
                    continue

            runs.append(
                _blank_record(
                    qid=qid,
                    replica=replica,
                    model=model,
                    file_path=str(p),
                    parse_ok=False,
                    parse_error=safe_str(raw.get("error")) or f"status={status}",
                    query=query,
                    mode=mode,
                    letter_to_method=mapping,
                )
            )
            continue

        judge_block, jerr = _get_judge_block(raw)
        if judge_block is None:
            runs.append(
                _blank_record(
                    qid=qid,
                    replica=replica,
                    model=model,
                    file_path=str(p),
                    parse_ok=False,
                    parse_error=jerr or "schema not found",
                    query=query,
                    mode=mode,
                    letter_to_method=mapping,
                )
            )
            continue

        if mode == "pairwise":
            verr = _validate_pairwise(judge_block)
            if verr:
                runs.append(
                    _blank_record(
                        qid=qid,
                        replica=replica,
                        model=model,
                        file_path=str(p),
                        parse_ok=False,
                        parse_error=verr,
                        query=query,
                        mode=mode,
                        letter_to_method=mapping,
                    )
                )
                continue

            decision = safe_str(judge_block.get("decision"), "")
            if decision not in ("A", "B", ""):
                decision = ""
            reason = safe_str(judge_block.get("reason"), "")
            runs.append(
                _blank_record(
                    qid=qid,
                    replica=replica,
                    model=model,
                    file_path=str(p),
                    parse_ok=True,
                    parse_error="",
                    query=query,
                    mode=mode,
                    decision_letter=decision,
                    reason=reason,
                    letter_to_method=mapping,
                )
            )
            continue

        verr = _validate_fiveway(judge_block)
        if verr:
            runs.append(
                _blank_record(
                    qid=qid,
                    replica=replica,
                    model=model,
                    file_path=str(p),
                    parse_ok=False,
                    parse_error=verr,
                    query=query,
                    mode=mode,
                    letter_to_method=mapping,
                )
            )
            continue

        rel = _normalize_metric_map(judge_block.get("relevance"), LETTERS_5)
        main_metric_raw = judge_block.get("usefulness_for_logic")
        if main_metric_raw is None:
            main_metric_raw = judge_block.get("answerability")
        if main_metric_raw is None:
            main_metric_raw = judge_block.get("sufficiency_for_reasoning")
        main_metric = _normalize_metric_map(main_metric_raw, LETTERS_5)

        coherence_raw = judge_block.get("coherence")
        coherence = _normalize_metric_map(coherence_raw, LETTERS_5) if coherence_raw is not None else dict(empty_metrics)
        noi = _normalize_metric_map(judge_block.get("noise"), LETTERS_5)
        ovl = _normalize_metric_map(judge_block.get("overall"), LETTERS_5)

        winner_letter = safe_str(judge_block.get("winner"), "")
        if winner_letter not in LETTERS_5:
            winner_letter = ""

        ranking_letters_raw = judge_block.get("ranking") if isinstance(judge_block.get("ranking"), list) else []
        ranking_letters = [safe_str(x) for x in ranking_letters_raw if safe_str(x) in LETTERS_5]

        failure_letters_raw = judge_block.get("failure_letters") if isinstance(judge_block.get("failure_letters"), list) else []
        failure_letters = sorted({safe_str(x) for x in failure_letters_raw if safe_str(x) in LETTERS_5})

        confidence = safe_int(judge_block.get("confidence"), None)
        rats_raw = judge_block.get("rationales") if isinstance(judge_block.get("rationales"), dict) else {}
        rationales = {L: safe_str(rats_raw.get(L), "") for L in LETTERS_5}

        runs.append(
            _blank_record(
                qid=qid,
                replica=replica,
                model=model,
                file_path=str(p),
                parse_ok=True,
                parse_error="",
                query=query,
                mode=mode,
                relevance=rel,
                usefulness_for_logic=main_metric,
                coherence=coherence,
                noise=noi,
                overall=ovl,
                winner_letter=winner_letter,
                ranking_letters=ranking_letters,
                failure_letters=failure_letters,
                confidence=confidence,
                rationales=rationales,
                letter_to_method=mapping,
            )
        )

    def _sort_key(r: RunRecord) -> Tuple[Any, Any, Any, Any]:
        rep = r.replica if r.replica is not None else 10**9
        return (r.qid, r.model, rep, r.file_path)

    return sorted(runs, key=_sort_key)


# ---------------------------
# Rows building
# ---------------------------

def build_runs_rows(runs: List[RunRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in runs:
        row = {
            "qid": r.qid,
            "replica": r.replica if r.replica is not None else "",
            "model": r.model,
            "query": r.query,
            "mode": r.mode,
            "parse_ok": int(r.parse_ok),
            "parse_error": r.parse_error,
            "source_file": r.file_path,
        }

        if r.mode == "pairwise":
            decision_method = r.letter_to_method.get(r.decision_letter, "") if r.decision_letter else ""
            row.update(
                {
                    "decision_letter": r.decision_letter,
                    "decision_method": decision_method,
                    "reason": r.reason,
                }
            )
        else:
            winner_method = r.letter_to_method.get(r.winner_letter, "") if r.winner_letter else ""
            ranking_methods = [r.letter_to_method.get(L, "") for L in r.ranking_letters]
            row.update(
                {
                    "winner_letter": r.winner_letter,
                    "winner_method": winner_method,
                    "confidence": r.confidence,
                    "failure_letters": ",".join(r.failure_letters),
                    "ranking_letters": ",".join(r.ranking_letters),
                    "ranking_methods": ",".join(ranking_methods),
                    "overall_A": r.overall.get("A"),
                    "overall_B": r.overall.get("B"),
                    "overall_C": r.overall.get("C"),
                    "overall_D": r.overall.get("D"),
                    "overall_E": r.overall.get("E"),
                    "coherence_A": r.coherence.get("A"),
                    "coherence_B": r.coherence.get("B"),
                    "coherence_C": r.coherence.get("C"),
                    "coherence_D": r.coherence.get("D"),
                    "coherence_E": r.coherence.get("E"),
                }
            )
        rows.append(row)
    return rows


def build_long_rows_fiveway(runs: List[RunRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in runs:
        if r.mode != "fiveway":
            continue
        for L in LETTERS_5:
            method = r.letter_to_method.get(L, "")
            rows.append(
                {
                    "qid": r.qid,
                    "replica": r.replica if r.replica is not None else "",
                    "model": r.model,
                    "query": r.query,
                    "letter": L,
                    "method": method,
                    "parse_ok": int(r.parse_ok),
                    "parse_error": r.parse_error,
                    "is_failure_letter": int(L in set(r.failure_letters)),
                    "relevance": r.relevance.get(L),
                    "usefulness_for_logic": r.usefulness_for_logic.get(L),
                    "coherence": r.coherence.get(L),
                    "noise": r.noise.get(L),
                    "overall": r.overall.get(L),
                    "rationale": r.rationales.get(L, ""),
                    "source_file": r.file_path,
                }
            )
    return rows


def build_summary_rows_fiveway(long_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Pooled aggregation for fiveway: per method compute mean/std across all run rows.
    """
    bucket: Dict[str, Dict[str, List[float]]] = {}

    for row in long_rows:
        if safe_int(row.get("parse_ok"), 0) != 1:
            continue
        method = safe_str(row.get("method"))
        if not method:
            continue
        bucket.setdefault(method, {"relevance": [], "usefulness_for_logic": [], "coherence": [], "noise": [], "overall": []})
        for m in ["relevance", "usefulness_for_logic", "coherence", "noise", "overall"]:
            v = safe_float(row.get(m), None)
            if v is None:
                continue
            bucket[method][m].append(v)

    out: List[Dict[str, Any]] = []
    for method, metrics in sorted(bucket.items(), key=lambda kv: kv[0]):
        out.append(
            {
                "method": method,
                "n": max(
                    len(metrics["overall"]),
                    len(metrics["relevance"]),
                    len(metrics["usefulness_for_logic"]),
                    len(metrics["coherence"]),
                    len(metrics["noise"]),
                ),
                "relevance_mean": mean(metrics["relevance"]),
                "relevance_std": std(metrics["relevance"]),
                "usefulness_for_logic_mean": mean(metrics["usefulness_for_logic"]),
                "usefulness_for_logic_std": std(metrics["usefulness_for_logic"]),
                "coherence_mean": mean(metrics["coherence"]),
                "coherence_std": std(metrics["coherence"]),
                "noise_mean": mean(metrics["noise"]),
                "noise_std": std(metrics["noise"]),
                "overall_mean": mean(metrics["overall"]),
                "overall_std": std(metrics["overall"]),
            }
        )
    return out


def build_majority_consensus_fiveway(
    runs: List[RunRecord],
    expected_qids: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build query-level consensus rows for fiveway judging.

    - winner_method is selected by majority vote over judge runs for a qid.
    - candidate metrics are averaged across judge runs per qid/letter.
    - if there is no strict majority winner, winner_method is left empty.
    """
    grouped: Dict[str, List[RunRecord]] = defaultdict(list)
    for r in runs:
        if r.mode == "fiveway" and r.parse_ok:
            grouped[r.qid].append(r)

    consensus_long_rows: List[Dict[str, Any]] = []
    consensus_winners: List[Dict[str, Any]] = []
    metric_names = ["relevance", "usefulness_for_logic", "coherence", "noise", "overall"]

    for qid in expected_qids:
        qruns = grouped.get(qid, [])
        if not qruns:
            continue

        query = qruns[0].query
        n_judges = len(qruns)

        vote_counts: Dict[str, int] = {}
        for r in qruns:
            m = r.letter_to_method.get(r.winner_letter, "") if r.winner_letter else ""
            if not m:
                m = "__TIE_OR_EMPTY__"
            vote_counts[m] = vote_counts.get(m, 0) + 1

        vote_items = sorted(vote_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top_method, top_count = vote_items[0] if vote_items else ("", 0)
        has_majority = (
            top_method != "__TIE_OR_EMPTY__"
            and top_count > (n_judges / 2.0)
            and (len(vote_items) == 1 or top_count > vote_items[1][1])
        )
        consensus_winner = top_method if has_majority else ""

        consensus_winners.append(
            {
                "qid": qid,
                "query": query,
                "n_runs": n_judges,
                "top_winner_method": consensus_winner,
                "top_winner_count": top_count,
                "top_winner_share": (top_count / n_judges) if n_judges else None,
                "winners_breakdown": ";".join([f"{k}:{v}" for k, v in vote_items]),
                "has_majority": int(has_majority),
            }
        )

        for letter in LETTERS_5:
            method = qruns[0].letter_to_method.get(letter, "")
            row: Dict[str, Any] = {
                "qid": qid,
                "query": query,
                "n_judges": n_judges,
                "letter": letter,
                "method": method,
                "parse_ok": 1,
                "parse_error": "",
                "is_failure_letter": 0,
                "relevance": None,
                "usefulness_for_logic": None,
                "coherence": None,
                "noise": None,
                "overall": None,
                "rationale": "",
                "source_file": "",
            }

            failure_votes = sum(1 for r in qruns if letter in set(r.failure_letters))
            row["is_failure_letter"] = int(failure_votes > (n_judges / 2.0))

            for metric in metric_names:
                vals = [safe_float(getattr(r, metric).get(letter), None) for r in qruns]
                row[metric] = mean(vals)

            best_rationale = ""
            best_score = -10**9
            for r in qruns:
                s = safe_float(r.overall.get(letter), None)
                if s is None:
                    continue
                if s > best_score:
                    best_score = s
                    best_rationale = r.rationales.get(letter, "")
            row["rationale"] = best_rationale

            consensus_long_rows.append(row)

    return consensus_long_rows, consensus_winners


def winners_fiveway_per_qid(runs_rows: List[Dict[str, Any]], expected_qids: List[str]) -> List[Dict[str, Any]]:
    """
    Per-qid winner distribution across runs (fiveway).
    Only includes qids that exist in expected_qids.
    """
    counts: Dict[str, Dict[str, int]] = {}
    totals: Dict[str, int] = {}
    queries: Dict[str, str] = {}

    for rr in runs_rows:
        if safe_str(rr.get("mode")) != "fiveway":
            continue
        qid = safe_str(rr.get("qid"))
        queries[qid] = safe_str(rr.get("query"))
        if safe_int(rr.get("parse_ok"), 0) != 1:
            continue
        totals[qid] = totals.get(qid, 0) + 1
        m = safe_str(rr.get("winner_method"))
        if not m:
            m = "__TIE_OR_EMPTY__"
        counts.setdefault(qid, {})
        counts[qid][m] = counts[qid].get(m, 0) + 1

    rows: List[Dict[str, Any]] = []
    for qid in expected_qids:
        total = totals.get(qid, 0)
        if total == 0:
            # For fiveway, missing outputs is not assumed to be a tie.
            continue
        items = sorted(counts.get(qid, {}).items(), key=lambda kv: (-kv[1], kv[0]))
        top_method, top_count = items[0] if items else ("", 0)
        rows.append(
            {
                "qid": qid,
                "query": queries.get(qid, ""),
                "n_runs": total,
                "top_winner_method": "" if top_method == "__TIE_OR_EMPTY__" else top_method,
                "top_winner_count": top_count,
                "top_winner_share": (top_count / total) if total else None,
                "winners_breakdown": ";".join([f"{k}:{v}" for k, v in items]),
            }
        )
    return rows


def winners_pairwise_per_qid_with_skips(
    runs_rows: List[Dict[str, Any]],
    expected_qids: List[str],
) -> List[Dict[str, Any]]:
    """
    Per-qid decision distribution across runs (pairwise).

    If a qid exists in expected_qids but has 0 judge runs (because you skipped identical contexts),
    we explicitly mark it as a deterministic skip tie: __SKIPPED_IDENTICAL__:1

    This meets your requirement: "if payload exists but outputs missing => winner doesn't exist => 1:1".
    """
    counts: Dict[str, Dict[str, int]] = {}
    totals: Dict[str, int] = {}
    queries: Dict[str, str] = {}

    for rr in runs_rows:
        if safe_str(rr.get("mode")) != "pairwise":
            continue
        qid = safe_str(rr.get("qid"))
        queries[qid] = safe_str(rr.get("query"))
        if safe_int(rr.get("parse_ok"), 0) != 1:
            continue

        totals[qid] = totals.get(qid, 0) + 1
        m = safe_str(rr.get("decision_method"))
        if not m:
            m = "__TIE_OR_EMPTY__"
        counts.setdefault(qid, {})
        counts[qid][m] = counts[qid].get(m, 0) + 1

    rows: List[Dict[str, Any]] = []
    for qid in expected_qids:
        total = totals.get(qid, 0)
        if total == 0:
            rows.append(
                {
                    "qid": qid,
                    "query": queries.get(qid, ""),
                    "n_runs": 0,
                    "top_decision_method": "",
                    "top_decision_count": 0,
                    "top_decision_share": 1.0,
                    "decisions_breakdown": "__SKIPPED_IDENTICAL__:1",
                }
            )
            continue

        items = sorted(counts.get(qid, {}).items(), key=lambda kv: (-kv[1], kv[0]))
        top_method, top_count = items[0] if items else ("", 0)

        rows.append(
            {
                "qid": qid,
                "query": queries.get(qid, ""),
                "n_runs": total,
                "top_decision_method": "" if top_method == "__TIE_OR_EMPTY__" else top_method,
                "top_decision_count": top_count,
                "top_decision_share": (top_count / total) if total else None,
                "decisions_breakdown": ";".join([f"{k}:{v}" for k, v in items]),
            }
        )
    return rows


# ---------------------------
# Markdown report
# ---------------------------

def build_md_report(
    runs: List[RunRecord],
    runs_rows: List[Dict[str, Any]],
    summary_fiveway: List[Dict[str, Any]],
    winners_fiveway: List[Dict[str, Any]],
    winners_pairwise: List[Dict[str, Any]],
    *,
    consensus_qids: Optional[int] = None,
) -> str:
    total = len(runs)
    ok = sum(1 for r in runs if r.parse_ok)
    bad = total - ok

    fiveway_runs = [r for r in runs if r.mode == "fiveway"]
    pairwise_runs = [r for r in runs if r.mode == "pairwise"]

    md: List[str] = []
    md.append("# LLM-as-Judge report")
    md.append("")
    md.append(f"- Total run files scanned: **{total}**")
    md.append(f"- Parsed OK: **{ok}**")
    md.append(f"- Parsed ERROR: **{bad}**")
    md.append(f"- Fiveway runs (files): **{len(fiveway_runs)}**")
    md.append(f"- Pairwise runs (files): **{len(pairwise_runs)}**")
    if consensus_qids is not None:
        md.append(f"- Consensus fiveway queries: **{consensus_qids}**")
    md.append("")

    # Pairwise summary: win rates + tie/skip rates across all OK runs
    if pairwise_runs:
        counts: Dict[str, int] = {}
        total_ok = 0
        ties = 0
        for rr in runs_rows:
            if safe_str(rr.get("mode")) != "pairwise":
                continue
            if safe_int(rr.get("parse_ok"), 0) != 1:
                continue
            total_ok += 1
            m = safe_str(rr.get("decision_method"))
            if not m:
                ties += 1
                continue
            counts[m] = counts.get(m, 0) + 1

        rows = []
        for m, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            rows.append({"method": m, "wins": c, "win_rate": round(c / total_ok, 4) if total_ok else 0})

        skipped = sum(1 for w in winners_pairwise if safe_str(w.get("decisions_breakdown")) == "__SKIPPED_IDENTICAL__:1")

        md.append("## Pairwise summary (A/B)")
        md.append("")
        md.append(f"- Total OK pairwise runs: **{total_ok}**")
        md.append(f"- Tie/empty decisions (from runs): **{ties}** ({round(ties/total_ok, 4) if total_ok else 0})")
        md.append(f"- Skipped identical payloads (no runs): **{skipped}**")
        md.append("")
        if rows:
            md.append(_md_table(rows, ["method", "wins", "win_rate"]))
        else:
            md.append("_No non-tie pairwise decisions._")
        md.append("")
        md.append("### Pairwise per-query breakdown (includes SKIPPED_IDENTICAL)")
        md.append("")
        md.append(_md_table(
            winners_pairwise,
            ["qid", "top_decision_method", "top_decision_share", "decisions_breakdown"]
        ))
        md.append("")

    # Fiveway summary: consensus over judge runs
    if fiveway_runs:
        counts: Dict[str, int] = {}
        total_ok = len(winners_fiveway)
        ties = sum(1 for w in winners_fiveway if not safe_str(w.get("top_winner_method")))
        for w in winners_fiveway:
            m = safe_str(w.get("top_winner_method"))
            if not m:
                continue
            counts[m] = counts.get(m, 0) + 1

        rows = []
        for m, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            rows.append({"method": m, "wins": c, "win_rate": round(c / total_ok, 4) if total_ok else 0})

        md.append("## Fiveway consensus summary (A-E, majority vote)")
        md.append("")
        md.append("- Aggregation: per query, winner by majority vote across judges; candidate scores are averaged across judges.")
        md.append(f"- Consensus queries: **{total_ok}**")
        md.append(f"- No-majority queries: **{ties}** ({round(ties/total_ok, 4) if total_ok else 0})")
        md.append("")
        if rows:
            md.append(_md_table(rows, ["method", "wins", "win_rate"]))
        else:
            md.append("_No non-tie fiveway winners._")
        md.append("")
        md.append("### Fiveway overall metric summary (pooled over consensus queries)")
        md.append("")
        if summary_fiveway:
            md.append(_md_table(
                summary_fiveway,
                [
                    "method", "n",
                    "relevance_mean", "relevance_std",
                    "usefulness_for_logic_mean", "usefulness_for_logic_std",
                    "coherence_mean", "coherence_std",
                    "noise_mean", "noise_std",
                    "overall_mean", "overall_std",
                ],
            ))
        else:
            md.append("_No fiveway summary data._")
        md.append("")
        md.append("### Fiveway per-query winner breakdown")
        md.append("")
        if winners_fiveway:
            md.append(_md_table(
                winners_fiveway,
                ["qid", "top_winner_method", "top_winner_share", "winners_breakdown"]
            ))
        else:
            md.append("_No fiveway per-query data._")
        md.append("")

    return "\n".join(md)


# ---------------------------
# CLI
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_outputs_dir", type=str, required=True)
    ap.add_argument("--judge_payloads_dir", type=str, required=True)
    ap.add_argument("--reports_dir", type=str, required=True)
    ap.add_argument("--prefix", type=str, default="judge")
    ap.add_argument("--fmt", type=str, default="csv", choices=["csv"])
    ap.add_argument("--exclude_ties", action="store_true", help="Exclude tie runs from report outputs.")
    args = ap.parse_args()

    judge_outputs_dir = Path(args.judge_outputs_dir)
    judge_payloads_dir = Path(args.judge_payloads_dir)
    reports_dir = Path(args.reports_dir)
    prefix = args.prefix

    # Expected QIDs must come from payloads (per user requirement).
    expected_qids = expected_qids_from_payloads(judge_payloads_dir)

    # Payloads mapping + query text, used for decoding.
    payloads_by_qid = load_payloads_recursive(judge_payloads_dir)

    runs = load_runs(judge_outputs_dir, payloads_by_qid)
    if args.exclude_ties:
        runs = [r for r in runs if not _is_tie_run(r)]
    runs_rows = build_runs_rows(runs)
    long_rows = build_long_rows_fiveway(runs)
    summary_rows = build_summary_rows_fiveway(long_rows)

    consensus_long_rows, consensus_winners_fiveway = build_majority_consensus_fiveway(runs, expected_qids)
    consensus_summary_rows = build_summary_rows_fiveway(consensus_long_rows)

    winners_fiveway = consensus_winners_fiveway
    winners_pairwise = winners_pairwise_per_qid_with_skips(runs_rows, expected_qids)

    # Write CSVs
    dump_csv(
        reports_dir / f"{prefix}_runs.csv",
        runs_rows,
        [
            "qid", "replica", "model", "query", "mode",
            "parse_ok", "parse_error",
            "winner_letter", "winner_method", "confidence",
            "failure_letters", "ranking_letters", "ranking_methods",
            "overall_A", "overall_B", "overall_C", "overall_D", "overall_E",
            "coherence_A", "coherence_B", "coherence_C", "coherence_D", "coherence_E",
            "decision_letter", "decision_method", "reason",
            "source_file",
        ],
    )

    dump_csv(
        reports_dir / f"{prefix}_long.csv",
        long_rows,
        [
            "qid", "replica", "model", "query", "letter", "method",
            "parse_ok", "parse_error", "is_failure_letter",
            "relevance", "usefulness_for_logic", "coherence", "noise", "overall",
            "rationale", "source_file",
        ],
    )

    dump_csv(
        reports_dir / f"{prefix}_summary.csv",
        consensus_summary_rows,
        [
            "method", "n",
            "relevance_mean", "relevance_std",
            "usefulness_for_logic_mean", "usefulness_for_logic_std",
            "coherence_mean", "coherence_std",
            "noise_mean", "noise_std",
            "overall_mean", "overall_std",
        ],
    )

    dump_csv(
        reports_dir / f"{prefix}_consensus_long.csv",
        consensus_long_rows,
        [
            "qid", "query", "n_judges", "letter", "method",
            "parse_ok", "parse_error", "is_failure_letter",
            "relevance", "usefulness_for_logic", "coherence", "noise", "overall",
            "rationale", "source_file",
        ],
    )

    dump_csv(
        reports_dir / f"{prefix}_consensus_summary.csv",
        consensus_summary_rows,
        [
            "method", "n",
            "relevance_mean", "relevance_std",
            "usefulness_for_logic_mean", "usefulness_for_logic_std",
            "coherence_mean", "coherence_std",
            "noise_mean", "noise_std",
            "overall_mean", "overall_std",
        ],
    )

    dump_csv(
        reports_dir / f"{prefix}_winners_fiveway.csv",
        winners_fiveway,
        [
            "qid", "query", "n_runs",
            "top_winner_method", "top_winner_count", "top_winner_share",
            "winners_breakdown", "has_majority",
        ],
    )

    dump_csv(
        reports_dir / f"{prefix}_winners_pairwise.csv",
        winners_pairwise,
        [
            "qid", "query", "n_runs",
            "top_decision_method", "top_decision_count", "top_decision_share",
            "decisions_breakdown",
        ],
    )

    # Markdown report
    md_text = build_md_report(
        runs=runs,
        runs_rows=runs_rows,
        summary_fiveway=consensus_summary_rows,
        winners_fiveway=winners_fiveway,
        winners_pairwise=winners_pairwise,
        consensus_qids=len(consensus_winners_fiveway),
    )
    if args.exclude_ties:
        md_text = md_text.replace("# LLM-as-Judge report", "# LLM-as-Judge report (ties excluded)", 1)
    dump_md(reports_dir / f"{prefix}_report.md", md_text)

    print(f"Wrote reports to: {reports_dir.resolve()}")
    print(f" - {prefix}_runs.csv")
    print(f" - {prefix}_long.csv")
    print(f" - {prefix}_summary.csv")
    print(f" - {prefix}_consensus_long.csv")
    print(f" - {prefix}_consensus_summary.csv")
    print(f" - {prefix}_winners_fiveway.csv")
    print(f" - {prefix}_winners_pairwise.csv")
    print(f" - {prefix}_report.md")


if __name__ == "__main__":
    main()
