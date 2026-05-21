"""
Shared helpers for calibration metrics + final aggregation.

The crucial logic here is the "perm inversion": a judge looking at perm AB
sees method_first as A and method_second as B; in perm BA the same physical
methods are swapped, so to compare like-for-like we must normalize every
decision to refer to (method_first, method_second).

Conventions used everywhere downstream:
  normalize_label(label, perm) -> "first" | "second" | "equal"
where the *first* / *second* are taken from comparison_id = "{first}_vs_{second}".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


AXES = ("relevance", "cleanliness", "sufficiency")
VALID_LABELS = ("A", "B", "equal")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def split_comparison_id(comparison_id: str) -> Tuple[str, str]:
    parts = comparison_id.split("_vs_")
    if len(parts) != 2:
        raise ValueError(f"Bad comparison_id: {comparison_id}")
    return parts[0], parts[1]


def normalize_label(label: str, perm: str) -> Optional[str]:
    """Translate a judge label (A/B/equal) seen under perm AB/BA into one of
    {"first","second","equal"} in the canonical (first, second) orientation."""
    if label not in VALID_LABELS:
        return None
    if label == "equal":
        return "equal"
    if perm == "AB":
        return "first" if label == "A" else "second"
    if perm == "BA":
        return "first" if label == "B" else "second"
    return None


def invert_label_ab(label: str) -> str:
    """If the AB orientation says X, the BA orientation should say invert(X)
    when the judge is permutation-consistent. equal -> equal."""
    if label == "A":
        return "B"
    if label == "B":
        return "A"
    return label


def load_pairs_index(pairs_path: Path) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(pairs_path)
    return {row["pair_id"]: row for row in rows}


def load_decisions(path: Path) -> Dict[str, Dict[str, Any]]:
    """Last-write-wins per pair_id. Only ok decisions populate labels."""
    out: Dict[str, Dict[str, Any]] = {}
    for rec in read_jsonl(path):
        pid = rec.get("pair_id")
        if isinstance(pid, str):
            out[pid] = rec
    return out


def axis_label(decision_rec: Dict[str, Any], axis: str) -> Optional[str]:
    if decision_rec.get("status") != "ok":
        return None
    labels = decision_rec.get("labels") or {}
    v = labels.get(axis)
    if v in VALID_LABELS:
        return v
    return None
