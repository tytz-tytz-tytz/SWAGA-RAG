"""Text-hygiene heuristics shared by the BM25 and classic RAG baselines.

These functions classify a chunk of retrieved text using only the text
itself (no structure/metadata), so the baselines can filter out
captions, table headers, headings, bullet-only fragments and obviously
incomplete snippets before returning context.
"""

from __future__ import annotations

import re
from typing import List


def normalize_text_for_dedup(text: str) -> str:
    """Lowercase, collapse whitespace and trim edge punctuation for dedup keys."""
    t = (text or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" \t\n\r.,;:!—-")
    return t


def looks_like_caption(text: str) -> bool:
    """Detect figure/table captions, e.g. "Рисунок 51 — ...", "Table 3 - ..."."""
    t = (text or "").strip().lower()
    if not t:
        return False
    prefixes = ("рисунок", "таблица", "figure", "table", "диаграмма", "схема", "листинг")
    return t.startswith(prefixes)


def looks_like_bullet_list(text: str) -> bool:
    """Detect chunks dominated by bullet/list formatting rather than prose."""
    t = (text or "").strip()
    if not t:
        return False

    bullet_markers = ["•", "—", "-", "*", "·"]
    bullet_count = sum(t.count(m) for m in bullet_markers)
    newline_count = t.count("\n")

    if bullet_count >= 3:
        return True
    if t.lstrip().startswith("•") and newline_count >= 1:
        return True
    if newline_count >= 2 and bullet_count >= 1:
        return True

    return False


def looks_like_table_header(text: str) -> bool:
    """Detect table-like header rows: many Title-Case tokens, no sentence punctuation."""
    t = (text or "").strip()
    if not t:
        return False

    # Sentence punctuation means it is probably prose, not a header row.
    if any(ch in t for ch in ".!?;"):
        return False

    tokens = [x for x in t.split() if x]
    if len(tokens) < 5:
        return False

    upper_initial = sum(1 for tok in tokens if tok[:1].isupper())
    ratio = upper_initial / max(1, len(tokens))

    if ratio >= 0.5 and len(tokens) >= 5:
        return True
    if len(tokens) >= 10:
        return True

    return False


def looks_like_heading(text: str) -> bool:
    """Detect short heading-like chunks: short, no sentence punctuation, mostly letters."""
    t = (text or "").strip()
    if not t:
        return False
    if len(t) > 80:
        return False
    if any(ch in t for ch in ".!?"):
        return False
    letters_spaces = sum(1 for ch in t if ch.isalpha() or ch.isspace())
    if letters_spaces / max(1, len(t)) >= 0.9 and t[:1].isupper():
        return True
    return False


_INCOMPLETE_PATTERNS: List[str] = [
    r"\(см\.\s*$",                    # ends with "(см."
    r"см\.\s*рисунок\s*\d+\)\s*:$",   # ends with "... (см. Рисунок 180):"
    r"см\.\s*подробнее.*$",           # "см. подробнее ..."
    r"в\s+разделе\s*$",               # ends with "в разделе"
    r"в\s+разделе\s*\(?$",            # ends with "в разделе ("
]


def looks_incomplete(text: str) -> bool:
    """Detect fragments that dangle into missing continuation ("см. подробнее...")."""
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(re.search(pat, t) for pat in _INCOMPLETE_PATTERNS)
