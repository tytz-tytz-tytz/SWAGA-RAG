# src/judge_prep/clean_cap.py
"""
Minimal, rank-faithful clean+cap for LLM-as-judge.

Goals:
- Treat retrieved outputs as the method's "product" (ranking + chunk choice matters).
- Apply identical logic across methods.
- Preserve chunk order exactly (top-k in rank order).
- Use a shared token budget per method.
- Cap primarily on CHUNK BOUNDARIES: never cut a chunk in the middle.
- Only exception: if the FIRST kept chunk alone exceeds the budget, we trim it
  safely to a sentence boundary so we don't end up with an empty context.
- Avoid aggressive filtering (no dedup, no keyword overlap, no table/header heuristics).
- Stable, reproducible behavior.

Output:
- Join chunks with the configured joiner (recommended: "\\n\\n---\\n\\n").
- If nothing usable remains: "[NO RELEVANT CONTEXT FOUND]".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


# =========================
# Config and result models
# =========================

@dataclass(frozen=True)
class CleanCapConfig:
    # Minimum acceptable final output length in characters (after cap).
    min_chars: int = 20

    # Token budget per method (same for all methods).
    token_budget_per_method: int = 350

    # Optional: tiktoken encoding name (used if tiktoken is installed).
    encoding_name: str = "cl100k_base"

    # How many top chunks to consider (None = no limit).
    top_k_chunks: Optional[int] = 7

    # Joiner used between kept chunks.
    joiner: str = "\n\n---\n\n"

    # Drop chunks that become too short after normalization (to avoid empty crumbs).
    min_chunk_chars: int = 5

    # If True, allow trimming ONLY the first chunk when it exceeds budget.
    allow_first_chunk_trim: bool = True


@dataclass(frozen=True)
class CleanCapStats:
    in_chunks: int
    considered_chunks: int
    kept_chunks: int
    dropped_empty: int
    truncated: bool
    truncated_first_chunk: bool


@dataclass(frozen=True)
class CleanCapResult:
    text: str
    tokens: int
    stats: CleanCapStats


# =========================
# Public API
# =========================

_NO_CONTEXT = "[NO RELEVANT CONTEXT FOUND]"


def clean_and_cap(chunks: List[str], config: CleanCapConfig, query: str = "") -> CleanCapResult:
    """
    Clean and cap a list of retrieved chunks (already ordered by retrieval rank).

    Notes:
    - `query` is accepted for call compatibility but intentionally unused here
      (no keyword filtering; we preserve retrieval behavior).
    """
    in_chunks = len(chunks)

    # 1) Take top-k chunks (rank-faithful).
    if config.top_k_chunks is None:
        considered = chunks
    else:
        considered = chunks[: max(0, int(config.top_k_chunks))]
    considered_chunks = len(considered)

    # 2) Normalize chunks lightly; drop empty ones.
    normalized: List[str] = []
    dropped_empty = 0
    for ch in considered:
        ch2 = _normalize_text(ch)
        if not ch2 or len(ch2) < max(0, int(config.min_chunk_chars)):
            dropped_empty += 1
            continue
        normalized.append(ch2)

    # 3) Cap by tokens on chunk boundaries.
    budget = max(0, int(config.token_budget_per_method))
    kept: List[str] = []
    used = 0
    truncated = False
    truncated_first_chunk = False

    for i, ch in enumerate(normalized):
        ch_tokens = _count_tokens(ch, cfg=config)

        # If it fits, keep it as-is.
        if used + ch_tokens <= budget:
            kept.append(ch)
            used += ch_tokens
            continue

        # Does not fit. We do NOT cut chunks in the middle.
        truncated = True

        # Special case: first chunk alone exceeds budget.
        if i == 0 and config.allow_first_chunk_trim and budget > 0:
            trimmed = _safe_sentence_trim(ch, token_limit=budget, cfg=config).strip()
            if trimmed and _count_tokens(trimmed, cfg=config) > 0:
                kept.append(trimmed)
                used = _count_tokens(trimmed, cfg=config)
                truncated_first_chunk = True
            # Either way, we stop after handling the first chunk.
        break

    text = config.joiner.join(kept).strip()
    text = _final_format(text)
    tokens = _count_tokens(text, cfg=config)

    # 4) Empty / too short handling.
    if not text or len(text) < max(0, int(config.min_chars)):
        text = _NO_CONTEXT
        tokens = _count_tokens(text, cfg=config)
        truncated = False
        truncated_first_chunk = False
        kept_chunks = 0
    else:
        kept_chunks = len(kept)

    return CleanCapResult(
        text=text,
        tokens=tokens,
        stats=CleanCapStats(
            in_chunks=in_chunks,
            considered_chunks=considered_chunks,
            kept_chunks=kept_chunks,
            dropped_empty=dropped_empty,
            truncated=truncated,
            truncated_first_chunk=truncated_first_chunk,
        ),
    )


# =========================
# Token counting (tiktoken optional)
# =========================

def _count_tokens(text: str, cfg: CleanCapConfig) -> int:
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding(cfg.encoding_name)
        return len(enc.encode(text))
    except Exception:
        # Fallback: rough token estimate (word/punct pieces).
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _prefix_by_tokens(text: str, token_limit: int, cfg: CleanCapConfig) -> str:
    """
    Return a prefix of `text` that is within `token_limit` tokens.
    Uses tiktoken when available, otherwise approximates by regex tokens.
    """
    if not text or token_limit <= 0:
        return ""

    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding(cfg.encoding_name)
        toks = enc.encode(text)[:token_limit]
        return enc.decode(toks)
    except Exception:
        matches = list(re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE))
        if len(matches) <= token_limit:
            return text
        end = matches[token_limit - 1].end()
        return text[:end]


# =========================
# Sentence-safe trim (only for first chunk overflow)
# =========================

_SENT_BOUNDARY_RE = re.compile(r"[.!?…]+(?:[\"'”»)\]]+)?\s*$", re.UNICODE)


def _safe_sentence_trim(text: str, token_limit: int, cfg: CleanCapConfig) -> str:
    """
    Trim `text` to <= token_limit while avoiding cutting mid-sentence.

    Strategy:
    - take a token-aware prefix that fits token_limit,
    - then search backwards for a sentence boundary (. ! ? …),
    - if none found, fall back to last whitespace.
    """
    if not text or token_limit <= 0:
        return ""

    if _count_tokens(text, cfg=cfg) <= token_limit:
        return text

    prefix = _prefix_by_tokens(text, token_limit, cfg=cfg).rstrip()
    if not prefix:
        return ""

    last = None
    for m in re.finditer(r"[.!?…]+", prefix):
        last = m.end()

    if last is not None and last >= max(20, int(0.3 * len(prefix))):
        candidate = prefix[:last].rstrip()
        if _SENT_BOUNDARY_RE.search(candidate + " "):
            return candidate

    ws = prefix.rfind(" ")
    if ws > 0:
        return prefix[:ws].rstrip()

    return prefix.strip()


# =========================
# Formatting / normalization
# =========================

def _final_format(text: str) -> str:
    """
    Very light formatting normalization:
    - normalize spaces,
    - normalize newlines (no more than 2 in a row).
    """
    t = (text or "").replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _normalize_text(s: str) -> str:
    """
    Light normalization without semantic filtering:
    - normalize whitespace and line breaks,
    - trim lines,
    - keep internal newlines (judge can benefit from structure).
    """
    if not s:
        return ""
    s = s.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    lines = [ln.strip() for ln in s.split("\n")]
    # Keep non-empty lines, preserve order.
    s2 = "\n".join([ln for ln in lines if ln != ""]).strip()
    return s2
