from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Set

from classic_rag.index.store import ClassicRAGIndex
from classic_rag.rag.retrieve import retrieve_with_scores
from rag_common.text_hygiene import (
    normalize_text_for_dedup,
    looks_like_caption,
    looks_like_bullet_list,
    looks_like_table_header,
    looks_like_heading,
    looks_incomplete,
)


@dataclass(frozen=True)
class HeuristicRAGConfig:
    # Final number of chunks returned (context budget).
    top_k: int = 10

    # Retrieve more candidates first, then filter down to top_k.
    candidate_multiplier: int = 6

    # Filter out very short chunks (headings/captions/noise).
    min_chars: int = 80

    # Deduplicate texts after normalization.
    deduplicate: bool = True

    # Drop common caption-like chunks (e.g., "Рисунок 51 — ...", "Table 3 - ...").
    drop_captions: bool = True

    # Drop chunks dominated by bullets / list formatting.
    drop_bullets: bool = True

    # Drop fragments ending with ":" (often incomplete / followed by a list).
    drop_colon_trailing: bool = True

    # Drop table-like header rows (many short "column name" tokens).
    drop_table_like: bool = True

    # Drop common incomplete phrases ("см. подробнее...", "в разделе", etc.).
    drop_incomplete_phrases: bool = True

    # Drop short heading-like chunks (e.g., "Редактирование событий").
    drop_headings: bool = True

    # In fallback, still keep some minimum length to avoid tiny junk.
    fallback_min_chars: int = 40


def select_heuristic_from_candidates(
    candidates: List[Tuple[str, str, float]],
    cfg: HeuristicRAGConfig = HeuristicRAGConfig(),
) -> List[Tuple[str, str, float]]:
    """
    Heuristic-enhanced dense retrieval without using any structure/metadata.

    Steps:
    1) Oversample candidates with dense retrieval.
    2) Filter obvious non-informative chunks (captions/lists/table headers/headings/incomplete fragments).
    3) Apply min length and dedup.
    4) Fallback to fill top_k if too strict (still avoid obvious junk and duplicates).
    """
    out: List[Tuple[str, str, float]] = []
    seen: Set[str] = set()

    def _accept_strict(txt: str) -> bool:
        if not txt:
            return False

        if cfg.drop_captions and looks_like_caption(txt):
            return False

        if cfg.drop_bullets and looks_like_bullet_list(txt):
            return False

        if cfg.drop_table_like and looks_like_table_header(txt):
            return False

        if cfg.drop_headings and looks_like_heading(txt):
            return False

        if cfg.drop_incomplete_phrases and looks_incomplete(txt):
            return False

        if cfg.drop_colon_trailing and txt.endswith(":"):
            return False

        if len(txt) < cfg.min_chars:
            return False

        return True

    # Pass 1: strict filtering
    for cid, text, score in candidates:
        txt = (text or "").strip()

        if not _accept_strict(txt):
            continue

        if cfg.deduplicate:
            key = normalize_text_for_dedup(txt)
            if key in seen:
                continue
            seen.add(key)

        out.append((cid, txt, float(score)))
        if len(out) >= cfg.top_k:
            break

    # Pass 2: fallback if filters were too strict.
    # Relax most constraints, but keep minimal hygiene and avoid tiny fragments.
    if len(out) < cfg.top_k:
        for cid, text, score in candidates:
            txt = (text or "").strip()
            if not txt:
                continue

            if cfg.drop_captions and looks_like_caption(txt):
                continue

            if cfg.drop_table_like and looks_like_table_header(txt):
                continue

            if cfg.drop_headings and looks_like_heading(txt):
                continue

            if len(txt) < cfg.fallback_min_chars:
                continue

            if cfg.deduplicate:
                key = normalize_text_for_dedup(txt)
                if key in seen:
                    continue
                seen.add(key)

            out.append((cid, txt, float(score)))
            if len(out) >= cfg.top_k:
                break

    return out


def retrieve_heuristic_with_scores(
    index: ClassicRAGIndex,
    query: str,
    cfg: HeuristicRAGConfig = HeuristicRAGConfig(),
) -> List[Tuple[str, str, float]]:
    cand_k = max(cfg.top_k * cfg.candidate_multiplier, cfg.top_k)
    candidates: List[Tuple[str, str, float]] = retrieve_with_scores(index, query, top_k=cand_k)
    return select_heuristic_from_candidates(candidates, cfg)


def retrieve_heuristic(
    index: ClassicRAGIndex,
    query: str,
    cfg: HeuristicRAGConfig = HeuristicRAGConfig(),
) -> List[str]:
    return [text for _cid, text, _score in retrieve_heuristic_with_scores(index, query, cfg)]
