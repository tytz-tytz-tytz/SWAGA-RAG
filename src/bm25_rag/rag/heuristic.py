from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Set

from bm25_rag.index.builder import tokenize
from bm25_rag.index.store import BM25Index
from bm25_rag.rag.retrieve import retrieve_with_scores, STOPWORDS
from rag_common.text_hygiene import (
    normalize_text_for_dedup,
    looks_like_caption,
    looks_like_table_header,
    looks_like_heading,
)


@dataclass(frozen=True)
class BM25HeuristicConfig:
    # Final number of chunks returned (context budget).
    top_k: int = 10

    # Retrieve more candidates first, then filter down to top_k.
    candidate_multiplier: int = 12

    # Filter out very short chunks (headings/captions/noise).
    min_chars: int = 80

    # Deduplicate texts after normalization.
    deduplicate: bool = True

    # Hygiene filters.
    drop_captions: bool = True
    drop_table_like: bool = True
    drop_headings: bool = True
    drop_colon_trailing: bool = True

    # IDF-based must-have constraint:
    # require that a doc contains at least one of the top-N rare query terms.
    use_idf_must: bool = True
    rare_terms_top_n: int = 2  # take top-1 or top-2 rare terms by IDF

    # If strict filtering yields too few results, relax automatically.
    relax_threshold: int = 4  # if < this many items, relax constraints

    # Fallback minimum length (still avoid tiny junk).
    fallback_min_chars: int = 40


def _significant_query_terms(query: str) -> List[str]:
    """
    Tokenize query and remove stopwords + 1-char tokens.
    Note: tokenize() may include normalization (depends on builder.py).
    """
    toks = tokenize(query)
    out: List[str] = []
    for t in toks:
        if len(t) <= 1:
            continue
        if t in STOPWORDS:
            continue
        out.append(t)
    # unique-preserving
    return list(dict.fromkeys(out))


def _pick_rare_terms_by_idf(index: BM25Index, terms: List[str], top_n: int) -> List[str]:
    """
    Pick top-N rare terms (highest IDF) among query terms that exist in the index.
    """
    scored: List[Tuple[str, float]] = []
    for t in terms:
        idf = index.idf.get(t)
        if idf is not None:
            scored.append((t, float(idf)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _idf in scored[: max(1, top_n)]]


def _contains_any_term(text: str, must_terms: List[str]) -> bool:
    if not must_terms:
        return True
    doc_terms = set(tokenize(text))
    return any(t in doc_terms for t in must_terms)


def retrieve_heuristic_with_scores(
    index: BM25Index,
    query: str,
    cfg: BM25HeuristicConfig = BM25HeuristicConfig(),
) -> List[Tuple[str, str, float]]:
    """
    Robust BM25+heuristics baseline.

    Key idea:
    - Use BM25 scores for ranking candidates.
    - Apply hygiene filters.
    - Use an IDF-based "must-have" constraint (rare query terms),
      but RELAX automatically if it becomes too strict.
    """
    cand_k = max(cfg.top_k * cfg.candidate_multiplier, cfg.top_k)
    candidates: List[Tuple[str, str, float]] = retrieve_with_scores(index, query, top_k=cand_k)

    sig_terms = _significant_query_terms(query)

    must_terms: List[str] = []
    if cfg.use_idf_must and sig_terms:
        must_terms = _pick_rare_terms_by_idf(index, sig_terms, cfg.rare_terms_top_n)

    out: List[Tuple[str, str, float]] = []
    seen: Set[str] = set()

    def hygiene_ok(txt: str) -> bool:
        if not txt:
            return False
        if cfg.drop_captions and looks_like_caption(txt):
            return False
        if cfg.drop_table_like and looks_like_table_header(txt):
            return False
        if cfg.drop_headings and looks_like_heading(txt):
            return False
        if cfg.drop_colon_trailing and txt.endswith(":"):
            return False
        return True

    # Pass 1: strict = hygiene + min_chars + must_terms
    for cid, text, score in candidates:
        txt = (text or "").strip()
        if not hygiene_ok(txt):
            continue
        if len(txt) < cfg.min_chars:
            continue
        if must_terms and not _contains_any_term(txt, must_terms):
            continue

        if cfg.deduplicate:
            key = normalize_text_for_dedup(txt)
            if key in seen:
                continue
            seen.add(key)

        out.append((cid, txt, float(score)))
        if len(out) >= cfg.top_k:
            break

    # If too few results, relax "must_terms" constraint (keep hygiene).
    if len(out) < cfg.relax_threshold:
        out2: List[Tuple[str, str, float]] = out[:]
        for cid, text, score in candidates:
            txt = (text or "").strip()
            if not hygiene_ok(txt):
                continue
            if len(txt) < cfg.min_chars:
                continue

            if cfg.deduplicate:
                key = normalize_text_for_dedup(txt)
                if key in seen:
                    continue
                seen.add(key)

            out2.append((cid, txt, float(score)))
            if len(out2) >= cfg.top_k:
                break
        out = out2

    # Final fallback: relax min_chars to fill top_k (still keep hygiene + dedup).
    if len(out) < cfg.top_k:
        for cid, text, score in candidates:
            txt = (text or "").strip()
            if not hygiene_ok(txt):
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


def retrieve_heuristic(
    index: BM25Index,
    query: str,
    cfg: BM25HeuristicConfig = BM25HeuristicConfig(),
) -> List[str]:
    return [text for _cid, text, _score in retrieve_heuristic_with_scores(index, query, cfg)]
