from __future__ import annotations

from typing import List, Tuple, Set
import numpy as np

from bm25_rag.index.builder import tokenize
from bm25_rag.index.store import BM25Index


# Minimal, practical stopword list (RU + a bit of EN).
# We keep it small to avoid overfiltering, but enough to reduce noise.
STOPWORDS: Set[str] = {
    # RU
    "и", "в", "во", "на", "к", "ко", "по", "из", "у", "о", "об", "от", "до",
    "для", "при", "без", "с", "со", "за", "над", "под", "про", "через",
    "это", "этот", "эта", "эти", "того", "тому", "тем", "таким", "такая",
    "как", "так", "же", "ли", "бы", "не", "ни", "нет", "есть", "будет",
    "или", "либо", "а", "но", "да", "то", "т", "д", "тд", "тп",
    "что", "чтобы", "который", "которая", "которые", "когда", "где", "куда",
    "какой", "какая", "какие", "каких", "какому",
    "нужно", "можно", "нельзя",
    "все", "всё", "их", "его", "ее", "её", "мы", "вы", "они", "она", "он",
    "я", "ты", "мне", "нам", "вам",
    "здесь", "там", "тут",
    "после", "перед", "между", "если", "чем",
    # EN (just in case)
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "is", "are", "be", "as", "by", "at", "from",
}


def _filter_query_tokens(tokens: List[str]) -> List[str]:
    """
    Remove stopwords and very short tokens from the query.
    This reduces lexical noise for BM25.
    """
    out: List[str] = []
    for t in tokens:
        if len(t) <= 1:
            continue
        if t in STOPWORDS:
            continue
        out.append(t)
    return out


def retrieve_with_scores(
    index: BM25Index,
    query: str,
    top_k: int = 10
) -> List[Tuple[str, str, float]]:
    """
    BM25 retrieval returning (chunk_id, text, score).

    Improvements vs naive BM25:
    - stopword removal in query tokens
    """
    q_tokens = tokenize(query)
    q_tokens = _filter_query_tokens(q_tokens)

    if not q_tokens:
        return []

    # Unique query terms are enough for BM25 scoring.
    q_terms = list(dict.fromkeys(q_tokens))

    N = len(index.ids)
    scores = np.zeros(N, dtype=np.float32)

    k1 = index.k1
    b = index.b
    avgdl = index.avgdl

    for term in q_terms:
        idf = index.idf.get(term)
        if idf is None:
            continue

        for i, tf_dict in enumerate(index.doc_freqs):
            tf = tf_dict.get(term, 0)
            if tf == 0:
                continue

            dl = len(index.doc_tokens[i])
            denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
            score = idf * (tf * (k1 + 1.0) / denom)
            scores[i] += score

    # Pick top-k by score
    if top_k >= N:
        top_idx = np.argsort(-scores)
    else:
        top_idx = np.argpartition(-scores, top_k)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

    return [(index.ids[i], index.texts[i], float(scores[i])) for i in top_idx if scores[i] > 0.0]


def retrieve(index: BM25Index, query: str, top_k: int = 10) -> List[str]:
    """
    BM25 retrieval returning only texts.
    """
    hits = retrieve_with_scores(index, query, top_k=top_k)
    return [t for _cid, t, _s in hits]
