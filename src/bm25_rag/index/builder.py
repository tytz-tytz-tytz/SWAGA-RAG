from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from bm25_rag.index.store import BM25Index


_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_]+")

# A conservative, "light" Russian stemming via suffix stripping.
# This is NOT lemmatization, but it helps unify forms like:
#   "виджет", "виджета", "виджету", "виджетом" -> "виджет"
_RU_SUFFIXES = [
    "иями", "ями", "ами",
    "ого", "его", "ому", "ему",
    "ыми", "ими",
    "иях", "ях", "ах",
    "ией", "ей", "ой",
    "иям", "ям", "ам",
    "ию", "ью",
    "ия", "ья",
    "ов", "ев",
    "ом", "ем",
    "ы", "и", "а", "я", "у", "ю", "е", "о",
]

def _is_cyrillic(token: str) -> bool:
    return any("а" <= ch <= "я" or "А" <= ch <= "Я" or ch in "ёЁ" for ch in token)

def _light_ru_stem(token: str) -> str:
    """
    Very lightweight stemming for Russian:
    strip a single common inflectional suffix if token is long enough.
    """
    t = token.lower()
    if len(t) < 5:
        return t

    if not _is_cyrillic(t):
        return t

    # strip the longest matching suffix first
    for suf in sorted(_RU_SUFFIXES, key=len, reverse=True):
        if len(t) - len(suf) >= 4 and t.endswith(suf):
            return t[: -len(suf)]

    return t


def tokenize(text: str) -> List[str]:
    """
    Tokenizer + light normalization:
    - keeps latin/cyrillic letters, digits, underscore
    - lowercases
    - applies light RU stemming (suffix stripping)
    """
    text = (text or "").lower()
    raw = _TOKEN_RE.findall(text)
    return [_light_ru_stem(tok) for tok in raw]


def build_bm25_index(
    docs: List[Tuple[str, str]],
    k1: float = 1.5,
    b: float = 0.75,
    *,
    verbose: bool = False,
    log_every: int = 10000,
) -> BM25Index:
    """
    Build BM25 index from (id, text) docs.
    Uses Okapi BM25 with IDF:
      idf(t) = log( (N - df + 0.5) / (df + 0.5) + 1 )
    """
    ids: List[str] = []
    texts: List[str] = []
    doc_tokens: List[List[str]] = []
    doc_freqs: List[Dict[str, int]] = []

    df = defaultdict(int)
    doc_lens: List[int] = []

    total_in = len(docs)
    for idx, (cid, text) in enumerate(docs, start=1):
        toks = tokenize(text)
        if not toks:
            if verbose and (idx % log_every == 0 or idx == total_in):
                print(f"[BM25] tokenized {idx}/{total_in} docs (kept={len(ids)})")
            continue

        ids.append(cid)
        texts.append(text)
        doc_tokens.append(toks)

        tf = Counter(toks)
        doc_freqs.append(dict(tf))

        doc_lens.append(len(toks))
        for term in tf.keys():
            df[term] += 1

        if verbose and (idx % log_every == 0 or idx == total_in):
            print(f"[BM25] tokenized {idx}/{total_in} docs (kept={len(ids)})")

    n_docs = len(ids)
    if n_docs == 0:
        raise ValueError("No documents to index (after tokenization).")

    avgdl = sum(doc_lens) / n_docs

    idf: Dict[str, float] = {}
    for term, dft in df.items():
        idf[term] = math.log((n_docs - dft + 0.5) / (dft + 0.5) + 1.0)

    return BM25Index(
        ids=ids,
        texts=texts,
        doc_tokens=doc_tokens,
        doc_freqs=doc_freqs,
        idf=idf,
        avgdl=avgdl,
        k1=k1,
        b=b,
    )


def save_index(index: BM25Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(index, f)


def load_index(path: Path) -> BM25Index:
    with path.open("rb") as f:
        return pickle.load(f)
