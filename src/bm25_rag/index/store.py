from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class BM25Index:
    """
    A simple BM25 index for offline retrieval.
    - ids/texts: aligned lists
    - doc_tokens: tokenized documents
    - doc_freqs: per-doc term frequencies
    - idf: inverse document frequency per term
    """
    ids: List[str]
    texts: List[str]

    doc_tokens: List[List[str]]
    doc_freqs: List[Dict[str, int]]

    idf: Dict[str, float]
    avgdl: float
    k1: float
    b: float
