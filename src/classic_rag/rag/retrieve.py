from typing import List, Tuple
import numpy as np

from classic_rag.index.store import ClassicRAGIndex

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def retrieve(index: ClassicRAGIndex, query: str, top_k: int = 10) -> List[str]:
    """
    Baseline dense retrieval: return top-k chunk texts using cosine similarity.
    """
    if SentenceTransformer is None:
        raise RuntimeError("Missing dependency: sentence-transformers")

    model = SentenceTransformer(index.model_name)
    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype=np.float32)  # (1, D)

    # Cosine similarity because embeddings are normalized.
    scores = (index.embeddings @ q.T).reshape(-1)
    top_idx = np.argsort(-scores)[:top_k]

    return [index.texts[i] for i in top_idx]


def retrieve_with_scores(
    index: ClassicRAGIndex,
    query: str,
    top_k: int = 50,
) -> List[Tuple[str, str, float]]:
    """
    Dense retrieval that returns (chunk_id, text, score) for the top-k hits.

    Notes:
    - score is cosine similarity since embeddings are L2-normalized.
    - top_k here is typically larger than the final context budget
      (we oversample candidates and then apply heuristics).
    """
    if SentenceTransformer is None:
        raise RuntimeError("Missing dependency: sentence-transformers")

    model = SentenceTransformer(index.model_name)
    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype=np.float32)  # (1, D)

    scores = (index.embeddings @ q.T).reshape(-1)
    top_idx = np.argsort(-scores)[:top_k]

    return [(index.ids[i], index.texts[i], float(scores[i])) for i in top_idx]