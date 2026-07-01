# src/index/embeddings.py

import os
from pathlib import Path
import numpy as np
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer

from rag_common.encoder_spec import DEFAULT_EMBED_MODEL, encoder_spec


def _resolve_local_snapshot(model_name: str) -> str:
    """Prefer an offline local snapshot built by the index builders
    (``$HF_HOME/models/<repo__slug>``) over a bare HF repo id, so query-time
    loading matches the corpus encoder without re-downloading. The logical
    model name is unchanged (kept for the index compatibility assert)."""
    if Path(model_name).exists():  # already a local path
        return model_name
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        cand = Path(hf_home) / "models" / model_name.replace("/", "__")
        if cand.exists() and any(cand.iterdir()):
            return str(cand)
    return model_name


class EmbeddingModel:
    """
    Обёртка над SentenceTransformer:
      - конфигурируемое имя модели (аргумент / $SWAGA_EMBED_MODEL / дефолт mpnet)
      - per-model query/passage префиксы (e5, bge) через rag_common.encoder_spec
      - автодетект CPU/GPU
      - возврат L2-нормализованных numpy-векторов
      - устойчивость к пустым строкам
      - совместимость с API.encode()

    Дефолт (mpnet) имеет пустые префиксы — поведение и эмбеддинги бит-в-бит
    как раньше.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        model_name:
            None -> use $SWAGA_EMBED_MODEL if set, else the default HF id.
                    May be a HF repo id or a local model directory (offline).
        device:
            None   -> autodetect GPU if available
            "cpu"  -> force CPU
            "cuda" -> force GPU
        """

        if model_name is None:
            model_name = os.environ.get("SWAGA_EMBED_MODEL", DEFAULT_EMBED_MODEL)

        self.model_name = model_name
        self.spec = encoder_spec(model_name)

        # Автодетект устройства
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        self.device = device

        load_target = _resolve_local_snapshot(model_name)
        print(f"[EmbeddingModel] Loading model {model_name} on {device} (from {load_target})...")
        self.model = SentenceTransformer(load_target, device=device)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    # -------------------------------------------------------------
    # embed(): основной метод → numpy-вектора
    # -------------------------------------------------------------
    def embed(self, texts: Union[str, List[str]], role: str = "query") -> np.ndarray:
        """
        Возвращает L2-normalized numpy-вектора.
        Поддерживает строку или список строк.

        role: "query" | "passage" — какой per-model префикс применить
        (для mpnet оба пустые, поведение не меняется).
        """

        if role == "query":
            prefix = self.spec.query_prefix
        elif role == "passage":
            prefix = self.spec.passage_prefix
        else:
            raise ValueError(f"Unknown role: {role!r} (expected 'query' or 'passage')")

        single_input = False

        # str → list
        if isinstance(texts, str):
            texts = [texts]
            single_input = True

        # пустая строка → " " (модель не любит ""); префикс применяется к обоим
        safe_texts = [
            prefix + (t if (isinstance(t, str) and t.strip()) else " ")
            for t in texts
        ]

        vecs = self.model.encode(
            safe_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # сразу cosine-ready
            show_progress_bar=False
        )

        return vecs[0] if single_input else vecs

    # -------------------------------------------------------------
    # encode(): совместимость с SentenceTransformer API
    # -------------------------------------------------------------
    def encode(self, texts: Union[str, List[str]], role: str = "query") -> np.ndarray:
        return self.embed(texts, role=role)
