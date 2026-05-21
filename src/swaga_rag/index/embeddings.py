# src/index/embeddings.py

import numpy as np
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Обёртка над SentenceTransformer (multilingual MPNet):
      - автодетект CPU/GPU
      - возврат numpy-векторов
      - L2-нормализация
      - устойчивость к пустым строкам
      - совместимость с API.encode()
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device: Optional[str] = None,
    ):
        """
        device:
            None   -> autodetect GPU if available
            "cpu"  -> force CPU
            "cuda" -> force GPU
        """

        # Автодетект устройства
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        self.device = device

        print(f"[EmbeddingModel] Loading model {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device)

    # -------------------------------------------------------------
    # embed(): основной метод → numpy-вектора
    # -------------------------------------------------------------
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Возвращает L2-normalized numpy-вектора.
        Поддерживает строку или список строк.
        """

        single_input = False

        # str → list
        if isinstance(texts, str):
            texts = [texts]
            single_input = True

        # пустая строка → " " (модель не любит "")
        safe_texts = [
            t if (isinstance(t, str) and t.strip()) else " "
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
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        return self.embed(texts)
