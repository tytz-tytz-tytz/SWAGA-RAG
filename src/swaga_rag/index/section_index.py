# src/index/section_index.py

from typing import Dict
import numpy as np
from .embeddings import EmbeddingModel
from ..data.models import Section


class SectionIndex:
    """
    Отвечает за вычисление эмбеддингов:
        - E_local (текст секции)
        - E_subtree (текст секции + дочерних)
    """

    def __init__(self, model: EmbeddingModel):
        self.model = model

    # -------------------------------------------------------------
    # Основная функция
    # -------------------------------------------------------------
    def compute_section_embeddings(self, sections: Dict[str, Section]) -> Dict[str, Section]:

        # Кэш для одинаковых текстов
        embed_cache = {}

        def get_emb(text: str):
            t = text.strip()
            if not t:
                return None  # пустой текст → None
            if t in embed_cache:
                return embed_cache[t]
            v = self.model.encode(t)
            embed_cache[t] = v
            return v

        print("[SectionIndex] Computing embeddings for sections...")

        counter = 0
        for sid, sec in sections.items():
            # local text embedding
            sec.E_local = get_emb(sec.local_text)

            # subtree embedding
            sec.E_subtree = get_emb(sec.subtree_text)

            counter += 1
            if counter % 20 == 0:
                print(f"  processed {counter}/{len(sections)} sections")

        print(f"[SectionIndex] DONE. Total sections: {len(sections)}")
        return sections
