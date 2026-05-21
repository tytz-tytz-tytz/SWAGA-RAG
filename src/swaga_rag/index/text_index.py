# src/index/text_index.py

from typing import Dict
import numpy as np
from ..data.models import TextNode
from .embeddings import EmbeddingModel


class TextIndex:
    """
    Вычисляет эмбеддинги для всех текстовых узлов:
      - chunk
      - caption
      - section_title (если будет)
      - list_item (если добавим позже)
    """

    def __init__(self, model: EmbeddingModel):
        self.model = model

    # -------------------------------------------------------------
    # Основная функция
    # -------------------------------------------------------------
    def compute_textnode_embeddings(
        self,
        text_nodes: Dict[str, TextNode],
    ) -> Dict[str, TextNode]:

        embed_cache = {}

        def get_emb(text: str):
            if not text or not text.strip():
                return None  # пустой текст → нет эмбеддинга
            if text in embed_cache:
                return embed_cache[text]
            v = self.model.encode(text)
            embed_cache[text] = v
            return v

        print("[TextIndex] Computing embeddings for text nodes...")

        total = len(text_nodes)
        processed = 0

        for nid, tn in text_nodes.items():
            tn.embedding = get_emb(tn.text)
            processed += 1
            if processed % 200 == 0:
                print(f"  processed {processed}/{total}")

        print(f"[TextIndex] DONE. Total text nodes: {total}")
        return text_nodes
