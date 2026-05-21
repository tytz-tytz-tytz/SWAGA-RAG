from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class ClassicRAGIndex:
    ids: List[str]
    texts: List[str]
    embeddings: np.ndarray  # (N, D), float32, normalized
    model_name: str
