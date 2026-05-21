# src/data/models.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np


@dataclass
class Edge:
    from_id: str
    to_id: str
    relation_type: str


@dataclass
class Section:
    id: str
    level: Optional[int] = None          # будет проставлено позже в hierarchy.py
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    local_text: str = ""                 # текст только этой секции
    subtree_text: str = ""               # текст всей подветки

    E_local: Optional[np.ndarray] = None
    E_subtree: Optional[np.ndarray] = None


@dataclass
class TextNode:
    id: str
    section_id: Optional[str]
    node_type: str                       # "section_title" | "chunk" | "list_item" | "caption"
    text: str
    embedding: Optional[np.ndarray] = None
    parent_id: Optional[str] = None
    order: Optional[float] = None
