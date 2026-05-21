# src/ontology/hierarchy.py

from typing import Dict, List
from ..data.models import Section, TextNode


# ---------------------------------------------------------
# Build hierarchy: parent/children уже проставлены в loaders
# ---------------------------------------------------------
def build_hierarchy(
    sections: Dict[str, Section],
    text_nodes: Dict[str, TextNode],
):
    """
    Сортирует text_nodes по секциям,
    собирает local_text и subtree_text для каждой Section.
    """

    # -----------------------------------------------------
    # 1. Собираем локальный текст каждой секции
    # -----------------------------------------------------
    for s in sections.values():
        texts = []

        # добавляем все текстовые узлы этой секции
        for tn in text_nodes.values():
            if tn.section_id == s.id:
                texts.append(tn.text)

        s.local_text = "\n".join(texts).strip()

    # -----------------------------------------------------
    # 2. Собираем subtree_text рекурсивно
    # -----------------------------------------------------
    def collect_subtree_text(sec_id: str) -> str:
        s = sections[sec_id]
        parts = []

        # локальный текст
        if s.local_text:
            parts.append(s.local_text)

        # рекурсивно добавляем дочерние секции
        for child_id in s.children_ids:
            child_text = collect_subtree_text(child_id)
            if child_text:
                parts.append(child_text)

        return "\n".join(parts).strip()

    for s in sections.values():
        s.subtree_text = collect_subtree_text(s.id)

    return sections, text_nodes


# ---------------------------------------------------------
# Helper: получить корневые секции (level=1)
# ---------------------------------------------------------
def get_root_sections(sections: Dict[str, Section]) -> List[Section]:
    return [s for s in sections.values() if s.level == 1]


# ---------------------------------------------------------
# Helper: получить цепочку родителей до корня
# ---------------------------------------------------------
def get_section_path(sections: Dict[str, Section], sec_id: str) -> List[Section]:
    path = []
    current = sections.get(sec_id)

    while current is not None:
        path.append(current)
        if current.parent_id is None:
            break
        current = sections.get(current.parent_id)

    return list(reversed(path))
