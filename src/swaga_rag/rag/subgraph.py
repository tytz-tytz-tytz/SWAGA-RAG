import re
from typing import Any, Dict, List, Optional, Tuple

from ..data.models import Section, TextNode

_INT_GROUP_RE = re.compile(r"\d+")


def _chunk_order_key(node_id: str) -> Tuple[int, ...]:
    """
    Ordinal sort key for chunks within a section, robust across corpus id
    schemes:

      - user docs : '...chN'                  -> ordered by N
      - Qasper    : '{doc}.{section}.{para}'  -> ordered by the trailing paragraph
      - BioASQ    : '{pmc}::...::pP::cC'       -> ordered by (P, C)

    Chunks inside one section share the same id prefix, so ordering by the full
    tuple of integer groups found in the id reproduces document order for every
    scheme above. Ids without any integer sort last (stably).
    """
    ints = _INT_GROUP_RE.findall(node_id)
    if not ints:
        return (1,)
    return (0,) + tuple(int(x) for x in ints)


class SubgraphConfig:
    """
    Configuration for chunk-window assembly around relevant chunks.

    A "subgraph" here is a contiguous slice of chunks belonging to the same
    parent section (HAS_CHUNK). Default behavior: take chunks from the start
    of the section up to the anchor chunk plus a small tail after it.
    If that span exceeds max_window_chunks, fall back to a symmetric window
    around the anchor (fallback_before / fallback_after).
    """

    def __init__(
        self,
        tail_after: int = 2,
        max_window_chunks: int = 8,
        fallback_before: int = 2,
        fallback_after: int = 2,
        **_: Any,
    ):
        self.tail_after = int(tail_after)
        self.max_window_chunks = int(max_window_chunks)
        self.fallback_before = int(fallback_before)
        self.fallback_after = int(fallback_after)


class SubgraphAssembler:
    """
    Build coherent chunk windows around ranked anchor chunks.

    For each ranked chunk:
      - find its parent section (via TextNode.section_id, set from HAS_CHUNK);
      - build a window over sibling chunks ordered by chunk index;
      - merge overlapping/adjacent windows within the same section.
    """

    def __init__(
        self,
        sections: Dict[str, Section],
        text_nodes: Dict[str, TextNode],
        cfg: SubgraphConfig,
    ):
        self.sections = sections
        self.text_nodes = text_nodes
        self.cfg = cfg

        chunks_by_section: Dict[str, List[str]] = {}
        for nid, tn in text_nodes.items():
            if tn.node_type != "chunk":
                continue
            sid = tn.section_id
            if sid is None:
                continue
            chunks_by_section.setdefault(sid, []).append(nid)
        for ids in chunks_by_section.values():
            ids.sort(key=_chunk_order_key)
        self._chunks_by_section = chunks_by_section

    def _window_for(self, n_chunks: int, anchor_pos: int) -> Tuple[int, int]:
        cfg = self.cfg
        start = 0
        end = min(anchor_pos + cfg.tail_after, n_chunks - 1)
        if (end - start + 1) > cfg.max_window_chunks:
            start = max(0, anchor_pos - cfg.fallback_before)
            end = min(n_chunks - 1, anchor_pos + cfg.fallback_after)
        return start, end

    def assemble(self, ranked_nodes: List[dict]) -> List[dict]:
        per_section: Dict[str, List[dict]] = {}
        for item in ranked_nodes:
            sid = item.get("section_id")
            if not isinstance(sid, str):
                continue
            if sid not in self._chunks_by_section:
                continue
            per_section.setdefault(sid, []).append(item)

        output: List[dict] = []
        for sid, anchors in per_section.items():
            sec_chunks = self._chunks_by_section[sid]
            id_to_pos = {nid: i for i, nid in enumerate(sec_chunks)}

            intervals: List[Tuple[int, int, List[dict]]] = []
            for a in anchors:
                pos = id_to_pos.get(a.get("node_id"))
                if pos is None:
                    continue
                s, e = self._window_for(len(sec_chunks), pos)
                intervals.append((s, e, [a]))

            if not intervals:
                continue

            intervals.sort(key=lambda x: x[0])
            merged: List[Tuple[int, int, List[dict]]] = []
            for s, e, ancs in intervals:
                if merged and s <= merged[-1][1] + 1:
                    ps, pe, prev_ancs = merged[-1]
                    merged[-1] = (ps, max(pe, e), prev_ancs + ancs)
                else:
                    merged.append((s, e, ancs))

            sec = self.sections.get(sid)
            parent_id: Optional[str] = getattr(sec, "parent_id", None) if sec else None
            title = ""
            if sec and sec.local_text:
                title = sec.local_text.split("\n")[0].strip()

            for s, e, ancs in merged:
                window_ids = sec_chunks[s : e + 1]
                texts: List[str] = []
                for nid in window_ids:
                    tn = self.text_nodes.get(nid)
                    if tn and isinstance(tn.text, str):
                        texts.append(tn.text)
                merged_text = "\n".join(t for t in texts if t).strip()
                best_score = max(float(a.get("score", 0.0)) for a in ancs)
                anchor_ids = [a.get("node_id") for a in ancs]
                output.append({
                    "section_id": sid,
                    "parent_section_id": parent_id,
                    "title": title,
                    "anchor_node_ids": anchor_ids,
                    "window_node_ids": window_ids,
                    "from_start": s == 0,
                    "text": merged_text,
                    "score": best_score,
                })

        output.sort(key=lambda x: -x["score"])
        return output


def window_chunk_id(window: dict) -> str:
    """
    Stable composite identifier for an assembled window:
    ``{section_id}::{first_chunk}..{last_chunk}``. Falls back to the section id
    or first anchor when the window range is unavailable. This matches the
    chunk_id convention used by run_queries_swaga_subgraphs.py.
    """
    sid = window.get("section_id")
    win_ids = window.get("window_node_ids") or []
    if isinstance(sid, str) and win_ids:
        return f"{sid}::{win_ids[0]}..{win_ids[-1]}"
    if isinstance(sid, str):
        return sid
    anchors = window.get("anchor_node_ids") or []
    return anchors[0] if anchors else ""


def windows_to_ranked_ids(windows: List[dict], order: str = "doc") -> List[str]:
    """
    Flatten assembled windows into a single ranked, de-duplicated list of
    member chunk ids for chunk-level retrieval metrics (variant A).

    Windows are consumed in their given order (``SubgraphAssembler.assemble``
    returns them sorted by descending score). Within each window:
      - ``order="doc"``: emit ``window_node_ids`` in document order;
      - ``order="anchors_first"``: emit every ``anchor_node_ids`` first (across
        all windows, preserving window order), then the remaining
        ``window_node_ids``.
    The first occurrence of each id wins; later duplicates are dropped.
    """
    seen: set = set()
    out: List[str] = []

    def _emit(nid: Any) -> None:
        if isinstance(nid, str) and nid and nid not in seen:
            seen.add(nid)
            out.append(nid)

    if order == "anchors_first":
        for w in windows:
            for nid in (w.get("anchor_node_ids") or []):
                _emit(nid)
        for w in windows:
            for nid in (w.get("window_node_ids") or []):
                _emit(nid)
    elif order == "doc":
        for w in windows:
            for nid in (w.get("window_node_ids") or []):
                _emit(nid)
    else:
        raise ValueError(f"Unknown window order: {order!r} (expected 'doc' or 'anchors_first')")

    return out


def window_output_items(windows: List[dict]) -> List[dict]:
    """
    Build judge/eval-friendly ``output_items`` from assembled windows, keeping
    the full window metadata (``window_node_ids``, ``anchor_node_ids``, text)
    so window-level evaluation (variant C) can recover which chunks each window
    covers.
    """
    items: List[dict] = []
    for w in windows:
        items.append({
            "chunk_id": window_chunk_id(w),
            "section_id": w.get("section_id"),
            "parent_section_id": w.get("parent_section_id"),
            "title": w.get("title"),
            "score": w.get("score"),
            "anchor_node_ids": w.get("anchor_node_ids"),
            "window_node_ids": w.get("window_node_ids") or [],
            "from_start": w.get("from_start"),
            "text": w.get("text"),
        })
    return items
