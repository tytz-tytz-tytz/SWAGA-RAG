"""Per-model query/passage prefix conventions for retrieval encoders.

Shared by the SWAGA-RAG and classic (dense) paths so that the corpus side
(passages, built offline) and the query side (runtime) apply consistent
prefixes for asymmetric retrieval models.

Conventions:
  - intfloat e5  : query -> "query: ",  passage -> "passage: "
  - BAAI bge-*-en: query -> retrieval instruction, passage -> (bare)
  - everything else (mpnet default, S-PubMedBert, ...): no prefixes,
    preserving byte-identical behaviour of pre-existing indexes/runs.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Official bge-v1.5 English retrieval query instruction.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class EncoderSpec:
    model_name: str
    query_prefix: str = ""
    passage_prefix: str = ""


def encoder_spec(model_name: str) -> EncoderSpec:
    """Return the query/passage prefix convention for a model id."""
    n = (model_name or "").lower()

    # intfloat/e5-* — symmetric query:/passage: prefixes.
    if "e5" in n:
        return EncoderSpec(model_name, "query: ", "passage: ")

    # BAAI/bge-*-en-v1.5 — query instruction, passages encoded bare.
    if "bge" in n and "en" in n:
        return EncoderSpec(model_name, _BGE_QUERY_INSTRUCTION, "")

    # Default (mpnet, S-PubMedBert-MS-MARCO, ...) — no prefixes.
    return EncoderSpec(model_name, "", "")


def model_slug(model_name: str) -> str:
    """Filesystem-safe short tag for per-model artifact directories."""
    return (model_name or "").split("/")[-1].strip() or "model"


def apply_prefix(prefix: str, text: str) -> str:
    """Prepend a (possibly empty) prefix to text; empty prefix is a no-op."""
    if not prefix:
        return text
    return prefix + (text or "")
