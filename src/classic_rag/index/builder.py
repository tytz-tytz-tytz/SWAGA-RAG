from pathlib import Path
import pickle
import numpy as np
import time
from typing import Optional
import os

from classic_rag.data.loaders import load_chunks_id_text
from classic_rag.index.store import ClassicRAGIndex
from rag_common.encoder_spec import encoder_spec

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _resolve_model_path(model_name: str, cache_folder: Optional[str]) -> str:
    """
    On Windows (non-admin), HF symlink cache may fail.
    We pre-download to a local directory and load model from there.
    """
    if os.name != "nt" or cache_folder is None or snapshot_download is None:
        return model_name

    root = Path(cache_folder)
    local_model_dir = root / "models" / model_name.replace("/", "__")
    local_model_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_model_dir),
        local_dir_use_symlinks=False,  # compatibility; ignored in new versions
        resume_download=True,
    )
    return str(local_model_dir)


def build_index(
    nodes_path: Path,
    model_name: str = DEFAULT_MODEL,
    *,
    device: str = "cpu",
    batch_size: int = 64,
    show_progress_bar: bool = True,
    cache_folder: Optional[str] = None,
    verbose: bool = True,
) -> ClassicRAGIndex:
    if SentenceTransformer is None:
        raise RuntimeError("Missing dependency: sentence-transformers")

    t0 = time.perf_counter()
    ids, texts = load_chunks_id_text(nodes_path)
    if verbose:
        print(f"[DenseIndex] loaded chunks={len(ids)} from {nodes_path}")
        print(f"[DenseIndex] model={model_name} device={device} batch_size={batch_size}")

    model_path = _resolve_model_path(model_name, cache_folder)
    if verbose and model_path != model_name:
        print(f"[DenseIndex] model resolved to local dir: {model_path}")
    model = SentenceTransformer(model_path, device=device, cache_folder=cache_folder)
    # Per-model passage prefix (e5/bge); empty for mpnet/MiniLM -> unchanged.
    passage_prefix = encoder_spec(model_name).passage_prefix
    enc_texts = [passage_prefix + t for t in texts] if passage_prefix else texts
    if verbose and passage_prefix:
        print(f"[DenseIndex] passage_prefix={passage_prefix!r}")
    t1 = time.perf_counter()
    emb = model.encode(
        enc_texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    emb = np.asarray(emb, dtype=np.float32)
    if verbose:
        print(f"[DenseIndex] embeddings shape={emb.shape}")
        print(f"[DenseIndex] encode_time={time.perf_counter()-t1:.1f}s total={time.perf_counter()-t0:.1f}s")

    return ClassicRAGIndex(ids=ids, texts=texts, embeddings=emb, model_name=model_name)


def save_index(index: ClassicRAGIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(index, f)


def load_index(path: Path) -> ClassicRAGIndex:
    with path.open("rb") as f:
        return pickle.load(f)
