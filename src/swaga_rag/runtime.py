from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(*parts: str | Path) -> Path:
    return REPO_ROOT.joinpath(*parts)


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_path(path)
