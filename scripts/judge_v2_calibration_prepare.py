"""
Step 6 (prep): generate a markdown file for manual calibration labeling.

Selects calibration query_ids from configs/judge_v2/judges.json, filters
pairs.jsonl to those queries (5 * 9 * 2 = 90 entries), shuffles with a fixed
seed and writes a markdown document where each pair has its contexts and a
three-axis checkbox group. Method names are hidden — the author should label
blind.

Output:
  artifacts/judge_v2/calibration/labeling.md
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _repo_paths import repo_path, resolve_repo_path  # noqa: E402


DEFAULT_PAIRS = repo_path("artifacts/judge_v2/pairs.jsonl")
DEFAULT_JUDGES_CFG = repo_path("configs/judge_v2/judges.json")
DEFAULT_OUT = repo_path("artifacts/judge_v2/calibration/labeling.md")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def render_pair(idx: int, total: int, pair: Dict[str, Any]) -> str:
    qid = pair["query_id"]
    pid = pair["pair_id"]
    query = pair["query"]
    ctx_a = pair["context_A"]
    ctx_b = pair["context_B"]
    return f"""---
## [{idx}/{total}] {pid}

**Запрос ({qid}):** {query}

### Контекст A

{ctx_a}

### Контекст B

{ctx_b}

### Оценки

- **relevance** (какой контекст содержит больше информации, относящейся непосредственно к запрошенной функциональности):
  - [ ] A
  - [ ] B
  - [ ] equal
- **cleanliness** (какой контекст ЧИЩЕ — содержит меньше информации, не относящейся к запрошенной функциональности; ставь X там, где чище):
  - [ ] A
  - [ ] B
  - [ ] equal
- **sufficiency** (на основе какого контекста можно составить более целостное представление о запрошенной функциональности):
  - [ ] A
  - [ ] B
  - [ ] equal

"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare calibration markdown.")
    p.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    p.add_argument("--judges_config", type=Path, default=DEFAULT_JUDGES_CFG)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pairs = _read_jsonl(resolve_repo_path(args.pairs))
    cfg = _read_json(resolve_repo_path(args.judges_config))
    calib_qids = set(cfg["calibration"]["query_ids"])
    seed = int(cfg["calibration"]["shuffle_seed"])

    # Manual labelling is AB-only — the author labels each (comparison, query)
    # once. BA counterparts are auto-derived by inversion in the parser. This
    # halves the manual workload (45 instead of 90 pairs) without weakening
    # the calibration: permutation consistency is measured on the *LLM* judges
    # (who run on both perms), not on the human.
    subset = [p for p in pairs if p["query_id"] in calib_qids and p["perm"] == "AB"]
    if not subset:
        raise RuntimeError("No AB pairs matched calibration query_ids")
    rng = random.Random(seed)
    rng.shuffle(subset)

    out_path = resolve_repo_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = f"""# Калибровочная разметка ({len(subset)} пар, AB-only)

Инструкция:
- Для каждой пары проставь ровно один `[x]` в КАЖДОЙ из трёх осей.
- Оценивай слепо: имена методов скрыты, тебя не должно интересовать что есть A и что есть B.
- Запрос виден сверху каждой пары.
- Когда закончишь — сохрани файл и запусти `judge_v2_calibration_parse.py`.

BA-перестановки автоматически выводятся парсером инверсией (A<->B, equal<->equal),
поэтому здесь только AB-пары.

Всего пар: {len(subset)}. Seed перемешивания: {seed}.

"""
    with out_path.open("w", encoding="utf-8") as f:
        f.write(header)
        for i, pair in enumerate(subset, start=1):
            f.write(render_pair(i, len(subset), pair))

    print(f"[DONE] Wrote {len(subset)} pairs to {out_path}")


if __name__ == "__main__":
    main()
