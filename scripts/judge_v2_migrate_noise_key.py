"""One-off migration: rename `noise` -> `low_noise` in manual.jsonl."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _repo_paths import repo_path  # noqa: E402

path = repo_path("artifacts/judge_v2/calibration/manual.jsonl")
lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

renamed = 0
for r in lines:
    lab = r.get("manual_labels") or {}
    if "low_noise" in lab and "cleanliness" not in lab:
        lab["cleanliness"] = lab.pop("low_noise")
        renamed += 1
    elif "noise" in lab and "cleanliness" not in lab:
        lab["cleanliness"] = lab.pop("noise")
        renamed += 1

with path.open("w", encoding="utf-8") as f:
    for r in lines:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"renamed records: {renamed} of total: {len(lines)}")
