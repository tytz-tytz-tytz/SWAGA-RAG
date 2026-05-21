"""Cross-tab manual vs each judge on the low_noise axis, to expose
systematic inversion if any."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _repo_paths import repo_path  # noqa: E402

CAL_DIR = repo_path("artifacts/judge_v2/calibration")
manual = {}
for line in (CAL_DIR / "manual.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    lab = r.get("manual_labels") or {}
    if "low_noise" in lab:
        manual[r["pair_id"]] = lab["low_noise"]

for jn in ("anthropic_haiku", "openai_4_1_mini", "gemini_2_5_flash"):
    path = CAL_DIR / f"llm_{jn}.jsonl"
    cm = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("status") != "ok":
            continue
        pid = rec["pair_id"]
        if pid not in manual:
            continue
        m = manual[pid]
        llm = (rec.get("labels") or {}).get("low_noise")
        if llm is None:
            continue
        cm[(m, llm)] += 1
    total = sum(cm.values())
    print(f"\n=== {jn} (n={total}) ===")
    print(f"{'manual\\llm':12s} {'A':>5s} {'B':>5s} {'equal':>5s}")
    for m_lab in ("A", "B", "equal"):
        row = " ".join(f"{cm.get((m_lab, l),0):5d}" for l in ("A","B","equal"))
        print(f"{m_lab:12s} {row}")
    agree = sum(v for (m,l),v in cm.items() if m==l)
    invert = sum(v for (m,l),v in cm.items() if (m,l) in [("A","B"),("B","A")])
    print(f"agreement: {agree}/{total} = {agree/total:.0%}")
    print(f"strict inversions (A<->B): {invert}/{total} = {invert/total:.0%}")
