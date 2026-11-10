"""Writes drills/big2-puzzles.json from the miner's raw.jsonl; prints the theme counts.
Usage: python -I write_puzzles.py RAW OUT SUMMARY (run by mine-puzzles.ps1)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from big2.teach.drills import THEMES, write_drills  # noqa: E402

raw, out, summary = sys.argv[1:4]
result = write_drills(raw, out, None)
result["missing_themes"] = [t for t in THEMES if t not in result["skills"]]
result["under_30"] = [t for t, n in result["skills"].items() if n < 30]
Path(summary).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
print(json.dumps(result, indent=1))
