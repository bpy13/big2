"""Adds the top-up run's endgame_counting puzzles to drills/big2-puzzles.json, keeping every row
already there (ratings included). New rows get ratings banded over 600-1400 among themselves.
Usage: python -I merge_endgame.py RAW PUZZLES  (RAW = runs/big2-puzzles-endgame/raw.jsonl)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from big2.teach.drills import band_ratings, read_raw, to_drill, validate_drill  # noqa: E402

raw, puzzles = sys.argv[1:3]
cands = [c for c in read_raw(raw) if c["type"] == "candidate"]
ratings = band_ratings(cands)
new = [to_drill(c, r) for c, r in zip(cands, ratings)]
path = Path(puzzles)
rows = json.loads(path.read_text(encoding="utf-8"))
have = {r["id"] for r in rows}
new = [d for d in new if d["id"] not in have and not validate_drill(d)]
path.write_text(json.dumps(rows + new, indent=1) + "\n", encoding="utf-8")
print(json.dumps({"added": len(new), "total": len(rows) + len(new),
                  "endgame_counting": sum(r["skill"] == "skill.endgame_counting" for r in rows + new)}))
