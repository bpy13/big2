"""Sharp-decision pass rate per band (plan Task 49).

    python experiments/sharp_pass.py runs/p6_mistakes

A decision is sharp when it is not forced, not flat and the second-best graded move loses at
least ``--line`` points (D102 inaccuracy line, 2.75). A deal-seat passes with at least
``--min-sharp`` sharp decisions. Prints one row per band: deals, deal-seats, sharp decisions,
deal-seats with >=1..5 sharp, deals with a passing seat, passing seats with at most 2 per deal.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

BANDS = ["net-t4.34", "easy", "medium", "hard", "net-t1", "expert", "net"]


def band_counts(path: Path, line: float, min_sharp: int) -> dict[str, int]:
    """Counts for one ``decisions-<band>.jsonl`` file."""
    seats: dict[tuple[int, int], int] = defaultdict(int)
    for raw in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(raw)
        key = (r["deal_id"], r["seat"])
        seats[key] += 0
        if r["forced"] or r["flat"]:
            continue
        m = r["moves"]
        if len(m) >= 2 and m[1]["loss"] >= line:
            seats[key] += 1
    per_deal: dict[int, list[int]] = defaultdict(list)
    for (d, _), s in seats.items():
        per_deal[d].append(s)
    res = {"deals": len(per_deal), "seats": len(seats), "sharp": sum(seats.values())}
    for t in range(1, 6):
        res[f"ge{t}"] = sum(1 for s in seats.values() if s >= t)
    res["deals_pass"] = sum(1 for v in per_deal.values() if any(x >= min_sharp for x in v))
    res["capped2"] = sum(min(2, sum(1 for x in v if x >= min_sharp)) for v in per_deal.values())
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--line", type=float, default=2.75)
    ap.add_argument("--min-sharp", type=int, default=3)
    a = ap.parse_args()
    tot: dict[str, int] = defaultdict(int)
    for b in BANDS:
        c = band_counts(a.dir / f"decisions-{b}.jsonl", a.line, a.min_sharp)
        print(b, *c.values())
        for k, v in c.items():
            tot[k] += v
    print("TOTAL", *tot.values())


if __name__ == "__main__":
    main()
