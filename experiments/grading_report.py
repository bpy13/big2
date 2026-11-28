"""Task 43's report: grade lines, the trivial flag, accuracy, APL and the luck-removed rating.

Called by ``experiments/grading_rating.py report`` (``build_report``). Reads Task 42's
``runs/p6_mistakes/decisions-<band>.jsonl``, Task 40's ladder games (``runs/p6_ladder/pairs``),
the new expert's games (``runs/p6_grading/pairs``) and the luck estimates
(``runs/p6_grading/luck-m<m>.jsonl``); writes ``runs/p6_grading/report.md``, ``tables.json``
and ``proposal.json``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "experiments"))

from big2.teach.grading import accuracy as acc  # noqa: E402
from big2.teach.grading import lines as gl  # noqa: E402
from big2.teach.grading import rating as rt  # noqa: E402
from big2.teach.grading.losses import D102_LINES, WORDS, read_decisions  # noqa: E402

BANDS = ["net-t4.34", "easy", "medium", "hard", "net-t1", "expert", "net"]
TOP_BAND = "net"
MEDIUM_BAND = "medium"
ZERO, TOP = "random", "net"
#: The six bots Brian asked about (plan Task 43, "ALSO 2026-11-28") and the new expert.
BOTS = [
    ("random", "random"),
    ("easy", "greedy (`easy`)"),
    ("medium", "Q-network (`medium`)"),
    ("hard", "ISMCTS-500 (`hard`)"),
    ("expert", "ISMCTS-10,000 (old `expert`)"),
    ("net", "hour-24 network, top move"),
    ("puct2000", "network + PUCT-2,000 (new expert, Task 42's reference)"),
]
STEP = 0.5
RATIO = 1.5
HALF_WIDTH = 100.0
PRECISION_DEALS = (10, 50, 100, 400, 2000)


def fmt(x: float | None, nd: int = 2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def pct(x: float | None, nd: int = 1) -> str:
    return "-" if x is None else f"{100 * x:.{nd}f} %"


def same_rung_games(out: Path) -> list[dict[str, Any]]:
    """Task 42's same-rung deals as games (all four seats the band), scores by replaying the
    stored history; cached in ``out/samerung-games.json``."""
    cache = out / "samerung-games.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    from big2.games.big2 import Rules, cards

    rules = Rules()
    games = []
    for band in BANDS:
        p = ROOT / "runs" / "p6_mistakes" / f"decisions-{band}.jsonl"
        with p.open(encoding="utf-8") as f:
            for line in f:
                if '"history"' not in line:
                    continue
                r = json.loads(line)
                st = rules.new_deal(r["deal_id"])
                for a in r["history"]:
                    st.apply(cards(a))
                assert st.is_terminal
                games.append(
                    {
                        "deal": r["deal_id"],
                        "players": [band] * 4,
                        "scores": [float(x) for x in st.scores],
                    }
                )
    cache.write_text(json.dumps(games), encoding="utf-8")
    return games


def build_report(out: Path, m: int) -> dict[str, Any]:
    from grading_rating import ladder_games, read_luck

    out.mkdir(parents=True, exist_ok=True)
    decisions = read_decisions([ROOT / "runs/p6_mistakes" / f"decisions-{b}.jsonl" for b in BANDS])
    rung_elo = {b: next(d["rung_elo"] for d in decisions if d["band"] == b) for b in BANDS}
    ladder = json.loads((ROOT / "runs/p6_ladder/ladder.json").read_text(encoding="utf-8"))
    rung_elo["random"] = next(r["elo"] for r in ladder["rungs"] if r["name"] == "random")
    res: dict[str, Any] = {"m": m, "bands": BANDS, "rung_elo": rung_elo}

    # 1. trivial shares
    triv = {}
    for b in BANDS:
        ds = [d for d in decisions if d["band"] == b]
        n = len(ds)
        forced = sum(bool(d["forced"]) for d in ds)
        flat = sum(bool(d.get("flat")) and not d["forced"] for d in ds)
        triv[b] = {
            "decisions": n,
            "forced": forced / n,
            "flat": flat / n,
            "trivial": (forced + flat) / n,
            "graded": n - forced - flat,
        }
    res["trivial"] = triv

    # 2. lines
    derived = gl.derive_lines(decisions, TOP_BAND, MEDIUM_BAND, step=STEP, ratio=RATIO)
    lines = derived["lines"]
    gl.check_lines(lines, RATIO)
    quant = {
        b: {str(s): gl.exceeded_on(ls, s) for s in (0.10, 0.05, 0.03, 0.01)}
        for b, ls in gl.losses_by_band(decisions).items()
    }
    res["lines"] = {
        "proposed": lines,
        "raw": derived["raw"],
        "noise": derived["noise"],
        "inputs": derived["inputs"],
        "d102": list(D102_LINES),
        "exceeded_on": quant,
    }
    res["word_shares"] = {
        "d102": gl.word_shares(decisions, D102_LINES),
        "proposed": gl.word_shares(decisions, lines),
    }
    res["same_word"] = {
        name: {key: gl.same_word_share(decisions, ln, key) for key in ("repeat", "rules")}
        for name, ln in (("d102", D102_LINES), ("proposed", lines))
    }

    # 3. accuracy, APL
    s = acc.accuracy_scale(lines[2])
    games_dec = acc.per_game(decisions, s, good_line=lines[1])
    band_acc = acc.per_band(games_dec)
    res["accuracy_s"] = s
    res["per_band"] = band_acc

    # 4. luck-removed rating
    luck_path = out / f"luck-m{m}.jsonl"
    luck_recs = read_luck(luck_path)
    luck = {k: v["mean"] for k, v in luck_recs.items()}
    games = ladder_games() + ladder_games(out / "pairs")
    same = same_rung_games(out)
    strengths = rt.fit_strengths(games, luck)
    ratings, c = rt.to_rating(strengths, ZERO, TOP)
    ranges = rt.bootstrap_ratings(games, luck, ZERO, TOP, resamples=300)
    msd = rt.margin_sd(games, luck, strengths)
    res["rating"] = {
        "luck": f"the hour-24 network plays all four seats of the deal; m={m} "
        + ("top-move game" if m == 0 else "sampled games at temperature 1"),
        "games": len(games),
        "deals": len({g["deal"] for g in games}),
        "strengths": strengths,
        "ratings": ratings,
        "ranges": ranges,
        "c": c,
        "margin_sd": msd,
        "deals_for_pm100": rt.deals_for_half_width(c, msd, HALF_WIDTH),
        "k_min": rt.K_MIN,
        "prior_deals": rt.PRIOR_DEALS,
        "half_width_after": {str(n): rt.half_width_after(c, msd, n) for n in PRECISION_DEALS},
        "steady_jitter": rt.steady_jitter(c, msd, rt.K_MIN),
        "variance_cut": rt.variance_cut(games, luck),
        "variance_cut_same_rung": {
            b: rt.variance_cut([g for g in same if g["players"][0] == b], luck) for b in BANDS
        },
    }
    if m > 0:
        # The sampled estimate is the deal's true expectation plus noise of variance
        # mean(sd^2) / m, so the cut at other sample counts follows from the one measured.
        noise = float(np.mean([x**2 for v in luck_recs.values() for x in v["sd"]]))
        vc = res["rating"]["variance_cut"]
        var_score = vc["score_sd"] ** 2
        limit = vc["cut"] + noise / (m * var_score)
        res["rating"]["luck_noise"] = {
            "playout_var": noise,
            "cut_limit": limit,
            "cut_at": {str(k): limit - noise / (k * var_score) for k in (1, 4, 16, 64)},
        }
    # pairing check: observed vs predicted margin of a vs b
    pair_rows = []
    by_pair: dict[tuple[str, str], list[float]] = {}
    for g in games:
        if g["deal"] not in luck:
            continue
        names = sorted(set(g["players"]))
        if len(names) != 2:
            continue
        a, b = names
        lk = luck[g["deal"]]
        ma = np.mean([g["scores"][i] - lk[i] for i, n in enumerate(g["players"]) if n == a])
        by_pair.setdefault((a, b), []).append(float(ma))
    for (a, b), v in sorted(by_pair.items()):
        pred = rt.expected_margin(strengths[a], [strengths[b], strengths[a], strengths[b]])
        pair_rows.append(
            {
                "a": a,
                "b": b,
                "games": len(v),
                "observed": float(np.mean(v)),
                "se": float(np.std(v, ddof=1) / math.sqrt(len(v))),
                "predicted": pred,
            }
        )
    res["rating"]["pairings"] = pair_rows
    # equal players: each name split by rotation parity
    halves = rt.split_by_rotation(games, lambda k: k % 2)
    sh = rt.fit_strengths(halves, luck)
    res["rating"]["halves"] = {
        n: {"A": c * (sh[f"{n}/A"] - strengths[ZERO]), "B": c * (sh[f"{n}/B"] - strengths[ZERO])}
        for n in strengths
        if f"{n}/A" in sh and f"{n}/B" in sh
    }
    # same-rung tables: mean margin by seat position (should be 0)
    seat_means = {}
    for b in BANDS:
        gs = [g for g in same if g["players"][0] == b and g["deal"] in luck]
        margins = np.array([[g["scores"][i] - luck[g["deal"]][i] for i in range(4)] for g in gs])
        seat_means[b] = {
            "deals": len(gs),
            "mean": [float(x) for x in margins.mean(0)],
            "se": [float(x) for x in margins.std(0, ddof=1) / math.sqrt(len(gs))],
        }
    res["rating"]["same_rung_seat_margin"] = seat_means

    # 5. rating from APL, orderings
    band_rating = {b: ratings[b] for b in BANDS if b in ratings}
    fit = acc.fit_log_apl(band_rating, {b: band_acc[b]["apl"] for b in BANDS})
    need = {
        b: acc.games_for_half_width(
            fit["b"], band_acc[b]["apl"], band_acc[b]["apl_game_sd"], HALF_WIDTH
        )
        for b in BANDS
    }
    res["elo_from_apl"] = {**fit, "games_for_pm100": need}
    order = {}
    for target_name, target in (("rung_elo", rung_elo), ("rating", ratings)):
        y = [target[b] for b in BANDS]
        order[target_name] = {
            "apl": acc.spearman([-band_acc[b]["apl"] for b in BANDS], y),
            "accuracy": acc.spearman([band_acc[b]["accuracy"] for b in BANDS], y),
            "best_good_share": acc.spearman([band_acc[b]["best_good_share"] for b in BANDS], y),
        }
    yg = [ratings[g["band"]] for g in games_dec]
    order["single_games_vs_rating"] = {
        "apl": acc.spearman([-g["apl"] for g in games_dec], yg),
        "accuracy": acc.spearman([g["accuracy"] for g in games_dec], yg),
        "best_good_share": acc.spearman([g["best_good_share"] for g in games_dec], yg),
    }
    res["ordering"] = order

    (out / "tables.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    (out / "report.md").write_text(markdown(res), encoding="utf-8")
    (out / "proposal.json").write_text(json.dumps(proposal(res), indent=1), encoding="utf-8")
    print(markdown(res))
    return res


def proposal(res: dict[str, Any]) -> dict[str, Any]:
    r = res["rating"]
    return {
        "task": "Task 43: grade lines, trivial flag, accuracy, luck-removed rating",
        "reference": "PUCT-2000, c_puct 2.5, belief=net, exact endgame at <= 12 hidden cards "
        "(decision 0010)",
        "trivial": {
            "rule": "forced (one legal move) or flat (every evaluated move's loss under 0.2 "
            "points or its 95 % range covers 0); no grade word, counts in no score",
            "share_by_band": {b: t["trivial"] for b, t in res["trivial"].items()},
        },
        "lines": dict(zip(gl.LINE_NAMES, res["lines"]["proposed"], strict=True)),
        "lines_words": list(WORDS),
        "lines_rule": res["lines"]["inputs"]
        | {"noise_half_width": res["lines"]["noise"], "raw": res["lines"]["raw"]},
        "d102": dict(zip(gl.LINE_NAMES, res["lines"]["d102"], strict=True)),
        "accuracy_s": res["accuracy_s"],
        "accuracy": "mean over non-trivial decisions of 100 * exp(-loss / accuracy_s)",
        "apl": "mean points lost over non-trivial decisions",
        "elo_from_apl": {
            "a": res["elo_from_apl"]["a"],
            "b": res["elo_from_apl"]["b"],
            "formula": "rating = a + b * ln(APL)",
            "rms_residual": res["elo_from_apl"]["rms_residual"],
            "games_for_pm100": res["elo_from_apl"]["games_for_pm100"],
        },
        "rating": {
            "scale": {"zero": ZERO, "top": TOP, "top_rating": rt.TOP_RATING},
            "luck": r["luck"],
            "margin": "seat's score minus its luck",
            "expected_margin": "s_player - mean(s over the four seats), s = rating / c",
            "c": r["c"],
            "update": "R <- R + max(1/(n+prior_deals+1), k_min) * c * (margin - expected_margin)",
            "k_min": r["k_min"],
            "prior_deals": r["prior_deals"],
            "half_width_after": r["half_width_after"],
            "steady_jitter": r["steady_jitter"],
            "margin_sd": r["margin_sd"],
            "deals_for_pm100": r["deals_for_pm100"],
            "bots": {
                name: {"rating": r["ratings"].get(name), "range95": r["ranges"].get(name)}
                for name, _ in BOTS
                if name in r["ratings"]
            },
            "rungs": {
                name: {"rating": v, "range95": r["ranges"].get(name)}
                for name, v in r["ratings"].items()
            },
            "variance_cut": r["variance_cut"],
        },
    }


def row(*cells: Any) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def sep(n: int, left: int = 1) -> str:
    return "|" + "---|" * left + "---:|" * (n - left)


def markdown(res: dict[str, Any]) -> str:
    L = res["lines"]
    r = res["rating"]
    o: list[str] = []
    o.append("### Trivial decisions\n")
    o.append(row("band", "decisions", "forced", "flat", "trivial", "graded"))
    o.append(sep(6))
    for b in res["bands"]:
        t = res["trivial"][b]
        o.append(
            row(
                b,
                f"{t['decisions']:,}",
                pct(t["forced"]),
                pct(t["flat"]),
                pct(t["trivial"]),
                f"{t['graded']:,}",
            )
        )
    o.append("\n### Lines\n")
    proposed = " / ".join(fmt(x, 1) for x in L["proposed"])
    d102 = " / ".join(str(x) for x in L["d102"])
    o.append(
        f"Noise half-width (median 1.96 x loss SE, played move not best): {L['noise']:.3f}. "
        f"Raw rule values: {', '.join(fmt(x) for x in L['raw'])}; rounded to {STEP} and lifted "
        f"to {RATIO}x: **{proposed}** (D102: {d102}).\n"
    )
    o.append(row("band", "loss exceeded on 10 %", "5 %", "3 %", "1 %"))
    o.append(sep(5))
    for b in res["bands"]:
        q = L["exceeded_on"][b]
        o.append(row(b, fmt(q["0.1"]), fmt(q["0.05"]), fmt(q["0.03"]), fmt(q["0.01"])))
    for name, label in (("d102", "D102"), ("proposed", "proposed")):
        o.append(f"\nWord shares over graded decisions, {label} lines:\n")
        o.append(row("band", *WORDS))
        o.append(sep(1 + len(WORDS)))
        for b in res["bands"]:
            w = res["word_shares"][name][b]
            o.append(row(b, *(pct(w[x]) for x in WORDS)))
        sw = res["same_word"][name]
        o.append(
            f"\nSame word on a new seed: {pct(sw['repeat'][0])} ({sw['repeat'][1]}); "
            f"under `belief=rules`: {pct(sw['rules'][0])} ({sw['rules'][1]})."
        )
    o.append("\n### Accuracy and APL\n")
    o.append(f"accuracy_s = {res['accuracy_s']:.3f} (a loss on the inaccuracy line scores 50).\n")
    o.append(
        row(
            "band",
            "rung Elo (Task 40)",
            "luck-removed rating",
            "graded decisions",
            "APL",
            "per-game APL SD",
            "accuracy",
            "per-game accuracy SD",
            "best-or-good share",
            "games for +-100",
        )
    )
    o.append(sep(10))
    for b in res["bands"]:
        p = res["per_band"][b]
        o.append(
            row(
                b,
                f"{res['rung_elo'][b]:.0f}",
                f"{r['ratings'].get(b, float('nan')):.0f}",
                f"{p['decisions']:,}",
                f"{p['apl']:.3f}",
                fmt(p["apl_game_sd"], 3),
                f"{p['accuracy']:.1f}",
                fmt(p["accuracy_game_sd"], 1),
                pct(p["best_good_share"]),
                f"{res['elo_from_apl']['games_for_pm100'][b]:,}",
            )
        )
    e = res["elo_from_apl"]
    o.append(
        f"\nRating from APL: rating = {e['a']:.0f} + ({e['b']:.0f}) x ln(APL); "
        f"RMS residual {e['rms_residual']:.0f} rating points.\n"
    )
    o.append("Spearman rank correlation of each measure with the rungs' order:\n")
    o.append(
        row(
            "measure",
            "vs Task 40 Elo (7 rungs)",
            "vs luck-removed rating (7 rungs)",
            "single games vs rating",
        )
    )
    o.append(sep(4))
    od = res["ordering"]
    for k in ("apl", "accuracy", "best_good_share"):
        o.append(
            row(
                k,
                fmt(od["rung_elo"][k], 3),
                fmt(od["rating"][k], 3),
                fmt(od["single_games_vs_rating"][k], 3),
            )
        )
    o.append("\n### Luck-removed rating\n")
    o.append(
        f"Luck: {r['luck']}. Games {r['games']:,} on {r['deals']:,} deals. c = {r['c']:.1f} "
        f"rating points per point of margin a deal; margin SD {r['margin_sd']:.2f}; deals for "
        f"+-100 (95 %): {r['deals_for_pm100']}.\n"
    )
    hw = ", ".join(f"{n} deals +-{x:.0f}" for n, x in r["half_width_after"].items())
    o.append(
        f"95 % half-width of a player's rating after {hw}; steady step k_min = "
        f"{r['k_min']:.4f} leaves a jitter SD of {r['steady_jitter']:.0f} on a constant player; "
        f"the start rating counts as {r['prior_deals']} deals.\n"
    )
    o.append(row("bot", "strength (points a deal)", "rating", "95 % range", "half A", "half B"))
    o.append(sep(6))
    for name in sorted(r["ratings"], key=lambda n: r["ratings"][n]):
        rg = r["ranges"].get(name, (float("nan"), float("nan")))
        h = r["halves"].get(name, {"A": float("nan"), "B": float("nan")})
        o.append(
            row(
                name,
                f"{r['strengths'][name]:+.2f}",
                f"{r['ratings'][name]:.0f}",
                f"[{rg[0]:.0f}, {rg[1]:.0f}]",
                f"{h['A']:.0f}",
                f"{h['B']:.0f}",
            )
        )
    v = r["variance_cut"]
    o.append(
        f"\nVariance cut over all seats: score SD {v['score_sd']:.2f} -> margin SD "
        f"{v['margin_sd']:.2f}, cut {pct(v['cut'])}; duplicate leave-one-out baseline cut "
        f"{pct(v.get('duplicate_cut'))} (margin SD {fmt(v.get('duplicate_margin_sd'))}).\n"
    )
    if "luck_noise" in r:
        ln = r["luck_noise"]
        at = ", ".join(f"m={k}: {pct(x)}" for k, x in ln["cut_at"].items())
        o.append(
            f"Self-play noise: one sampled game's score varies with SD "
            f"{math.sqrt(ln['playout_var']):.2f} around the deal's expectation, so the cut at "
            f"other sample counts is {at}; limit (exact expectation) {pct(ln['cut_limit'])}.\n"
        )
    o.append(
        "Same-rung tables (Task 42's deals): variance cut and mean margin by seat (should be 0):\n"
    )
    o.append(row("band", "deals", "cut", "seat 0", "seat 1", "seat 2", "seat 3"))
    o.append(sep(7))
    for b in res["bands"]:
        sm = r["same_rung_seat_margin"][b]
        vc = r["variance_cut_same_rung"][b]
        cells = [f"{m:+.2f} +- {s:.2f}" for m, s in zip(sm["mean"], sm["se"], strict=True)]
        o.append(row(b, sm["deals"], pct(vc["cut"]), *cells))
    o.append("\nPairings: observed mean margin of a (per game, two seats) vs the additive model:\n")
    o.append(row("a", "b", "games", "observed", "+- SE", "predicted"))
    o.append(sep(6, 2))
    for p in r["pairings"]:
        o.append(
            row(
                p["a"],
                p["b"],
                p["games"],
                f"{p['observed']:+.2f}",
                f"{p['se']:.2f}",
                f"{p['predicted']:+.2f}",
            )
        )
    return "\n".join(o) + "\n"
