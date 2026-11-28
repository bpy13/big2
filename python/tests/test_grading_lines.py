"""Grade lines, the trivial flag, accuracy and the luck-removed rating (plan Task 43)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from big2.teach.grading import accuracy as acc
from big2.teach.grading import lines as gl
from big2.teach.grading import rating as rt
from big2.teach.grading.losses import word


def decision(band: str, loss: float, se: float = 0.25, **kw) -> dict:
    d = {
        "band": band,
        "deal_id": kw.pop("deal", 1),
        "seat": kw.pop("seat", 0),
        "forced": False,
        "flat": False,
        "loss": loss,
        "loss_se": se if loss > 0 else 0.0,
        "moves": [
            {"action": "a", "loss": 0.0, "loss_se": 0.0},
            {"action": "b", "loss": max(loss, 1.0), "loss_se": se},
        ],
    }
    d.update(kw)
    return d


def synthetic(n: int = 1000) -> list[dict]:
    """Two bands whose non-trivial losses are known quantiles: ``top`` has losses 0..n-1 scaled
    to 0..10, ``medium`` 0..20. Forced and flat decisions are sprinkled in."""
    out = []
    for band, top in (("top", 10.0), ("medium", 20.0)):
        for i in range(n):
            out.append(decision(band, top * i / (n - 1), deal=i // 10, seat=i % 4))
        out.append(decision(band, 99.0, forced=True))
        out.append(decision(band, 99.0, flat=True))
    return out


def test_trivial_is_forced_or_flat():
    assert gl.trivial({"forced": True, "flat": False})
    assert gl.trivial({"forced": False, "flat": True})
    assert not gl.trivial({"forced": False, "flat": False})
    assert gl.is_flat([{"loss": 0.1, "loss_se": 0.0}, {"loss": 0.4, "loss_se": 0.3}])
    assert not gl.is_flat([{"loss": 0.0, "loss_se": 0.0}, {"loss": 1.0, "loss_se": 0.2}])


def test_lines_give_known_percentiles_back():
    ds = synthetic()
    res = gl.derive_lines(ds, "top", "medium", step=0.0, ratio=1.0)
    great, good, inacc, mistake = res["lines"]
    assert great == pytest.approx(1.96 * 0.25, abs=1e-9)  # the noise half-width, above 0.2
    assert good == pytest.approx(9.0, abs=0.02)  # top exceeds 9 on 10 % of 0..10
    assert inacc == pytest.approx(9.7, abs=0.02)  # top exceeds 9.7 on 3 %
    assert mistake == pytest.approx(19.4, abs=0.02)  # medium exceeds 19.4 on 3 %
    # forced and flat decisions were ignored (their 99.0 losses did not move the lines)
    assert all(x < 20 for x in res["lines"])


def test_lines_are_monotone_and_rounded():
    ds = synthetic()
    res = gl.derive_lines(ds, "top", "medium", step=0.5, ratio=1.5)
    lines = res["lines"]
    gl.check_lines(lines, 1.5)
    assert lines == sorted(lines)
    assert all(abs(x / 0.5 - round(x / 0.5)) < 1e-9 for x in lines)
    # the 1.5x rule lifted the inaccuracy line above the raw 9.7
    assert lines[2] >= 1.5 * lines[1]
    with pytest.raises(ValueError):
        gl.check_lines([0.5, 2.5, 3.0, 10.0], 1.5)
    with pytest.raises(ValueError):
        gl.check_lines([0.5, 2.5, 7.0], 1.5)


def test_word_shares_and_repeat():
    ds = [
        decision("b", 0.0),
        decision("b", 1.0),
        decision("b", 8.0),
        decision("b", 50.0, forced=True),
    ]
    ds[1]["repeat"] = {"loss": 0.1}  # good -> great under these lines
    ds[2]["repeat"] = {"loss": 9.0}
    lines = (0.5, 2.5, 7.0, 10.5)
    shares = gl.word_shares(ds, lines)["b"]
    assert shares["great"] == pytest.approx(1 / 3)
    assert shares["good"] == pytest.approx(1 / 3)
    assert shares["mistake"] == pytest.approx(1 / 3)
    same, n = gl.same_word_share(ds, lines)
    assert (same, n) == (0.5, 2)
    assert word(10.6, lines) == "blunder"


def test_accuracy_scale_and_game_measures():
    s = acc.accuracy_scale(7.0)
    assert acc.move_accuracy(7.0, s) == pytest.approx(50.0)
    assert acc.move_accuracy(0.0, s) == 100.0
    ds = [
        decision("b", 0.0, deal=1, seat=0),
        decision("b", 7.0, deal=1, seat=0),
        decision("b", 3.0, deal=1, seat=1),
        decision("b", 1.0, deal=1, seat=1, forced=True),
    ]
    games = acc.per_game(ds, s, good_line=2.5)
    assert len(games) == 2
    g0 = next(g for g in games if g["seat"] == 0)
    assert g0["apl"] == 3.5 and g0["accuracy"] == pytest.approx(75.0)
    assert g0["best_good_share"] == 0.5
    band = acc.per_band(games)["b"]
    assert band["decisions"] == 3 and band["apl"] == pytest.approx(10 / 3)


def test_spearman_and_log_apl_fit():
    assert acc.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert acc.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert acc.spearman([1, 1, 2], [1, 2, 3]) is not None
    rating = {"a": 0.0, "b": 1500.0, "c": 3000.0}
    aplv = {"a": math.e**3, "b": math.e**1.5, "c": 1.0}  # rating = 3000 - 1000 log(APL)
    fit = acc.fit_log_apl(rating, aplv)
    assert fit["a"] == pytest.approx(3000.0) and fit["b"] == pytest.approx(-1000.0)
    assert fit["rms_residual"] == pytest.approx(0.0, abs=1e-6)
    assert acc.rating_from_apl(1.0, fit["a"], fit["b"]) == pytest.approx(3000.0)
    assert acc.games_for_half_width(-1000.0, 2.0, 1.0, 100.0) == math.ceil((1.96 * 1000 / 200) ** 2)


def _tables(rng: np.random.Generator, strengths: dict[str, float], deals: int, noise: float):
    """Synthetic duplicate games: each deal's luck is a random zero-sum vector; a seat's score
    is luck + its strength minus the table mean plus noise."""
    names = list(strengths)
    games, luck = [], {}
    for d in range(deals):
        lk = rng.normal(0, 25, 4)
        lk -= lk.mean()
        luck[d] = list(lk)
        order = list(rng.permutation(names))
        for rot in range(4):
            seats = [order[(s - rot) % 4] for s in range(4)]
            s = np.array([strengths[n] for n in seats])
            scores = lk + s - s.mean() + rng.normal(0, noise, 4)
            scores -= scores.mean()
            games.append({"deal": d, "players": seats, "scores": list(scores)})
    return games, luck


def test_rating_recovers_strengths_and_is_not_capped():
    rng = np.random.default_rng(3)
    true = {"random": -20.0, "mid": -5.0, "net": 0.0, "better": 4.0}
    games, luck = _tables(rng, true, deals=600, noise=6.0)
    s = rt.fit_strengths(games, luck, anchor=("net", 0.0))
    for n, v in true.items():
        assert s[n] == pytest.approx(v, abs=0.6)
    ratings, c = rt.to_rating(s, "random", "net")
    assert ratings["random"] == 0.0 and ratings["net"] == 3000.0
    assert ratings["better"] > 3000.0  # a player above the network rates above it
    assert c == pytest.approx(3000 / 20, rel=0.05)
    # equal players come out equal: each name split by rotation parity
    halves = rt.split_by_rotation(games, lambda k: k % 2)
    sh = rt.fit_strengths(halves, luck)
    for n in true:
        assert sh[f"{n}/A"] == pytest.approx(sh[f"{n}/B"], abs=1.0)
    # the luck term removes most of the score variance
    cut = rt.variance_cut(games, luck)
    assert cut["cut"] > 0.8
    ranges = rt.bootstrap_ratings(games, luck, "random", "net", resamples=50)
    lo, hi = ranges["mid"]
    assert lo < ratings["mid"] < hi


def test_update_and_deals_needed():
    c = 150.0  # rating points per point of margin
    # a 1500 player at a table of three 1500s: expected margin 0; wins 2 points -> rating up
    r1 = rt.update(1500.0, 0, 2.0, [1500.0] * 3, c)
    assert r1 == pytest.approx(1500.0 + 1.0 * c * 2.0)
    # after many deals the step is k_min
    r2 = rt.update(1500.0, 10_000, 2.0, [1500.0] * 3, c, k_min=0.02)
    assert r2 == pytest.approx(1500.0 + 0.02 * c * 2.0)
    # expected margin against weaker opponents is positive, so the same win moves less
    r3 = rt.update(1500.0, 0, 2.0, [600.0] * 3, c)
    assert r3 < r1
    assert rt.expected_margin(10.0, [0.0, 0.0, 0.0]) == pytest.approx(7.5)
    assert rt.deals_for_half_width(c, 20.0, 100.0) == math.ceil((1.96 * 150 * 20 / 100) ** 2)
