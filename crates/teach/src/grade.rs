//! The paired grader (hard-problems §8, teaching-and-deployment §2.2).
//!
//! **Paired deals.** [`evaluate_paired`] samples deals of the hidden cards from
//! the actor's view and plays *every* legal action out on each of them, with the
//! same random stream for every action. A move's loss is then the mean of the
//! per-deal differences "best − played", whose noise is much smaller than that of
//! two independent means: what the deal does to both moves cancels.
//!
//! **Thresholds.** Lines are multiples of σ, the standard deviation of one
//! seat's score in one deal of expert self-play ([`SIGMA_TIERED`], measured in
//! Task 6): `good` < 0.025σ, `inaccuracy` < 0.1σ, `mistake` < 0.25σ, `blunder`
//! from 0.25σ. `best` is the table's best row. The app's current lines (1 / 3 / 9
//! points) are [`Thresholds::APP`]; open question 3 asks Brian which ships.
//!
//! **Unclear.** When the loss's 95 % interval crosses a line, the grade is
//! `unclear` and [`Graded::range`] names the two grades it lies between.
//!
//! **Seeds.** Every random draw comes from `ChaCha8Rng` seeded from
//! [`PairedConfig::seed`], which gives the same stream on every platform, so a
//! grade made on the phone repeats on a PC from its stored seed.

use crate::table::{MoveRow, MoveTable};
use big2_big2::bots::Greedy;
use big2_big2::plan::min_plays;
use big2_big2::{Action, State, NUM_PLAYERS};
use big2_games::{play_out, Game, Policy};
use big2_search::{infoset_endgame, InfosetConfig, Objective, UniformSampler};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// σ of one seat's score in one deal under `tiered` scoring, duty on: 27.45
/// points (ISMCTS self-play, 16,000 seat-games; `build-phase1.md` §2).
pub const SIGMA_TIERED: f64 = 27.45;

/// Exploration noise of the default rollout policy, as in the app's solvers.
pub const ROLLOUT_EPSILON: f64 = 0.1;

/// Grade lines in points: a loss under `good` is `good`, under `inaccuracy` is
/// `inaccuracy`, under `mistake` is `mistake`, else `blunder`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thresholds {
    pub good: f64,
    pub inaccuracy: f64,
    pub mistake: f64,
}

impl Thresholds {
    /// The app's current lines under `tiered`: ⅓, 1 and 3 cards at 3 points per card.
    pub const APP: Thresholds = Thresholds {
        good: 1.0,
        inaccuracy: 3.0,
        mistake: 9.0,
    };

    /// Lines at 0.025σ, 0.1σ and 0.25σ.
    pub fn from_sigma(sigma: f64) -> Thresholds {
        Thresholds {
            good: 0.025 * sigma,
            inaccuracy: 0.1 * sigma,
            mistake: 0.25 * sigma,
        }
    }

    /// The grade of a loss of `loss` points (> 0) against the best move.
    pub fn grade_of(&self, loss: f64) -> &'static str {
        if loss < self.good {
            "good"
        } else if loss < self.inaccuracy {
            "inaccuracy"
        } else if loss < self.mistake {
            "mistake"
        } else {
            "blunder"
        }
    }
}

impl Default for Thresholds {
    /// σ-scaled lines at [`SIGMA_TIERED`]: 0.69 / 2.75 / 6.86 points.
    fn default() -> Self {
        Thresholds::from_sigma(SIGMA_TIERED)
    }
}

/// Settings of [`evaluate_paired`] and [`grade_paired`].
#[derive(Clone, Debug)]
pub struct PairedConfig {
    /// Sampled deals; every legal action is played out once on each.
    pub deals: u64,
    /// Seed of every random draw (store it beside the grade).
    pub seed: u64,
    /// Grade lines.
    pub thresholds: Thresholds,
    /// When set, the exact information-set endgame runs too, and a proven
    /// winner is reported as [`MoveTable::forced`]. The values stay the
    /// sampled ones, so a grade never depends on whether the endgame fired.
    pub endgame: Option<InfosetConfig>,
    /// [`grade_paired`] only: while the grade is `unclear`, double the deals up
    /// to this many (teaching-and-deployment §2.3). Values at or below `deals`
    /// turn the doubling off.
    pub max_deals: u64,
}

impl Default for PairedConfig {
    /// 300 deals (the app's review budget), seed 0, σ lines, no exact endgame,
    /// no doubling.
    fn default() -> Self {
        PairedConfig {
            deals: 300,
            seed: 0,
            thresholds: Thresholds::default(),
            endgame: None,
            max_deals: 0,
        }
    }
}

/// The grade of one action.
#[derive(Clone, Debug, PartialEq)]
pub struct Graded {
    /// The action graded.
    pub played: Action,
    /// The table's best action.
    pub best: Action,
    /// Points lost against `best`: the mean of the paired differences (0 for the best).
    pub loss: f64,
    /// Half-width of the 95 % interval of `loss`.
    pub interval: f64,
    /// `best`, `good`, `inaccuracy`, `mistake`, `blunder` or `unclear`.
    pub grade: &'static str,
    /// For `unclear`: the grades at the two ends of the interval (lower first).
    pub range: Option<(&'static str, &'static str)>,
    /// Sampled deals behind the grade.
    pub deals: u64,
    /// Seed of the evaluation.
    pub seed: u64,
}

/// Mixes a seed and an index into a new seed (SplitMix64 finaliser).
fn mix(seed: u64, i: u64) -> u64 {
    let mut z = seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// [`evaluate_paired_with`] with the app's rollout policy (greedy, ε = 0.1).
pub fn evaluate_paired(state: &State, cfg: &PairedConfig) -> MoveTable {
    evaluate_paired_with(
        state,
        cfg,
        &Greedy {
            epsilon: ROLLOUT_EPSILON,
        },
    )
}

/// Values every legal action of the player to act on the same `cfg.deals`
/// sampled deals, each played out to the end by `rollout` with a random stream
/// shared by all actions.
///
/// # Panics
/// Panics if the deal is over.
pub fn evaluate_paired_with<P: Policy<State>>(
    state: &State,
    cfg: &PairedConfig,
    rollout: &P,
) -> MoveTable {
    let actor = state.current_player().expect("the deal is not over");
    let legal = state.legal_actions_vec();
    let n = cfg.deals.max(1);
    let rules = state.rules();
    // per_deal[i][d]: every seat's score after action i on deal d.
    let mut scores = vec![Vec::with_capacity(n as usize); legal.len()];
    let mut wins = vec![0u64; legal.len()];
    let mut plays = [0.0f64; NUM_PLAYERS];
    for d in 0..n {
        let deal_seed = mix(cfg.seed, d);
        let mut rng = ChaCha8Rng::seed_from_u64(deal_seed);
        let deal = state.determinize(actor, &mut rng);
        for (p, x) in plays.iter_mut().enumerate() {
            *x += min_plays(deal.hand(p), rules) as f64;
        }
        for (i, &a) in legal.iter().enumerate() {
            // The same stream for every action: common random numbers.
            let mut roll = ChaCha8Rng::seed_from_u64(mix(deal_seed, 0x5EED));
            let mut s = deal.clone();
            s.apply(a);
            let o = play_out(&mut s, rollout, &mut roll);
            let mut v = [0.0; NUM_PLAYERS];
            v.copy_from_slice(&o.scores[..NUM_PLAYERS]);
            scores[i].push(v);
            wins[i] += u64::from(o.winner == Some(actor));
        }
    }
    for x in &mut plays {
        *x /= n as f64;
    }
    plays[actor] = min_plays(state.hand(actor), rules) as f64;

    let nf = n as f64;
    let mut rows: Vec<(MoveRow, Vec<[f64; NUM_PLAYERS]>)> = legal
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            let mine: Vec<f64> = scores[i].iter().map(|v| v[actor]).collect();
            let mean = mine.iter().sum::<f64>() / nf;
            let mut seat_means = [0.0; NUM_PLAYERS];
            for v in &scores[i] {
                for p in 0..NUM_PLAYERS {
                    seat_means[p] += v[p] / nf;
                }
            }
            let row = MoveRow {
                action: a,
                mean,
                ci95: ci95(&mine),
                win: wins[i] as f64 / nf,
                seat_means: Some(seat_means),
            };
            (row, std::mem::take(&mut scores[i]))
        })
        .collect();
    rows.sort_by(|a, b| b.0.mean.total_cmp(&a.0.mean));

    let (rows, per_deal): (Vec<MoveRow>, Vec<Vec<[f64; NUM_PLAYERS]>>) = rows.into_iter().unzip();
    // The proof is about the exact search's best action; keep it only when the
    // sampled table's best action is that one, or ties with it in the exact
    // values and wins in every deal too (then the proven winner is the actor).
    let forced = cfg.endgame.as_ref().and_then(|ec| {
        let r = infoset_endgame(state, actor, ec, Objective::Score, &UniformSampler)?;
        let f = r.forced.filter(|_| r.exact)?;
        let top = &r.analysis.moves[0];
        let ours = r
            .analysis
            .moves
            .iter()
            .find(|m| m.action == rows[0].action)?;
        let same = ours.action == top.action
            || (f.winner == actor
                && (ours.mean_score - top.mean_score).abs() < 1e-9
                && ours.win_rate == 1.0);
        same.then_some(f)
    });
    MoveTable {
        actor,
        rows,
        deals: n,
        plays_to_out: plays,
        forced,
        per_deal: Some(per_deal),
    }
}

/// Half-width of the 95 % interval of the mean of `xs` (sample variance).
pub(crate) fn ci95(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return 0.0;
    }
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64;
    1.96 * (var / n as f64).sqrt()
}

/// Grades `played` against the table's best row with `thresholds`.
///
/// With per-deal values (a paired table) the loss and its interval come from
/// the paired differences; otherwise from the two means, with the interval
/// `√(ci_best² + ci_played²)` (independent samples).
///
/// # Panics
/// Panics if `played` is not in the table.
pub fn grade_from_table(
    table: &MoveTable,
    played: Action,
    thresholds: &Thresholds,
    seed: u64,
) -> Graded {
    let i = table.index_of(played).expect("the played action is legal");
    let best = table.best();
    let mut g = Graded {
        played,
        best: best.action,
        loss: 0.0,
        interval: 0.0,
        grade: "best",
        range: None,
        deals: table.deals,
        seed,
    };
    if table.rows.len() == 1 {
        return g;
    }
    // Mean and 95 % half-width of row a − row b, paired when per-deal values exist.
    let gap = |a: usize, b: usize| -> (f64, f64) {
        match &table.per_deal {
            Some(pd) => {
                let me = table.actor;
                let diffs: Vec<f64> = pd[a]
                    .iter()
                    .zip(&pd[b])
                    .map(|(x, y)| x[me] - y[me])
                    .collect();
                (diffs.iter().sum::<f64>() / diffs.len() as f64, ci95(&diffs))
            }
            None => {
                let (x, y) = (&table.rows[a], &table.rows[b]);
                (x.mean - y.mean, x.ci95.hypot(y.ci95))
            }
        }
    };
    let (loss, interval) = if i == 0 {
        // The best row loses nothing on these deals, but the runner-up may be
        // better in truth: its true loss can reach (runner-up − played) at the
        // interval's upper end.
        let (margin, ci) = gap(0, 1);
        (0.0, (ci - margin).max(0.0))
    } else {
        gap(0, i)
    };
    g.loss = loss;
    g.interval = interval;
    let (low, high) = grade_span(loss, interval, thresholds);
    if low == high {
        g.grade = low;
    } else {
        g.grade = "unclear";
        g.range = Some((low, high));
    }
    g
}

/// The grades at the two ends of a loss's 95 % interval, `best` first.
///
/// `best` means the move cannot be told apart from the best one (the interval
/// reaches 0) and surely loses less than the `good` line; `good` needs a loss
/// the interval shows to be real but under the line. So a near-tie between two
/// moves is `best` for both, whichever of them the deals happened to rank first.
fn grade_span(loss: f64, interval: f64, t: &Thresholds) -> (&'static str, &'static str) {
    let (lo, hi) = (loss - interval, loss + interval);
    let low = if lo <= 0.0 { "best" } else { t.grade_of(lo) };
    let high = t.grade_of(hi);
    if low == "best" && high == "good" {
        ("best", "best")
    } else {
        (low, high)
    }
}

/// Grades `played`, an action of the player to act in `state`, on `cfg.deals`
/// paired deals with the app's rollout policy, doubling the deals while the
/// grade is `unclear` up to `cfg.max_deals`. A forced move (one legal action)
/// grades `best` without any search.
pub fn grade_paired(state: &State, played: Action, cfg: &PairedConfig) -> Graded {
    grade_paired_table(state, played, cfg).0
}

/// [`grade_paired`] that also returns the last move table (`None` for a forced
/// move), so reasons and the luck ledger can reuse it.
///
/// Deal `d` is drawn from the seed and `d` alone, so a doubled run replays the
/// first deals exactly and adds new ones.
pub fn grade_paired_table(
    state: &State,
    played: Action,
    cfg: &PairedConfig,
) -> (Graded, Option<MoveTable>) {
    let legal = state.legal_actions_vec();
    if legal.len() == 1 {
        let g = Graded {
            played,
            best: legal[0],
            loss: 0.0,
            interval: 0.0,
            grade: "best",
            range: None,
            deals: 0,
            seed: cfg.seed,
        };
        return (g, None);
    }
    let mut c = cfg.clone();
    loop {
        let table = evaluate_paired(state, &c);
        let g = grade_from_table(&table, played, &c.thresholds, c.seed);
        if g.grade != "unclear" || c.deals * 2 > c.max_deals {
            return (g, Some(table));
        }
        c.deals *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use big2_big2::card::parse_mask;
    use big2_big2::RuleSet;

    fn h(s: &str) -> u64 {
        parse_mask(s).unwrap()
    }

    fn act(s: &State, cards: &str) -> Action {
        let m = if cards.is_empty() { 0 } else { h(cards) };
        s.legal_actions_vec()
            .into_iter()
            .find(|a| a.cards() == m)
            .unwrap()
    }

    fn cfg(deals: u64, seed: u64) -> PairedConfig {
        PairedConfig {
            deals,
            seed,
            ..PairedConfig::default()
        }
    }

    #[test]
    fn grade_thresholds_scale_with_sigma() {
        let t = Thresholds::from_sigma(20.0);
        assert_eq!((t.good, t.inaccuracy, t.mistake), (0.5, 2.0, 5.0));
        let d = Thresholds::default();
        assert!((d.good - 0.68625).abs() < 1e-12 && (d.mistake - 6.8625).abs() < 1e-12);
        assert_eq!(t.grade_of(0.1), "good");
        assert_eq!(t.grade_of(1.0), "inaccuracy");
        assert_eq!(t.grade_of(4.9), "mistake");
        assert_eq!(t.grade_of(5.0), "blunder");
        assert_eq!(Thresholds::APP.grade_of(2.0), "inaccuracy");
    }

    #[test]
    fn grade_of_a_forced_move_is_best() {
        // Seat 0 cannot beat the 2s: the pass is the only action.
        let s = State::from_position(
            RuleSet::default(),
            [h("3c 4d"), h("5d 5c"), h("6d 6c"), h("7d 7c")],
            0,
            Some((3, h("2s"))),
        )
        .unwrap();
        let g = grade_paired(&s, Action::Pass, &cfg(50, 1));
        assert_eq!(
            (g.grade, g.loss, g.interval, g.deals),
            ("best", 0.0, 0.0, 0)
        );
    }

    /// Seat 0 leads with 3c 2h 2s: the pair of 2s cannot be beaten, so leading it
    /// and then the 3c wins at once; leading a single gives the others a chance.
    fn choice() -> State {
        State::from_position(
            RuleSet::default(),
            [h("3c 2s 2h"), h("Ad Kd"), h("4d 4c"), h("5d 5c")],
            0,
            None,
        )
        .unwrap()
    }

    #[test]
    fn grade_paired_finds_a_blunder_and_the_best_move() {
        let s = choice();
        let c = cfg(40, 3);
        let table = evaluate_paired(&s, &c);
        // The pair of 2s wins at once in every deal (no unseen pair beats it, then 3c).
        let best = table.best().action;
        let g = grade_paired(&s, best, &c);
        assert_eq!(g.grade, "best");
        // Leading the 3c lets an opponent take the lead with cards to spare.
        let g = grade_paired(&s, act(&s, "3c"), &c);
        assert!(g.loss > 0.0, "{g:?}");
        assert_ne!(g.grade, "best");
        assert_eq!(g.seed, 3);
        assert_eq!(g.deals, 40);
    }

    #[test]
    fn grade_paired_is_repeatable_from_its_seed() {
        let s = choice();
        let a = evaluate_paired(&s, &cfg(30, 9));
        let b = evaluate_paired(&s, &cfg(30, 9));
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.per_deal, b.per_deal);
        let c = evaluate_paired(&s, &cfg(30, 10));
        assert!(a.per_deal != c.per_deal || a.rows.len() == 1);
    }

    #[test]
    fn grade_paired_rows_share_their_deals() {
        let s = choice();
        let t = evaluate_paired(&s, &cfg(25, 4));
        let pd = t.per_deal.as_ref().unwrap();
        assert_eq!(pd.len(), t.rows.len());
        assert!(pd.iter().all(|v| v.len() == 25));
        for (r, v) in t.rows.iter().zip(pd) {
            assert!((r.mean - v.iter().map(|x| x[0]).sum::<f64>() / 25.0).abs() < 1e-9);
            let seats = r.seat_means.unwrap();
            assert!((seats[0] - r.mean).abs() < 1e-9);
            assert!(seats.iter().sum::<f64>().abs() < 1e-9, "scores sum to 0");
        }
        assert!(t.rows.windows(2).all(|w| w[0].mean >= w[1].mean));
        assert_eq!(t.plays_to_out[0], 2.0); // 3c and the pair of 2s
    }

    fn manual(per_deal: Vec<Vec<f64>>) -> MoveTable {
        let s = choice();
        let legal = s.legal_actions_vec();
        let rows = per_deal
            .iter()
            .zip(&legal)
            .map(|(v, &a)| MoveRow {
                action: a,
                mean: v.iter().sum::<f64>() / v.len() as f64,
                ci95: 0.0,
                win: 0.0,
                seat_means: None,
            })
            .collect();
        MoveTable {
            actor: 0,
            rows,
            deals: per_deal[0].len() as u64,
            plays_to_out: [0.0; 4],
            forced: None,
            per_deal: Some(
                per_deal
                    .iter()
                    .map(|v| v.iter().map(|&x| [x, 0.0, 0.0, -x]).collect())
                    .collect(),
            ),
        }
    }

    #[test]
    fn grade_is_unclear_when_the_interval_crosses_a_line() {
        let t = Thresholds::from_sigma(20.0); // 0.5 / 2 / 5
                                              // Differences 0 and 4 alternate: loss 2 ± 1.3, across the 2-point line.
        let best = vec![4.0; 20];
        let played: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 4.0 } else { 0.0 })
            .collect();
        let table = manual(vec![best.clone(), played]);
        let g = grade_from_table(&table, table.rows[1].action, &t, 0);
        assert!((g.loss - 2.0).abs() < 1e-12);
        assert!(g.interval > 0.5 && g.interval < 1.5, "{g:?}");
        assert_eq!(g.grade, "unclear");
        assert_eq!(g.range, Some(("inaccuracy", "mistake")));
        // The same mean with no spread is a clear grade.
        let table = manual(vec![best, vec![2.0; 20]]);
        let g = grade_from_table(&table, table.rows[1].action, &t, 0);
        assert_eq!((g.grade, g.interval), ("mistake", 0.0));
    }

    #[test]
    fn grade_from_an_analysis_table_uses_both_intervals() {
        let s = choice();
        let legal = s.legal_actions_vec();
        let rows = vec![
            MoveRow {
                action: legal[0],
                mean: 3.0,
                ci95: 0.3,
                win: 0.5,
                seat_means: None,
            },
            MoveRow {
                action: legal[1],
                mean: 0.0,
                ci95: 0.4,
                win: 0.1,
                seat_means: None,
            },
        ];
        let table = MoveTable {
            actor: 0,
            rows,
            deals: 100,
            plays_to_out: [0.0; 4],
            forced: None,
            per_deal: None,
        };
        let g = grade_from_table(&table, legal[1], &Thresholds::from_sigma(20.0), 0);
        assert!((g.interval - 0.5).abs() < 1e-12);
        assert_eq!(g.grade, "mistake");
    }

    #[test]
    fn grade_paired_reports_a_forced_endgame() {
        let s = choice();
        let c = PairedConfig {
            deals: 20,
            seed: 1,
            endgame: Some(InfosetConfig::default()),
            ..PairedConfig::default()
        };
        let t = evaluate_paired(&s, &c);
        let f = t
            .forced
            .expect("leading the 2s pair then the 3c wins in every deal");
        assert_eq!(f.winner, 0);
    }
}
