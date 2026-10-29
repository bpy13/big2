//! Exact endgame from the observer's point of view: every possible deal, solved.
//!
//! [`crate::endgame`] solves one *fully known* position. Near the end of a deal the
//! observer still does not know who holds which of the hidden cards, but there are
//! few enough of them to list every possibility: with 12 hidden cards split 4–4–4
//! among three opponents there are 12! / (4!·4!·4!) = 34,650 deals. This module
//! enumerates them ([`big2_games::Game::for_each_deal`]), solves each one with
//! max^n search ([`Endgame::solve_actions`]), and averages every legal action's
//! result over the deals, weighted by the sampler ([`Sampler::weight`]: uniform,
//! or 0 for deals the rules exclude with [`crate::RulesMask`]).
//!
//! That is the *exact* value of each action under two assumptions the Monte Carlo
//! searches make too: every possible deal is equally likely, and from there on
//! everyone plays as if all cards were open (max^n). It replaces sampling noise by
//! certainty; it does not model that the opponents cannot see the cards either.
//!
//! **Forced results.** When every deal gives the same winner after the best action,
//! the result is reported as [`Forced`]: e.g. "you go out in 5 plays whatever the
//! others hold" (`quality: {"exact": true}` for the app later).
//!
//! **Budget.** All deals share one node budget (`max_nodes`, positions expanded);
//! if it runs out the result says `exact: false` and the callers
//! ([`crate::ismcts_with`], [`crate::puct_with`]) fall back to their normal search.
//! Runs on the calling thread (no rayon), so it works in WebAssembly.

use crate::analysis::{Analysis, MoveStats, Objective};
use crate::belief::Sampler;
use crate::endgame::Endgame;
use big2_games::{Player, Transposable};
use web_time::Instant;

/// Settings of the information-set endgame.
#[derive(Clone, Debug)]
pub struct InfosetConfig {
    /// Solve only when the observer cannot see at most this many cards.
    pub threshold: usize,
    /// Solve only when all hands together hold at most this many cards
    /// ([`big2_games::Game::endgame_size`]): the cost of each deal's max^n search
    /// grows with every card in play, the observer's own included.
    pub max_cards: usize,
    /// Positions the max^n search may expand, over all deals together.
    pub max_nodes: u64,
}

impl Default for InfosetConfig {
    fn default() -> Self {
        InfosetConfig {
            threshold: 12,
            max_cards: 10,
            max_nodes: 1_000_000,
        }
    }
}

/// A result that holds in every possible deal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Forced {
    /// Who goes out first after the best action, in every deal.
    pub winner: Player,
    /// Moves (plays and passes, the best action included) until the deal ends on
    /// the max^n line, in the longest of the deals.
    pub plays: u32,
}

/// What [`infoset_endgame`] found.
#[derive(Clone, Debug)]
pub struct InfosetResult<A> {
    /// Every legal action with its average over the deals, best first. `visits`
    /// counts the deals with a non-zero weight (the same for every action).
    pub analysis: Analysis<A>,
    /// `true` when every deal was solved; `false` when the node budget ran out
    /// (then `analysis` covers only the deals solved before that).
    pub exact: bool,
    /// Set when `exact` and the best action's winner is the same in every deal.
    pub forced: Option<Forced>,
    /// Deals listed (including those of weight 0).
    pub deals: u64,
    /// Deals the sampler weighted 0 (impossible under the rules).
    pub excluded: u64,
    /// Positions expanded.
    pub nodes: u64,
}

/// Weighted sums for one action.
#[derive(Clone, Copy, Default)]
struct WSums {
    n: u64,
    w: f64,
    score: f64,
    score_sq: f64,
    wins: f64,
}

/// Solves `state` for `observer` (who must be to move) over every deal of the
/// hidden cards, or returns `None` when it does not apply: the game is over, it is
/// not the observer's turn, the game cannot list deals, more than
/// `cfg.threshold` cards are hidden, or more than `cfg.max_cards` are in play.
pub fn infoset_endgame<G, S>(
    state: &G,
    observer: Player,
    cfg: &InfosetConfig,
    objective: Objective,
    sampler: &S,
) -> Option<InfosetResult<G::Action>>
where
    G: Transposable,
    S: Sampler<G>,
{
    if state.current_player() != Some(observer) {
        return None;
    }
    if state.hidden_count(observer)? > cfg.threshold || state.endgame_size() > cfg.max_cards {
        return None;
    }
    let start = Instant::now();
    let legal = state.legal_actions_vec();
    let mut sums = vec![WSums::default(); legal.len()];
    // Winner after each action, while every deal so far agrees (`None` once two differ).
    let mut winners: Vec<Option<Option<Player>>> = vec![None; legal.len()];
    let mut agree = vec![true; legal.len()];
    let mut endgame = Endgame::<G>::new(objective);
    let mut budget = cfg.max_nodes;
    let (mut deals, mut excluded, mut exact) = (0u64, 0u64, true);

    let listed = state.for_each_deal(observer, &mut |d: &G| {
        deals += 1;
        if !exact {
            return; // the budget ran out: nothing more is solved
        }
        let w = sampler.weight(d);
        if w <= 0.0 {
            excluded += 1;
            return;
        }
        let Some(solved) = endgame.solve_actions_within(d, &mut budget) else {
            exact = false;
            return;
        };
        for (i, (a, o)) in solved.iter().enumerate() {
            debug_assert_eq!(
                *a, legal[i],
                "the observer's actions do not depend on the deal"
            );
            let s = &mut sums[i];
            let score = o.scores[observer];
            s.n += 1;
            s.w += w;
            s.score += w * score;
            s.score_sq += w * score * score;
            s.wins += w * f64::from(u8::from(o.winner == Some(observer)));
            match winners[i] {
                None => winners[i] = Some(o.winner),
                Some(prev) if prev != o.winner => agree[i] = false,
                Some(_) => {}
            }
        }
    });
    if !listed {
        return None;
    }

    let mean_utility = |s: &WSums| {
        let w = s.w.max(f64::MIN_POSITIVE);
        match objective {
            Objective::Score => s.score / w,
            Objective::Win => s.wins / w,
        }
    };
    let mut order: Vec<usize> = (0..legal.len()).collect();
    // Best first; equal values keep the legal-move order (a stable sort).
    order.sort_by(|&x, &y| mean_utility(&sums[y]).total_cmp(&mean_utility(&sums[x])));
    let moves = order
        .iter()
        .map(|&i| move_stats(legal[i], &sums[i]))
        .collect();

    let best = order[0];
    let mut forced = None;
    if exact && sums[best].n > 0 && agree[best] {
        if let Some(Some(winner)) = winners[best] {
            // The longest max^n line after the best action over the deals (the
            // positions are in the table already, so this costs little).
            let mut plays = 0u32;
            let mut ok = true;
            state.for_each_deal(observer, &mut |d: &G| {
                if !ok || sampler.weight(d) <= 0.0 {
                    return;
                }
                let mut child = d.clone();
                child.apply(legal[best]);
                match endgame.line_length(&child, &mut budget) {
                    Some(n) => plays = plays.max(n + 1),
                    None => ok = false,
                }
            });
            if ok {
                forced = Some(Forced { winner, plays });
            }
        }
    }

    Some(InfosetResult {
        analysis: Analysis {
            player: observer,
            moves,
            simulations: sums[best].n,
            elapsed: start.elapsed(),
        },
        exact,
        forced,
        deals,
        excluded,
        nodes: cfg.max_nodes - budget,
    })
}

fn move_stats<A: Copy>(action: A, s: &WSums) -> MoveStats<A> {
    let w = s.w.max(f64::MIN_POSITIVE);
    let mean = s.score / w;
    let var = (s.score_sq / w - mean * mean).max(0.0);
    let p = s.wins / w;
    let n = s.n.max(1) as f64;
    MoveStats {
        action,
        visits: s.n,
        mean_score: mean,
        // Spread over the deals, not sampling error: the value itself is exact.
        score_ci95: 1.96 * (var / n).sqrt(),
        win_rate: p,
        win_ci95: 1.96 * (p * (1.0 - p) / n).sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::UniformSampler;
    use crate::endgame::solve_naive;
    use big2_big2::card::parse_mask;
    use big2_big2::{Action, RuleSet, State};
    use big2_games::Game;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;

    fn position(hands: [&str; 4], current: usize) -> State {
        let h = hands.map(|s| {
            if s.is_empty() {
                0
            } else {
                parse_mask(s).unwrap()
            }
        });
        State::from_position(RuleSet::default(), h, current, None).unwrap()
    }

    /// The plan's test (Task 20): seat 0 leads with 2♠ 5♣ 5♦ against three single
    /// cards. Whatever the others hold, 2♠ (or the pair of 5s) wins every trick:
    /// lead, three passes, lead the rest and go out, 5 moves. A forced win.
    #[test]
    fn infoset_reports_a_hand_provable_forced_win() {
        let s = position(["2s 5c 5d", "3d", "9h", "Kc"], 0);
        let r = infoset_endgame(
            &s,
            0,
            &InfosetConfig::default(),
            Objective::Score,
            &UniformSampler,
        )
        .expect("3 hidden cards is under the threshold");
        assert!(r.exact);
        assert_eq!(r.deals, 6, "3 cards, one each: 3! deals");
        assert_eq!(r.excluded, 0);
        assert_eq!(r.analysis.simulations, 6);
        assert_eq!(
            r.forced,
            Some(Forced {
                winner: 0,
                plays: 5
            }),
            "{:?}",
            r.analysis.moves
        );
        let best = &r.analysis.moves[0];
        assert_eq!(best.win_rate, 1.0);
        // Seat 0 out, the others one card each (penalty 1): total 3 − 4 · 0 = +3.
        assert_eq!(best.mean_score, 3.0);
        // Leading a 5 as a single is not allowed (last-card duty: seat 1 holds one
        // card), so the legal moves are the 2♠, the pair, and nothing else.
        assert_eq!(r.analysis.moves.len(), 2, "{:?}", r.analysis.moves);
    }

    /// With 13 hidden cards (above the default threshold of 12) nothing is solved.
    #[test]
    fn infoset_does_not_solve_13_hidden_cards() {
        let s = position(["3d 4d", "5d 6d 7d 8d 9d", "Td Jd Qd Kd", "Ad 2d 3c 4c"], 0);
        assert_eq!(s.hidden_count(0), Some(13));
        assert!(infoset_endgame(
            &s,
            0,
            &InfosetConfig::default(),
            Objective::Score,
            &UniformSampler
        )
        .is_none());
        // 6 hidden cards but 19 in play (over `max_cards` 10): not solved either.
        let m = position(
            [
                "3d 4d 5d 6d 7d 8d 9d Td Jd Qd Kd Ad 2d",
                "3c 4c",
                "5c 6c",
                "7c 8c",
            ],
            0,
        );
        assert_eq!((m.hidden_count(0), m.endgame_size()), (Some(6), 19));
        assert!(infoset_endgame(
            &m,
            0,
            &InfosetConfig::default(),
            Objective::Score,
            &UniformSampler
        )
        .is_none());
        // Not the observer's turn: not applicable either.
        let t = position(["3d", "5d", "6d", "7d"], 1);
        assert!(infoset_endgame(
            &t,
            0,
            &InfosetConfig::default(),
            Objective::Score,
            &UniformSampler
        )
        .is_none());
    }

    /// Every deal is listed once, keeps the observer's hand and the hand sizes, and
    /// the averages equal plain max^n over the same deals.
    #[test]
    fn infoset_averages_max_n_over_every_deal() {
        let s = position(["3c 7h Jd", "4s 8d", "5h 9c", "6c Th"], 0);
        let hidden = s.hand(1) | s.hand(2) | s.hand(3);
        let mut seen = HashSet::new();
        let mut naive = vec![(0.0, 0.0); s.legal_actions_vec().len()];
        assert!(s.for_each_deal(0, &mut |d: &State| {
            assert_eq!(d.hand(0), s.hand(0));
            assert_eq!(d.hand_sizes(), s.hand_sizes());
            assert_eq!(d.hand(1) | d.hand(2) | d.hand(3), hidden);
            assert!(seen.insert(d.hands()));
            for (i, a) in d.legal_actions_vec().into_iter().enumerate() {
                let mut c = d.clone();
                c.apply(a);
                let o = solve_naive(&c, Objective::Score);
                naive[i].0 += o.scores[0];
                naive[i].1 += f64::from(u8::from(o.winner == Some(0)));
            }
        }));
        assert_eq!(seen.len(), 90, "6 cards split 2-2-2: 6!/(2!2!2!)");
        let r = infoset_endgame(
            &s,
            0,
            &InfosetConfig::default(),
            Objective::Score,
            &UniformSampler,
        )
        .unwrap();
        assert!(r.exact);
        let legal = s.legal_actions_vec();
        for m in &r.analysis.moves {
            let i = legal.iter().position(|a| *a == m.action).unwrap();
            assert!((m.mean_score - naive[i].0 / 90.0).abs() < 1e-9, "{m:?}");
            assert!((m.win_rate - naive[i].1 / 90.0).abs() < 1e-9, "{m:?}");
            assert_eq!(m.visits, 90);
        }
        // Best first.
        for w in r.analysis.moves.windows(2) {
            assert!(w[0].mean_score >= w[1].mean_score);
        }
    }

    /// A tiny node budget is not enough: the result says it is not exact.
    #[test]
    fn infoset_reports_inexact_when_the_budget_runs_out() {
        let mut rng = SmallRng::seed_from_u64(3);
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        while s.hidden_count(s.current_player().unwrap()).unwrap() > 9 {
            let legal = s.legal_actions_vec();
            s.apply(legal[rng.gen_range(0..legal.len())]);
            if s.current_player().is_none() {
                panic!("deal ended early");
            }
        }
        let me = s.current_player().unwrap();
        let cfg = InfosetConfig {
            threshold: 12,
            max_cards: 52,
            max_nodes: 10,
        };
        let r = infoset_endgame(&s, me, &cfg, Objective::Score, &UniformSampler).unwrap();
        assert!(!r.exact);
        assert!(r.forced.is_none());
        assert!(r.deals > 0);
    }

    /// `infoset_endgame` in the ISMCTS and PUCT settings answers at the root with the
    /// exact result (one "simulation" per deal); off, the searches run as before.
    #[test]
    fn infoset_hook_answers_inside_ismcts_and_puct() {
        use crate::{ismcts, puct, IsmctsConfig, PuctConfig};
        use big2_big2::bots::Greedy;
        use big2_big2::pvnet::{synthetic_graph, PvNet};
        use big2_nn::Precision;

        let s = position(["2s 5c 5d", "3d", "9h", "Kc"], 0);
        let on = Some(InfosetConfig::default());
        let greedy = Greedy { epsilon: 0.1 };
        let cfg = IsmctsConfig {
            iterations: 500,
            ..Default::default()
        };
        assert_eq!(ismcts(&s, 0, &cfg, &greedy).simulations, 500);
        let exact = ismcts(
            &s,
            0,
            &IsmctsConfig {
                infoset_endgame: on.clone(),
                ..cfg
            },
            &greedy,
        );
        assert_eq!(exact.simulations, 6);
        assert_eq!(exact.moves[0].win_rate, 1.0);

        let net =
            PvNet::from_bytes(&synthetic_graph(16, 8, 8, 16, 3).to_bytes(Precision::F32)).unwrap();
        let pc = PuctConfig {
            simulations: 100,
            ..Default::default()
        };
        assert_eq!(puct(&s, 0, &pc, &net).analysis.simulations, 100);
        let p = puct(
            &s,
            0,
            &PuctConfig {
                infoset_endgame: on,
                ..pc
            },
            &net,
        );
        assert_eq!(p.analysis.simulations, 6);
        assert!(p.first_answer.is_none());
    }

    /// With the rules mask, deals the history excludes get no weight.
    #[test]
    fn infoset_with_the_rules_mask_skips_impossible_deals() {
        use crate::belief::RulesMask;
        use big2_games::Rewind;
        // Random seeded games until one ends up with ≤ 9 cards in all hands (small
        // enough to solve in a debug build) and some deal excluded by a last-card
        // duty single.
        for seed in 0..2_000u64 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            while s.current_player().is_some() {
                if s.endgame_size() <= 9 {
                    break;
                }
                let legal = s.legal_actions_vec();
                let singles: Vec<Action> = legal
                    .iter()
                    .copied()
                    .filter(|a| a.cards().count_ones() == 1)
                    .collect();
                let pool = if singles.is_empty() { &legal } else { &singles };
                s.apply(pool[rng.gen_range(0..pool.len())]);
            }
            let Some(me) = s.current_player() else {
                continue;
            };
            let mut impossible = 0;
            s.for_each_deal(me, &mut |d: &State| {
                impossible += u64::from(!d.history_consistent())
            });
            if impossible == 0 {
                continue;
            }
            let cfg = InfosetConfig {
                threshold: 12,
                max_cards: 52,
                max_nodes: 2_000_000,
            };
            let r = infoset_endgame(&s, me, &cfg, Objective::Score, &RulesMask::default()).unwrap();
            if !r.exact {
                continue;
            }
            assert_eq!(r.excluded, impossible);
            assert_eq!(r.analysis.simulations, r.deals - impossible);
            return;
        }
        panic!("no position with an excluded deal found");
    }
}
