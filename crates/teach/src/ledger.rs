//! The luck ledger (teaching-and-deployment §2.4): one seat's result in one deal
//! split into what the cards gave, what the opponents' choices gave, what the
//! seat's own choices cost, and the rest.
//!
//! * `deal_strength` — the seat's expected score at the start of the deal, from
//!   its own view (its hand known, the other hands sampled), every seat played
//!   by the rollout policy. The average over all deals is 0 (scores sum to 0 and
//!   the seats are symmetric), so this is "how good the cards were".
//! * `opponents_errors` — at every opponent decision, the seat's mean score after
//!   the move the opponent played minus after the opponent's best move (the
//!   opponent's own move table, from the opponent's view). Positive: the
//!   opponents' choices helped the seat.
//! * `own_errors` — at every decision of the seat, the points its move lost
//!   against its best move (the sum of its review losses). It is subtracted.
//! * `residual` — the rest: how the hidden cards actually lay, how play went
//!   beyond what the move tables expected, and sampling noise.
//!
//! `score = deal_strength + opponents_errors − own_errors + residual` holds
//! exactly, because `residual` is defined as what the other three terms leave.
//! The size of `residual` against the other terms is the ledger's quality
//! measure (`build-phase4.md` §1 reports it).

use crate::grade::{evaluate_paired_with, PairedConfig};
use crate::table::MoveTable;
use big2_big2::{Action, State};
use big2_games::{play_out, Game, Policy};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// One decision of a deal: its move table (`None` for a forced move), who
/// acted, and the action played.
pub type DecisionTable = (Option<MoveTable>, usize, Action);

/// One decision of the deal as the ledger saw it.
#[derive(Clone, Debug, PartialEq)]
pub struct LedgerStep {
    /// Index of the decision in the deal (0 = the opening play).
    pub turn: usize,
    /// Who acted.
    pub actor: usize,
    pub played: Action,
    /// The actor's best action by the actor's move table.
    pub best: Action,
    /// The step's contribution to the seat's ledger: `−loss` at the seat's own
    /// decisions, `played − best` for the seat at an opponent's decision; 0 for
    /// a forced move.
    pub delta: f64,
}

/// A seat's result split into four terms that sum to `score`.
#[derive(Clone, Debug, PartialEq)]
pub struct Ledger {
    pub seat: usize,
    /// The seat's final score.
    pub score: f64,
    pub deal_strength: f64,
    pub opponents_errors: f64,
    pub own_errors: f64,
    pub residual: f64,
    pub steps: Vec<LedgerStep>,
}

impl Ledger {
    /// Builds the ledger from the move table of every decision of the deal (in
    /// order, with the action played), the seat's final score and its deal
    /// strength. A decision with one legal action may carry no table (`None`).
    pub fn from_tables(
        seat: usize,
        score: f64,
        deal_strength: f64,
        decisions: &[DecisionTable],
    ) -> Ledger {
        let mut steps = Vec::with_capacity(decisions.len());
        let (mut opp, mut own) = (0.0, 0.0);
        for (turn, (table, actor, played)) in decisions.iter().enumerate() {
            let (best, delta) = match table {
                None => (*played, 0.0),
                Some(t) => {
                    let b = t.best();
                    let p = t.row(*played).expect("the played action is in its table");
                    let delta = if *actor == seat {
                        -(b.mean - p.mean)
                    } else {
                        let (bs, ps) = (b.seat_means, p.seat_means);
                        match (bs, ps) {
                            (Some(bs), Some(ps)) => ps[seat] - bs[seat],
                            _ => 0.0,
                        }
                    };
                    (b.action, delta)
                }
            };
            if *actor == seat {
                own -= delta;
            } else {
                opp += delta;
            }
            steps.push(LedgerStep {
                turn,
                actor: *actor,
                played: *played,
                best,
                delta,
            });
        }
        Ledger {
            seat,
            score,
            deal_strength,
            opponents_errors: opp,
            own_errors: own,
            residual: score - (deal_strength + opp - own),
            steps,
        }
    }

    /// `deal_strength + opponents_errors − own_errors + residual`, which equals
    /// `score` up to floating-point rounding.
    pub fn total(&self) -> f64 {
        self.deal_strength + self.opponents_errors - self.own_errors + self.residual
    }
}

/// The seat's expected score at `start` from its own view: `deals` deals of
/// the other hands, each played out from `start` by `rollout`.
pub fn deal_strength<P: Policy<State>>(
    start: &State,
    seat: usize,
    deals: u64,
    seed: u64,
    rollout: &P,
) -> f64 {
    let n = deals.max(1);
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xDEA1_5EED);
    let mut sum = 0.0;
    for _ in 0..n {
        let mut d = start.determinize(seat, &mut rng);
        sum += play_out(&mut d, rollout, &mut rng).scores[seat];
    }
    sum / n as f64
}

/// The move table of every decision of a finished deal: `start` is the deal
/// before the first play, `actions` every action in order. Forced decisions
/// (one legal action) get no table. Tables use `cfg` with the seed mixed with
/// the decision index.
///
/// # Errors
/// An action that is illegal where it is played.
pub fn deal_tables<P: Policy<State>>(
    start: &State,
    actions: &[Action],
    cfg: &PairedConfig,
    rollout: &P,
) -> Result<(Vec<DecisionTable>, State), String> {
    let mut state = start.clone();
    let mut out = Vec::with_capacity(actions.len());
    for (t, &a) in actions.iter().enumerate() {
        let actor = state
            .current_player()
            .ok_or_else(|| format!("action {t} comes after the end of the deal"))?;
        let legal = state.legal_actions_vec();
        if !legal.contains(&a) {
            return Err(format!("action {t} ({a}) is illegal"));
        }
        let table = (legal.len() > 1).then(|| {
            let c = PairedConfig {
                seed: cfg.seed.wrapping_add(t as u64),
                ..cfg.clone()
            };
            evaluate_paired_with(&state, &c, rollout)
        });
        out.push((table, actor, a));
        state.apply(a);
    }
    Ok((out, state))
}

/// The luck ledger of `seat` for a finished deal (`start` before the first play,
/// then `actions`), with move tables from [`deal_tables`] and the deal strength
/// on `cfg.deals` deals.
///
/// # Errors
/// An illegal action, or `actions` that do not finish the deal.
pub fn luck_ledger<P: Policy<State>>(
    start: &State,
    actions: &[Action],
    seat: usize,
    cfg: &PairedConfig,
    rollout: &P,
) -> Result<Ledger, String> {
    let (tables, end) = deal_tables(start, actions, cfg, rollout)?;
    let outcome = end.outcome().ok_or("the actions do not finish the deal")?;
    let strength = deal_strength(start, seat, cfg.deals, cfg.seed, rollout);
    Ok(Ledger::from_tables(
        seat,
        outcome.scores[seat],
        strength,
        &tables,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use big2_big2::bots::Greedy;
    use big2_big2::RuleSet;
    use rand::Rng;

    /// A random deal played to the end by a noisy greedy policy.
    fn random_deal(seed: u64) -> (State, Vec<Action>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let start = State::new_deal(RuleSet::default(), &mut rng);
        let mut s = start.clone();
        let mut actions = Vec::new();
        let policy = Greedy { epsilon: 0.3 };
        while !s.is_terminal() {
            let legal = s.legal_actions_vec();
            let a = policy.choose(&s, &legal, &mut rng);
            actions.push(a);
            s.apply(a);
        }
        (start, actions)
    }

    #[test]
    fn ledger_sums_exactly_on_200_random_deals() {
        let cfg = PairedConfig {
            deals: 1,
            ..PairedConfig::default()
        };
        let rollout = Greedy::default();
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        for d in 0..200 {
            let (start, actions) = random_deal(d);
            let seat = rng.gen_range(0..4);
            let l = luck_ledger(
                &start,
                &actions,
                seat,
                &PairedConfig {
                    seed: d,
                    ..cfg.clone()
                },
                &rollout,
            )
            .unwrap();
            assert!((l.total() - l.score).abs() <= 1e-9, "deal {d}: {l:?}");
            assert!(l.own_errors >= 0.0, "own errors are losses: {l:?}");
            assert_eq!(l.steps.len(), actions.len());
        }
    }

    #[test]
    fn ledger_terms_come_from_the_steps() {
        let (start, actions) = random_deal(5);
        let cfg = PairedConfig {
            deals: 4,
            seed: 2,
            ..PairedConfig::default()
        };
        let l = luck_ledger(&start, &actions, 1, &cfg, &Greedy::default()).unwrap();
        let own: f64 = l
            .steps
            .iter()
            .filter(|s| s.actor == 1)
            .map(|s| -s.delta)
            .sum();
        let opp: f64 = l
            .steps
            .iter()
            .filter(|s| s.actor != 1)
            .map(|s| s.delta)
            .sum();
        assert!((own - l.own_errors).abs() < 1e-12);
        assert!((opp - l.opponents_errors).abs() < 1e-12);
        // A step where the actor played its best move adds nothing.
        assert!(l
            .steps
            .iter()
            .filter(|s| s.played == s.best)
            .all(|s| s.delta == 0.0));
    }

    #[test]
    fn ledger_rejects_unfinished_or_illegal_deals() {
        let (start, actions) = random_deal(7);
        let cfg = PairedConfig {
            deals: 1,
            ..PairedConfig::default()
        };
        let r = luck_ledger(&start, &actions[..3], 0, &cfg, &Greedy::default());
        assert!(r.is_err());
        let mut bad = actions.clone();
        bad[0] = Action::Pass; // the opening leader cannot pass
        assert!(luck_ledger(&start, &bad, 0, &cfg, &Greedy::default()).is_err());
    }
}
