//! Solver tests on Big Two positions.

use poker_big2::bots::Greedy;
use poker_big2::card::parse_mask;
use poker_big2::{Action, RuleSet, State};
use poker_games::{Game, Transposable};
use poker_search::endgame::{solve_naive, Endgame};
use poker_search::{ismcts, pimc, IsmctsConfig, Objective, PimcConfig};
use rand::seq::SliceRandom;
use rand::{rngs::SmallRng, Rng, SeedableRng};

fn hands(h: [&str; 4]) -> [u64; 4] {
    h.map(|x| parse_mask(x).unwrap())
}

/// Random small position: each player holds 1–`max_cards` cards; nobody has played yet
/// in the current trick.
fn random_small_position(rng: &mut SmallRng, max_cards: usize) -> State {
    let mut deck: Vec<u8> = (0..52).collect();
    deck.shuffle(rng);
    let mut it = deck.into_iter();
    let hs: [u64; 4] = std::array::from_fn(|_| {
        let n = rng.gen_range(1..=max_cards);
        (0..n).fold(0, |m, _| m | 1u64 << it.next().unwrap())
    });
    State::from_position(RuleSet::default(), hs, rng.gen_range(0..4), None).unwrap()
}

#[test]
fn endgame_with_table_matches_naive_maxn() {
    let mut rng = SmallRng::seed_from_u64(11);
    let mut solver = Endgame::<State>::new(Objective::Score);
    for _ in 0..200 {
        let s = random_small_position(&mut rng, 3);
        let fast = solver.solve(&s, u64::MAX).unwrap();
        let slow = solve_naive(&s, Objective::Score);
        assert_eq!(fast, slow, "position {s:?}");
    }
    assert!(solver.table_len() > 0);
}

#[test]
fn endgame_respects_the_node_budget() {
    let mut rng = SmallRng::seed_from_u64(5);
    let s = State::new_deal(RuleSet::default(), &mut rng);
    let mut solver = Endgame::<State>::new(Objective::Score);
    assert!(
        solver.solve(&s, 1_000).is_none(),
        "a full deal cannot be solved in 1,000 nodes"
    );
}

#[test]
fn endgame_finds_a_forced_win() {
    // P0 leads with 2s + 3d. Leading 2s first (unbeatable) then 3d wins; leading 3d
    // first lets P1 (holding one card, 4d) go out.
    let s = State::from_position(
        RuleSet::default(),
        hands(["3d 2s", "4d", "5d 5c", "6d 6c"]),
        0,
        None,
    )
    .unwrap();
    let mut solver = Endgame::<State>::new(Objective::Win);
    let o = solver.solve(&s, u64::MAX).unwrap();
    assert_eq!(o.winner, Some(0));
    let per_action = solver.solve_actions(&s, u64::MAX).unwrap();
    let after_3d = per_action
        .iter()
        .find(|(a, _)| a.cards() == parse_mask("3d").unwrap())
        .unwrap();
    assert_eq!(after_3d.1.winner, Some(1));
}

/// A position where P0 can go out right now by playing the pair.
fn winning_now() -> State {
    State::from_position(
        RuleSet::default(),
        hands(["9d 9c", "Ah", "Kd Kc", "Qd Qs Qh"]),
        0,
        None,
    )
    .unwrap()
}

#[test]
fn ismcts_takes_an_immediate_win() {
    let cfg = IsmctsConfig {
        iterations: 2_000,
        ..Default::default()
    };
    let a = ismcts(&winning_now(), 0, &cfg, &Greedy { epsilon: 0.1 });
    assert_eq!(a.best().cards(), parse_mask("9d 9c").unwrap());
    assert_eq!(a.moves[0].win_rate, 1.0);
    assert_eq!(a.simulations, 2_000);
    assert_eq!(a.moves.len(), winning_now().legal_actions_vec().len());
}

#[test]
fn pimc_takes_an_immediate_win() {
    let cfg = PimcConfig {
        determinizations: 50,
        ..Default::default()
    };
    let a = pimc(&winning_now(), 0, &cfg, &Greedy { epsilon: 0.1 });
    assert_eq!(a.best().cards(), parse_mask("9d 9c").unwrap());
    assert_eq!(a.moves[0].win_rate, 1.0);
}

#[test]
fn ismcts_is_reproducible_with_a_fixed_seed() {
    let mut rng = SmallRng::seed_from_u64(9);
    let s = State::new_deal(RuleSet::default(), &mut rng);
    let p = s.current_player().unwrap();
    let cfg = IsmctsConfig {
        iterations: 3_000,
        threads: 2,
        seed: 42,
        ..Default::default()
    };
    let a = ismcts(&s, p, &cfg, &Greedy { epsilon: 0.1 });
    let b = ismcts(&s, p, &cfg, &Greedy { epsilon: 0.1 });
    let summary = |x: &poker_search::Analysis<Action>| {
        x.moves
            .iter()
            .map(|m| (m.action, m.visits))
            .collect::<Vec<_>>()
    };
    assert_eq!(summary(&a), summary(&b));
}

#[test]
fn ismcts_never_peeks_at_hidden_cards() {
    // Two states that look identical to P0 (same P0 hand, same public info, different
    // hidden hands) must get identical analyses with the same seed.
    let mut rng = SmallRng::seed_from_u64(1);
    let s1 = State::new_deal(RuleSet::default(), &mut rng);
    let p = s1.current_player().unwrap();
    let s2 = s1.determinize(p, &mut rng);
    assert_ne!(s1.hands(), s2.hands());
    assert_ne!(s1.position_key(), s2.position_key());
    let cfg = IsmctsConfig {
        iterations: 2_000,
        seed: 3,
        ..Default::default()
    };
    let a = ismcts(&s1, p, &cfg, &Greedy { epsilon: 0.1 });
    let b = ismcts(&s2, p, &cfg, &Greedy { epsilon: 0.1 });
    let summary = |x: &poker_search::Analysis<Action>| {
        x.moves
            .iter()
            .map(|m| (m.action, m.visits, m.mean_score.to_bits()))
            .collect::<Vec<_>>()
    };
    assert_eq!(summary(&a), summary(&b));
}
