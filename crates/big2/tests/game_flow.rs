//! Turn flow, scoring and determinization tests. Each test cites the section of
//! `docs/rules/big2.md` it checks.

use big2_big2::card::{parse_mask, FULL_DECK};
use big2_big2::{Action, IllegalAction, PassRule, RuleSet, Scoring, State};
use big2_games::{play_out, Game, UniformRandom};
use proptest::prelude::*;
use rand::{rngs::SmallRng, SeedableRng};

/// Deal fixed hands written as text, one per seat.
fn deal(rules: RuleSet, hands: [&str; 4]) -> State {
    State::from_hands(rules, hands.map(|h| parse_mask(h).unwrap())).unwrap()
}

/// A deal where seat 0 holds 3♦ and every seat holds one full rank group per suit
/// pattern, chosen so plays are easy to follow in tests.
fn simple_deal(rules: RuleSet) -> State {
    deal(
        rules,
        [
            "3d 4d 5d 6d 7d 8d 9d Td Jd Qd Kd Ad 2d", // all diamonds
            "3c 4c 5c 6c 7c 8c 9c Tc Jc Qc Kc Ac 2c",
            "3h 4h 5h 6h 7h 8h 9h Th Jh Qh Kh Ah 2h",
            "3s 4s 5s 6s 7s 8s 9s Ts Js Qs Ks As 2s",
        ],
    )
}

fn play(s: &mut State, cards: &str) {
    let a = s.action_for_cards(parse_mask(cards).unwrap()).unwrap();
    s.apply(a);
}

fn pass(s: &mut State) {
    s.apply(Action::Pass);
}

#[test]
fn holder_of_three_of_diamonds_leads_and_must_play_it() {
    // rules §2
    let s = simple_deal(RuleSet::default());
    assert_eq!(s.current_player(), Some(0));
    assert_eq!(
        s.action_for_cards(parse_mask("4d").unwrap()),
        Err(IllegalAction::MustIncludeThreeOfDiamonds)
    );
    assert!(s.legal_actions_vec().iter().all(|a| a.cards() & 1 == 1));
    assert!(
        !s.legal_actions_vec().contains(&Action::Pass),
        "rules §5.1: leader cannot pass"
    );
}

#[test]
fn three_passes_end_the_trick_and_the_last_player_leads() {
    // rules §5.3
    let mut s = simple_deal(RuleSet::default());
    play(&mut s, "3d");
    play(&mut s, "3c");
    pass(&mut s);
    pass(&mut s);
    assert_eq!(s.current_player(), Some(0));
    pass(&mut s);
    assert_eq!(
        s.current_player(),
        Some(1),
        "player 1 made the last play and leads"
    );
    assert!(s.current_play().is_none());
    assert!(!s.legal_actions_vec().contains(&Action::Pass));
}

#[test]
fn a_passed_player_may_reenter_by_default() {
    // rules §5.4
    let mut s = simple_deal(RuleSet::default());
    play(&mut s, "3d");
    pass(&mut s); // P1 passes
    play(&mut s, "3h"); // P2
    play(&mut s, "3s"); // P3
    play(&mut s, "4d"); // P0
    assert_eq!(s.current_player(), Some(1));
    play(&mut s, "4c"); // P1 re-enters the same trick
    assert_eq!(s.current_play().unwrap().0, 1);
}

#[test]
fn locked_out_players_are_skipped() {
    // rules §5, variant locked_out
    let rules = RuleSet {
        pass_rule: PassRule::LockedOut,
        ..Default::default()
    };
    let mut s = simple_deal(rules);
    play(&mut s, "3d"); // P0
    pass(&mut s); // P1 locked
    play(&mut s, "3h"); // P2
    play(&mut s, "3s"); // P3
    play(&mut s, "4d"); // P0
    assert_eq!(s.current_player(), Some(2), "P1 is skipped");
    pass(&mut s); // P2 locked
    pass(&mut s); // P3 locked → back to P0, who holds the trick
    assert_eq!(s.current_player(), Some(0));
    assert!(s.current_play().is_none(), "trick over, P0 leads");
    assert_eq!(s.locked_out(), [false; 4]);
}

#[test]
fn last_card_duty_forces_highest_single() {
    // rules §5, variant last_card_duty. Seats hold one suit each (see simple_deal).
    let rules = RuleSet {
        last_card_duty: true,
        ..Default::default()
    };
    let mut s = simple_deal(rules);
    let all_pass = |s: &mut State| (0..3).for_each(|_| pass(s));
    play(&mut s, "3d 4d 5d 6d 7d"); // P0
    play(&mut s, "3c 4c 5c 6c 7c"); // P1: same straight, clubs beat diamonds
    all_pass(&mut s); // P1 leads
    play(&mut s, "8c 9c Tc Jc Qc");
    all_pass(&mut s);
    play(&mut s, "Kc");
    all_pass(&mut s);
    play(&mut s, "Ac"); // P1 now holds only 2c
    pass(&mut s); // P2
    pass(&mut s); // P3
    play(&mut s, "2d"); // P0
    all_pass(&mut s); // P0 leads; next player P1 has one card
    assert_eq!(s.current_player(), Some(0));
    assert_eq!(s.hand_sizes()[1], 1);
    assert_eq!(
        s.action_for_cards(parse_mask("8d").unwrap()),
        Err(IllegalAction::LastCardDuty)
    );
    assert!(s.action_for_cards(parse_mask("Ad").unwrap()).is_ok());
    assert!(
        s.action_for_cards(parse_mask("8d 9d Td Jd Qd").unwrap())
            .is_ok(),
        "five-card plays unaffected"
    );
    let singles: Vec<_> = s
        .legal_actions_vec()
        .into_iter()
        .filter(|a| a.cards().count_ones() == 1)
        .collect();
    assert_eq!(singles.len(), 1);
    assert_eq!(singles[0].cards(), parse_mask("Ad").unwrap());

    // Without the variant, any single may be led.
    let mut free = simple_deal(RuleSet {
        last_card_duty: false,
        ..RuleSet::default()
    });
    for t in s.history() {
        free.apply(free.action_for_cards(t.action.cards()).unwrap());
    }
    assert!(free.action_for_cards(parse_mask("8d").unwrap()).is_ok());
}

#[test]
fn illegal_actions_are_explained() {
    let mut s = simple_deal(RuleSet::default());
    assert!(matches!(
        s.action_for_cards(parse_mask("3c").unwrap()),
        Err(IllegalAction::NotInHand(_))
    ));
    assert!(matches!(
        s.action_for_cards(parse_mask("3d 4d").unwrap()),
        Err(IllegalAction::NotACombination(_))
    ));
    assert_eq!(
        s.check(&Action::Pass),
        Err(IllegalAction::CannotPassWhenLeading)
    );
    play(&mut s, "3d");
    assert!(matches!(
        s.action_for_cards(parse_mask("3s 4s").unwrap()),
        Err(IllegalAction::NotInHand(_))
    ));
    assert!(matches!(
        s.action_for_cards(parse_mask("4c 5c").unwrap()),
        Err(IllegalAction::NotACombination(_))
    ));
    assert!(matches!(
        s.action_for_cards(parse_mask("4c 4d").unwrap()),
        Err(IllegalAction::NotInHand(_))
    ));
    s.apply(Action::Pass);
    s.apply(Action::Pass);
    s.apply(Action::Pass);
    assert_eq!(
        s.check(&Action::Pass),
        Err(IllegalAction::CannotPassWhenLeading)
    );
    s.apply(
        s.action_for_cards(parse_mask("4d 5d 6d 7d 8d").unwrap())
            .unwrap(),
    );
    assert!(matches!(
        s.action_for_cards(parse_mask("3c").unwrap()),
        Err(IllegalAction::DoesNotBeat { .. }) | Err(IllegalAction::NotACombination(_))
    ));
    assert!(matches!(
        s.action_for_cards(parse_mask("3c 4h 5c 6c 7c").unwrap()),
        Err(IllegalAction::NotInHand(_))
    ));
    assert!(matches!(
        s.action_for_cards(parse_mask("3c 4c 5c 6c 7c").unwrap()),
        Err(IllegalAction::DoesNotBeat { .. })
    ));
}

#[test]
fn scoring_matches_the_worked_example() {
    // rules §7.2: winner 0 cards, B 3, C 8, D 13 → +71, +59, +7, −137 (tiered)
    let pen = [0u32, 3, 8, 13].map(|n| Scoring::Tiered.penalty(n));
    let total: u32 = pen.iter().sum();
    let scores: Vec<i64> = pen.iter().map(|&p| total as i64 - 4 * p as i64).collect();
    assert_eq!(scores, [71, 59, 7, -137]);
}

#[test]
fn a_full_game_ends_when_someone_goes_out() {
    // rules §6, using the engine's own outcome
    let mut rng = SmallRng::seed_from_u64(3);
    for scoring in [Scoring::Tiered, Scoring::Original, Scoring::WinLoss] {
        let mut s = State::new_deal(
            RuleSet {
                scoring,
                ..Default::default()
            },
            &mut rng,
        );
        let out = play_out(&mut s, &UniformRandom, &mut rng);
        let w = out.winner.unwrap();
        assert_eq!(s.hand(w), 0);
        assert!(s.legal_actions_vec().is_empty());
        assert_eq!(out.scores.iter().sum::<f64>(), 0.0);
        assert!(out.scores[w] >= out.scores.iter().cloned().fold(f64::MIN, f64::max));
    }
}

fn arb_rules() -> impl Strategy<Value = RuleSet> {
    (
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        0..3usize,
        any::<bool>(),
    )
        .prop_map(|(st, fl, pr, sc, lcd)| RuleSet {
            straights: if st {
                big2_big2::Straights::WithLowTwos
            } else {
                big2_big2::Straights::NoTwos
            },
            flush_order: if fl {
                big2_big2::FlushOrder::RankFirst
            } else {
                big2_big2::FlushOrder::SuitFirst
            },
            pass_rule: if pr {
                PassRule::MayReenter
            } else {
                PassRule::LockedOut
            },
            scoring: Scoring::ALL[sc],
            last_card_duty: lcd,
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Invariants over random games under random rule variants: cards are never
    /// created or lost, every generated action passes the independent legality
    /// check, the game terminates, and scores sum to zero.
    #[test]
    fn random_games_keep_invariants(seed in any::<u64>(), rules in arb_rules()) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut s = State::new_deal(rules, &mut rng);
        let mut legal = Vec::new();
        let mut steps = 0;
        while !s.is_terminal() {
            s.legal_actions(&mut legal);
            prop_assert!(!legal.is_empty());
            for a in &legal {
                prop_assert_eq!(s.check(a), Ok(()));
            }
            let a = legal[rand::Rng::gen_range(&mut rng, 0..legal.len())];
            s.apply(a);
            let hands = s.hands();
            let union = hands.iter().fold(s.played(), |m, h| m | h);
            prop_assert_eq!(union, FULL_DECK);
            prop_assert_eq!(hands.iter().map(|h| h.count_ones()).sum::<u32>() + s.played().count_ones(), 52);
            steps += 1;
            prop_assert!(steps < 2000, "game did not terminate");
        }
        let out = s.outcome().unwrap();
        prop_assert_eq!(out.scores.iter().sum::<f64>(), 0.0);
        prop_assert!(out.scores.iter().all(|x| x.abs() <= s.max_abs_score()));
    }

    /// Determinization keeps everything the observer knows and re-deals the rest.
    #[test]
    fn determinize_preserves_the_observers_view(seed in any::<u64>(), steps in 0usize..40, observer in 0usize..4) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        let mut legal = Vec::new();
        for _ in 0..steps {
            if s.is_terminal() { break; }
            s.legal_actions(&mut legal);
            s.apply(legal[rand::Rng::gen_range(&mut rng, 0..legal.len())]);
        }
        let d = s.determinize(observer, &mut rng);
        prop_assert_eq!(d.hand(observer), s.hand(observer));
        prop_assert_eq!(d.hand_sizes(), s.hand_sizes());
        prop_assert_eq!(d.played(), s.played());
        prop_assert_eq!(d.current_player(), s.current_player());
        prop_assert_eq!(d.hands().iter().fold(d.played(), |m, h| m | h), FULL_DECK);
        if s.is_opening() {
            let cur = s.current_player().unwrap();
            prop_assert!(d.hand(cur) & 1 == 1, "the player to open holds 3d");
        }
        // Legal actions only depend on the current player's hand and public state,
        // so they agree whenever the observer is to move.
        if s.current_player() == Some(observer) {
            prop_assert_eq!(d.legal_actions_vec(), s.legal_actions_vec());
        }
    }
}

#[test]
fn from_position_sets_up_a_mid_game_state() {
    let hands = ["3d 4d", "5c", "Kh Ks", "2s"].map(|h| parse_mask(h).unwrap());
    let s = State::from_position(
        RuleSet::default(),
        hands,
        0,
        Some((3, parse_mask("Ac").unwrap())),
    )
    .unwrap();
    assert_eq!(s.current_player(), Some(0));
    assert_eq!(s.played().count_ones(), 52 - 6);
    // 3d and 4d can't beat Ac, so P0 must pass
    assert_eq!(s.legal_actions_vec(), vec![Action::Pass]);
    assert!(State::from_position(
        RuleSet::default(),
        hands,
        0,
        Some((0, parse_mask("Ac").unwrap()))
    )
    .is_err());
    assert!(State::from_position(
        RuleSet::default(),
        hands,
        0,
        Some((3, parse_mask("2s").unwrap()))
    )
    .is_err());
}
