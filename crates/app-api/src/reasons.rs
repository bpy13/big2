//! Reasons for a move, derived from the position and the solver's numbers.
//!
//! Every key here is a fact the code checks; nothing is guessed. The list of
//! keys and what each one means is in `README.md` ("Reason keys").

use crate::shapes::{cards_to_json, combo_key, round2, Reason};
use big2_big2::card::{format_mask, rank_of, FULL_DECK};
use big2_big2::movegen::legal_combos;
use big2_big2::plan::{decompositions, min_plays};
use big2_big2::{classify, Action, Combo, RuleSet, State};
use big2_cards::mask::{highest, lowest};
use big2_games::Game;
use big2_search::MoveStats;
use std::collections::HashMap;

/// Big Two rank index of a 2.
const RANK_TWO: u8 = 12;

fn has_two(cards: u64) -> bool {
    (0..4).any(|s| cards & (1u64 << (RANK_TWO * 4 + s)) != 0)
}

fn count_twos(cards: u64) -> u32 {
    (0..4)
        .filter(|s| cards & (1u64 << (RANK_TWO * 4 + s)) != 0)
        .count() as u32
}

/// Reasons that follow from the position alone (before the move is made).
///
/// `legal` is the full legal-action list for the player to act, so the reasons
/// can contrast the move with its alternatives.
pub fn position_reasons(state: &State, action: Action, legal: &[Action]) -> Vec<Reason> {
    let mut out = Vec::new();
    let Some(player) = state.current_player() else {
        return out;
    };
    if legal.len() == 1 {
        out.push(Reason::new("reason.forced"));
        return out;
    }
    let hand = state.hand(player);
    let plays: Vec<_> = legal
        .iter()
        .filter_map(|a| match a {
            Action::Play(c) => Some(*c),
            Action::Pass => None,
        })
        .collect();

    let combo = match action {
        Action::Pass => {
            // Passing while a play exists: the player keeps those cards for later.
            out.push(Reason::new("reason.pass_holds").arg("playable", plays.len() as u64));
            return out;
        }
        Action::Play(c) => c,
    };

    let left = hand.count_ones() - combo.size();
    if left == 0 {
        out.push(Reason::new("reason.goes_out"));
    } else {
        out.push(Reason::new("reason.cards_left").arg("cards_left", left as u64));
    }

    // No combination of the same size among the cards not yet seen beats this
    // play, so no opponent can beat it (whatever they hold) and the player leads again.
    let unseen = FULL_DECK & !state.played() & !hand;
    let mut beating = Vec::new();
    legal_combos(unseen, Some(&combo), 0, state.rules(), &mut beating);
    if beating.is_empty() && left > 0 {
        out.push(Reason::new("reason.keeps_control"));
    }

    if state.current_play().is_some() {
        // The weakest combination that still beats the current play.
        let min_key = plays.iter().map(|c| c.key).min();
        if min_key == Some(combo.key) && plays.len() > 1 {
            out.push(Reason::new("reason.lowest_beating"));
        }
    } else if let Some(low) = lowest(hand) {
        // Leading with the hand's weakest card, which is the hardest to shed later.
        if combo.cards & (1u64 << low) != 0 {
            out.push(Reason::new("reason.sheds_weakest").arg("card", format_mask(1u64 << low)));
        }
    }

    // Keeps every 2 while some other legal play would spend one.
    if has_two(hand)
        && !has_two(combo.cards)
        && plays
            .iter()
            .any(|c| c.cards != combo.cards && has_two(c.cards))
    {
        out.push(Reason::new("reason.saves_two").arg("twos_kept", count_twos(hand) as u64));
    }

    // Keeps the hand's highest card while some other legal play would spend it.
    if let Some(top) = highest(hand) {
        let top_bit = 1u64 << top;
        if combo.cards & top_bit == 0
            && rank_of(top) != RANK_TWO
            && plays.iter().any(|c| c.cards & top_bit != 0)
        {
            out.push(Reason::new("reason.saves_high_card").arg("card", format_mask(top_bit)));
        }
    }

    // An opponent is close to going out.
    let sizes = state.hand_sizes();
    if let Some((p, n)) = (0..4)
        .filter(|&p| p != player && sizes[p] > 0)
        .map(|p| (p, sizes[p]))
        .min_by_key(|&(_, n)| n)
    {
        if n <= 2 {
            out.push(
                Reason::new("reason.opponent_nearly_out")
                    .arg("player", p as u64)
                    .arg("cards_left", n as u64),
            );
        }
    }
    out
}

/// Reasons from the solver's numbers: the margin of the best move over the
/// runner-up, or how much a move loses against the best.
pub fn solver_reasons(
    this: &MoveStats<Action>,
    best: &MoveStats<Action>,
    second: Option<&MoveStats<Action>>,
) -> Vec<Reason> {
    let mut out = Vec::new();
    if this.action == best.action {
        if let Some(s) = second {
            let margin = best.mean_score - s.mean_score;
            if margin > 0.0 {
                out.push(Reason::new("reason.score_margin").arg("points", round2(margin)));
            }
        }
    } else {
        let loss = best.mean_score - this.mean_score;
        if loss > 0.0 {
            out.push(Reason::new("reason.score_loss").arg("points", round2(loss)));
        }
    }
    out
}

// -- causes and merits: the played move against the best one ------------------

/// Hands larger than this skip the sure-out search (it is exponential in the
/// hand size; ten cards take well under a millisecond).
const SURE_OUT_MAX_CARDS: u32 = 10;

/// What one move does, measured the same way for the played and the best move,
/// so a cause is stated only when it is true of one and false of the other.
struct MoveFacts {
    /// Plays still needed to empty the hand: 1 for a play (0 for a pass) plus
    /// the fewest plays that shed what is left ([`min_plays`]).
    plays_to_out: u32,
    /// A pass, or a play that leaves cards in hand and that some combination
    /// among the unseen cards beats.
    loses_lead: bool,
    /// Combinations among the unseen cards that beat the play (0 for a pass).
    beaters: usize,
    /// The 2s the move spends.
    twos: u64,
    /// The hand's highest card, if the move spends it.
    top_card: Option<u8>,
    /// An opponent who could go out on the next turn because of this move:
    /// `(seat, cards_left)`.
    lets_out: Option<(usize, u32)>,
    /// Singles of the best plan for the cards left that an unseen single beats.
    loose_singles: u32,
    /// The lowest of those loose singles.
    low_single: Option<u8>,
    /// Plays (this one included) to go out with every play but the last one
    /// unbeatable by the unseen cards; `None` if no such line exists.
    sure_out: Option<u32>,
}

/// Combinations among `unseen` that beat `combo`.
fn beaters_of(unseen: u64, combo: &Combo, rules: &RuleSet) -> usize {
    let mut out = Vec::new();
    legal_combos(unseen, Some(combo), 0, rules, &mut out);
    out.len()
}

/// Fewest plays to empty `hand` leading every time, each play but the last
/// one unbeatable by `unseen`. `None` if no such line exists (or the hand is
/// too large to search).
fn sure_out_plays(
    hand: u64,
    unseen: u64,
    rules: &RuleSet,
    memo: &mut HashMap<u64, Option<u32>>,
) -> Option<u32> {
    if hand == 0 {
        return Some(0);
    }
    if hand.count_ones() > SURE_OUT_MAX_CARDS {
        return None;
    }
    if classify(hand, rules).is_some() {
        return Some(1);
    }
    if let Some(&known) = memo.get(&hand) {
        return known;
    }
    let mut leads = Vec::new();
    legal_combos(hand, None, 0, rules, &mut leads);
    let mut best: Option<u32> = None;
    for c in leads {
        if beaters_of(unseen, &c, rules) > 0 {
            continue;
        }
        if let Some(n) = sure_out_plays(hand & !c.cards, unseen, rules, memo) {
            best = Some(best.map_or(n + 1, |b| b.min(n + 1)));
        }
    }
    memo.insert(hand, best);
    best
}

/// Every 2 in the deck.
const TWOS: u64 = 0xFu64 << (RANK_TWO * 4);

fn move_facts(state: &State, player: usize, action: Action) -> MoveFacts {
    let rules = state.rules();
    let hand = state.hand(player);
    let unseen = FULL_DECK & !state.played() & !hand;
    let cards = action.cards();
    let left = hand & !cards;
    let sizes = state.hand_sizes();
    let opponents_from = |seat: usize| (1..4).map(move |k| (seat + k) % 4);

    let (beaters, plays_to_out, lets_out, sure_out) = match action {
        Action::Play(c) => {
            let beaters = if left == 0 {
                0
            } else {
                beaters_of(unseen, &c, rules)
            };
            // An opponent holding exactly as many cards as the play could beat
            // it with their whole hand.
            let lets_out = if left > 0 && beaters > 0 {
                opponents_from(player)
                    .find(|&q| sizes[q] == c.size())
                    .map(|q| (q, sizes[q]))
            } else {
                None
            };
            let sure_out = if left == 0 {
                Some(1)
            } else if beaters == 0 {
                sure_out_plays(left, unseen, rules, &mut HashMap::new()).map(|n| n + 1)
            } else {
                None
            };
            (beaters, 1 + min_plays(left, rules), lets_out, sure_out)
        }
        Action::Pass => {
            let locked = state.locked_out();
            let lets_out = state.current_play().and_then(|(owner, c)| {
                let can_beat = beaters_of(unseen, &c, rules) > 0;
                opponents_from(player)
                    .find(|&q| {
                        sizes[q] > 0
                            && if q == owner {
                                // The trick may come back to them with one card.
                                sizes[q] == 1
                            } else {
                                can_beat && !locked[q] && sizes[q] == c.size()
                            }
                    })
                    .map(|q| (q, sizes[q]))
            });
            (0, min_plays(hand, rules), lets_out, None)
        }
    };

    let top_card = highest(hand).filter(|&t| cards & (1u64 << t) != 0);
    let loose: Vec<Combo> = decompositions(left, rules, 1)
        .first()
        .map(|d| {
            d.combos
                .iter()
                .filter(|g| g.size() == 1 && beaters_of(unseen, g, rules) > 0)
                .copied()
                .collect()
        })
        .unwrap_or_default();
    let low_single = loose.iter().filter_map(|g| lowest(g.cards)).min();

    MoveFacts {
        plays_to_out,
        loses_lead: matches!(action, Action::Pass) || beaters > 0,
        beaters,
        twos: cards & TWOS,
        top_card,
        lets_out,
        loose_singles: loose.len() as u32,
        low_single,
        sure_out,
    }
}

/// The group of the hand's best plan that `cards` takes only part of.
fn broken_group(hand: u64, cards: u64, rules: &RuleSet) -> Option<Combo> {
    decompositions(hand, rules, 1).first().and_then(|d| {
        d.combos
            .iter()
            .find(|g| g.cards & cards != 0 && g.cards & !cards != 0)
            .copied()
    })
}

fn combo_json(c: &Combo) -> serde_json::Value {
    serde_json::json!({
        "cards": cards_to_json(c.cards),
        "combo": combo_key(c.kind),
    })
}

/// Causes and merits of the played move against the best move.
///
/// Returns `(causes, merits)`: `causes` are facts true of `played` and false
/// of `best` (for the played move's `reasons`); `merits` are facts true of
/// `best` and false of `played` (for `best_reasons`). Both are empty when the
/// two moves are the same. The keys are listed in `README.md` ("Causes and
/// merits"); causes come strongest first, so the first one can be the headline.
pub fn contrast_reasons(state: &State, played: Action, best: Action) -> (Vec<Reason>, Vec<Reason>) {
    let mut causes = Vec::new();
    let mut merits = Vec::new();
    let Some(player) = state.current_player() else {
        return (causes, merits);
    };
    if played == best {
        return (causes, merits);
    }
    let rules = state.rules();
    let hand = state.hand(player);
    let p = move_facts(state, player, played);
    let b = move_facts(state, player, best);

    if let (Some((seat, n)), None) = (p.lets_out, b.lets_out) {
        causes.push(
            Reason::new("reason.lets_opponent_out")
                .arg("player", seat as u64)
                .arg("cards_left", n as u64),
        );
    }
    if p.plays_to_out > b.plays_to_out {
        let extra = (p.plays_to_out - b.plays_to_out) as u64;
        match broken_group(hand, played.cards(), rules) {
            Some(g) => causes.push(
                Reason::new("reason.breaks_group")
                    .arg("group", combo_json(&g))
                    .arg("extra_plays", extra),
            ),
            None => causes.push(Reason::new("reason.more_plays_to_out").arg("extra_plays", extra)),
        }
        merits.push(
            Reason::new("reason.fewer_plays_to_out")
                .arg("plays", b.plays_to_out as u64)
                .arg("played_plays", p.plays_to_out as u64),
        );
    }
    if p.loses_lead && !b.loses_lead {
        causes.push(Reason::new("reason.can_be_beaten").arg("beaters", p.beaters as u64));
    }
    if let (None, Some(n)) = (p.sure_out, b.sure_out) {
        causes.push(Reason::new("reason.misses_sure_out").arg("plays", n as u64));
        merits.push(Reason::new("reason.sure_out").arg("plays", n as u64));
    }
    if p.twos != 0 && b.twos == 0 {
        causes.push(Reason::new("reason.spends_two").arg("cards", cards_to_json(p.twos)));
    }
    if let (Some(t), None) = (p.top_card, b.top_card) {
        // A top 2 is already said by `spends_two`.
        if !(rank_of(t) == RANK_TWO && b.twos == 0) {
            causes.push(Reason::new("reason.spends_high_card").arg("card", format_mask(1u64 << t)));
        }
    }
    if p.loose_singles > b.loose_singles {
        causes.push(
            Reason::new("reason.leaves_loose_singles")
                .arg("singles", p.loose_singles as u64)
                .arg("best_singles", b.loose_singles as u64),
        );
    }
    if let Some(low) = p.low_single {
        if b.low_single.is_none_or(|bl| low < bl) {
            causes
                .push(Reason::new("reason.keeps_low_single").arg("card", format_mask(1u64 << low)));
        }
    }
    if let (Action::Pass, Action::Play(_), Some((owner, _))) = (played, best, state.current_play())
    {
        causes.push(
            Reason::new("reason.pass_returns_lead")
                .arg("player", owner as u64)
                .arg("cards_left", state.hand_sizes()[owner] as u64),
        );
    }
    (causes, merits)
}
